"""
DINOv2 whole-image region descriptors.

This module runs DINOv2 once on a full RGB image, extracts dense patch tokens,
and pools those patch embeddings into descriptors that comply with an input
segmentation map. It is intended for offline watermarking evaluation where
reproducibility and segmentation alignment matter more than throughput.

Patch-to-region mapping
-----------------------
The image is resized once to the DINOv2 model input size. DINOv2 patch tokens
then correspond to non-overlapping patch_size x patch_size rectangles in that
resized image. To map those rectangles back to segmentation regions, this module
projects each token rectangle into the original image coordinate system and
computes exact area overlap against original segmentation pixels, treated as
unit squares.

This avoids a lossy resize of the segmentation map and gives deterministic
soft overlap weights. Hard majority assignment is derived from the same overlap
matrix, with ties resolved by the sorted region ID order. With soft assignment,
``area_weighted_mean`` uses fractional overlap weights, while ``mean`` treats
every overlapping patch as one selected patch.

Assumptions
-----------
- Hugging Face DINOv2 checkpoints use RGB inputs normalized by the associated
  image processor, typically ImageNet mean/std.
- The input size must be divisible by the model patch size. This keeps every
  model input pixel covered by exactly one patch token.
- Output patch embeddings are returned on CPU as float32 tensors to simplify
  downstream reproducibility and serialization.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PoolingMode = Literal["mean", "area_weighted_mean", "max"]
PatchAssignmentMode = Literal["majority_region", "soft_area_weights"]
ScalarMode = Literal[
    "fixed_random_projection",
    "cosine_to_anchor",
    "first_principal_component_projection_given_embeddings",
]
SmallRegionPolicy = Literal["drop", "nearest_patch"]


@dataclass(frozen=True)
class ImageMetadata:
    """Metadata needed to map model patch coordinates back to the image."""

    original_size: tuple[int, int]
    model_input_size: tuple[int, int]
    resize_scale_y: float
    resize_scale_x: float
    resize_method: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    grayscale_converted: bool


@dataclass(frozen=True)
class PatchMetadata:
    """Metadata describing the dense DINOv2 patch token grid."""

    grid_size: tuple[int, int]
    num_patches: int
    embedding_dim: int | None
    patch_size: int
    stride: int
    patch_size_original: tuple[float, float]
    token_layout: str = "row_major_without_cls"
    prefix_tokens_skipped: int = 1


@dataclass
class PatchRegionWeights:
    """Sparse-enough patch-to-region overlap representation.

    Attributes
    ----------
    region_ids:
        Region IDs represented by rows of ``weights``.
    weights:
        Tensor of shape ``(num_regions, num_patches)`` in row-major patch order.
        For ``soft_area_weights``, entries are fractions of each patch covered by
        a region. For ``majority_region``, entries are 0/1 assignments.
    """

    region_ids: list[int]
    weights: torch.Tensor
    patch_assignment_mode: PatchAssignmentMode
    all_present_region_ids: list[int]
    candidate_region_ids: list[int]
    absent_target_region_ids: list[int]
    ignored_region_ids: list[int]
    region_pixel_counts: dict[int, int]

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly metadata for descriptor results."""

        row_sums = self.weights.sum(dim=1).detach().cpu().numpy()
        return {
            "patch_assignment_mode": self.patch_assignment_mode,
            "region_ids": list(self.region_ids),
            "all_present_region_ids": list(self.all_present_region_ids),
            "candidate_region_ids": list(self.candidate_region_ids),
            "absent_target_region_ids": list(self.absent_target_region_ids),
            "ignored_region_ids": list(self.ignored_region_ids),
            "region_pixel_counts": dict(self.region_pixel_counts),
            "region_patch_weight_sums": {
                int(rid): float(row_sums[i]) for i, rid in enumerate(self.region_ids)
            },
            "weights_shape": tuple(int(v) for v in self.weights.shape),
            "patch_order": "row_major",
        }


@dataclass
class DinoV2DescriptorResult:
    """Result returned by :meth:`DinoV2WholeImageRegionDescriptor.describe_image`."""

    region_embeddings: dict[int, torch.Tensor]
    scalar_descriptors: dict[int, float] | None
    patch_embeddings: torch.Tensor | None
    metadata: dict[str, Any]


class DinoV2WholeImageRegionDescriptor:
    """Generate DINOv2 dense patch and segmentation-compliant region descriptors.

    The model is loaded lazily and cached for reuse. Each call to
    :meth:`describe_image` performs one model forward pass for the full image,
    then pools patch tokens according to the provided segmentation map.

    Parameters
    ----------
    model_name:
        Hugging Face checkpoint name or local path. The default is explicit but
        users who need strict checkpoint pinning should also pass
        ``model_revision`` as a commit hash.
    device:
        ``"cuda"``, ``"mps"``, ``"cpu"``, a torch device, or ``None`` for
        auto-detect. Auto-detect prefers CUDA, then Apple Metal/MPS, then CPU.
    pooling_mode:
        ``"mean"``, ``"area_weighted_mean"``, or ``"max"``.
    patch_assignment_mode:
        ``"soft_area_weights"`` for fractional patch/region overlaps or
        ``"majority_region"`` for one region per patch.
    scalar_mode:
        Optional scalar reduction mode. Kept separate from embedding extraction.
    random_seed:
        Seed used for ``fixed_random_projection``.
    use_fp16:
        Enables CUDA autocast for the DINOv2 forward pass only.
    batch_size:
        Reserved for future batched-image extension. The current implementation
        intentionally processes one image at a time.
    input_size:
        Model input size as int or ``(height, width)``. If omitted, the DINOv2
        config image size is used after model loading.
    patch_size:
        Optional patch size override, mainly useful for tests or custom local
        encoders. Real DINOv2 loads this from the checkpoint config.
    allow_grayscale:
        If false, grayscale images raise. If true, grayscale inputs are expanded
        to RGB deterministically.
    ignore_background:
        If true, ``background_id`` is excluded from region descriptors.
    small_region_policy:
        ``"drop"`` omits regions with no meaningful patch coverage and records
        metadata. ``"nearest_patch"`` uses the nearest patch to the region
        centroid.
    min_region_weight:
        Minimum total patch weight needed for normal pooling. Increase this to
        drop very tiny regions under soft assignment.
    scalar_anchor:
        Optional anchor for ``cosine_to_anchor``. If omitted, describe_image uses
        the mean patch embedding for that image.
    model_revision:
        Hugging Face revision or commit hash. Passing a commit hash is the most
        reproducible option.
    local_files_only:
        Passed to Hugging Face loading for offline, cache-only evaluation.
    """

    DEFAULT_MODEL_NAME = "facebook/dinov2-base"
    DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | torch.device | None = None,
        pooling_mode: PoolingMode = "mean",
        patch_assignment_mode: PatchAssignmentMode = "soft_area_weights",
        scalar_mode: ScalarMode | None = None,
        random_seed: int = 0,
        use_fp16: bool = False,
        batch_size: int = 1,
        input_size: int | tuple[int, int] | None = None,
        patch_size: int | None = None,
        allow_grayscale: bool = False,
        ignore_background: bool = False,
        background_id: int = 0,
        small_region_policy: SmallRegionPolicy = "drop",
        min_region_weight: float = 1e-8,
        scalar_anchor: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        model_revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.device = self._resolve_device(device)
        self.pooling_mode = self._validate_choice(
            pooling_mode, "pooling_mode", ("mean", "area_weighted_mean", "max")
        )
        self.patch_assignment_mode = self._validate_choice(
            patch_assignment_mode,
            "patch_assignment_mode",
            ("majority_region", "soft_area_weights"),
        )
        self.scalar_mode = self._validate_optional_choice(
            scalar_mode,
            "scalar_mode",
            (
                "fixed_random_projection",
                "cosine_to_anchor",
                "first_principal_component_projection_given_embeddings",
            ),
        )
        self.random_seed = int(random_seed)
        self.use_fp16 = bool(use_fp16)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.input_size = self._normalize_size_tuple(input_size) if input_size is not None else None
        self.patch_size = int(patch_size) if patch_size is not None else None
        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")

        self.allow_grayscale = bool(allow_grayscale)
        self.ignore_background = bool(ignore_background)
        self.background_id = int(background_id)
        self.small_region_policy = self._validate_choice(
            small_region_policy, "small_region_policy", ("drop", "nearest_patch")
        )
        self.min_region_weight = float(min_region_weight)
        if self.min_region_weight < 0:
            raise ValueError("min_region_weight must be non-negative.")

        self.scalar_anchor = self._optional_1d_tensor(scalar_anchor, "scalar_anchor")

        self._model: Any | None = None
        self._processor: Any | None = None
        self._model_loaded = False
        self._image_mean = self.DEFAULT_IMAGE_MEAN
        self._image_std = self.DEFAULT_IMAGE_STD
        self._resolved_checkpoint_revision: str | None = None
        self._random_projection_cache: dict[int, torch.Tensor] = {}

    def load_model(self) -> Any:
        """Load and cache the DINOv2 model and image processor.

        Returns
        -------
        Any
            The loaded Hugging Face model in eval mode on ``self.device``.

        Notes
        -----
        The resolved checkpoint commit hash is stored in metadata when
        available from Transformers. For strict reproducibility, pass
        ``model_revision`` as a commit hash and run with cached artifacts.
        """

        if self._model_loaded:
            return self._model

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv2 loading requires the 'transformers' package. "
                "Install it alongside torch to use this descriptor."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.model_revision is not None:
            load_kwargs["revision"] = self.model_revision

        self._processor = AutoImageProcessor.from_pretrained(self.model_name, **load_kwargs)
        self._model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        self._model.eval()
        self._model.to(self.device)

        config = getattr(self._model, "config", None)
        config_patch_size = getattr(config, "patch_size", None)
        if config_patch_size is not None:
            self.patch_size = int(config_patch_size)
        if self.patch_size is None:
            self.patch_size = 14

        if self.input_size is None:
            config_image_size = getattr(config, "image_size", 224)
            self.input_size = self._normalize_size_tuple(config_image_size)

        self._validate_input_size_against_patch_size()

        image_mean = getattr(self._processor, "image_mean", None)
        image_std = getattr(self._processor, "image_std", None)
        if image_mean is not None and image_std is not None:
            if len(image_mean) != 3 or len(image_std) != 3:
                raise ValueError(
                    "Expected a 3-channel image processor normalization for RGB DINOv2 inputs."
                )
            self._image_mean = tuple(float(v) for v in image_mean)
            self._image_std = tuple(float(v) for v in image_std)

        self._resolved_checkpoint_revision = getattr(config, "_commit_hash", None) or self.model_revision
        self._model_loaded = True
        return self._model

    def preprocess_image(self, image: Image.Image | np.ndarray | torch.Tensor) -> tuple[torch.Tensor, ImageMetadata]:
        """Validate, resize, and normalize an RGB image for DINOv2.

        Parameters
        ----------
        image:
            PIL RGB image, numpy array ``(H, W, 3)``, or torch tensor in
            unambiguous HWC/CHW RGB layout.

        Returns
        -------
        tuple[torch.Tensor, ImageMetadata]
            A tensor of shape ``(1, 3, input_h, input_w)`` on ``self.device`` and
            metadata needed for segmentation-to-token mapping.
        """

        patch_size = self._get_patch_size()
        input_h, input_w = self._get_input_size()
        self._validate_input_size_against_patch_size(patch_size=patch_size, input_size=(input_h, input_w))

        rgb_float, grayscale_converted = self._image_to_rgb_float_numpy(image)
        original_h, original_w = rgb_float.shape[:2]
        if original_h <= 0 or original_w <= 0:
            raise ValueError("image must have positive spatial dimensions.")

        tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        if (original_h, original_w) != (input_h, input_w):
            tensor = F.interpolate(
                tensor,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        mean = torch.tensor(self._image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._image_std, dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        metadata = ImageMetadata(
            original_size=(int(original_h), int(original_w)),
            model_input_size=(int(input_h), int(input_w)),
            resize_scale_y=float(input_h / original_h),
            resize_scale_x=float(input_w / original_w),
            resize_method="torch.nn.functional.interpolate bilinear antialias=True align_corners=False",
            normalization_mean=self._image_mean,
            normalization_std=self._image_std,
            grayscale_converted=grayscale_converted,
        )
        return tensor.to(self.device), metadata

    def extract_patch_embeddings(
        self, image: Image.Image | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, PatchMetadata, ImageMetadata]:
        """Run DINOv2 once and return dense patch embeddings.

        Returns
        -------
        tuple[torch.Tensor, PatchMetadata, ImageMetadata]
            Patch embeddings with shape ``(grid_h, grid_w, dim)`` on CPU, plus
            patch and image metadata.
        """

        self.load_model()
        pixel_values, image_metadata = self.preprocess_image(image)
        input_h, input_w = image_metadata.model_input_size
        patch_size = self._get_patch_size()
        grid_h = input_h // patch_size
        grid_w = input_w // patch_size
        expected_patches = grid_h * grid_w

        assert self._model is not None
        autocast_enabled = self.use_fp16 and self.device.type == "cuda"
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            outputs = self._model(pixel_values=pixel_values)

        sequence = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise RuntimeError(
                "Expected DINOv2 last_hidden_state with shape (1, sequence_length, dim); "
                f"got {tuple(sequence.shape)}."
            )

        config = getattr(self._model, "config", None)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        prefix_tokens = 1 + num_register_tokens
        patch_tokens = sequence[:, prefix_tokens : prefix_tokens + expected_patches, :]

        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "Could not extract the expected number of DINOv2 patch tokens. "
                f"Expected {expected_patches} tokens for grid {(grid_h, grid_w)}, "
                f"but sequence shape is {tuple(sequence.shape)} with {prefix_tokens} prefix tokens."
            )

        patch_embeddings = (
            patch_tokens.squeeze(0)
            .detach()
            .to(dtype=torch.float32, device="cpu")
            .reshape(grid_h, grid_w, -1)
            .contiguous()
        )
        patch_metadata = PatchMetadata(
            grid_size=(int(grid_h), int(grid_w)),
            num_patches=int(expected_patches),
            embedding_dim=int(patch_embeddings.shape[-1]),
            patch_size=int(patch_size),
            stride=int(patch_size),
            patch_size_original=(
                float(patch_size * image_metadata.original_size[0] / input_h),
                float(patch_size * image_metadata.original_size[1] / input_w),
            ),
            prefix_tokens_skipped=prefix_tokens,
        )
        return patch_embeddings, patch_metadata, image_metadata

    def compute_patch_region_weights(
        self,
        segmentation: Any,
        image_metadata: ImageMetadata,
        patch_metadata: PatchMetadata,
        target_region_ids: Iterable[int] | None = None,
    ) -> PatchRegionWeights:
        """Compute region overlap weights for each DINOv2 patch token.

        The segmentation is kept in original image space. Each model patch cell
        is projected back to original continuous coordinates, then exact overlap
        with original segmentation pixels is accumulated.

        Parameters
        ----------
        segmentation:
            2D integer region map or an object with a ``region_map`` attribute.
        image_metadata:
            Metadata returned by :meth:`preprocess_image`.
        patch_metadata:
            Metadata returned by :meth:`extract_patch_embeddings`.
        target_region_ids:
            Optional subset of original region IDs to return.

        Returns
        -------
        PatchRegionWeights
            Rows correspond to requested region IDs; columns correspond to
            row-major patch tokens.
        """

        seg = self._coerce_segmentation(segmentation)
        original_h, original_w = image_metadata.original_size
        if seg.shape != (original_h, original_w):
            raise ValueError(
                "segmentation shape must match the original image spatial shape. "
                f"Expected {(original_h, original_w)}, got {tuple(seg.shape)}."
            )

        present_values, present_counts = np.unique(seg, return_counts=True)
        all_present = [int(v) for v in present_values.tolist()]
        region_pixel_counts = {
            int(region_id): int(count)
            for region_id, count in zip(present_values.tolist(), present_counts.tolist())
        }

        ignored_region_ids: list[int] = []
        candidate_region_ids = list(all_present)
        if self.ignore_background and self.background_id in candidate_region_ids:
            ignored_region_ids.append(self.background_id)
            candidate_region_ids = [rid for rid in candidate_region_ids if rid != self.background_id]

        if target_region_ids is None:
            requested_region_ids = list(candidate_region_ids)
        else:
            requested_region_ids = []
            seen_requested: set[int] = set()
            for rid in target_region_ids:
                int_rid = int(rid)
                if int_rid not in seen_requested:
                    requested_region_ids.append(int_rid)
                    seen_requested.add(int_rid)
            if self.ignore_background:
                requested_region_ids = [rid for rid in requested_region_ids if rid != self.background_id]

        absent_target_region_ids = [
            rid for rid in requested_region_ids if rid not in all_present or rid in ignored_region_ids
        ]
        output_region_ids = [rid for rid in requested_region_ids if rid in candidate_region_ids]

        candidate_overlap = self._compute_soft_overlap_matrix(
            seg=seg,
            candidate_region_ids=candidate_region_ids,
            image_metadata=image_metadata,
            patch_metadata=patch_metadata,
        )

        if self.patch_assignment_mode == "majority_region":
            assigned = np.zeros_like(candidate_overlap, dtype=np.float32)
            if candidate_overlap.size > 0:
                winners = np.argmax(candidate_overlap, axis=0)
                winner_values = np.max(candidate_overlap, axis=0)
                valid_patch_mask = winner_values > 0
                patch_indices = np.arange(candidate_overlap.shape[1])[valid_patch_mask]
                assigned[winners[valid_patch_mask], patch_indices] = 1.0
            candidate_weights = assigned
        else:
            candidate_weights = candidate_overlap.astype(np.float32, copy=False)

        candidate_index = {rid: idx for idx, rid in enumerate(candidate_region_ids)}
        output_indices = [candidate_index[rid] for rid in output_region_ids]
        if output_indices:
            output_weights = candidate_weights[output_indices]
        else:
            output_weights = np.zeros((0, patch_metadata.num_patches), dtype=np.float32)

        return PatchRegionWeights(
            region_ids=output_region_ids,
            weights=torch.from_numpy(output_weights.astype(np.float32, copy=False)),
            patch_assignment_mode=self.patch_assignment_mode,
            all_present_region_ids=all_present,
            candidate_region_ids=candidate_region_ids,
            absent_target_region_ids=absent_target_region_ids,
            ignored_region_ids=ignored_region_ids,
            region_pixel_counts=region_pixel_counts,
        )

    def pool_region_embeddings(
        self,
        patch_embeddings: torch.Tensor,
        segmentation: Any,
        image_metadata: ImageMetadata,
        patch_metadata: PatchMetadata,
        target_region_ids: Iterable[int] | None = None,
        patch_region_weights: PatchRegionWeights | None = None,
        return_metadata: bool = False,
    ) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[str, Any]]:
        """Pool dense patch embeddings into one embedding per segmentation region.

        Parameters
        ----------
        patch_embeddings:
            Tensor of shape ``(grid_h, grid_w, dim)`` or ``(num_patches, dim)``.
        segmentation:
            2D integer region map or object with ``region_map``.
        image_metadata, patch_metadata:
            Metadata returned by extraction/preprocessing.
        target_region_ids:
            Optional subset of regions.
        patch_region_weights:
            Precomputed patch/region weights. If omitted, they are computed.
        return_metadata:
            If true, also return pooling status metadata.

        Returns
        -------
        dict or tuple
            ``region_id -> embedding``. If ``return_metadata`` is true, a
            ``(mapping, metadata)`` tuple is returned.
        """

        flat_embeddings = self._flatten_patch_embeddings(patch_embeddings, patch_metadata)
        if patch_region_weights is None:
            patch_region_weights = self.compute_patch_region_weights(
                segmentation=segmentation,
                image_metadata=image_metadata,
                patch_metadata=patch_metadata,
                target_region_ids=target_region_ids,
            )

        seg = self._coerce_segmentation(segmentation)
        weights = patch_region_weights.weights.to(dtype=torch.float32, device=flat_embeddings.device)
        region_embeddings: dict[int, torch.Tensor] = {}
        region_status: dict[int, str] = {}
        nearest_patch_assignments: dict[int, int] = {}

        for row_idx, region_id in enumerate(patch_region_weights.region_ids):
            row = weights[row_idx]
            total_weight = float(row.sum().item())
            if total_weight <= self.min_region_weight:
                if self.small_region_policy == "nearest_patch":
                    patch_index = self._nearest_patch_index_for_region(
                        seg=seg,
                        region_id=region_id,
                        image_metadata=image_metadata,
                        patch_metadata=patch_metadata,
                    )
                    if patch_index is not None:
                        region_embeddings[region_id] = flat_embeddings[patch_index].clone()
                        region_status[region_id] = "nearest_patch"
                        nearest_patch_assignments[region_id] = int(patch_index)
                        continue
                region_status[region_id] = "dropped_small_or_uncovered"
                continue

            positive = row > 0
            if self.pooling_mode == "max":
                if not bool(positive.any()):
                    region_status[region_id] = "dropped_small_or_uncovered"
                    continue
                region_embeddings[region_id] = flat_embeddings[positive].max(dim=0).values
            else:
                if self.pooling_mode == "mean":
                    if self.patch_assignment_mode == "soft_area_weights":
                        pool_weights = positive.to(dtype=torch.float32)
                    else:
                        pool_weights = row
                elif self.pooling_mode == "area_weighted_mean":
                    pool_weights = row
                else:
                    raise ValueError(f"Unknown pooling_mode '{self.pooling_mode}'.")

                denom = pool_weights.sum()
                if float(denom.item()) <= 0:
                    region_status[region_id] = "dropped_small_or_uncovered"
                    continue
                region_embeddings[region_id] = (flat_embeddings * pool_weights[:, None]).sum(dim=0) / denom

            region_status[region_id] = "valid"

        for rid in patch_region_weights.absent_target_region_ids:
            region_status[int(rid)] = "absent_or_ignored"

        metadata = {
            "pooling_mode": self.pooling_mode,
            "patch_assignment_mode": self.patch_assignment_mode,
            "small_region_policy": self.small_region_policy,
            "min_region_weight": self.min_region_weight,
            "valid_region_ids": [int(rid) for rid in region_embeddings.keys()],
            "dropped_region_ids": [
                int(rid)
                for rid, status in region_status.items()
                if status in {"dropped_small_or_uncovered", "absent_or_ignored"}
            ],
            "nearest_patch_assignments": nearest_patch_assignments,
            "region_status": region_status,
        }

        if return_metadata:
            return region_embeddings, metadata
        return region_embeddings

    def reduce_region_embeddings_to_scalars(
        self,
        region_embeddings: Mapping[int, torch.Tensor | np.ndarray | Sequence[float]],
        scalar_mode: ScalarMode | None = None,
        anchor_embedding: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        return_metadata: bool = False,
    ) -> dict[int, float] | tuple[dict[int, float], dict[str, Any]]:
        """Reduce each region embedding vector to one scalar descriptor.

        Supported modes
        ---------------
        ``fixed_random_projection``
            Projects every embedding onto one seeded random unit vector. This is
            reproducible for the same seed and embedding dimension.
        ``cosine_to_anchor``
            Computes cosine similarity to an external anchor. If no anchor is
            passed, the mean of the provided region embeddings is used here; in
            :meth:`describe_image`, the default anchor is the mean patch
            embedding for the image.
        ``first_principal_component_projection_given_embeddings``
            Computes PC1 from the current image's region embeddings and projects
            centered embeddings onto it. This is deterministic for a fixed set
            of regions but less comparable when the region set changes.
        """

        mode = self._validate_optional_choice(
            scalar_mode if scalar_mode is not None else self.scalar_mode,
            "scalar_mode",
            (
                "fixed_random_projection",
                "cosine_to_anchor",
                "first_principal_component_projection_given_embeddings",
            ),
        )
        if mode is None:
            raise ValueError(
                "scalar_mode is required to reduce region embeddings. "
                "Pass scalar_mode in __init__ or to this method."
            )

        if not region_embeddings:
            metadata = {"scalar_mode": mode, "num_regions": 0}
            return ({}, metadata) if return_metadata else {}

        region_ids = [int(rid) for rid in region_embeddings.keys()]
        matrix = torch.stack(
            [self._as_1d_float_tensor(region_embeddings[rid], f"region_embeddings[{rid}]") for rid in region_ids],
            dim=0,
        ).to(dtype=torch.float32, device="cpu")
        dim = int(matrix.shape[1])

        scalar_metadata: dict[str, Any] = {
            "scalar_mode": mode,
            "num_regions": len(region_ids),
            "embedding_dim": dim,
        }

        if mode == "fixed_random_projection":
            projection = self._fixed_random_unit_vector(dim)
            values = matrix @ projection
            scalar_metadata.update(
                {
                    "random_seed": self.random_seed,
                    "projection_norm": float(torch.linalg.norm(projection).item()),
                    "reproducibility_note": (
                        "Stable for the same seed and embedding dimension; independent of region set."
                    ),
                }
            )
        elif mode == "cosine_to_anchor":
            if anchor_embedding is None:
                anchor = matrix.mean(dim=0)
                anchor_source = "mean_region_embedding"
            else:
                anchor = self._as_1d_float_tensor(anchor_embedding, "anchor_embedding")
                anchor_source = "provided_anchor"
            if anchor.numel() != dim:
                raise ValueError(
                    f"anchor_embedding dimension {anchor.numel()} does not match region embedding dim {dim}."
                )
            anchor = anchor.to(dtype=torch.float32, device="cpu")
            denom = torch.linalg.norm(matrix, dim=1) * torch.linalg.norm(anchor)
            values = torch.zeros(len(region_ids), dtype=torch.float32)
            valid = denom > 0
            values[valid] = (matrix[valid] @ anchor) / denom[valid]
            scalar_metadata.update(
                {
                    "anchor_source": anchor_source,
                    "anchor_norm": float(torch.linalg.norm(anchor).item()),
                    "reproducibility_note": (
                        "Stable for a fixed external anchor. If derived from image embeddings, "
                        "the anchor can change with edits."
                    ),
                }
            )
        elif mode == "first_principal_component_projection_given_embeddings":
            if matrix.shape[0] == 1:
                values = torch.zeros(1, dtype=torch.float32)
                explained = 0.0
            else:
                centered = matrix - matrix.mean(dim=0, keepdim=True)
                _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                pc1 = vh[0]
                largest_abs_index = int(torch.argmax(torch.abs(pc1)).item())
                if pc1[largest_abs_index] < 0:
                    pc1 = -pc1
                values = centered @ pc1
                total_var = float(torch.sum(singular_values**2).item())
                explained = (
                    float((singular_values[0] ** 2).item() / total_var) if total_var > 0 else 0.0
                )
            scalar_metadata.update(
                {
                    "pc1_explained_variance_ratio": explained,
                    "reproducibility_note": (
                        "Deterministic for the same current-image region embeddings, "
                        "but PC1 changes if the region set changes."
                    ),
                }
            )
        else:
            raise ValueError(f"Unknown scalar_mode '{mode}'.")

        scalars = {rid: float(values[i].item()) for i, rid in enumerate(region_ids)}
        if return_metadata:
            return scalars, scalar_metadata
        return scalars

    def describe_image(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        segmentation: Any,
        return_patch_embeddings: bool = False,
        return_scalars: bool = False,
        target_region_ids: Iterable[int] | None = None,
    ) -> DinoV2DescriptorResult:
        """Run the full whole-image descriptor pipeline.

        Returns region embeddings for all segmentation IDs or for
        ``target_region_ids`` if provided. DINOv2 is run exactly once.
        """

        patch_embeddings, patch_metadata, image_metadata = self.extract_patch_embeddings(image)
        seg = self._coerce_segmentation(segmentation)
        if seg.shape != image_metadata.original_size:
            raise ValueError(
                "segmentation shape must match original image size before descriptor extraction. "
                f"Expected {image_metadata.original_size}, got {tuple(seg.shape)}."
            )

        patch_region_weights = self.compute_patch_region_weights(
            segmentation=seg,
            image_metadata=image_metadata,
            patch_metadata=patch_metadata,
            target_region_ids=target_region_ids,
        )
        pooled = self.pool_region_embeddings(
            patch_embeddings=patch_embeddings,
            segmentation=seg,
            image_metadata=image_metadata,
            patch_metadata=patch_metadata,
            target_region_ids=target_region_ids,
            patch_region_weights=patch_region_weights,
            return_metadata=True,
        )
        region_embeddings, pooling_metadata = pooled

        scalar_descriptors: dict[int, float] | None = None
        scalar_metadata: dict[str, Any] | None = None
        if return_scalars:
            if self.scalar_mode == "cosine_to_anchor" and self.scalar_anchor is None:
                anchor = patch_embeddings.reshape(-1, patch_embeddings.shape[-1]).mean(dim=0)
                anchor_source = "mean_patch_embedding"
            else:
                anchor = self.scalar_anchor
                anchor_source = "provided_anchor" if anchor is not None else None
            reduced = self.reduce_region_embeddings_to_scalars(
                region_embeddings=region_embeddings,
                anchor_embedding=anchor,
                return_metadata=True,
            )
            scalar_descriptors, scalar_metadata = reduced
            if scalar_metadata is not None and anchor_source is not None:
                scalar_metadata["anchor_source"] = anchor_source

        metadata = {
            "image": asdict(image_metadata),
            "patches": asdict(patch_metadata),
            "model": self._model_metadata(),
            "mapping": patch_region_weights.to_metadata(),
            "pooling": pooling_metadata,
            "scalar": scalar_metadata,
            "target_region_ids": None if target_region_ids is None else [int(rid) for rid in target_region_ids],
        }

        return DinoV2DescriptorResult(
            region_embeddings=region_embeddings,
            scalar_descriptors=scalar_descriptors,
            patch_embeddings=patch_embeddings if return_patch_embeddings else None,
            metadata=metadata,
        )

    def _compute_soft_overlap_matrix(
        self,
        seg: np.ndarray,
        candidate_region_ids: Sequence[int],
        image_metadata: ImageMetadata,
        patch_metadata: PatchMetadata,
    ) -> np.ndarray:
        """Compute exact patch/region area fractions in original image space."""

        num_regions = len(candidate_region_ids)
        num_patches = patch_metadata.num_patches
        weights = np.zeros((num_regions, num_patches), dtype=np.float32)
        if num_regions == 0 or num_patches == 0:
            return weights

        region_to_row = {int(region_id): idx for idx, region_id in enumerate(candidate_region_ids)}
        original_h, original_w = image_metadata.original_size
        input_h, input_w = image_metadata.model_input_size
        grid_h, grid_w = patch_metadata.grid_size
        patch_size = patch_metadata.patch_size

        patch_idx = 0
        for gy in range(grid_h):
            y0 = gy * patch_size * original_h / input_h
            y1 = (gy + 1) * patch_size * original_h / input_h
            row_start = max(0, int(math.floor(y0)))
            row_end = min(original_h, int(math.ceil(y1)))
            row_indices = np.arange(row_start, row_end, dtype=np.float64)
            y_overlap = np.minimum(row_indices + 1.0, y1) - np.maximum(row_indices, y0)
            y_overlap = np.clip(y_overlap, 0.0, None)

            for gx in range(grid_w):
                x0 = gx * patch_size * original_w / input_w
                x1 = (gx + 1) * patch_size * original_w / input_w
                col_start = max(0, int(math.floor(x0)))
                col_end = min(original_w, int(math.ceil(x1)))
                col_indices = np.arange(col_start, col_end, dtype=np.float64)
                x_overlap = np.minimum(col_indices + 1.0, x1) - np.maximum(col_indices, x0)
                x_overlap = np.clip(x_overlap, 0.0, None)

                if row_end <= row_start or col_end <= col_start:
                    patch_idx += 1
                    continue

                area = y_overlap[:, None] * x_overlap[None, :]
                total_area = float(area.sum())
                if total_area <= 0:
                    patch_idx += 1
                    continue

                labels = seg[row_start:row_end, col_start:col_end]
                for label in np.unique(labels):
                    row = region_to_row.get(int(label))
                    if row is None:
                        continue
                    weights[row, patch_idx] = float(area[labels == label].sum() / total_area)

                patch_idx += 1

        return weights

    def _nearest_patch_index_for_region(
        self,
        seg: np.ndarray,
        region_id: int,
        image_metadata: ImageMetadata,
        patch_metadata: PatchMetadata,
    ) -> int | None:
        rows, cols = np.nonzero(seg == region_id)
        if rows.size == 0:
            return None

        centroid_y = float(rows.mean() + 0.5)
        centroid_x = float(cols.mean() + 0.5)
        original_h, original_w = image_metadata.original_size
        input_h, input_w = image_metadata.model_input_size
        grid_h, grid_w = patch_metadata.grid_size
        patch_size = patch_metadata.patch_size

        best_index: int | None = None
        best_dist = float("inf")
        patch_idx = 0
        for gy in range(grid_h):
            center_y = (gy + 0.5) * patch_size * original_h / input_h
            for gx in range(grid_w):
                center_x = (gx + 0.5) * patch_size * original_w / input_w
                dist = (center_y - centroid_y) ** 2 + (center_x - centroid_x) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_index = patch_idx
                patch_idx += 1
        return best_index

    def _model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision_requested": self.model_revision,
            "checkpoint_revision_resolved": self._resolved_checkpoint_revision,
            "device": str(self.device),
            "use_fp16": self.use_fp16,
            "batch_size": self.batch_size,
            "checkpoint_pinning_note": (
                "For exact reproducibility, pass model_revision as a Hugging Face commit hash "
                "and use local_files_only=True after caching the checkpoint."
            ),
        }

    def _fixed_random_unit_vector(self, dim: int) -> torch.Tensor:
        cached = self._random_projection_cache.get(dim)
        if cached is not None:
            return cached
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)
        vector = torch.randn(dim, generator=generator, dtype=torch.float32)
        norm = torch.linalg.norm(vector)
        if float(norm.item()) == 0:
            raise RuntimeError("Generated a zero random projection vector, which should be impossible.")
        vector = vector / norm
        self._random_projection_cache[dim] = vector
        return vector

    def _flatten_patch_embeddings(self, patch_embeddings: torch.Tensor, patch_metadata: PatchMetadata) -> torch.Tensor:
        if not isinstance(patch_embeddings, torch.Tensor):
            patch_embeddings = torch.as_tensor(patch_embeddings)
        patch_embeddings = patch_embeddings.detach().to(dtype=torch.float32, device="cpu")
        if patch_embeddings.ndim == 3:
            grid_h, grid_w, _ = patch_embeddings.shape
            if (int(grid_h), int(grid_w)) != patch_metadata.grid_size:
                raise ValueError(
                    f"patch_embeddings grid {tuple(patch_embeddings.shape[:2])} does not match "
                    f"patch metadata grid {patch_metadata.grid_size}."
                )
            flat = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
        elif patch_embeddings.ndim == 2:
            flat = patch_embeddings
        else:
            raise ValueError(
                "patch_embeddings must have shape (grid_h, grid_w, dim) or (num_patches, dim); "
                f"got {tuple(patch_embeddings.shape)}."
            )
        if int(flat.shape[0]) != patch_metadata.num_patches:
            raise ValueError(
                f"patch_embeddings contain {flat.shape[0]} patches, expected {patch_metadata.num_patches}."
            )
        return flat.contiguous()

    def _image_to_rgb_float_numpy(self, image: Image.Image | np.ndarray | torch.Tensor) -> tuple[np.ndarray, bool]:
        grayscale_converted = False

        if isinstance(image, Image.Image):
            if image.mode == "RGB":
                array = np.asarray(image)
            elif image.mode in {"L", "I", "I;16", "F"} and self.allow_grayscale:
                array = np.asarray(image.convert("RGB"))
                grayscale_converted = True
            elif image.mode in {"L", "I", "I;16", "F"}:
                raise ValueError("Received a grayscale PIL image. Set allow_grayscale=True to expand it to RGB.")
            else:
                raise ValueError(
                    f"Expected a PIL RGB image, got mode '{image.mode}'. Convert explicitly before calling."
                )
            return self._normalize_numeric_image_array(array), grayscale_converted

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 2:
                if not self.allow_grayscale:
                    raise ValueError("Received a grayscale tensor. Set allow_grayscale=True to expand it to RGB.")
                tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
                grayscale_converted = True
            elif tensor.ndim == 3:
                if tensor.shape[-1] == 3 and tensor.shape[0] != 3:
                    pass
                elif tensor.shape[0] == 3 and tensor.shape[-1] != 3:
                    tensor = tensor.permute(1, 2, 0)
                elif tensor.shape[-1] == 1 and self.allow_grayscale:
                    tensor = tensor.repeat(1, 1, 3)
                    grayscale_converted = True
                elif tensor.shape[0] == 1 and tensor.shape[-1] != 1 and self.allow_grayscale:
                    tensor = tensor.permute(1, 2, 0).repeat(1, 1, 3)
                    grayscale_converted = True
                else:
                    raise ValueError(
                        "Torch image tensor must have unambiguous RGB layout HxWx3 or 3xHxW. "
                        f"Got shape {tuple(tensor.shape)}."
                    )
            else:
                raise ValueError(f"Torch image tensor must be 2D or 3D, got shape {tuple(tensor.shape)}.")
            return self._normalize_numeric_image_array(tensor.numpy()), grayscale_converted

        array = np.asarray(image)
        if array.ndim == 2:
            if not self.allow_grayscale:
                raise ValueError("Received a grayscale numpy image. Set allow_grayscale=True to expand it to RGB.")
            array = np.repeat(array[:, :, None], 3, axis=2)
            grayscale_converted = True
        elif array.ndim == 3:
            if array.shape[2] == 3:
                pass
            elif array.shape[2] == 1 and self.allow_grayscale:
                array = np.repeat(array, 3, axis=2)
                grayscale_converted = True
            else:
                raise ValueError(
                    "Numpy image array must be RGB channel-last with shape (H, W, 3). "
                    f"Got shape {tuple(array.shape)}."
                )
        else:
            raise ValueError(f"Numpy image array must be 2D or 3D, got shape {tuple(array.shape)}.")
        return self._normalize_numeric_image_array(array), grayscale_converted

    @staticmethod
    def _normalize_numeric_image_array(array: np.ndarray) -> np.ndarray:
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected RGB image array with shape (H, W, 3), got {tuple(array.shape)}.")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Image array must be numeric, got dtype {array.dtype}.")

        arr = array.astype(np.float32, copy=False)
        if not np.isfinite(arr).all():
            raise ValueError("Image array contains NaN or infinite values.")
        min_value = float(arr.min())
        max_value = float(arr.max())
        if min_value < 0:
            raise ValueError("Image values must be non-negative.")
        if np.issubdtype(array.dtype, np.integer):
            if max_value > 255:
                raise ValueError(
                    "Integer RGB inputs are expected in [0, 255]. "
                    f"Observed maximum value {max_value}."
                )
            arr = arr / 255.0
        else:
            if max_value <= 1.0:
                pass
            elif max_value <= 255.0:
                arr = arr / 255.0
            else:
                raise ValueError(
                    "Floating RGB inputs must be in [0, 1] or [0, 255]. "
                    f"Observed maximum value {max_value}."
                )
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _coerce_segmentation(segmentation: Any) -> np.ndarray:
        region_map = getattr(segmentation, "region_map", segmentation)
        if isinstance(region_map, torch.Tensor):
            region_map = region_map.detach().cpu().numpy()
        seg = np.asarray(region_map)
        if seg.ndim != 2:
            raise ValueError(f"segmentation must be a 2D integer map, got shape {tuple(seg.shape)}.")
        if not np.issubdtype(seg.dtype, np.integer):
            raise TypeError(f"segmentation must contain integer region IDs, got dtype {seg.dtype}.")
        return np.ascontiguousarray(seg.astype(np.int64, copy=False))

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is not available.")
        if resolved.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise ValueError("device='mps' was requested but Apple Metal/MPS is not available.")
        return resolved

    @staticmethod
    def _validate_choice(value: str, name: str, choices: Sequence[str]) -> Any:
        if value not in choices:
            raise ValueError(f"{name} must be one of {tuple(choices)}, got '{value}'.")
        return value

    @staticmethod
    def _validate_optional_choice(value: str | None, name: str, choices: Sequence[str]) -> Any:
        if value is None:
            return None
        if value not in choices:
            raise ValueError(f"{name} must be one of {tuple(choices)} or None, got '{value}'.")
        return value

    @staticmethod
    def _normalize_size_tuple(size: int | Sequence[int] | None) -> tuple[int, int]:
        if size is None:
            raise ValueError("size must not be None here.")
        if isinstance(size, int):
            h = w = int(size)
        else:
            if len(size) != 2:
                raise ValueError(f"input_size must be an int or (height, width), got {size}.")
            h, w = int(size[0]), int(size[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"input_size must be positive, got {(h, w)}.")
        return h, w

    def _get_patch_size(self) -> int:
        if self.patch_size is None:
            self.load_model()
        assert self.patch_size is not None
        return self.patch_size

    def _get_input_size(self) -> tuple[int, int]:
        if self.input_size is None:
            self.load_model()
        assert self.input_size is not None
        return self.input_size

    def _validate_input_size_against_patch_size(
        self,
        patch_size: int | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        patch = patch_size if patch_size is not None else self._get_patch_size()
        size = input_size if input_size is not None else self._get_input_size()
        h, w = size
        if h % patch != 0 or w % patch != 0:
            raise ValueError(
                f"input_size {size} must be divisible by patch_size {patch} so segmentation "
                "mapping covers the exact DINOv2 token grid."
            )

    @staticmethod
    def _optional_1d_tensor(
        value: torch.Tensor | np.ndarray | Sequence[float] | None, name: str
    ) -> torch.Tensor | None:
        if value is None:
            return None
        return DinoV2WholeImageRegionDescriptor._as_1d_float_tensor(value, name)

    @staticmethod
    def _as_1d_float_tensor(value: torch.Tensor | np.ndarray | Sequence[float], name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be a 1D vector, got shape {tuple(tensor.shape)}.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return tensor.contiguous()


__all__ = [
    "DinoV2WholeImageRegionDescriptor",
    "DinoV2DescriptorResult",
    "ImageMetadata",
    "PatchMetadata",
    "PatchRegionWeights",
]
