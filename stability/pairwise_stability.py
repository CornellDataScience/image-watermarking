import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stability/pairwise_stability.py
#
# PURPOSE
# -------
# Core per-image pairwise stability evaluation.
#
# A watermark is encoded as a set of region-pair inequalities:
#   sign(descriptor(region_i) - descriptor(region_j))  →  1 bit
#
# This module answers: for a given before/after image pair, which of those
# pairwise orderings survive the edit?
#
# It separates three outcomes per pair (i, j):
#
#   "stable"              — both regions survived in the after-image AND the
#                           sign of (d_i - d_j) is the same as before.
#                           The watermark bit carried by this pair survived.
#
#   "descriptor_flip"     — both regions survived (their IoU match was strong
#                           enough), but the sign of (d_i - d_j) reversed.
#                           The bit is wrong. This implicates the descriptor
#                           function, not the segmentation.
#
#   "segmentation_failure"— at least one of the two regions could not be
#                           matched in the after-image (IoU below threshold,
#                           or region count difference). The bit is lost
#                           before the descriptor even matters. This
#                           implicates the segmentation function.
#
# Keeping these three categories separate is the key design decision:
# a combo with high segmentation_failure is a bad segmentation choice;
# a combo with high descriptor_flip is a bad descriptor choice. The two
# failure modes require different fixes, so they must be counted separately.
#
# --------------------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------------------
#
# before_image : np.ndarray  (H, W, 3)  BGR uint8
#     The original image. Segmentation and descriptors will be computed on it.
#
# segment_func : callable(image: np.ndarray) -> SegmentationResult
#     Takes a BGR image, returns a SegmentationResult with:
#       .region_map   : (H, W) int array, each pixel labelled 0..num_regions-1
#       .num_regions  : int
#     Pluggable so the evaluation driver can swap SLIC / k-means / watershed
#     without touching this file.
#
# descriptor_func : callable(image: np.ndarray, seg: SegmentationResult) -> descriptors
#     Returns either scalar descriptors or embedding descriptors keyed/indexed by
#     region. Scalar descriptors use the original sign(d_i - d_j) test.
#     Embedding descriptors use the direction of the pairwise embedding
#     difference vector: (e_i - e_j) before is stable when it remains aligned
#     with (e_i' - e_j') after.
#     Pluggable for the same reason as segment_func.
#
# after_image : np.ndarray | None   (default None)
#     The edited / transformed version of the image (explicit).
#     Provide this when running on real before/after pairs (e.g. FragFake).
#     Exactly one of after_image or transform_fn must be non-None.
#
# transform_fn : callable(image: np.ndarray) -> np.ndarray | None  (default None)
#     A function that derives the after-image from the before-image.
#     Convenience for synthetic transforms (JPEG, blur, etc.) — the caller
#     does not need to apply the transform externally before calling here.
#     Exactly one of after_image or transform_fn must be non-None.
#
# correspondence_map : dict[int, int | None] | None   (default None)
#     Maps before-region index → after-region index (or None if unmatched).
#     If None, match_regions() is called internally to compute it via
#     IoU + Hungarian assignment (see stability/region_matching.py).
#     Pass it explicitly if you have already computed it (e.g. to reuse the
#     same region matching across multiple descriptor combos on the same pair).
#
# image_id : str   (default "")
#     An identifier for this image, threaded into every PairResult.image_id.
#     Required by compute_metrics() to count distinct images when computing
#     usable_pair_yield (stable pairs per image).
#
# iou_threshold : float   (default 0.5)
#     Forwarded to match_regions(). Before-to-after region matches with
#     IoU below this value are treated as unmatched (None), meaning both
#     regions in any pair involving them are tagged segmentation_failure.
#     0.5 is the COCO detection standard; see STABILITY_PLAN §3 for rationale.
#
# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------
#
# list[PairResult]
#     One entry per ordered pair (i, j) with i < j over range(num_regions)
#     in the before-image. Each PairResult carries:
#       .image_id      : str    — the image_id passed in
#       .pair_id       : (i, j) — both indices are before-image region labels
#       .before_margin : float  — |d_i - d_j|  in the before-image
#       .after_margin  : float | None  — |d_{i'} - d_{j'}| in the after-image,
#                        or None if status == "segmentation_failure"
#       .status        : "stable" | "descriptor_flip" | "segmentation_failure"
#
# --------------------------------------------------------------------------
# ALGORITHM (step by step)
# --------------------------------------------------------------------------
#
# Step 1 — Produce the after-image.
#   If after_image is not None, use it directly.
#   If transform_fn is not None, compute after_image = transform_fn(before_image).
#   If both are None, raise ValueError.
#
# Step 2 — Segment both images.
#   seg_before = segment_func(before_image)   →  n regions (0..n-1)
#   seg_after  = segment_func(after_image)    →  m regions (0..m-1)
#   Note: n and m may differ because segmentation is re-run on the edited image.
#   This is intentional — we want to measure segmentation robustness as a
#   first-class property (STABILITY_PLAN §6, open question 3).
#
# Step 3 — Compute descriptors.
#   before_descs = descriptor_func(before_image, seg_before)
#   after_descs  = descriptor_func(after_image,  seg_after)
#   Descriptors may be scalar values or 1-D embedding vectors per region.
#
# Step 4 — Establish region correspondence.
#   If correspondence_map is None:
#     correspondence_map = match_regions(seg_before, seg_after, iou_threshold)
#   This returns a dict {before_idx: after_idx | None} for every before-region.
#   before-regions with no confident IoU match map to None.
#
# Step 5 — Evaluate every pair (i, j) with i < j.
#   For i in range(n):
#     For j in range(i+1, n):
#       i_prime = correspondence_map[i]
#       j_prime = correspondence_map[j]
#
#       If i_prime is None OR j_prime is None:
#         → PairResult(
#               image_id      = image_id,
#               pair_id       = (i, j),
#               before_margin = abs(before_descs[i] - before_descs[j]),
#               after_margin  = None,
#               status        = "segmentation_failure",
#           )
#
#       Else:
#         For scalar descriptors, compare sign(d_i - d_j) before/after.
#         For embedding descriptors, compare cosine alignment between the
#         before and after pair-difference vectors.
#         → PairResult(
#               image_id,
#               (i,j),
#               before_margin,
#               after_margin,
#               status,
#               embedding_cosine_similarity=cosine(before_delta, after_delta),
#           )
#
# Step 6 — Return the full list[PairResult].
#
# --------------------------------------------------------------------------
# NOTES
# --------------------------------------------------------------------------
#
# - Near-zero-margin pairs (before_margin ≈ 0) are recorded without filtering.
#   Filtering belongs to compute_metrics(min_margin_threshold=...) in the
#   aggregation layer, not here. That keeps this function policy-free.
#
# - Pairs where before_sign == 0 (d_i == d_j exactly) will never flip because
#   the sign is 0 both before and after (barring floating-point coincidence).
#   They inflate stable counts slightly but their before_margin is 0, so they
#   will be excluded by any non-zero min_margin_threshold in compute_metrics.
#
# - This function does NOT call compute_metrics. One call to run_stability_test
#   produces raw PairResult objects for one image pair. The driver accumulates
#   these across many pairs and calls compute_metrics once per combo.

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    cv2 = None

from core.types import SegmentationResult
from stability.evaluation_metrics import PairResult
from stability.region_matching import match_regions


DescriptorKind = Literal["scalar", "embedding"]


@dataclass(frozen=True)
class _RegionDescriptors:
    kind: DescriptorKind
    values: dict[int, float | np.ndarray]


def _as_numpy_descriptor(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} must contain numeric descriptor values, got object dtype.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric descriptor values, got dtype {array.dtype}.")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _coerce_single_descriptor(value: Any, *, name: str) -> float | np.ndarray:
    array = _as_numpy_descriptor(value, name=name)
    if array.ndim == 0:
        return float(array.item())
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D embedding vector, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} embedding vector must not be empty.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _extract_descriptor_payload(raw_descriptors: Any) -> Any:
    """Accept plain descriptors or DinoV2DescriptorResult-like objects."""

    scalar_descriptors = getattr(raw_descriptors, "scalar_descriptors", None)
    if scalar_descriptors is not None:
        return scalar_descriptors

    region_embeddings = getattr(raw_descriptors, "region_embeddings", None)
    if region_embeddings is not None:
        return region_embeddings

    return raw_descriptors


def _coerce_region_descriptors(raw_descriptors: Any, *, num_regions: int, label: str) -> _RegionDescriptors:
    descriptors = _extract_descriptor_payload(raw_descriptors)

    if isinstance(descriptors, Mapping):
        values = {
            int(region_id): _coerce_single_descriptor(value, name=f"{label}[{region_id}]")
            for region_id, value in descriptors.items()
        }
    else:
        if hasattr(descriptors, "detach") and callable(descriptors.detach):
            descriptors = descriptors.detach().cpu().numpy()

        array = np.asarray(descriptors)
        if array.dtype != object and np.issubdtype(array.dtype, np.number):
            array = array.astype(np.float64, copy=False)
            if not np.isfinite(array).all():
                raise ValueError(f"{label} contains NaN or infinite values.")
            if array.ndim == 1:
                if array.shape[0] != num_regions:
                    raise ValueError(
                        f"{label} has {array.shape[0]} scalar descriptors; expected {num_regions}."
                    )
                values = {idx: float(array[idx]) for idx in range(num_regions)}
            elif array.ndim == 2:
                if array.shape[0] != num_regions:
                    raise ValueError(
                        f"{label} has {array.shape[0]} embedding descriptors; expected {num_regions}."
                    )
                if array.shape[1] == 0:
                    raise ValueError(f"{label} embedding vectors must not be empty.")
                values = {
                    idx: np.ascontiguousarray(array[idx], dtype=np.float64)
                    for idx in range(num_regions)
                }
            else:
                raise ValueError(
                    f"{label} must be a 1-D scalar array or 2-D embedding array, got shape {array.shape}."
                )
        else:
            sequence = list(descriptors)
            if len(sequence) != num_regions:
                raise ValueError(f"{label} has {len(sequence)} descriptors; expected {num_regions}.")
            values = {
                idx: _coerce_single_descriptor(value, name=f"{label}[{idx}]")
                for idx, value in enumerate(sequence)
            }

    if not values:
        return _RegionDescriptors(kind="scalar", values={})

    has_embeddings = any(not isinstance(value, float) for value in values.values())
    has_scalars = any(isinstance(value, float) for value in values.values())
    if has_embeddings and has_scalars:
        raise ValueError(f"{label} mixes scalar and embedding descriptors.")

    if has_embeddings:
        dims = {int(np.asarray(value).shape[0]) for value in values.values() if not isinstance(value, float)}
        if len(dims) != 1:
            raise ValueError(f"{label} embedding descriptors must all have the same dimension.")
        return _RegionDescriptors(kind="embedding", values=values)

    return _RegionDescriptors(kind="scalar", values=values)


def _scalar_pair_stats(
    before_i: float,
    before_j: float,
    after_i: float,
    after_j: float,
) -> tuple[float, float, str]:
    before_diff = before_i - before_j
    after_diff = after_i - after_j

    before_sign = 1 if before_diff > 0 else (-1 if before_diff < 0 else 0)
    after_sign = 1 if after_diff > 0 else (-1 if after_diff < 0 else 0)

    before_margin = abs(before_diff)
    after_margin = abs(after_diff)
    status = "descriptor_flip" if before_sign != after_sign else "stable"
    return before_margin, after_margin, status


def _embedding_pair_stats(
    before_i: np.ndarray,
    before_j: np.ndarray,
    after_i: np.ndarray,
    after_j: np.ndarray,
    *,
    embedding_similarity_threshold: float,
) -> tuple[float, float, float, str]:
    before_delta = before_i - before_j
    after_delta = after_i - after_j
    before_margin = float(np.linalg.norm(before_delta))
    after_margin = float(np.linalg.norm(after_delta))

    if before_margin == 0.0 and after_margin == 0.0:
        similarity = 1.0
    elif before_margin == 0.0 or after_margin == 0.0:
        similarity = -1.0
    else:
        similarity = float(np.dot(before_delta, after_delta) / (before_margin * after_margin))
        similarity = max(-1.0, min(1.0, similarity))

    status = "stable" if similarity >= embedding_similarity_threshold else "descriptor_flip"
    return before_margin, after_margin, similarity, status


def _before_margin_for_missing_pair(
    descriptors: _RegionDescriptors,
    i: int,
    j: int,
) -> float:
    left = descriptors.values.get(i)
    right = descriptors.values.get(j)
    if left is None or right is None:
        return 0.0
    if descriptors.kind == "scalar":
        return abs(float(left) - float(right))
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def _resize_to_before_shape(after_image: np.ndarray, before_image: np.ndarray) -> np.ndarray:
    h, w = before_image.shape[:2]
    if cv2 is not None:
        return cv2.resize(after_image, (w, h), interpolation=cv2.INTER_LINEAR)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to resize after_image when its shape differs from before_image. "
            "Install opencv-python or provide same-size images."
        ) from exc

    pil_mode = None
    if after_image.ndim == 2:
        pil_mode = "L"
    resized = Image.fromarray(after_image, mode=pil_mode).resize((w, h), Image.BILINEAR)
    return np.asarray(resized, dtype=after_image.dtype)


def run_stability_test(
    before_image: np.ndarray,
    segment_func: Callable,
    descriptor_func: Callable,
    after_image: Optional[np.ndarray] = None,
    transform_fn: Optional[Callable] = None,
    correspondence_map: Optional[dict] = None,
    image_id: str = "",
    iou_threshold: float = 0.5,
    embedding_similarity_threshold: float = 0.0,
    seg_before: Optional[SegmentationResult] = None,
    seg_after: Optional[SegmentationResult] = None,
) -> list[PairResult]:
    # Step 1 — Produce the after-image.
    if after_image is not None and transform_fn is not None:
        raise ValueError("Provide exactly one of after_image or transform_fn, not both.")
    if after_image is None and transform_fn is None:
        raise ValueError("Provide exactly one of after_image or transform_fn.")
    if after_image is None:
        after_image = transform_fn(before_image)

    if after_image.shape[:2] != before_image.shape[:2]:
        after_image = _resize_to_before_shape(after_image, before_image)

    # Step 2 — Segment both images.
    if seg_before is None:
        seg_before = segment_func(before_image)
    if seg_after is None:
        seg_after = segment_func(after_image)

    # Step 3 — Compute descriptors.
    before_descs = _coerce_region_descriptors(
        descriptor_func(before_image, seg_before),
        num_regions=seg_before.num_regions,
        label="before descriptors",
    )
    after_descs = _coerce_region_descriptors(
        descriptor_func(after_image, seg_after),
        num_regions=seg_after.num_regions,
        label="after descriptors",
    )
    if before_descs.kind != after_descs.kind:
        raise ValueError(
            "descriptor_func returned different descriptor kinds before and after: "
            f"{before_descs.kind} vs {after_descs.kind}."
        )

    # Step 4 — Establish region correspondence.
    if correspondence_map is None:
        correspondence_map = match_regions(seg_before, seg_after, iou_threshold)

    # Step 5 — Evaluate every pair (i, j) with i < j.
    results = []
    n = seg_before.num_regions
    for i in range(n):
        for j in range(i + 1, n):
            i_prime = correspondence_map.get(i)
            j_prime = correspondence_map.get(j)

            before_margin = _before_margin_for_missing_pair(before_descs, i, j)

            if (
                i_prime is None
                or j_prime is None
                or i not in before_descs.values
                or j not in before_descs.values
                or i_prime not in after_descs.values
                or j_prime not in after_descs.values
            ):
                results.append(PairResult(
                    image_id=image_id,
                    pair_id=(i, j),
                    before_margin=before_margin,
                    after_margin=None,
                    status="segmentation_failure",
                ))
            else:
                if before_descs.kind == "scalar":
                    before_margin, after_margin, status = _scalar_pair_stats(
                        float(before_descs.values[i]),
                        float(before_descs.values[j]),
                        float(after_descs.values[i_prime]),
                        float(after_descs.values[j_prime]),
                    )
                else:
                    before_margin, after_margin, embedding_cosine_similarity, status = _embedding_pair_stats(
                        np.asarray(before_descs.values[i]),
                        np.asarray(before_descs.values[j]),
                        np.asarray(after_descs.values[i_prime]),
                        np.asarray(after_descs.values[j_prime]),
                        embedding_similarity_threshold=embedding_similarity_threshold,
                    )

                results.append(PairResult(
                    image_id=image_id,
                    pair_id=(i, j),
                    before_margin=before_margin,
                    after_margin=after_margin,
                    status=status,
                    embedding_cosine_similarity=(
                        embedding_cosine_similarity if before_descs.kind == "embedding" else None
                    ),
                ))

    return results
