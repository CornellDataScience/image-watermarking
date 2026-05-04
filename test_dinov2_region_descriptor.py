from __future__ import annotations

import numpy as np
import pytest
import torch

from descriptors.dinov2_region_descriptor import (
    DinoV2WholeImageRegionDescriptor,
    ImageMetadata,
    PatchMetadata,
)


class ToyWholeImageDescriptor(DinoV2WholeImageRegionDescriptor):
    """Offline deterministic encoder for descriptor tests.

    It preserves the public pipeline but replaces DINOv2 with fixed dense patch
    embeddings so tests validate mapping, pooling, scalar reduction, and
    determinism without downloading a checkpoint.
    """

    def __init__(self, patch_embeddings: torch.Tensor, **kwargs) -> None:
        super().__init__(
            model_name="toy-dinov2",
            device="cpu",
            input_size=(8, 8),
            patch_size=4,
            **kwargs,
        )
        self._toy_patch_embeddings = patch_embeddings.to(dtype=torch.float32)
        self._model_loaded = True

    def load_model(self):
        self._model_loaded = True
        return None

    def extract_patch_embeddings(self, image):
        _, image_metadata = self.preprocess_image(image)
        grid_h, grid_w, dim = self._toy_patch_embeddings.shape
        patch_metadata = PatchMetadata(
            grid_size=(grid_h, grid_w),
            num_patches=grid_h * grid_w,
            embedding_dim=dim,
            patch_size=4,
            stride=4,
            patch_size_original=(
                4 * image_metadata.original_size[0] / image_metadata.model_input_size[0],
                4 * image_metadata.original_size[1] / image_metadata.model_input_size[1],
            ),
            prefix_tokens_skipped=1,
        )
        return self._toy_patch_embeddings.clone(), patch_metadata, image_metadata


@pytest.fixture()
def toy_patch_embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture()
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)


def quadrant_segmentation() -> np.ndarray:
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[:4, :4] = 10
    seg[:4, 4:] = 20
    seg[4:, :4] = 30
    seg[4:, 4:] = 40
    return seg


def test_basic_functionality_returns_region_and_patch_embeddings(rgb_image, toy_patch_embeddings):
    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="area_weighted_mean",
        patch_assignment_mode="soft_area_weights",
        scalar_mode="fixed_random_projection",
        random_seed=7,
    )
    result = descriptor.describe_image(
        rgb_image,
        quadrant_segmentation(),
        return_patch_embeddings=True,
        return_scalars=True,
    )

    assert result.patch_embeddings is not None
    assert tuple(result.patch_embeddings.shape) == (2, 2, 3)
    assert set(result.region_embeddings) == {10, 20, 30, 40}
    assert all(tuple(emb.shape) == (3,) for emb in result.region_embeddings.values())
    assert result.scalar_descriptors is not None
    assert set(result.scalar_descriptors) == {10, 20, 30, 40}
    assert result.metadata["patches"]["patch_size"] == 4


def test_determinism_repeated_calls_same_inputs(rgb_image, toy_patch_embeddings):
    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="area_weighted_mean",
        patch_assignment_mode="soft_area_weights",
        scalar_mode="fixed_random_projection",
        random_seed=123,
    )

    first = descriptor.describe_image(rgb_image, quadrant_segmentation(), return_scalars=True)
    second = descriptor.describe_image(rgb_image, quadrant_segmentation(), return_scalars=True)

    assert first.region_embeddings.keys() == second.region_embeddings.keys()
    for region_id in first.region_embeddings:
        assert torch.allclose(first.region_embeddings[region_id], second.region_embeddings[region_id])
    assert first.scalar_descriptors == pytest.approx(second.scalar_descriptors)


def test_region_sensitivity_different_regions_differ(rgb_image, toy_patch_embeddings):
    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="area_weighted_mean",
        patch_assignment_mode="soft_area_weights",
    )
    result = descriptor.describe_image(rgb_image, quadrant_segmentation())

    assert not torch.allclose(result.region_embeddings[10], result.region_embeddings[20])
    assert not torch.allclose(result.region_embeddings[10], result.region_embeddings[30])


def test_small_region_majority_assignment_drops_uncovered_tiny_region(rgb_image, toy_patch_embeddings):
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[0, 0] = 99

    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="mean",
        patch_assignment_mode="majority_region",
        small_region_policy="drop",
    )
    result = descriptor.describe_image(rgb_image, seg)

    assert 0 in result.region_embeddings
    assert 99 not in result.region_embeddings
    assert result.metadata["pooling"]["region_status"][99] == "dropped_small_or_uncovered"


def test_nearest_patch_policy_handles_tiny_uncovered_region(rgb_image, toy_patch_embeddings):
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[7, 7] = 99

    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="mean",
        patch_assignment_mode="majority_region",
        small_region_policy="nearest_patch",
    )
    result = descriptor.describe_image(rgb_image, seg, target_region_ids=[99])

    assert 99 in result.region_embeddings
    assert torch.allclose(result.region_embeddings[99], toy_patch_embeddings[1, 1])
    assert result.metadata["pooling"]["region_status"][99] == "nearest_patch"


def test_fixed_random_projection_reproducible_with_same_seed(toy_patch_embeddings):
    region_embeddings = {
        1: torch.tensor([1.0, 2.0, 3.0]),
        2: torch.tensor([4.0, 5.0, 6.0]),
    }
    first = DinoV2WholeImageRegionDescriptor(
        model_name="toy",
        device="cpu",
        input_size=(8, 8),
        patch_size=4,
        scalar_mode="fixed_random_projection",
        random_seed=55,
    )
    second = DinoV2WholeImageRegionDescriptor(
        model_name="toy",
        device="cpu",
        input_size=(8, 8),
        patch_size=4,
        scalar_mode="fixed_random_projection",
        random_seed=55,
    )

    scalars_first = first.reduce_region_embeddings_to_scalars(region_embeddings)
    scalars_second = second.reduce_region_embeddings_to_scalars(region_embeddings)

    assert scalars_first == pytest.approx(scalars_second)


def test_patch_region_mapping_correct_for_left_right_layout(rgb_image, toy_patch_embeddings):
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[:, :4] = 1
    seg[:, 4:] = 2

    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="area_weighted_mean",
        patch_assignment_mode="majority_region",
    )
    _, image_metadata = descriptor.preprocess_image(rgb_image)
    patch_metadata = PatchMetadata(
        grid_size=(2, 2),
        num_patches=4,
        embedding_dim=3,
        patch_size=4,
        stride=4,
        patch_size_original=(4.0, 4.0),
    )

    weights = descriptor.compute_patch_region_weights(seg, image_metadata, patch_metadata)

    assert weights.region_ids == [1, 2]
    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(weights.weights, expected)


def test_soft_mapping_splits_patch_by_exact_area(rgb_image, toy_patch_embeddings):
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[:, :2] = 1
    seg[:, 2:] = 2

    descriptor = ToyWholeImageDescriptor(
        toy_patch_embeddings,
        pooling_mode="area_weighted_mean",
        patch_assignment_mode="soft_area_weights",
    )
    image_metadata = ImageMetadata(
        original_size=(8, 8),
        model_input_size=(8, 8),
        resize_scale_y=1.0,
        resize_scale_x=1.0,
        resize_method="test",
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        grayscale_converted=False,
    )
    patch_metadata = PatchMetadata(
        grid_size=(2, 2),
        num_patches=4,
        embedding_dim=3,
        patch_size=4,
        stride=4,
        patch_size_original=(4.0, 4.0),
    )

    weights = descriptor.compute_patch_region_weights(seg, image_metadata, patch_metadata)

    assert weights.region_ids == [1, 2]
    expected = torch.tensor(
        [
            [0.5, 0.0, 0.5, 0.0],
            [0.5, 1.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(weights.weights, expected)
