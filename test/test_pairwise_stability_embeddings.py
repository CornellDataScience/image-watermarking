import numpy as np

from core.types import SegmentationResult
from stability.pairwise_stability import run_stability_test


class _EmbeddingDescriptorResult:
    def __init__(self, region_embeddings):
        self.scalar_descriptors = None
        self.region_embeddings = region_embeddings


def _two_region_segmentation():
    return SegmentationResult(
        region_map=np.array([[0, 1]], dtype=np.int64),
        num_regions=2,
    )


def _unused_segment_func(_image):
    raise AssertionError("precomputed segmentations should be used")


def test_pairwise_stability_preserves_existing_scalar_descriptor_path():
    before = np.zeros((1, 2, 3), dtype=np.uint8)
    after = np.ones((1, 2, 3), dtype=np.uint8)
    seg = _two_region_segmentation()

    def descriptor(image, _seg):
        if image[0, 0, 0] == 0:
            return [2.0, 1.0]
        return [3.0, 1.0]

    results = run_stability_test(
        before,
        segment_func=_unused_segment_func,
        descriptor_func=descriptor,
        after_image=after,
        correspondence_map={0: 0, 1: 1},
        seg_before=seg,
        seg_after=seg,
    )

    assert len(results) == 1
    assert results[0].status == "stable"
    assert results[0].before_margin == 1.0
    assert results[0].after_margin == 2.0
    assert results[0].embedding_cosine_similarity is None


def test_pairwise_stability_accepts_embedding_descriptors_when_direction_is_stable():
    before = np.zeros((1, 2, 3), dtype=np.uint8)
    after = np.ones((1, 2, 3), dtype=np.uint8)
    seg = _two_region_segmentation()

    def descriptor(image, _seg):
        if image[0, 0, 0] == 0:
            return {
                0: np.array([1.0, 0.0]),
                1: np.array([0.0, 1.0]),
            }
        return {
            0: np.array([2.0, 0.0]),
            1: np.array([0.0, 2.0]),
        }

    results = run_stability_test(
        before,
        segment_func=_unused_segment_func,
        descriptor_func=descriptor,
        after_image=after,
        correspondence_map={0: 0, 1: 1},
        seg_before=seg,
        seg_after=seg,
    )

    assert len(results) == 1
    assert results[0].status == "stable"
    assert np.isclose(results[0].before_margin, np.sqrt(2.0))
    assert np.isclose(results[0].after_margin, np.sqrt(8.0))
    assert np.isclose(results[0].embedding_cosine_similarity, 1.0)


def test_pairwise_stability_flags_embedding_descriptor_direction_flip():
    before = np.zeros((1, 2, 3), dtype=np.uint8)
    after = np.ones((1, 2, 3), dtype=np.uint8)
    seg = _two_region_segmentation()

    def descriptor(image, _seg):
        if image[0, 0, 0] == 0:
            return {
                0: np.array([1.0, 0.0]),
                1: np.array([0.0, 1.0]),
            }
        return {
            0: np.array([0.0, 2.0]),
            1: np.array([2.0, 0.0]),
        }

    results = run_stability_test(
        before,
        segment_func=_unused_segment_func,
        descriptor_func=descriptor,
        after_image=after,
        correspondence_map={0: 0, 1: 1},
        seg_before=seg,
        seg_after=seg,
    )

    assert len(results) == 1
    assert results[0].status == "descriptor_flip"
    assert np.isclose(results[0].before_margin, np.sqrt(2.0))
    assert np.isclose(results[0].after_margin, np.sqrt(8.0))
    assert np.isclose(results[0].embedding_cosine_similarity, -1.0)


def test_pairwise_stability_accepts_dinov2_like_descriptor_result_objects():
    before = np.zeros((1, 2, 3), dtype=np.uint8)
    after = np.ones((1, 2, 3), dtype=np.uint8)
    seg = _two_region_segmentation()

    def descriptor(image, _seg):
        if image[0, 0, 0] == 0:
            embeddings = {
                0: np.array([1.0, 0.0]),
                1: np.array([0.0, 1.0]),
            }
        else:
            embeddings = {
                0: np.array([2.0, 0.0]),
                1: np.array([0.0, 2.0]),
            }
        return _EmbeddingDescriptorResult(embeddings)

    results = run_stability_test(
        before,
        segment_func=_unused_segment_func,
        descriptor_func=descriptor,
        after_image=after,
        correspondence_map={0: 0, 1: 1},
        seg_before=seg,
        seg_after=seg,
    )

    assert len(results) == 1
    assert results[0].status == "stable"
    assert np.isclose(results[0].embedding_cosine_similarity, 1.0)
