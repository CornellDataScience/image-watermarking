"""
test_LBPdescriptor.py
-----------
pytest test suite for lbp_descriptor.lbp_descriptor.

Run with:
    python -m pytest test_lbp.py -v
"""

import numpy as np
import pytest
from skimage.filters import gaussian

from lbp_descriptor import lbp_descriptor


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

P = 8
NUM_BINS = P + 2   # 10 uniform bins for P=8
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------

def test_smoke():
    """Descriptor runs without error on a random grayscale image and returns a 1-D array."""
    image = RNG.integers(0, 256, size=(64, 64), dtype=np.uint8)
    result = lbp_descriptor(image)

    assert isinstance(result, np.ndarray), "result must be a numpy array"
    assert result.ndim == 1, f"result must be 1-D, got ndim={result.ndim}"


# ---------------------------------------------------------------------------
# 2. Output shape test
# ---------------------------------------------------------------------------

def test_output_shape():
    """
    For a 128×128 image with grid_size=4, the descriptor length must equal
    grid_size² × num_bins = 16 × 10 = 160.
    """
    grid_size = 4
    image = RNG.integers(0, 256, size=(128, 128), dtype=np.uint8)
    result = lbp_descriptor(image, grid_size=grid_size, P=P)

    expected_length = grid_size * grid_size * NUM_BINS
    assert result.shape == (expected_length,), (
        f"expected shape ({expected_length},), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# 3. Flat image test
# ---------------------------------------------------------------------------

def test_flat_image():
    """
    A completely uniform image (all same pixel value) produces LBP code = P
    for every pixel (all neighbours equal the centre → all 8 bits set → P ones
    → uniform code P). After normalization each region histogram should have
    bin[P] == 1.0 and all other bins == 0.0.
    """
    image = np.zeros((64, 64), dtype=np.uint8)  # all zeros → LBP code = P
    grid_size = 4
    result = lbp_descriptor(image, grid_size=grid_size, P=P)

    # Reshape into (num_regions, num_bins) so we can inspect each region
    num_regions = grid_size * grid_size
    region_hists = result.reshape(num_regions, NUM_BINS)

    for i, hist in enumerate(region_hists):
        assert np.isclose(hist[P], 1.0, atol=1e-6), (
            f"region {i}: expected bin[{P}]=1.0, got {hist[P]:.6f}"
        )
        non_p_bins = np.concatenate([hist[:P], hist[P + 1:]])
        assert np.allclose(non_p_bins, 0.0, atol=1e-6), (
            f"region {i}: expected all non-P bins to be 0.0, got {non_p_bins}"
        )


# ---------------------------------------------------------------------------
# 4. Distinctness test
# ---------------------------------------------------------------------------

def test_distinctness():
    """
    The descriptor for a smooth (blurred) image must differ meaningfully from
    a noisy/textured image; their L2 distance must exceed a conservative threshold.
    """
    base = RNG.random((128, 128))

    smooth = gaussian(base, sigma=5)

    noise = RNG.random((128, 128)) * 0.5   # strong additive noise
    noisy = np.clip(base + noise, 0.0, 1.0)

    desc_smooth = lbp_descriptor(smooth, grid_size=4, P=P)
    desc_noisy = lbp_descriptor(noisy, grid_size=4, P=P)

    l2_distance = np.linalg.norm(desc_smooth - desc_noisy)
    threshold = 0.1
    assert l2_distance > threshold, (
        f"descriptors are too similar (L2={l2_distance:.4f} ≤ {threshold}); "
        "smooth and noisy images should produce distinct descriptors"
    )


# ---------------------------------------------------------------------------
# 5. Normalization test
# ---------------------------------------------------------------------------

def test_normalization():
    """
    Each region's histogram (a (P+2)-element slice of the descriptor) must sum
    to approximately 1.0, confirming per-region probability normalization.
    """
    image = RNG.integers(0, 256, size=(128, 128), dtype=np.uint8)
    grid_size = 4
    result = lbp_descriptor(image, grid_size=grid_size, P=P)

    num_regions = grid_size * grid_size
    region_hists = result.reshape(num_regions, NUM_BINS)

    region_sums = region_hists.sum(axis=1)
    assert np.allclose(region_sums, 1.0, atol=1e-6), (
        f"region histogram sums should be 1.0; got: {region_sums}"
    )
