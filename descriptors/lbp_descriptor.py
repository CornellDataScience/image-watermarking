"""
lbp_descriptor.py
-----------------
Implements a Local Binary Pattern (LBP) texture descriptor for grayscale images.
The image is divided into a grid of non-overlapping regions; each region contributes
a normalized histogram of uniform LBP codes. All region histograms are concatenated
into a single feature vector.

Dependencies: numpy, scikit-image
"""

import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern


# ---------------------------------------------------------------------------
# Core descriptor function
# ---------------------------------------------------------------------------

def lbp_descriptor(
    image: np.ndarray,
    grid_size: int = 4,
    P: int = 8,
    R: float = 1.0,
) -> np.ndarray:
    """
    Compute a concatenated LBP histogram descriptor for a grayscale image.

    The image is divided into a ``grid_size × grid_size`` grid of non-overlapping
    regions.  For each region a normalized histogram of uniform LBP codes is
    computed, then all histograms are concatenated into one 1-D feature vector
    of length ``grid_size² × (P + 2)``.

    Parameters
    ----------
    image : np.ndarray
        Input image.  Either a 2-D grayscale array (H×W) or a 3-D RGB array
        (H×W×3).  uint8 or float values are both accepted.
    grid_size : int, optional
        Number of rows *and* columns to divide the image into.  Default is 4,
        producing a 4×4 = 16 region grid.
    P : int, optional
        Number of circularly-symmetric neighbour points for LBP.  Default is 8.
    R : float, optional
        Radius of the circle of neighbours.  Default is 1.0.

    Returns
    -------
    np.ndarray, shape (grid_size * grid_size * (P + 2),)
        Concatenated normalized LBP histograms.  Each block of ``(P + 2)`` values
        is a per-region histogram that sums to 1.

    Notes
    -----
    With ``method='uniform'`` the LBP codes are:

    * **0 … P** — the P+1 uniform patterns (≤ 2 circular 0↔1 transitions),
      indexed by the number of '1' bits in the pattern.
    * **P+1** — catch-all bin for all non-uniform patterns.

    So ``num_bins = P + 2``.
    """

    # Step 1 — Convert to grayscale float64 in [0, 1].
    # local_binary_pattern expects a 2-D array; rgb2gray also handles the
    # luminance conversion (0.2126 R + 0.7152 G + 0.0722 B).
    if image.ndim == 3:
        gray = color.rgb2gray(image.astype(np.float64))
    else:
        gray = image.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0

    # Step 2 — Compute the full-image LBP map.
    # method='uniform' maps each pixel to an integer in [0, P+1]:
    #   codes 0..P  → uniform patterns (≤ 2 transitions), value = count of 1-bits
    #   code  P+1   → all non-uniform patterns (> 2 transitions)
    lbp_image = local_binary_pattern(gray, P, R, method='uniform')

    num_bins = P + 2  # uniform codes 0..P  plus one non-uniform bin

    # Step 3 — Tile the LBP map into a grid_size × grid_size grid.
    # np.array_split divides rows (then columns) into grid_size strips as evenly
    # as possible; any leftover pixels are folded into the last strip.
    row_strips = np.array_split(lbp_image, grid_size, axis=0)

    # Step 4 — For each region, build a normalized LBP histogram.
    histograms: list[np.ndarray] = []
    for row_strip in row_strips:
        col_strips = np.array_split(row_strip, grid_size, axis=1)
        for region in col_strips:
            # Histogram over integer bins [0, 1, …, P+1].
            # np.histogram bin edges: [0, 1, 2, …, P+2] → P+2 bins total.
            hist, _ = np.histogram(region, bins=num_bins, range=(0, num_bins))

            # Normalize by the number of pixels so the histogram sums to 1.
            # This makes the descriptor invariant to region size differences
            # caused by integer rounding when grid_size does not divide evenly.
            hist = hist.astype(np.float64)
            pixel_count = region.size
            if pixel_count > 0:
                hist /= pixel_count

            histograms.append(hist)

    # Step 5 — Concatenate all region histograms into one feature vector.
    # Final shape: (grid_size² × num_bins,)
    descriptor = np.concatenate(histograms)

    return descriptor


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Quick sanity-check demo using scikit-image's built-in test images."""
    from skimage import data as skdata
    import argparse

    parser = argparse.ArgumentParser(description="LBP descriptor demo.")
    parser.add_argument(
        "--grid-size", type=int, default=4, metavar="G",
        help="Divide the image into G×G regions (default: 4).",
    )
    args = parser.parse_args()

    images = {
        "camera (textured)": skdata.camera(),      # classic grayscale test image
        "coins (structured)": skdata.coins(),       # coins on plain background
    }

    g = args.grid_size
    num_bins = 8 + 2  # P=8 uniform

    print(f"Grid: {g}×{g}  |  descriptor length: {g * g * num_bins}\n")

    for name, img in images.items():
        desc = lbp_descriptor(img, grid_size=g)
        print(f"  {name:<28}  shape={desc.shape}  min={desc.min():.4f}  max={desc.max():.4f}")


def compute_lbp_descriptor(image, seg, stat='mean') -> list[float]:
    """
    Region-level LBP descriptor compatible with the evaluation interface.

    Computes the full-image LBP map once, then summarises each region's
    LBP codes into a single scalar using the chosen statistic.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) or (H, W)  uint8
    seg   : SegmentationResult
    stat  : str
        Summary statistic to use per region:
        'mean'       — mean LBP code value
        'entropy'    — Shannon entropy of the region's LBP histogram
        'nonuniform' — fraction of pixels with a non-uniform LBP code (code == P+1)
        'edge'       — fraction of pixels with an edge-like code (1 <= code <= P-1)

    Returns
    -------
    list[float] of length seg.num_regions
    """
    P = 8
    num_bins = P + 2  # codes 0..P plus non-uniform catch-all at P+1

    if image.ndim == 3:
        gray = color.rgb2gray(image.astype(np.float64))
    else:
        gray = image.astype(np.float64)
        if gray.max() > 1.0:
            gray /= 255.0

    lbp_image = local_binary_pattern((gray * 255).astype(np.uint8), P, R=1.0, method='uniform')

    descriptors = []
    for k in range(seg.num_regions):
        mask = seg.region_map == k
        region_lbp = lbp_image[mask]

        # Guard: watershed can produce label IDs with zero pixels assigned.
        if len(region_lbp) == 0:
            descriptors.append(0.0)
            continue

        if stat == 'mean':
            scalar = float(region_lbp.mean())

        elif stat == 'entropy':
            hist, _ = np.histogram(region_lbp, bins=num_bins, range=(0, num_bins))
            hist = hist / hist.sum()
            scalar = float(-np.sum(hist * np.log(hist + 1e-10)))

        elif stat == 'nonuniform':
            scalar = float((region_lbp == P + 1).sum() / len(region_lbp))

        elif stat == 'edge':
            scalar = float(((region_lbp >= 1) & (region_lbp <= P - 1)).sum() / len(region_lbp))

        else:
            raise ValueError(f"Unknown stat '{stat}'. Choose from: mean, entropy, nonuniform, edge.")

        descriptors.append(scalar)

    return descriptors


def compute_lbp_mean(image, seg) -> list[float]:
    return compute_lbp_descriptor(image, seg, stat='mean')

def compute_lbp_entropy(image, seg) -> list[float]:
    return compute_lbp_descriptor(image, seg, stat='entropy')

def compute_lbp_nonuniform(image, seg) -> list[float]:
    return compute_lbp_descriptor(image, seg, stat='nonuniform')

def compute_lbp_edge(image, seg) -> list[float]:
    return compute_lbp_descriptor(image, seg, stat='edge')


if __name__ == "__main__":
    main()
