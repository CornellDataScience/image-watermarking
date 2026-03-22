import numpy as np
import cv2
from scipy import ndimage
from core.types import SegmentationResult


def segment_image(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel: int = 5,
) -> SegmentationResult:
    """
    Segment an image into regions using Canny edge detection.

    Canny edges are treated as region boundaries. Connected components of
    non-edge pixels are then labeled — everything on the same side of an
    edge boundary becomes one region.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, C).
    low_threshold : int
        Lower hysteresis threshold for Canny (weak edge cutoff).
    high_threshold : int
        Upper hysteresis threshold for Canny (strong edge cutoff).
    blur_kernel : int
        Size of the Gaussian blur kernel applied before edge detection.
        Must be an odd integer. Larger values reduce noise-driven edges.

    Returns
    -------
    SegmentationResult
        region_map : np.ndarray of shape (H, W), dtype int
            Each pixel holds an integer region ID in [0, num_regions - 1].
        num_regions : int
            Total number of distinct regions found.
    """
    # --- Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # --- Canny edge map ---
    # Produces a binary mask: 255 at edges, 0 elsewhere
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # --- Build the non-edge mask ---
    # Pixels that are NOT on an edge boundary can belong to a region.
    # Edge pixels themselves act as dividers between regions.
    non_edge_mask = edges == 0  # True where there is no edge

    # --- Connected components ---
    # Label every connected blob of non-edge pixels as a distinct region.
    # Pixels separated by an edge boundary will receive different labels.
    region_map, num_regions = ndimage.label(non_edge_mask)

    # ndimage.label uses 1-based IDs; shift to 0-based to match the contract.
    # Edge pixels currently hold 0; after the shift they become -1.
    # We reassign them to the nearest labeled neighbor so every pixel has a
    # valid region ID (full coverage — Rule 2 of the interface contract).
    region_map = region_map - 1  # shift: labels now 0-based, edges = -1

    if np.any(region_map == -1):
        # Nearest-neighbor fill: each edge pixel inherits the label of the
        # closest non-edge pixel, effectively "closing" the boundaries.
        distance, nearest_idx = ndimage.distance_transform_edt(
            region_map == -1, return_distances=True, return_indices=True
        )
        region_map[region_map == -1] = region_map[
            tuple(nearest_idx[:, region_map == -1])
        ]

    # Sanity checks (mirror the interface contract rules)
    assert region_map.shape == image.shape[:2], "Shape mismatch"
    assert region_map.min() == 0, "IDs must start at 0"
    assert region_map.max() == num_regions - 1, "IDs must be contiguous"

    return SegmentationResult(region_map.astype(int), num_regions)
