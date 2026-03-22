import numpy as np
import cv2
from scipy import ndimage
from core.types import SegmentationResult


def segment_image(
    image: np.ndarray,
    threshold: float = 0.1,
    blur_kernel: int = 5,
) -> SegmentationResult:
    """
    Segment an image into regions using Sobel edge detection.

    Sobel operators compute the image gradient in X and Y directions.
    The gradient magnitude is thresholded to produce edge boundaries.
    Connected components of non-edge pixels are then labeled — everything
    on the same side of an edge boundary becomes one region.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, C).
    threshold : float
        Fraction of the maximum gradient magnitude used as the edge cutoff.
        Range [0, 1]. Higher values keep only the strongest edges, producing
        fewer, larger regions. Lower values keep weaker edges too, producing
        more, smaller regions.
    blur_kernel : int
        Size of the Gaussian blur kernel applied before gradient computation.
        Must be an odd integer. Larger values smooth out noise.

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

    # --- Sobel gradient computation ---
    # Compute the gradient in X and Y using Sobel kernels (ksize=3).
    # cv2.CV_64F keeps the full signed range so negative gradients aren't clipped.
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # horizontal edges
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # vertical edges

    # Gradient magnitude: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # --- Threshold the magnitude to get a binary edge mask ---
    # Pixels with magnitude above (threshold * max) are classified as edges.
    edge_cutoff = threshold * magnitude.max()
    edges = magnitude >= edge_cutoff  # True where there is a strong gradient

    # --- Build the non-edge mask ---
    non_edge_mask = ~edges  # True where there is no edge boundary

    # --- Connected components ---
    # Label every connected blob of non-edge pixels as a distinct region.
    region_map, num_regions = ndimage.label(non_edge_mask)

    # ndimage.label uses 1-based IDs; shift to 0-based to match the contract.
    # Edge pixels currently hold 0; after the shift they become -1.
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
