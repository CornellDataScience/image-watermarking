import numpy as np
from core.types import SegmentationResult

def mock_segment_image(image, grid_size=8):
    """
    Mock segmentation used for development.

    Divides the image into a grid of equal-sized regions.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C)

    grid_size : int
        Number of regions along each dimension

    Returns
    -------
    region_map : np.ndarray
        (H, W) array assigning region IDs

    num_regions : int
        Total number of regions
    """

    H, W = image.shape[:2]

    region_map = np.zeros((H, W), dtype=int)

    region_id = 0

    for i in range(grid_size):
        for j in range(grid_size):

            y0 = i * H // grid_size
            y1 = (i + 1) * H // grid_size

            x0 = j * W // grid_size
            x1 = (j + 1) * W // grid_size

            region_map[y0:y1, x0:x1] = region_id

            region_id += 1

    num_regions = region_id

    return SegmentationResult(region_map, num_regions)