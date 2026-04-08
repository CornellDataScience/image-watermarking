import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.types import SegmentationResult


def mock_compute_descriptors(image, segmentation):
    """
    Mock descriptor: mean grayscale intensity per region.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C), BGR format.
    segmentation : SegmentationResult

    Returns
    -------
    list[float]
        One descriptor per region.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptors = []

    for r in range(segmentation.num_regions):
        mask = segmentation.region_map == r
        if mask.any():
            descriptors.append(float(gray[mask].mean()))
        else:
            descriptors.append(0.0)

    return descriptors
