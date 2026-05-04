import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional
from core.types import SegmentationResult


def compute_iou_matrix(
    seg_before: SegmentationResult,
    seg_after: SegmentationResult,
) -> np.ndarray:
    """Return an (n_before, n_after) matrix of pairwise IoU scores."""
    n = seg_before.num_regions
    m = seg_after.num_regions

    flat_b = seg_before.region_map.ravel()
    flat_a = seg_after.region_map.ravel()

    ids_b = np.arange(n)
    ids_a = np.arange(m)

    masks_b = flat_b[None, :] == ids_b[:, None]   # (n, HW) bool
    masks_a = flat_a[None, :] == ids_a[:, None]   # (m, HW) bool

    intersection = masks_b.astype(np.int32) @ masks_a.astype(np.int32).T  # (n, m)

    sizes_b = masks_b.sum(axis=1)  # (n,)
    sizes_a = masks_a.sum(axis=1)  # (m,)
    union = sizes_b[:, None] + sizes_a[None, :] - intersection  # (n, m)

    return np.where(union > 0, intersection / union, 0.0)


def match_regions(
    seg_before: SegmentationResult,
    seg_after: SegmentationResult,
    iou_threshold: float = 0.5,
) -> Dict[int, Optional[int]]:
    """Hungarian-matched correspondence from before-region ids to after-region ids.

    Matches below iou_threshold are dropped and map to None.
    """
    iou = compute_iou_matrix(seg_before, seg_after)
    row_ind, col_ind = linear_sum_assignment(-iou)

    correspondence: Dict[int, Optional[int]] = {i: None for i in range(seg_before.num_regions)}
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_threshold:
            correspondence[r] = c

    return correspondence