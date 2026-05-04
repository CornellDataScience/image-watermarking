# watermark/centroid_matching.py
#
# Responsible for: finding the correspondence between before-image region IDs
# (stored in the sidecar as centroids) and after-image region IDs (computed
# fresh from SLIC on the edited image).
#
# This is the DECODER-SIDE replacement for the IoU + Hungarian matching used
# in stability/region_matching.py during evaluation.
#
# At EVALUATION time (region_matching.py):
#   - Both before and after segmentation maps are available in full.
#   - IoU can be computed for every (before_region, after_region) pair.
#
# At DECODE time (here):
#   - The original image is NOT available.
#   - The sidecar only stores centroids (cx, cy) for referenced before-image regions.
#   - We find the nearest after-image centroid for each sidecar centroid.
#   - If drift > centroid_threshold pixels → None (erasure), handled by RS.
#
# IMPORTANT: 1-to-1 matching is NOT enforced.
#   Multiple before-regions can match to the same after-region near edit boundaries.
#   match[r1] == match[r2] means the pair casts a wrong vote (not an erasure).
#   K=7 majority voting absorbs a small number of such collisions.

import numpy as np

CENTROID_THRESHOLD_DEFAULT = 40.0   # pixels


def compute_region_centroids(seg) -> dict:
    """
    Compute the (x, y) centroid of every region in a segmentation result.

    Parameters
    ----------
    seg : SegmentationResult
        .region_map  : (H, W) int array, region labels 0..num_regions-1
        .num_regions : int

    Returns
    -------
    dict[int, tuple[float, float]]
        region_id → (mean_col, mean_row) in pixel coordinates.
        x = column index (horizontal), y = row index (vertical).
    """
    centroids = {}
    for k in range(seg.num_regions):
        ys, xs = np.where(seg.region_map == k)
        if len(xs) == 0:
            # Empty region (can occur with watershed on transformed images).
            centroids[k] = (0.0, 0.0)
        else:
            centroids[k] = (float(xs.mean()), float(ys.mean()))
    return centroids


def match_regions_by_centroid(
    sidecar_centroids,          # dict[int, tuple[float, float]] — before-image centroids from sidecar
    after_centroids,            # dict[int, tuple[float, float]] — after-image centroids (fresh from SLIC)
    centroid_threshold=CENTROID_THRESHOLD_DEFAULT,
) -> dict:
    """
    For each before-image region in the sidecar, find the nearest after-image
    region by Euclidean centroid distance.

    Parameters
    ----------
    sidecar_centroids : dict[int, (float, float)]
        Before-image region ID → (cx, cy).  Loaded from the sidecar.
    after_centroids : dict[int, (float, float)]
        After-image region ID → (cx, cy).  Computed by compute_region_centroids.
    centroid_threshold : float
        Max drift in pixels before treating the region as an erasure (None).

    Returns
    -------
    dict[int, int | None]
        before-region ID → after-region ID, or None if drift > threshold.
    """
    if not after_centroids:
        return {r: None for r in sidecar_centroids}

    # Build vectorized arrays from after_centroids for batch distance computation.
    # Option B (cKDTree) would slot in here for large n; O(300×200) is fine for n=200.
    after_ids = list(after_centroids.keys())
    after_xy = np.array([after_centroids[s] for s in after_ids], dtype=np.float64)  # (n_after, 2)

    match = {}
    for r, (cx_r, cy_r) in sidecar_centroids.items():
        dists = np.sqrt((after_xy[:, 0] - cx_r) ** 2 + (after_xy[:, 1] - cy_r) ** 2)
        best_idx = int(np.argmin(dists))
        min_dist = float(dists[best_idx])
        match[r] = after_ids[best_idx] if min_dist <= centroid_threshold else None

    return match