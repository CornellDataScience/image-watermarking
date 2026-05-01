# watermark/centroid_matching.py
#
# Responsible for: finding the correspondence between before-image region IDs
# (stored in the sidecar as centroids) and after-image region IDs (computed
# fresh from SLIC on the edited image).
#
# This is the DECODER-SIDE replacement for the IoU + Hungarian matching used
# in stability/region_matching.py during evaluation.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY CENTROID MATCHING INSTEAD OF IoU
# ─────────────────────────────────────────────────────────────────────────────
#
#   At EVALUATION time (region_matching.py):
#     - Both before and after segmentation maps are available in full.
#     - IoU can be computed for every (before_region, after_region) pair.
#     - Hungarian assignment gives the globally optimal 1-to-1 matching.
#
#   At DECODE time (here):
#     - The original image is NOT available.
#     - The sidecar only stores centroids (cx, cy) for ~300 before-image regions.
#     - The after-image segmentation is computed fresh from SLIC.
#     - We have NO pixel-level overlap information.
#
#   Solution: nearest-centroid lookup.
#     For each before-image region r with sidecar centroid C[r]:
#       find the after-image region s whose centroid D[s] is closest to C[r].
#       match[r] = s  if distance(C[r], D[s]) ≤ centroid_threshold
#       match[r] = None  if distance > centroid_threshold  → treat as erasure
#
#   Why this works:
#     SLIC centroids are spatially stable.  Even when AI editing changes
#     region boundaries or texture, the center of mass of a region moves
#     only a few pixels for background/unedited areas.  Empirically,
#     segmentation survival is 43–88% across all strata (RESULTS_EXTENDED.md),
#     which means the centroid of a surviving region is close to where it was.
#     Regions that DO move a lot are destroyed by the edit — but those register
#     as large centroid drift → distance > threshold → erasure, which RS handles.
#
# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
#
#   centroid_threshold : float = 40.0
#       Maximum Euclidean distance (in pixels) between before-centroid and
#       best-match after-centroid before the match is rejected as an erasure.
#       Design rationale:
#         - SLIC n=200 on a typical 640×480 image → average region ≈ 32×32 px
#         - 40 px ≈ 1.25 region diameters
#         - Large enough to tolerate minor SLIC boundary drift
#         - Small enough to reject wrong-region matches in dense areas
#       This should be tunable via Sidecar.metadata["centroid_threshold"].
#
# ─────────────────────────────────────────────────────────────────────────────
# DATA FLOW
# ─────────────────────────────────────────────────────────────────────────────
#
#   sidecar.centroids : dict[int, (float, float)]
#       Before-image region centroids.  Keys are before-image region IDs.
#
#   after_centroids : dict[int, (float, float)]
#       After-image region centroids.  Keys are after-image region IDs (0..199).
#       Computed by compute_region_centroids(seg_after) (defined below).
#
#   match : dict[int, int | None]
#       Maps each before-image region ID to either:
#         - an after-image region ID (successful match), or
#         - None (centroid drift exceeded threshold → erasure)
#
# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

CENTROID_THRESHOLD_DEFAULT = 40.0   # pixels


def compute_region_centroids(
    seg,          # SegmentationResult — output of slic_superpixels on any image
                  #   seg.region_map: (H, W) int array
                  #   seg.num_regions: int
) -> dict:        # dict[int, tuple[float, float]]
                  #   region_id → (mean_x, mean_y) in pixel coordinates
    # For each region k in range(seg.num_regions):
    #   Find all pixel positions where seg.region_map == k.
    #   ys, xs = np.where(seg.region_map == k)
    #   centroid = (float(xs.mean()), float(ys.mean()))
    #   centroids[k] = centroid
    #
    # This is identical to what the encoder computes in Phase 1 Step 2.
    # At encode time: called on the before-image → stored in sidecar.
    # At decode time: called on the after-image → used for matching.
    #
    # NOTE: x = column index, y = row index.  Be consistent with how the
    # encoder computed before-image centroids so that the Euclidean distance
    # comparison is meaningful.  If encoder uses (col, row), decoder must too.
    pass


def match_regions_by_centroid(
    sidecar_centroids,   # dict[int, tuple[float, float]]
                         #   before-image region ID → (cx, cy)
                         #   loaded from the sidecar
    after_centroids,     # dict[int, tuple[float, float]]
                         #   after-image region ID → (cx, cy)
                         #   computed by compute_region_centroids(seg_after)
    centroid_threshold=CENTROID_THRESHOLD_DEFAULT,  # float, pixels
) -> dict:               # dict[int, int | None]
                         #   before-image region ID → after-image region ID, or None
    # Algorithm:
    #
    # Step 1: Build a lookup structure from after_centroids for fast nearest-neighbor.
    #   Option A (simple, fine for n=200):
    #     For each before-region r, iterate all after-regions s, compute distance.
    #     O(|sidecar_centroids| × |after_centroids|) = O(300 × 200) = 60,000 ops.
    #     Totally acceptable for n=200.
    #   Option B (faster, only if n grows large):
    #     scipy.spatial.cKDTree(list(after_centroids.values()))
    #     Then query for each sidecar centroid.
    #   Use Option A for now; comment where Option B would be inserted.
    #
    # Step 2: For each before-image region r in sidecar_centroids:
    #   cx_r, cy_r = sidecar_centroids[r]
    #   Find s* = argmin_s sqrt((cx_r - cx_s)² + (cy_r - cy_s)²) over all s in after_centroids.
    #   min_dist = that minimum distance.
    #
    # Step 3: Apply threshold.
    #   if min_dist <= centroid_threshold:
    #       match[r] = s*
    #   else:
    #       match[r] = None   # erasure: region moved too far or was destroyed
    #
    # Step 4: Return match dict.
    #
    # IMPORTANT: do NOT enforce 1-to-1 matching here.
    #   Multiple before-regions can match to the same after-region.
    #   This can happen near edit boundaries where SLIC redraws region borders.
    #   It is NOT a correctness problem because:
    #     - Each (r1, r2) pair in a bit position compares e[match[r1]] vs e[match[r2]].
    #     - If match[r1] == match[r2], the comparison gives a tie (e==e), which will
    #       vote incorrectly.  This is an error, not an erasure.
    #     - K=7 majority voting absorbs a small number of such wrong votes.
    #   A future optimization could detect and suppress duplicate-match pairs,
    #   converting them to erasures instead of wrong votes.  Not needed for v1.
    pass
