import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# stability/pairwise_stability.py
#
# PURPOSE
# -------
# Core per-image pairwise stability evaluation.
#
# A watermark is encoded as a set of region-pair inequalities:
#   sign(descriptor(region_i) - descriptor(region_j))  →  1 bit
#
# This module answers: for a given before/after image pair, which of those
# pairwise orderings survive the edit?
#
# It separates three outcomes per pair (i, j):
#
#   "stable"              — both regions survived in the after-image AND the
#                           sign of (d_i - d_j) is the same as before.
#                           The watermark bit carried by this pair survived.
#
#   "descriptor_flip"     — both regions survived (their IoU match was strong
#                           enough), but the sign of (d_i - d_j) reversed.
#                           The bit is wrong. This implicates the descriptor
#                           function, not the segmentation.
#
#   "segmentation_failure"— at least one of the two regions could not be
#                           matched in the after-image (IoU below threshold,
#                           or region count difference). The bit is lost
#                           before the descriptor even matters. This
#                           implicates the segmentation function.
#
# Keeping these three categories separate is the key design decision:
# a combo with high segmentation_failure is a bad segmentation choice;
# a combo with high descriptor_flip is a bad descriptor choice. The two
# failure modes require different fixes, so they must be counted separately.
#
# --------------------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------------------
#
# before_image : np.ndarray  (H, W, 3)  BGR uint8
#     The original image. Segmentation and descriptors will be computed on it.
#
# segment_func : callable(image: np.ndarray) -> SegmentationResult
#     Takes a BGR image, returns a SegmentationResult with:
#       .region_map   : (H, W) int array, each pixel labelled 0..num_regions-1
#       .num_regions  : int
#     Pluggable so the evaluation driver can swap SLIC / k-means / watershed
#     without touching this file.
#
# descriptor_func : callable(image: np.ndarray, seg: SegmentationResult) -> list[float]
#     Returns a list of length num_regions where element k is a single scalar
#     descriptor value for region k. The sign of differences between region
#     descriptor values is what the watermark encodes.
#     Pluggable for the same reason as segment_func.
#
# after_image : np.ndarray | None   (default None)
#     The edited / transformed version of the image (explicit).
#     Provide this when running on real before/after pairs (e.g. FragFake).
#     Exactly one of after_image or transform_fn must be non-None.
#
# transform_fn : callable(image: np.ndarray) -> np.ndarray | None  (default None)
#     A function that derives the after-image from the before-image.
#     Convenience for synthetic transforms (JPEG, blur, etc.) — the caller
#     does not need to apply the transform externally before calling here.
#     Exactly one of after_image or transform_fn must be non-None.
#
# correspondence_map : dict[int, int | None] | None   (default None)
#     Maps before-region index → after-region index (or None if unmatched).
#     If None, match_regions() is called internally to compute it via
#     IoU + Hungarian assignment (see stability/region_matching.py).
#     Pass it explicitly if you have already computed it (e.g. to reuse the
#     same region matching across multiple descriptor combos on the same pair).
#
# image_id : str   (default "")
#     An identifier for this image, threaded into every PairResult.image_id.
#     Required by compute_metrics() to count distinct images when computing
#     usable_pair_yield (stable pairs per image).
#
# iou_threshold : float   (default 0.5)
#     Forwarded to match_regions(). Before-to-after region matches with
#     IoU below this value are treated as unmatched (None), meaning both
#     regions in any pair involving them are tagged segmentation_failure.
#     0.5 is the COCO detection standard; see STABILITY_PLAN §3 for rationale.
#
# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------
#
# list[PairResult]
#     One entry per ordered pair (i, j) with i < j over range(num_regions)
#     in the before-image. Each PairResult carries:
#       .image_id      : str    — the image_id passed in
#       .pair_id       : (i, j) — both indices are before-image region labels
#       .before_margin : float  — |d_i - d_j|  in the before-image
#       .after_margin  : float | None  — |d_{i'} - d_{j'}| in the after-image,
#                        or None if status == "segmentation_failure"
#       .status        : "stable" | "descriptor_flip" | "segmentation_failure"
#
# --------------------------------------------------------------------------
# ALGORITHM (step by step)
# --------------------------------------------------------------------------
#
# Step 1 — Produce the after-image.
#   If after_image is not None, use it directly.
#   If transform_fn is not None, compute after_image = transform_fn(before_image).
#   If both are None, raise ValueError.
#
# Step 2 — Segment both images.
#   seg_before = segment_func(before_image)   →  n regions (0..n-1)
#   seg_after  = segment_func(after_image)    →  m regions (0..m-1)
#   Note: n and m may differ because segmentation is re-run on the edited image.
#   This is intentional — we want to measure segmentation robustness as a
#   first-class property (STABILITY_PLAN §6, open question 3).
#
# Step 3 — Compute descriptors.
#   before_descs = descriptor_func(before_image, seg_before)  →  list of n floats
#   after_descs  = descriptor_func(after_image,  seg_after)   →  list of m floats
#   Element k of each list is the scalar descriptor for region k.
#
# Step 4 — Establish region correspondence.
#   If correspondence_map is None:
#     correspondence_map = match_regions(seg_before, seg_after, iou_threshold)
#   This returns a dict {before_idx: after_idx | None} for every before-region.
#   before-regions with no confident IoU match map to None.
#
# Step 5 — Evaluate every pair (i, j) with i < j.
#   For i in range(n):
#     For j in range(i+1, n):
#       i_prime = correspondence_map[i]
#       j_prime = correspondence_map[j]
#
#       If i_prime is None OR j_prime is None:
#         → PairResult(
#               image_id      = image_id,
#               pair_id       = (i, j),
#               before_margin = abs(before_descs[i] - before_descs[j]),
#               after_margin  = None,
#               status        = "segmentation_failure",
#           )
#
#       Else:
#         before_diff   = before_descs[i] - before_descs[j]
#         after_diff    = after_descs[i_prime] - after_descs[j_prime]
#         before_sign   = +1 if before_diff > 0 else (-1 if before_diff < 0 else 0)
#         after_sign    = +1 if after_diff  > 0 else (-1 if after_diff  < 0 else 0)
#         before_margin = abs(before_diff)
#         after_margin  = abs(after_diff)
#         status        = "descriptor_flip" if before_sign != after_sign else "stable"
#         → PairResult(image_id, (i,j), before_margin, after_margin, status)
#
# Step 6 — Return the full list[PairResult].
#
# --------------------------------------------------------------------------
# NOTES
# --------------------------------------------------------------------------
#
# - Near-zero-margin pairs (before_margin ≈ 0) are recorded without filtering.
#   Filtering belongs to compute_metrics(min_margin_threshold=...) in the
#   aggregation layer, not here. That keeps this function policy-free.
#
# - Pairs where before_sign == 0 (d_i == d_j exactly) will never flip because
#   the sign is 0 both before and after (barring floating-point coincidence).
#   They inflate stable counts slightly but their before_margin is 0, so they
#   will be excluded by any non-zero min_margin_threshold in compute_metrics.
#
# - This function does NOT call compute_metrics. One call to run_stability_test
#   produces raw PairResult objects for one image pair. The driver accumulates
#   these across many pairs and calls compute_metrics once per combo.

import cv2
import numpy as np
from typing import Optional, Callable

from core.types import SegmentationResult
from stability.evaluation_metrics import PairResult
from stability.region_matching import match_regions


def run_stability_test(
    before_image: np.ndarray,
    segment_func: Callable,
    descriptor_func: Callable,
    after_image: Optional[np.ndarray] = None,
    transform_fn: Optional[Callable] = None,
    correspondence_map: Optional[dict] = None,
    image_id: str = "",
    iou_threshold: float = 0.5,
) -> list[PairResult]:
    # Step 1 — Produce the after-image.
    if after_image is not None and transform_fn is not None:
        raise ValueError("Provide exactly one of after_image or transform_fn, not both.")
    if after_image is None and transform_fn is None:
        raise ValueError("Provide exactly one of after_image or transform_fn.")
    if after_image is None:
        after_image = transform_fn(before_image)

    if after_image.shape[:2] != before_image.shape[:2]:
        h, w = before_image.shape[:2]
        after_image = cv2.resize(after_image, (w, h), interpolation=cv2.INTER_LINEAR)

    # Step 2 — Segment both images.
    seg_before = segment_func(before_image)
    seg_after = segment_func(after_image)

    # Step 3 — Compute descriptors.
    before_descs = descriptor_func(before_image, seg_before)
    after_descs = descriptor_func(after_image, seg_after)

    # Step 4 — Establish region correspondence.
    if correspondence_map is None:
        correspondence_map = match_regions(seg_before, seg_after, iou_threshold)

    # Step 5 — Evaluate every pair (i, j) with i < j.
    results = []
    n = seg_before.num_regions
    for i in range(n):
        for j in range(i + 1, n):
            i_prime = correspondence_map[i]
            j_prime = correspondence_map[j]

            before_margin = abs(before_descs[i] - before_descs[j])

            if i_prime is None or j_prime is None:
                results.append(PairResult(
                    image_id=image_id,
                    pair_id=(i, j),
                    before_margin=before_margin,
                    after_margin=None,
                    status="segmentation_failure",
                ))
            else:
                before_diff = before_descs[i] - before_descs[j]
                after_diff = after_descs[i_prime] - after_descs[j_prime]

                before_sign = 1 if before_diff > 0 else (-1 if before_diff < 0 else 0)
                after_sign = 1 if after_diff > 0 else (-1 if after_diff < 0 else 0)

                after_margin = abs(after_diff)
                status = "descriptor_flip" if before_sign != after_sign else "stable"

                results.append(PairResult(
                    image_id=image_id,
                    pair_id=(i, j),
                    before_margin=before_margin,
                    after_margin=after_margin,
                    status=status,
                ))

    return results