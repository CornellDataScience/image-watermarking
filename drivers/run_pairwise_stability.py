import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# drivers/run_pairwise_stability.py
#
# PURPOSE
# -------
# Dataset-level evaluation driver. Wires together:
#   - A matrix of (segment_func, descriptor_func) combos to evaluate
#   - A set of before/after image pairs
#   - stability/pairwise_stability.py  (per-image loop)
#   - stability/evaluation_metrics.py  (aggregation and reporting)
#
# The output is a ranked table showing which (segmentation, descriptor) combo
# produces the most stable pairwise orderings across edits.
#
# --------------------------------------------------------------------------
# TWO OPERATING MODES
# --------------------------------------------------------------------------
#
# Mode A — Synthetic transforms  [WIRED NOW — use this first]
#   Load N images from IMAGE_DIR.
#   For each image, apply every transform in get_default_transformations().
#   Each (original, transformed) pair becomes one before/after pair.
#   Metadata for each pair: {"image_id": ..., "transform_name": ...}.
#   Stratum key for compute_metrics_by_stratum: "transform_name".
#   This tests the floor: can the combo survive JPEG compression and blur?
#
# Mode B — Real before/after pairs (FragFake or similar)  [STUB — fill in later]
#   Load pairs from a dataset directory structured as:
#     <dataset_dir>/original/<id>.jpg   → before image
#     <dataset_dir>/edited/<id>.jpg     → after image
#   Each pair has metadata: edit_type, severity_level, editor, etc.
#   Stratum key: "severity_level" (or "edit_type").
#   This tests semantic robustness beyond pixel-level noise.
#
# --------------------------------------------------------------------------
# COMBO MATRIX
# --------------------------------------------------------------------------
#
# A combo is a (segment_func, descriptor_func) pair tested together.
# We evaluate the full cross-product: all segment functions × all descriptor
# functions. Each combo gets a human-readable name for reporting.
#
# Example (expand as groups add more implementations):
#   segment_funcs    = [slic_superpixels, k_means, watershed_segmentation]
#   descriptor_funcs = [compute_lbp_descriptor, compute_dwt_descriptor]
#   → 6 combos: "slic_superpixels + dwt_descriptor", etc.
#
# The COMBO_MATRIX list at the top of this file is the place to add/remove
# entries. Format: (name: str, segment_func: callable, descriptor_func: callable)
#
# --------------------------------------------------------------------------
# AGGREGATION AND RANKING
# --------------------------------------------------------------------------
#
# After collecting all PairResult objects for a combo, call:
#   metrics = compute_metrics(all_pair_results, min_margin_threshold=MIN_MARGIN)
#
# This returns an EvaluationMetrics with four key numbers (see STABILITY_PLAN §4):
#   mean_flip_rate              — lower is better. Primary constraint.
#   mean_margin_retention       — closer to 1.0 is better.
#   segmentation_survival_rate  — higher is better.
#   usable_pair_yield           — higher is better. This is the bit budget.
#
# Also call compute_metrics_by_stratum to get a per-transform breakdown, which
# shows whether a combo is good everywhere or only on easy transforms.
#
# Ranking rule (from STABILITY_PLAN §4):
#   Among combos with mean_flip_rate < FLIP_RATE_TARGET, sort by
#   usable_pair_yield descending. Flip rate is the constraint; yield is the
#   objective. A combo with great yield but terrible flip rate is useless.
#
# --------------------------------------------------------------------------
# ALGORITHM (step by step)
# --------------------------------------------------------------------------
#
# Step 1 — Build COMBO_MATRIX.
#   List of (combo_name, segment_func, descriptor_func) tuples.
#   One entry per (segmentation, descriptor) combination to evaluate.
#
# Step 2 — Build IMAGE_PATHS.
#   List of file paths to images for Mode A. In Mode B, this becomes a list
#   of (before_path, after_path, metadata_dict) tuples instead.
#
# Step 3 — Load images and build the pair list.
#   For each image path in IMAGE_PATHS:
#     img = cv2.imread(path)
#     for transform_name, transform_fn in get_default_transformations():
#       image_id = f"{stem}__{transform_name}"  (e.g. "cat__JPEG Q50")
#       pairs.append((img, None, transform_fn, image_id))
#       id_to_stratum[image_id] = transform_name
#
#   pairs is now a list of (before_img, after_img_or_None, transform_fn_or_None, image_id).
#
#   Mode B replaces step 3 with:
#     for before_path, after_path, meta in dataset_pairs:
#       image_id = meta["id"]
#       pairs.append((cv2.imread(before_path), cv2.imread(after_path), None, image_id))
#       id_to_stratum[image_id] = meta["severity_level"]
#
# Step 4 — Main evaluation loop: for each combo, for each pair.
#   results_per_combo = {}
#   For (combo_name, seg_fn, desc_fn) in COMBO_MATRIX:
#     all_pair_results = []
#     For (before_img, after_img, transform_fn, image_id) in pairs:
#       pair_results = run_stability_test(
#           before_image     = before_img,
#           segment_func     = seg_fn,
#           descriptor_func  = desc_fn,
#           after_image      = after_img,       # None in Mode A
#           transform_fn     = transform_fn,    # None in Mode B
#           image_id         = image_id,
#       )
#       all_pair_results.extend(pair_results)
#     results_per_combo[combo_name] = all_pair_results
#
# Step 5 — Aggregate metrics per combo.
#   combo_metrics = {}
#   For combo_name, pair_results in results_per_combo.items():
#     metrics = compute_metrics(pair_results, min_margin_threshold=MIN_MARGIN)
#     stratum_metrics = compute_metrics_by_stratum(
#         pair_results,
#         stratum_key_fn = lambda pr: id_to_stratum[pr.image_id],
#         min_margin_threshold = MIN_MARGIN,
#     )
#     combo_metrics[combo_name] = (metrics, stratum_metrics)
#
# Step 6 — Rank combos.
#   Sort combo_metrics.items() by:
#     primary   : mean_flip_rate < FLIP_RATE_TARGET  (True sorts before False)
#     secondary : usable_pair_yield descending
#
# Step 7 — Print ranked report.
#   For each combo in ranked order:
#     print_metrics_summary(metrics, combo_name=combo_name)
#     Optionally print stratum_metrics table (one row per stratum).
#
# --------------------------------------------------------------------------
# CONSTANTS (tune these before running)
# --------------------------------------------------------------------------
#
# IMAGE_DIR       : directory containing images for Mode A evaluation.
# MIN_MARGIN      : minimum |d_i - d_j| to include a pair in flip-rate
#                   computation. Pairs below this are excluded from the
#                   numerator and denominator of mean_flip_rate.
#                   Start at 0.0 (no filter), then revisit after first run
#                   (STABILITY_PLAN open question 2).
# FLIP_RATE_TARGET: the flip rate ceiling for a combo to be considered "good".
#                   Start at 0.10 (10%). Adjust based on results.
# MAX_IMAGES      : cap on how many images to load from IMAGE_DIR.
#                   Use a small number (e.g. 5) for a quick smoke test, then
#                   raise to 50-500 for real evaluation.

import cv2
import numpy as np
from pathlib import Path

from stability.pairwise_stability import run_stability_test
from stability.evaluation_metrics import compute_metrics, compute_metrics_by_stratum, print_metrics_summary
from stability.transformations import get_default_transformations
from stability.fragfake_loader import iter_fragfake_pairs

from regions.approach_regions import slic_superpixels, k_means, watershed_segmentation
from descriptors.lbp_descriptor import compute_lbp_mean, compute_lbp_entropy, compute_lbp_nonuniform, compute_lbp_edge
from descriptors.dwt_descriptor import compute_dwt_ll, compute_dwt_lh, compute_dwt_hl, compute_dwt_hh


# --------------------------------------------------------------------------
# CONSTANTS — edit these before running
# --------------------------------------------------------------------------

MODE             = 'B'              # 'A' = synthetic transforms, 'B' = FragFake
IMAGE_DIR        = "data/"          # Mode A: directory containing test images
FRAGFAKE_DIR     = "data/fragfake"  # Mode B: root of downloaded FragFake dataset
MIN_MARGIN       = 0.05              # raise after first run to filter near-zero pairs
FLIP_RATE_TARGET = 0.10             # combos above this flip rate are ranked last
MAX_IMAGES       = 10              # Mode A: cap on images loaded from IMAGE_DIR


# --------------------------------------------------------------------------
# COMBO MATRIX — add / remove combos here
# --------------------------------------------------------------------------
# Format: (human-readable name, segment_func, descriptor_func)
# segment_func    : callable(image) -> SegmentationResult
# descriptor_func : callable(image, SegmentationResult) -> list[float]

COMBO_MATRIX = [
    # ("slic + lbp_mean",        slic_superpixels,      compute_lbp_mean),
    # ("slic + lbp_entropy",     slic_superpixels,      compute_lbp_entropy),
    # ("slic + lbp_nonuniform",  slic_superpixels,      compute_lbp_nonuniform),
    # ("slic + lbp_edge",        slic_superpixels,      compute_lbp_edge),
    ("slic + dwt_ll",          slic_superpixels,      compute_dwt_ll),
    ("slic + dwt_lh",          slic_superpixels,      compute_dwt_lh),
    ("slic + dwt_hl",          slic_superpixels,      compute_dwt_hl),
    ("slic + dwt_hh",          slic_superpixels,      compute_dwt_hh),
    # ("kmeans + lbp_mean",      k_means,               compute_lbp_mean),
    # ("kmeans + lbp_entropy",   k_means,               compute_lbp_entropy),
    ("kmeans + dwt_ll",        k_means,               compute_dwt_ll),
    ("kmeans + dwt_hh",        k_means,               compute_dwt_hh),
    # ("watershed + lbp_mean",   watershed_segmentation, compute_lbp_mean),
    # ("watershed + lbp_entropy",watershed_segmentation, compute_lbp_entropy),
    ("watershed + dwt_ll",     watershed_segmentation, compute_dwt_ll),
    ("watershed + dwt_hh",     watershed_segmentation, compute_dwt_hh),
]


# --------------------------------------------------------------------------
# PAIR BUILDERS
# --------------------------------------------------------------------------

def build_synthetic_pairs(image_dir: str, max_images: int):
    """
    Mode A: load images from image_dir, generate one before/after pair per
    image × transform. Returns (pairs, id_to_stratum).

    pairs        : list of (before_img, after_img=None, transform_fn, image_id)
    id_to_stratum: dict mapping image_id -> transform_name
    """
    image_dir = Path(image_dir)
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    transforms = get_default_transformations()
    pairs = []
    id_to_stratum = {}

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARN: failed to load {path}, skipping")
            continue
        stem = path.stem
        for transform_name, transform_fn in transforms:
            image_id = f"{stem}__{transform_name}"
            pairs.append((img, None, transform_fn, image_id))
            id_to_stratum[image_id] = transform_name

    print(f"Mode A: {len(image_paths)} images × {len(transforms)} transforms = {len(pairs)} pairs")
    return pairs, id_to_stratum


def build_fragfake_pairs(fragfake_dir: str):
    """
    Mode B: load real before/after pairs from the FragFake dataset.
    Returns (pairs, id_to_stratum).

    pairs        : list of (before_img, after_img, transform_fn=None, image_id)
    id_to_stratum: dict mapping image_id -> "difficulty__edit_type"
    """
    pairs = []
    id_to_stratum = {}

    for pair in iter_fragfake_pairs(fragfake_dir):
        image_id = pair.image_id
        stratum = f"{pair.difficulty}__{pair.edit_type}"
        pairs.append((pair.before, pair.after, None, image_id))
        id_to_stratum[image_id] = stratum

    if not pairs:
        raise FileNotFoundError(
            f"No pairs found in {fragfake_dir}. "
            "Run drivers/download_fragfake.py first."
        )

    print(f"Mode B: {len(pairs)} FragFake pairs loaded")
    return pairs, id_to_stratum


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    # Step 1 — Build pairs
    if MODE == 'A':
        pairs, id_to_stratum = build_synthetic_pairs(IMAGE_DIR, MAX_IMAGES)
    else:
        pairs, id_to_stratum = build_fragfake_pairs(FRAGFAKE_DIR)

    if not COMBO_MATRIX:
        raise ValueError("COMBO_MATRIX is empty — add at least one combo.")

    # Step 2 — Evaluation loop: for each combo, run on every pair
    combo_metrics = {}
    for combo_name, seg_fn, desc_fn in COMBO_MATRIX:
        print(f"\nEvaluating: {combo_name} ...")
        all_pair_results = []

        for before_img, after_img, transform_fn, image_id in pairs:
            try:
                results = run_stability_test(
                    before_image    = before_img,
                    segment_func    = seg_fn,
                    descriptor_func = desc_fn,
                    after_image     = after_img,
                    transform_fn    = transform_fn,
                    image_id        = image_id,
                )
                all_pair_results.extend(results)
            except Exception as e:
                print(f"  WARN: {combo_name} failed on {image_id}: {e}")
                continue

        if not all_pair_results:
            print(f"  WARN: no results for {combo_name}, skipping")
            continue

        # Step 3 — Aggregate metrics
        metrics = compute_metrics(all_pair_results, min_margin_threshold=MIN_MARGIN)
        stratum_metrics = compute_metrics_by_stratum(
            all_pair_results,
            stratum_key_fn=lambda pr: id_to_stratum[pr.image_id],
            min_margin_threshold=MIN_MARGIN,
        )
        combo_metrics[combo_name] = (metrics, stratum_metrics)

    # Step 4 — Rank: flip_rate < target first, then by usable_pair_yield descending
    ranked = sorted(
        combo_metrics.items(),
        key=lambda x: (x[1][0].mean_flip_rate >= FLIP_RATE_TARGET, -x[1][0].usable_pair_yield),
    )

    # Step 5 — Print ranked report
    print("\n" + "=" * 70)
    print(f"RANKED RESULTS  (flip rate target: {FLIP_RATE_TARGET:.0%})")
    print("=" * 70)

    for rank, (combo_name, (metrics, stratum_metrics)) in enumerate(ranked, 1):
        passed = metrics.mean_flip_rate < FLIP_RATE_TARGET
        status = "PASS" if passed else "FAIL"
        print(f"\n#{rank} [{status}]  {combo_name}")
        print_metrics_summary(metrics, combo_name=combo_name)

        print(f"  Per-stratum breakdown:")
        print(f"  {'Stratum':<30}  {'Flip Rate':>10}  {'Yield':>8}  {'Seg Surv':>10}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*10}")
        for stratum, sm in sorted(stratum_metrics.items()):
            print(
                f"  {stratum:<30}  {sm.mean_flip_rate:>10.2%}  "
                f"{sm.usable_pair_yield:>8.1f}  {sm.segmentation_survival_rate:>10.2%}"
            )


if __name__ == "__main__":
    main()