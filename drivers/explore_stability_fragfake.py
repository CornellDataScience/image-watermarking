import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import cv2

from regions.approach_regions import slic_superpixels
from descriptors.mock_descriptors import mock_compute_descriptors
from stability.fragfake_loader import iter_fragfake_pairs
from stability.pairwise_stability import (
    run_stability_test, select_stable_pairs, print_stability_report
)


parser = argparse.ArgumentParser(
    description="Run the stability pipeline on one FragFake before/after pair.",
)
parser.add_argument(
    "--image-id", default=None,
    help=(
        "Pick a specific pair by image_id (e.g. 'snowboard_000000096638'). "
        "If omitted, uses the first pair the loader yields. "
        "List available IDs with: "
        "`python -c \"import json; [print(json.loads(l)['image_id']) "
        "for l in open('data/fragfake/manifest.jsonl')]\"`"
    ),
)
args = parser.parse_args()

# Pull one real FragFake before/after pair instead of dog.jpg + JPEG.
# Using the loader keeps this resilient to whichever originals the unique-
# pairing assignment lands on for a given --seed.
pairs = iter_fragfake_pairs("data/fragfake")
if args.image_id is not None:
    pair = next((p for p in pairs if p.image_id == args.image_id), None)
    if pair is None:
        raise SystemExit(
            f"image_id {args.image_id!r} not found in data/fragfake/manifest.jsonl"
        )
else:
    pair = next(pairs)
print(f"using pair: {pair.image_id} ({pair.editor}/{pair.difficulty}/{pair.edit_type})")
before, after = pair.before, pair.after

# run_stability_test expects "transformations" to be functions of (image -> image).
# We already have the after image, so the "transformation" just returns it.
transforms = [(f"FragFake {pair.editor} {pair.difficulty} {pair.edit_type}",
               lambda _im: after)]

result = run_stability_test(
    before,
    segment_func=slic_superpixels,
    descriptor_func=mock_compute_descriptors,
    transformations=transforms,
    segment_func_name="SLIC + mean-gray",
)
print_stability_report(result)

stable = select_stable_pairs(result, max_flip_rate=0.0, min_margin=5.0)
print(f"{len(stable)} stable pairs survived the AI edit")

seg_before = slic_superpixels(before)
seg_after  = slic_superpixels(after)
print(f"before regions: {seg_before.num_regions}, after: {seg_after.num_regions}")

# NOTE: high flip rates here are expected. run_stability_test assumes region
# indices match across before/after, which doesn't hold for AI edits. Region
# matching (STABILITY_PLAN.md §3) is the planned fix; this script just
# demonstrates the existing pipeline running on one real FragFake pair.