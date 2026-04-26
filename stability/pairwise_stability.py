import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, Sequence

from core.types import SegmentationResult
from stability.evaluation_metrics import PairResult
from stability.region_matching import match_regions


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, Sequence

from core.types import SegmentationResult
from stability.evaluation_metrics import PairResult
from stability.region_matching import match_regions


def run_stability_test(
    image_id: str,
    seg_before: SegmentationResult,
    before_desc: Sequence[float],
    seg_after: SegmentationResult,
    after_desc: Sequence[float],
    correspondence_map: Optional[dict[int, Optional[int]]] = None,
) -> list[PairResult]:
    """
    Evaluate per-region descriptor stability across a before/after image pair.

    Now also records unmatched after-regions as 'new_region'.
    """
    if correspondence_map is None:
        correspondence_map = match_regions(seg_before, seg_after)

    results: list[PairResult] = []
    matched_after_ids: set[int] = set()

    # --- Process before -> after matches ---
    for before_id, after_id in correspondence_map.items():
        before_margin = float(before_desc[before_id])

        if after_id is None:
            results.append(
                PairResult(
                    image_id=image_id,
                    pair_id=(before_id, -1),
                    before_margin=before_margin,
                    after_margin=None,
                    status="segmentation_failure",
                )
            )
        else:
            matched_after_ids.add(after_id)
            after_margin = float(after_desc[after_id])
            flipped = (before_margin >= 0) != (after_margin >= 0)

            results.append(
                PairResult(
                    image_id=image_id,
                    pair_id=(before_id, after_id),
                    before_margin=before_margin,
                    after_margin=after_margin,
                    status="descriptor_flip" if flipped else "stable",
                )
            )

    # --- Handle unmatched after-regions ---
    all_after_ids = set(range(seg_after.num_regions))
    unmatched_after = all_after_ids - matched_after_ids

    for after_id in unmatched_after:
        results.append(
            PairResult(
                image_id=image_id,
                pair_id=(-1, after_id),
                before_margin=None,
                after_margin=float(after_desc[after_id]),
                status="new_region",
            )
        )

    return results
