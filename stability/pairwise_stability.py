import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    """Evaluate per-region descriptor stability across a before/after image pair.

    Parameters
    ----------
    image_id:
        Identifier for the source image.
    seg_before / seg_after:
        Segmentation results for the original and watermarked images.
    before_desc / after_desc:
        Signed scalar margin per region (index matches region id).
        Sign encodes the embedded bit; magnitude encodes confidence.
    correspondence_map:
        Optional pre-computed mapping from before-region id to after-region id
        (or None for unmatched).  When omitted, computed via IoU + Hungarian.

    Returns
    -------
    One PairResult per before-region with status:
      - 'stable'             : region survived and sign is preserved
      - 'descriptor_flip'    : region survived but sign flipped
      - 'segmentation_failure': no matching after-region found (IoU < 0.5)
    """
    if correspondence_map is None:
        correspondence_map = match_regions(seg_before, seg_after)

    results: list[PairResult] = []
    for before_id, after_id in correspondence_map.items():
        before_margin = float(before_desc[before_id])

        if after_id is None:
            results.append(PairResult(
                image_id=image_id,
                pair_id=(before_id, -1),
                before_margin=before_margin,
                after_margin=None,
                status='segmentation_failure',
            ))
        else:
            after_margin = float(after_desc[after_id])
            flipped = (before_margin >= 0) != (after_margin >= 0)
            results.append(PairResult(
                image_id=image_id,
                pair_id=(before_id, after_id),
                before_margin=before_margin,
                after_margin=after_margin,
                status='descriptor_flip' if flipped else 'stable',
            ))

    return results
