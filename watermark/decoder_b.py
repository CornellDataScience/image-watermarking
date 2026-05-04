# watermark/decoder_b.py
#
# Option 2 (Design B) decoding pipeline.
# Structurally mirrors decoder.py (Option 3) but uses IoU + Hungarian matching
# (from stability/region_matching.py) instead of centroid proximity.
#
# IoU matching is more accurate than centroid proximity because it checks
# whether before/after regions have substantial pixel overlap, rather than
# just whether their centers are close.  This matters on heavy AI edits
# where region centroids can shift significantly.
#
# Pipeline:
#   edited_image + SidecarB
#     → read params from sidecar.metadata
#     → SLIC segmentation on edited image    (regions/approach_regions.py)
#     → after-image DWT-LL descriptors       (descriptors/dwt_descriptor.py)
#     → IoU + Hungarian region matching      (stability/region_matching.py)
#     → majority vote per bit position
#     → RS decode with explicit erasures     (watermark/reed_solomon.py)
#     → return recovered message bytes

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.types import SegmentationResult
from regions.approach_regions import slic_superpixels
from descriptors.dwt_descriptor import compute_raw_dwt_ll
from stability.region_matching import match_regions
from watermark.reed_solomon import ERASURE, bits_to_bytes_with_erasures, rs_decode_bytes


class DecodeErrorB(Exception):
    """Raised when Reed-Solomon cannot recover the message (edit too destructive)."""
    pass


def decode_watermark_b(edited_image, sidecar) -> bytes:
    """
    Option 2 decoder: recover a watermarked byte payload using IoU matching.

    Parameters
    ----------
    edited_image : np.ndarray (H, W, 3) uint8, BGR
    sidecar      : SidecarB — loaded via sidecar_b_from_file()

    Returns
    -------
    bytes — recovered original payload.

    Raises
    ------
    DecodeErrorB — if RS cannot recover (edit exceeded RS capacity).
    """
    meta = sidecar.metadata
    rs_overhead    = meta.get("rs_overhead",   2.0)
    iou_threshold  = meta.get("iou_threshold", 0.5)
    message_length = meta.get("message_length", None)
    n_encoded_bits = len(sidecar.pairs)

    # Step 1: Resize edited image to before-image dimensions if they differ.
    before_h = meta.get("image_height")
    before_w = meta.get("image_width")
    if before_h and before_w:
        after_h, after_w = edited_image.shape[:2]
        if (after_h, after_w) != (before_h, before_w):
            import cv2 as _cv2
            edited_image = _cv2.resize(
                edited_image, (before_w, before_h), interpolation=_cv2.INTER_LINEAR
            )

    # Step 2: Segment the edited image with the same SLIC params.
    seg_after = slic_superpixels(edited_image)

    # Step 3: Compute after-image DWT-LL descriptors.
    after_descriptors = compute_raw_dwt_ll(edited_image, seg_after)

    # Step 4: Reconstruct SegmentationResult for the before-image from the sidecar.
    seg_before = SegmentationResult(
        region_map=sidecar.region_map.astype(int),
        num_regions=sidecar.num_regions,
    )

    # Step 5: IoU + Hungarian matching — before-region → after-region (or None).
    #   match[r] = s   → before-region r corresponds to after-region s (IoU >= threshold)
    #   match[r] = None → no after-region overlaps enough → erasure
    match = match_regions(seg_before, seg_after, iou_threshold=iou_threshold)

    # Step 6: Majority vote per bit position.
    decoded_bits = []
    erasure_count = 0

    for pairs_j in sidecar.pairs:
        votes = []
        for (r1, r2) in pairs_j:
            s1 = match.get(r1)
            s2 = match.get(r2)
            if s1 is None or s2 is None:
                continue  # erased witness — skip rather than cast a wrong vote
            votes.append(1 if after_descriptors[s1] > after_descriptors[s2] else 0)

        if len(votes) == 0:
            decoded_bits.append(ERASURE)
            erasure_count += 1
        else:
            # Tie defaults to 0; k=7 (odd) makes ties impossible.
            decoded_bits.append(1 if sum(votes) > len(votes) / 2 else 0)

    # Step 7: Convert bit votes to bytes and collect erasure positions for RS.
    raw_bytes, erasure_byte_positions = bits_to_bytes_with_erasures(decoded_bits)

    try:
        recovered = rs_decode_bytes(raw_bytes, erasure_byte_positions, overhead=rs_overhead)
    except Exception as e:
        raise DecodeErrorB(
            f"Reed-Solomon decode failed: {e}\n"
            f"  {erasure_count}/{n_encoded_bits} bit positions erased "
            f"({100 * erasure_count / n_encoded_bits:.1f}%)\n"
            f"  The edit may have been too destructive for the RS capacity."
        ) from e

    # Step 8: Slice to original message length and return.
    if message_length is not None:
        return bytes(recovered[:message_length])
    return bytes(recovered)
