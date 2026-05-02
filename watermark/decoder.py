# watermark/decoder.py
#
# Full decoding pipeline.  Runs on the AI-edited image + the sidecar.
# Does NOT need the original image.  Does NOT need the secret key.
#
# Image color space note:
#   Accepts BGR images (as returned by cv2.imread), same as the encoder.
#   Must be consistent with encoder color space for centroid and descriptor
#   comparisons to be meaningful.
#
# Pipeline:
#   edited_image + sidecar
#     → read params from sidecar.metadata
#     → SLIC segmentation on edited image    (regions/approach_regions.py)
#     → after-image centroids                (watermark/centroid_matching.py)
#     → after-image DWT-LL descriptors       (descriptors/dwt_descriptor.py)
#     → nearest-centroid region matching     (watermark/centroid_matching.py)
#     → majority vote per bit position       (internal)
#     → RS decode with explicit erasures     (watermark/reed_solomon.py)
#     → return recovered message bytes

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regions.approach_regions import slic_superpixels
from descriptors.dwt_descriptor import compute_raw_dwt_ll
from watermark.centroid_matching import compute_region_centroids, match_regions_by_centroid
from watermark.reed_solomon import ERASURE, bits_to_bytes_with_erasures, rs_decode_bytes


class DecodeError(Exception):
    """Raised when Reed-Solomon cannot recover the message (edit too destructive)."""
    pass


def decode_watermark(edited_image, sidecar) -> bytes:
    """
    Recover a watermarked byte payload from an AI-edited image.

    Parameters
    ----------
    edited_image : np.ndarray (H, W, 3) uint8, BGR
    sidecar      : Sidecar — loaded via sidecar_from_file() or deserialize_sidecar()

    Returns
    -------
    bytes — recovered original payload (e.g. b"Cornell")
            Decode with .decode("utf-8") to get the original string.

    Raises
    ------
    DecodeError — if RS cannot recover (edit exceeded RS capacity).
    """
    # Step 1: Read all params from sidecar.metadata (no hardcoded values here).
    meta = sidecar.metadata
    rs_overhead        = meta.get("rs_overhead",        2.0)
    centroid_threshold = meta.get("centroid_threshold", 40.0)
    message_length     = meta.get("message_length",     None)
    n_encoded_bits     = len(sidecar.pairs)

    # Step 2: Resize edited image to before-image dimensions if they differ.
    #   Centroid coordinates in the sidecar are in before-image pixel space.
    #   If the AI editor outputs at a different resolution, centroids won't align.
    before_h = meta.get("image_height")
    before_w = meta.get("image_width")
    if before_h and before_w:
        after_h, after_w = edited_image.shape[:2]
        if (after_h, after_w) != (before_h, before_w):
            import cv2 as _cv2
            edited_image = _cv2.resize(edited_image, (before_w, before_h), interpolation=_cv2.INTER_LINEAR)

    # Step 3: Segment the edited image with the same SLIC params as the encoder.
    seg_after = slic_superpixels(edited_image)

    # Step 3: Compute after-image region centroids.
    #   SLIC renumbers regions from scratch on the new image — after-region 42 is NOT
    #   the same area as before-region 42.  match_regions_by_centroid bridges this gap.
    after_centroids = compute_region_centroids(seg_after)

    # Step 4: Compute after-image DWT-LL descriptors.
    #   after_descriptors[s] = LL energy of after-image region s.
    #   These are compared against each other (not against before-image values)
    #   to recover the original ordinal relationships.
    after_descriptors = compute_raw_dwt_ll(edited_image, seg_after)

    # Step 5: Match before-regions (from sidecar) to after-regions via centroid proximity.
    #   match[r] = s   → before-region r corresponds to after-region s
    #   match[r] = None → region drifted > centroid_threshold px → erasure
    match = match_regions_by_centroid(
        sidecar.centroids,
        after_centroids,
        centroid_threshold=centroid_threshold,
    )

    # Step 6: Majority vote per bit position.
    #   For each bit j, K pairs each cast a vote (0 or 1) based on e[s1] vs e[s2].
    #   Pairs with a missing region (match = None) are skipped entirely — an erasure
    #   is far better than a wrong vote.
    decoded_bits = []
    erasure_count = 0

    for pairs_j in sidecar.pairs:
        votes = []
        for (r1, r2) in pairs_j:
            s1 = match.get(r1)
            s2 = match.get(r2)
            if s1 is None or s2 is None:
                continue  # this witness is erased — do not vote
            votes.append(1 if after_descriptors[s1] > after_descriptors[s2] else 0)

        if len(votes) == 0:
            decoded_bits.append(ERASURE)
            erasure_count += 1
        else:
            # Tie (even K, equal votes) defaults to 0.  K=7 (odd) makes ties impossible.
            decoded_bits.append(1 if sum(votes) > len(votes) / 2 else 0)

    # Step 7: Convert bit votes to bytes and collect erasure positions for RS.
    raw_bytes, erasure_byte_positions = bits_to_bytes_with_erasures(decoded_bits)

    try:
        recovered = rs_decode_bytes(raw_bytes, erasure_byte_positions, overhead=rs_overhead)
    except Exception as e:
        raise DecodeError(
            f"Reed-Solomon decode failed: {e}\n"
            f"  {erasure_count}/{n_encoded_bits} bit positions erased "
            f"({100 * erasure_count / n_encoded_bits:.1f}%)\n"
            f"  The edit may have been too destructive for the RS capacity."
        ) from e

    # Step 8: Slice to original message length and return.
    if message_length is not None:
        return bytes(recovered[:message_length])
    return bytes(recovered)
