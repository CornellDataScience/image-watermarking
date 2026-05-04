# watermark/encoder_b.py
#
# Option 2 (Design B) encoding pipeline.
# Structurally identical to encoder.py (Option 3) except it builds a SidecarB
# that stores the full before-image region_map and per-region descriptor values
# instead of only region centroids.
#
# The larger sidecar enables the decoder to use IoU + Hungarian matching
# (more accurate than centroid proximity) at the cost of sidecar size.
#
# Pipeline:
#   image + message + key
#     → SLIC segmentation        (regions/approach_regions.py)
#     → DWT-LL descriptors       (descriptors/dwt_descriptor.py)
#     → pair pool + shuffle      (watermark/pair_pool.py)
#     → RS encode message        (watermark/reed_solomon.py)
#     → assign pairs to bits     (watermark/pair_pool.py)
#     → build + return SidecarB  (watermark/sidecar_b.py)

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass

from regions.approach_regions import slic_superpixels
from descriptors.dwt_descriptor import compute_raw_dwt_ll
from watermark.pair_pool import build_pair_pool, assign_pairs_to_bits
from watermark.reed_solomon import rs_encode_bytes, bytes_to_bits
from watermark.sidecar_b import build_sidecar_b, SidecarB


@dataclass
class EncodeParamsB:
    n_segments: int = 200        # SLIC superpixel count
    compactness: int = 20        # SLIC spatial regularization
    min_margin: float = 0.05     # minimum |d[i]-d[j]| for a pair to be used
    k: int = 7                   # witnesses per bit
    rs_overhead: float = 2.0     # encoded_bits = payload_bits × this
    iou_threshold: float = 0.5   # min IoU for a valid region match at decode time


def encode_watermark_b(image, message: bytes, key, params=None) -> SidecarB:
    """
    Option 2 encoder: embed a byte payload by recording pairwise region orderings.
    Returns a SidecarB that stores the full region_map for IoU-based decoding.

    Parameters
    ----------
    image   : np.ndarray (H, W, 3) uint8, BGR (as returned by cv2.imread)
    message : bytes  — payload to embed (max 127 bytes at 2× RS overhead)
    key     : int | str | bytes  — secret seed for pair-pool shuffling
    params  : EncodeParamsB | None — uses defaults if None

    Returns
    -------
    SidecarB — call sidecar_b_to_file(sidecar, path) to persist.
    """
    if params is None:
        params = EncodeParamsB()

    # Step 1: Segment the image into ~200 SLIC superpixels.
    seg = slic_superpixels(image)

    # Step 2: Compute DWT-LL energy descriptor for every region.
    descriptors = compute_raw_dwt_ll(image, seg)

    # Step 3: Build the pair pool from descriptors, filter by margin, shuffle with key.
    pool = build_pair_pool(
        descriptors,
        key=key,
        min_margin=params.min_margin,
        k=params.k,
    )

    # Step 4: Reed-Solomon encode the message.
    encoded_bytes = rs_encode_bytes(message, overhead=params.rs_overhead)
    encoded_bits = bytes_to_bits(encoded_bytes)
    n_encoded_bits = len(encoded_bits)

    # Step 5: Verify the image has enough usable pairs.
    if pool.total_capacity < n_encoded_bits:
        raise ValueError(
            f"Not enough usable pairs to embed this message.\n"
            f"  Pool capacity : {pool.total_capacity} bits\n"
            f"  Message needs : {n_encoded_bits} bits\n"
            f"  Try a shorter message, smaller K, or a different image."
        )

    # Step 6: Assign K pairs from the pool to each encoded bit position.
    pairs_for_bits = assign_pairs_to_bits(pool, encoded_bits, k=params.k)

    # Step 7: Build SidecarB (stores region_map + descriptors for IoU matching).
    metadata = {
        "method":         "option2",
        "n_segments":     params.n_segments,
        "compactness":    params.compactness,
        "k":              params.k,
        "rs_overhead":    params.rs_overhead,
        "min_margin":     params.min_margin,
        "iou_threshold":  params.iou_threshold,
        "message_length": len(message),
        "image_height":   image.shape[0],
        "image_width":    image.shape[1],
    }
    return build_sidecar_b(
        region_map=seg.region_map,
        num_regions=seg.num_regions,
        descriptors=descriptors,
        pairs_for_bits=pairs_for_bits,
        metadata=metadata,
    )
