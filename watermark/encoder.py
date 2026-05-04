# watermark/encoder.py
#
# Full encoding pipeline.  Runs once on the original image, produces a Sidecar.
# Does NOT modify the image — the watermark lives in the sidecar, not the pixels.
#
# Image color space note:
#   Accepts BGR images (as returned by cv2.imread) because that is what the
#   evaluation pipeline uses throughout.  slic_superpixels converts BGR→RGB
#   internally.  compute_dwt_ll receives BGR but uses it consistently, so
#   ordinal relationships are preserved between encoder and decoder.
#
# Pipeline:
#   image + message + key
#     → SLIC segmentation        (regions/approach_regions.py)
#     → region centroids         (watermark/centroid_matching.py)
#     → DWT-LL descriptors       (descriptors/dwt_descriptor.py)
#     → pair pool + shuffle      (watermark/pair_pool.py)
#     → RS encode message        (watermark/reed_solomon.py)
#     → assign pairs to bits     (watermark/pair_pool.py)
#     → build + return Sidecar   (watermark/sidecar.py)

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass

from regions.approach_regions import slic_superpixels
from descriptors.dwt_descriptor import compute_raw_dwt_ll
from watermark.centroid_matching import compute_region_centroids
from watermark.pair_pool import build_pair_pool, assign_pairs_to_bits
from watermark.reed_solomon import rs_encode_bytes, bytes_to_bits
from watermark.sidecar import build_sidecar, Sidecar


@dataclass
class EncodeParams:
    n_segments: int = 200        # SLIC superpixel count
    compactness: int = 20        # SLIC spatial regularization
    min_margin: float = 0.05     # minimum |d[i]-d[j]| for a pair to be used
    k: int = 7                   # witnesses per bit (use 11 for hard MagicBrush)
    rs_overhead: float = 2.0     # encoded_bits = payload_bits × this
    centroid_threshold: float = 40.0  # max centroid drift (px) before erasure at decode time


def encode_watermark(image, message, key, params=None) -> Sidecar:
    """
    Embed a byte payload into an image by recording pairwise region orderings.

    Parameters
    ----------
    image   : np.ndarray (H, W, 3) uint8, BGR (as returned by cv2.imread)
    message : bytes  — payload to embed, e.g. b"Cornell"  (max 127 bytes at 2× RS overhead)
    key     : int | str | bytes  — secret seed for pair-pool shuffling
    params  : EncodeParams | None  — uses defaults if None

    Returns
    -------
    Sidecar — call sidecar_to_file(sidecar, path) or serialize_sidecar(sidecar) to persist.
    """
    if params is None:
        params = EncodeParams()

    # Step 1: Segment the image into ~200 SLIC superpixels.
    seg = slic_superpixels(image)

    # Step 2: Compute (cx, cy) centroid for every region.
    #   These get stored in the sidecar so the decoder can do nearest-centroid lookup.
    centroids_all = compute_region_centroids(seg)

    # Step 3: Compute DWT-LL energy descriptor for every region.
    #   One float per region; the ordinal relationships among these are what carry the bits.
    descriptors = compute_raw_dwt_ll(image, seg)

    # Step 4: Build the pair pool from descriptors, filter by margin, shuffle with key.
    pool = build_pair_pool(
        descriptors,
        key=key,
        min_margin=params.min_margin,
        k=params.k,
    )

    # Step 5: Reed-Solomon encode the message.
    encoded_bytes = rs_encode_bytes(message, overhead=params.rs_overhead)
    encoded_bits = bytes_to_bits(encoded_bytes)
    n_encoded_bits = len(encoded_bits)

    # Step 6: Verify the image has enough usable pairs.
    if pool.total_capacity < n_encoded_bits:
        raise ValueError(
            f"Not enough usable pairs to embed this message.\n"
            f"  Pool capacity : {pool.total_capacity} bits\n"
            f"  Message needs : {n_encoded_bits} bits\n"
            f"  Try a shorter message, smaller K, or a different image."
        )

    # Step 7: Assign K pairs from the pool to each encoded bit position.
    pairs_for_bits = assign_pairs_to_bits(pool, encoded_bits, k=params.k)

    # Step 8: Build the sidecar (filter centroids to only referenced regions).
    metadata = {
        "n_segments":         params.n_segments,
        "compactness":        params.compactness,
        "k":                  params.k,
        "rs_overhead":        params.rs_overhead,
        "min_margin":         params.min_margin,
        "centroid_threshold": params.centroid_threshold,
        "message_length":     len(message),
        "image_height":       image.shape[0],
        "image_width":        image.shape[1],
    }
    return build_sidecar(centroids_all, pairs_for_bits, metadata)
