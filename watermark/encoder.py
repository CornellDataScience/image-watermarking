# watermark/encoder.py
#
# Responsible for: the full encoding pipeline.
# Runs ONCE on the original before-image, produces a Sidecar object.
# Does NOT modify the image at all — the watermark is read-only.
#
# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE (Phase 0 + Phase 1 from approach3_encoding_inital_explaination.md)
# ─────────────────────────────────────────────────────────────────────────────
#
#   Input:
#     image    : np.ndarray (H, W, 3) uint8, RGB color space
#     message  : bytes — the payload to embed (e.g. 60 bytes)
#     key      : int | bytes | str — secret seed for pair-pool shuffling
#     params   : EncodeParams — tunable hyperparameters (see below)
#
#   Output:
#     Sidecar — the object containing centroids + pair table + metadata.
#               Serialize with sidecar.serialize_sidecar() to write to disk/EXIF.
#
#   Steps:
#     1. Segment before-image with SLIC                     [→ regions/approach_regions.py]
#     2. Compute region centroids                            [→ centroid_matching.py]
#     3. Compute DWT-LL descriptor per region               [→ descriptors/dwt_descriptor.py]
#     4. Build and shuffle pair pool                         [→ pair_pool.py]
#     5. RS-encode the message bytes → bit array            [→ reed_solomon.py]
#     6. Verify the pool has enough pairs for the bit array [→ pair_pool.py]
#     7. Assign pairs to bits                               [→ pair_pool.py]
#     8. Build and return Sidecar                           [→ sidecar.py]
#
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS FROM EXISTING CODEBASE
# ─────────────────────────────────────────────────────────────────────────────
#
#   from regions.approach_regions import slic_superpixels
#       slic_superpixels(image) -> SegmentationResult
#       image must be RGB uint8 (H, W, 3)
#       Returns SegmentationResult with .region_map (H,W int) and .num_regions (int).
#
#   from descriptors.dwt_descriptor import compute_dwt_ll
#       compute_dwt_ll(image, seg) -> list[float]
#       Returns one float per region: the DWT LL energy, normalized by total energy.
#       This is the chosen descriptor per RESULTS_EXTENDED.md (only one below
#       10% flip rate in all 16 strata; worst case 7.63% on MagicBrush hard).
#
# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class EncodeParams:
    # SLIC segmentation parameters (must match exactly at decode time)
    n_segments: int = 200       # number of superpixels
    compactness: int = 20       # SLIC spatial regularization

    # Pair pool parameters
    min_margin: float = 0.05    # minimum |d[i] - d[j]| for a pair to be used
    k: int = 7                  # witnesses per bit (use 11 for hard MagicBrush)

    # Reed-Solomon overhead
    rs_overhead: float = 2.0    # encoded_bits = payload_bits * rs_overhead

    # Decoder-side erasure threshold (stored in metadata so decoder can read it)
    centroid_threshold: float = 40.0  # pixels; max centroid drift before erasure


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def encode_watermark(
    image,      # np.ndarray (H, W, 3) uint8, RGB
    message,    # bytes — payload to embed, e.g. b"hello world..."
    key,        # int | bytes | str — secret seed
    params=None,  # EncodeParams | None — uses defaults if None
) -> object:    # Sidecar — import from watermark.sidecar
    # If params is None, instantiate EncodeParams() with defaults.

    # ── Step 1: Segment the before-image ─────────────────────────────────────
    #   seg = slic_superpixels(image)
    #     → seg.region_map: (H, W) int array, values 0..n_segments-1
    #     → seg.num_regions: int (should be ~200 for most images)
    #
    #   NOTE: slic_superpixels takes RGB.  If the caller passes BGR (e.g. from
    #   cv2.imread), they must convert first: image = image[:, :, ::-1]
    #   Document this contract clearly in the docstring.

    # ── Step 2: Compute region centroids ─────────────────────────────────────
    #   from watermark.centroid_matching import compute_region_centroids
    #   centroids_all = compute_region_centroids(seg)
    #     → dict[int, (float, float)]: all 200 region centroids
    #
    #   These are computed here (encoder side) so that:
    #   (a) The encoder stores them in the sidecar.
    #   (b) The decoder can use them for nearest-centroid matching.
    #   The decoder independently runs compute_region_centroids on the
    #   after-image to get after_centroids, then calls match_regions_by_centroid.

    # ── Step 3: Compute DWT-LL descriptors ───────────────────────────────────
    #   from descriptors.dwt_descriptor import compute_dwt_ll
    #   descriptors = compute_dwt_ll(image, seg)
    #     → list[float], length = seg.num_regions
    #     → descriptors[k] = LL energy for region k, normalized by total energy
    #
    #   This is the same function called in stability evaluation (run_pairwise_stability.py).
    #   No changes needed to that function.

    # ── Step 4: Build and shuffle the pair pool ───────────────────────────────
    #   from watermark.pair_pool import build_pair_pool
    #   pool = build_pair_pool(descriptors, key=key, min_margin=params.min_margin, k=params.k)
    #     → PairPool with pool_1, pool_0, total_capacity
    #
    #   pool.total_capacity is the max number of bits that can be embedded.
    #   If this is less than n_encoded_bits (computed in Step 5), raise ValueError.

    # ── Step 5: RS-encode the message ────────────────────────────────────────
    #   from watermark.reed_solomon import rs_encode_bytes, bytes_to_bits
    #   encoded_bytes = rs_encode_bytes(message, overhead=params.rs_overhead)
    #     → bytes, length = len(message) * params.rs_overhead = 60 * 2 = 120
    #   encoded_bits = bytes_to_bits(encoded_bytes)
    #     → list[int], length = len(encoded_bytes) * 8 = 960
    #
    #   n_encoded_bits = len(encoded_bits)  (= 960)

    # ── Step 6: Verify pool capacity ─────────────────────────────────────────
    #   if pool.total_capacity < n_encoded_bits:
    #       raise ValueError(
    #           f"Not enough usable pairs: pool supports {pool.total_capacity} bits "
    #           f"but message requires {n_encoded_bits} bits. "
    #           f"Try a shorter message, smaller K, or a different image."
    #       )
    #
    #   This can fail for very uniform images (few pairs above min_margin),
    #   very small images, or messages that are too long.

    # ── Step 7: Assign pairs to bits ─────────────────────────────────────────
    #   from watermark.pair_pool import assign_pairs_to_bits
    #   pairs_for_bits = assign_pairs_to_bits(pool, encoded_bits, k=params.k)
    #     → list[list[tuple[int, int]]], length = n_encoded_bits
    #     → pairs_for_bits[j] = list of K (region_i, region_j) tuples for bit j

    # ── Step 8: Build the sidecar ─────────────────────────────────────────────
    #   from watermark.sidecar import build_sidecar, Sidecar
    #   metadata = {
    #       "n_segments":          params.n_segments,
    #       "compactness":         params.compactness,
    #       "k":                   params.k,
    #       "rs_overhead":         params.rs_overhead,
    #       "min_margin":          params.min_margin,
    #       "centroid_threshold":  params.centroid_threshold,
    #       "message_length":      len(message),
    #   }
    #   sidecar = build_sidecar(centroids_all, pairs_for_bits, metadata)
    #     → Sidecar(centroids={subset of centroids_all}, pairs=pairs_for_bits, metadata=metadata)
    #
    #   Return sidecar.
    #
    #   The caller then decides how to persist it:
    #     sidecar_to_file(sidecar, "photo.wm")   # companion file
    #     or embed serialize_sidecar(sidecar) in JPEG EXIF tag 0x9C9B
    pass
