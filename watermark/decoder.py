# watermark/decoder.py
#
# Responsible for: the full decoding pipeline.
# Runs on the AI-EDITED image + the sidecar.
# Does NOT need the original image.
# Does NOT need the secret key (the sidecar already has the pair table).
#
# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE (Phase 2 from approach3_encoding_inital_explaination.md)
# ─────────────────────────────────────────────────────────────────────────────
#
#   Input:
#     edited_image : np.ndarray (H, W, 3) uint8, RGB color space
#     sidecar      : Sidecar — loaded from file or EXIF; contains centroids + pairs + metadata
#
#   Output:
#     bytes — the recovered original payload (e.g. 60 bytes)
#             Raises DecodeError if RS cannot recover (too many uncorrectable errors).
#
#   Steps:
#     1. Read decode params from sidecar.metadata
#     2. Segment edited image with SLIC (same params as encoder)          [→ regions/approach_regions.py]
#     3. Compute after-image centroids                                    [→ centroid_matching.py]
#     4. Compute after-image DWT-LL descriptors                           [→ descriptors/dwt_descriptor.py]
#     5. Nearest-centroid matching: before-regions → after-regions        [→ centroid_matching.py]
#     6. Majority vote per bit position                                    [internal]
#     7. RS decode with explicit erasure positions                         [→ reed_solomon.py]
#     8. Return recovered message bytes
#
# ─────────────────────────────────────────────────────────────────────────────
# ERROR TYPES
# ─────────────────────────────────────────────────────────────────────────────
#
#   DecodeError(Exception):
#       Raised when Reed-Solomon cannot recover the message.
#       This means the edit was too destructive (beyond the RS capacity).
#       At 2× overhead, RS can handle up to 50% erasures + 8% errors simultaneously.
#       If the actual edit exceeded this (e.g. full image replacement), raise this.
#
#   The decoder should NOT silently return garbage bytes.  If RS fails, raise.
#
# ─────────────────────────────────────────────────────────────────────────────
# MAJORITY VOTING — DETAILED EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
#
#   For each bit position j (0 to n_encoded_bits-1):
#
#     pairs_j = sidecar.pairs[j]   # list of K (r1, r2) tuples
#     votes = []
#
#     for (r1, r2) in pairs_j:
#         s1 = match[r1]   # after-image region matched to before-region r1
#         s2 = match[r2]   # after-image region matched to before-region r2
#
#         if s1 is None or s2 is None:
#             # One or both regions were destroyed by the edit (centroid drift > threshold)
#             # Skip entirely — do NOT vote.  An erasure is better than a wrong vote.
#             continue
#
#         # Compare DWT-LL energies in the after-image
#         vote = 1 if after_descriptors[s1] > after_descriptors[s2] else 0
#         votes.append(vote)
#
#     if len(votes) == 0:
#         # All K witnesses erased — this bit position is completely unknown
#         decoded_bits[j] = ERASURE    # = -1, the sentinel from reed_solomon.py
#     else:
#         # Majority of surviving witnesses
#         decoded_bits[j] = 1 if sum(votes) > len(votes) / 2 else 0
#         # Tie-break (even K, equal votes): default to 0.
#         # With K=7 (odd), ties are impossible.
#
#   Result: decoded_bits is a list of length n_encoded_bits,
#   each element is 0, 1, or ERASURE (-1).
#
#   The number of ERASURE positions is passed to rs_decode_bytes explicitly.
#   RS uses erasure positions to correct 2× more efficiently than unknown errors.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW SIDECAR METADATA DRIVES THE DECODER
# ─────────────────────────────────────────────────────────────────────────────
#
#   sidecar.metadata["n_segments"]          → SLIC n_segments (must match encoder)
#   sidecar.metadata["compactness"]         → SLIC compactness (must match encoder)
#   sidecar.metadata["k"]                   → witnesses per bit (for validation only)
#   sidecar.metadata["rs_overhead"]         → RS redundancy ratio (for rs_decode_bytes)
#   sidecar.metadata["centroid_threshold"]  → max drift before erasure (pixels)
#   sidecar.metadata["message_length"]      → expected output length (for slicing RS output)
#
#   If any metadata key is missing, fall back to the same defaults as EncodeParams.
#   This ensures backward compatibility if an older sidecar omits some fields.
#
# ─────────────────────────────────────────────────────────────────────────────

from watermark.reed_solomon import ERASURE


class DecodeError(Exception):
    # Raised when RS cannot recover.  Message describes why:
    #   "Reed-Solomon decode failed: too many uncorrectable errors (N erasures, M errors)"
    pass


def decode_watermark(
    edited_image,  # np.ndarray (H, W, 3) uint8, RGB
                   # The AI-edited version of the watermarked image.
                   # Does NOT need to be the same resolution as the before-image
                   # as long as the spatial layout is similar enough for centroid matching.
    sidecar,       # Sidecar — output of encode_watermark, loaded from file or EXIF.
                   # Import: from watermark.sidecar import deserialize_sidecar, sidecar_from_file
) -> bytes:        # The recovered original payload bytes (e.g. 60 bytes)
    # ── Step 1: Read params from sidecar metadata ─────────────────────────────
    #   n_segments         = sidecar.metadata.get("n_segments", 200)
    #   compactness        = sidecar.metadata.get("compactness", 20)
    #   rs_overhead        = sidecar.metadata.get("rs_overhead", 2.0)
    #   centroid_threshold = sidecar.metadata.get("centroid_threshold", 40.0)
    #   message_length     = sidecar.metadata.get("message_length", None)
    #
    #   n_encoded_bits = len(sidecar.pairs)   # e.g. 960; ground truth from the pair table

    # ── Step 2: Segment the edited image ─────────────────────────────────────
    #   from regions.approach_regions import slic_superpixels
    #   seg_after = slic_superpixels(edited_image)
    #     Uses the same SLIC parameters as the encoder (read from metadata).
    #     seg_after.region_map and seg_after.num_regions define the after-image regions.
    #
    #   NOTE: SLIC renumbers regions from scratch on the new image.
    #   Region 42 in the before-image is NOT region 42 in the after-image.
    #   The centroid matching in Step 5 bridges this gap.

    # ── Step 3: Compute after-image centroids ─────────────────────────────────
    #   from watermark.centroid_matching import compute_region_centroids
    #   after_centroids = compute_region_centroids(seg_after)
    #     → dict[int, (float, float)]: after-image region ID → (cx, cy)
    #     These are compared against sidecar.centroids (before-image) in Step 5.

    # ── Step 4: Compute after-image DWT-LL descriptors ───────────────────────
    #   from descriptors.dwt_descriptor import compute_dwt_ll
    #   after_descriptors = compute_dwt_ll(edited_image, seg_after)
    #     → list[float], length = seg_after.num_regions
    #     → after_descriptors[s] = LL energy for after-image region s
    #
    #   These are the values used in majority voting (Step 6).
    #   The comparison e[s1] > e[s2] is the core ordinal relationship check.

    # ── Step 5: Nearest-centroid matching ─────────────────────────────────────
    #   from watermark.centroid_matching import match_regions_by_centroid
    #   match = match_regions_by_centroid(
    #       sidecar.centroids,      # before-image centroids from the sidecar
    #       after_centroids,        # after-image centroids computed in Step 3
    #       centroid_threshold=centroid_threshold
    #   )
    #   → dict[int, int | None]
    #     match[r] = s  : before-region r corresponds to after-region s
    #     match[r] = None : region r was destroyed / moved too far → erasure

    # ── Step 6: Majority vote per bit position ────────────────────────────────
    #   decoded_bits = []
    #   erasure_count = 0
    #   for j, pairs_j in enumerate(sidecar.pairs):
    #       votes = []
    #       for (r1, r2) in pairs_j:
    #           s1 = match.get(r1)  # None if r1 not in match (shouldn't happen but guard)
    #           s2 = match.get(r2)
    #           if s1 is None or s2 is None:
    #               continue      # erasure: skip this witness entirely
    #           vote = 1 if after_descriptors[s1] > after_descriptors[s2] else 0
    #           votes.append(vote)
    #       if len(votes) == 0:
    #           decoded_bits.append(ERASURE)
    #           erasure_count += 1
    #       else:
    #           decoded_bits.append(1 if sum(votes) > len(votes) / 2 else 0)
    #
    #   At this point decoded_bits is a list of length n_encoded_bits,
    #   with some elements = ERASURE (-1).
    #   Log: f"{erasure_count}/{n_encoded_bits} bits erased ({100*erasure_count/n_encoded_bits:.1f}%)"

    # ── Step 7: RS decode with explicit erasure positions ─────────────────────
    #   from watermark.reed_solomon import bits_to_bytes_with_erasures, rs_decode_bytes
    #
    #   raw_bytes, erasure_byte_positions = bits_to_bytes_with_erasures(decoded_bits)
    #     → raw_bytes: bytes of length n_encoded_bits // 8 (e.g. 120)
    #       Bytes at erasure positions are set to 0x00 (placeholder).
    #     → erasure_byte_positions: list[int] of byte indices where erasures occurred.
    #
    #   try:
    #       recovered = rs_decode_bytes(raw_bytes, erasure_byte_positions, overhead=rs_overhead)
    #   except Exception as e:
    #       raise DecodeError(
    #           f"Reed-Solomon decode failed: {e}. "
    #           f"({erasure_count} bit erasures, edit may exceed RS capacity)"
    #       )
    #
    #   recovered is a bytearray of length original_message_length (e.g. 60).

    # ── Step 8: Return recovered bytes ───────────────────────────────────────
    #   If message_length is known from metadata, slice: return bytes(recovered[:message_length])
    #   Otherwise return bytes(recovered) and let the caller interpret the length.
    pass
