# watermark/reed_solomon.py
#
# Responsible for: wrapping the Reed-Solomon encode/decode calls so the
# encoder and decoder don't need to know about the `reedsolo` library directly.
#
# Library: reedsolo (pip install reedsolo)
#   GitHub: https://github.com/tomerfiliba-org/reedsolomon
#   The `reedsolo` library operates on byte sequences with configurable
#   error-correction capacity.  We configure it for the erasure-dominated
#   regime this system operates in.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY REED-SOLOMON HERE
# ─────────────────────────────────────────────────────────────────────────────
#
#   The evaluation in RESULTS_EXTENDED.md shows the error channel is:
#     - ~50% erasure rate on hard MagicBrush edits (regions destroyed → known missing bits)
#     - ~8% random flip rate among surviving pairs (descriptor ordering wrong)
#
#   RS corrects twice as many erasures as random errors for the same overhead.
#   At 2× overhead (960 encoded bits for 480 payload bits):
#     - Can correct up to 240 random errors, OR
#     - Can correct up to 480 known erasures, OR
#     - Can correct a mix (erasures count as half the budget)
#   At 50% erasure + 8% errors simultaneously:
#     - 480 erasures uses the full erasure budget → barely passes
#     - This is the design target per approach3_encoding_inital_explaination.md
#
#   The key insight: pass EXPLICIT ERASURE POSITIONS to the RS decoder.
#   When a region pair produces no votes (all K witnesses erased), the decoder
#   marks that bit as an erasure.  RS can use this positional information to
#   recover 2× more bits than if they were treated as unknown random errors.
#
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
#
#   RS_OVERHEAD : float = 2.0
#       The redundancy ratio.  n_encoded_bits = n_payload_bits * RS_OVERHEAD.
#       At 2.0, half the encoded bits are payload and half are parity.
#       This is set conservatively for the hard MagicBrush worst case.
#
#   n_payload_bits  = message_bytes * 8    (e.g. 60 bytes → 480 bits)
#   n_encoded_bits  = n_payload_bits * RS_OVERHEAD  (e.g. 480 → 960 bits)
#   n_parity_bytes  = (n_encoded_bits - n_payload_bits) / 8  (for reedsolo config)
#
#   reedsolo works at the byte level, not the bit level.
#   So encode at the BYTE level: 60 payload bytes → 60+60 = 120 encoded bytes.
#   Then convert bytes → bits for the pair assignment step.
#   At decode: convert decoded bits → bytes → pass to RS byte decoder.
#
# ─────────────────────────────────────────────────────────────────────────────
# BIT ↔ BYTE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────
#
#   RS operates on bytes.  The pair pool operates on individual bits.
#   So there are two conversions:
#
#   Encode path:
#     message (bytes, len=60)
#       → rs_encode_bytes → encoded_bytes (bytes, len=120)
#       → bytes_to_bits   → encoded_bits  (list[int], len=960)
#       → assigned to pairs via pair_pool.assign_pairs_to_bits
#
#   Decode path:
#     raw_bit_votes (list[int|ERASURE], len=960) from majority voting
#       → bits_to_bytes with erasure handling
#           → raw_bytes (bytes, len=120) with erasure byte positions
#       → rs_decode_bytes(raw_bytes, erasure_positions)
#           → recovered_bytes (bytes, len=60)
#
#   ERASURE sentinel: use the integer -1 (or a module-level constant ERASURE = -1)
#   to represent a bit position where all K witnesses were erased.
#   The decoder collects these positions and passes them to rs_decode_bytes.
#
# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

RS_OVERHEAD = 2.0    # redundancy ratio: encoded = payload × this
ERASURE = -1         # sentinel value for an unknown/erased bit


def rs_encode_bytes(
    message,         # bytes — raw payload, e.g. 60 bytes
    overhead=RS_OVERHEAD,  # float — redundancy ratio
) -> bytes:          # bytes — RS-encoded payload, e.g. 120 bytes
    # Step 1: Compute n_parity_bytes = len(message) * (overhead - 1).
    #   For 60 bytes and 2.0 overhead: n_parity_bytes = 60.
    #   reedsolo uses n_error_correct_symbols = n_parity_bytes // 2
    #   because each correction symbol can fix one error OR one erasure.
    #   At n_parity_bytes = 60: can correct 30 errors or 60 erasures.
    #   Since we're operating at the byte level, this scales to bit level
    #   when bytes_to_bits is applied.

    # Step 2: Initialize RSCodec with the right nsize.
    #   from reedsolo import RSCodec
    #   rsc = RSCodec(nsym=n_parity_bytes, nsize=255)
    #   NOTE: reedsolo's nsize=255 is the GF field size; message chunks
    #   must be ≤ 255 - nsym bytes.  For nsym=60, max chunk = 195 bytes.
    #   60 bytes < 195 so no chunking needed.

    # Step 3: Encode.
    #   encoded = rsc.encode(message)
    #   Returns bytearray of length len(message) + n_parity_bytes.
    #   The first len(message) bytes are the original message (systematic code),
    #   the remaining n_parity_bytes are the RS parity.

    # Step 4: Return bytes(encoded).
    pass


def rs_decode_bytes(
    encoded,          # bytes — 120 bytes of RS-encoded data from the decoder
                      #   Some bytes may be corrupted; some positions are known erasures.
    erasure_positions, # list[int] — byte-level positions of known erasures.
                      #   These come from converting the bit-level ERASURE sentinels
                      #   to byte positions (see bits_to_bytes_with_erasures).
    overhead=RS_OVERHEAD,
) -> bytes:           # bytes — recovered original message, e.g. 60 bytes
    # Step 1: Initialize RSCodec with same parameters as encode.

    # Step 2: Decode with explicit erasure positions.
    #   decoded, _, _ = rsc.decode(encoded, erase_pos=erasure_positions)
    #   reedsolo's decode() accepts erase_pos as a list of byte indices.
    #   Passing these allows RS to correct twice as many erasures as errors.

    # Step 3: Return bytes(decoded)[:original_message_length].
    #   rsc.decode returns the full codeword; slice to message length.
    #   original_message_length = len(encoded) // overhead = len(encoded) // 2
    pass


def bytes_to_bits(
    data,   # bytes — e.g. 120 RS-encoded bytes
) -> list:  # list[int] — 0s and 1s, length = len(data) * 8
            #   MSB first within each byte (big-endian bit order)
    # For each byte b in data:
    #   for bit_idx in range(7, -1, -1):
    #       bits.append((b >> bit_idx) & 1)
    # Returns a flat list of bits in the order pairs will be assigned.
    pass


def bits_to_bytes_with_erasures(
    bits,   # list[int] — decoded bit votes from majority voting.
            #   Each element is 0, 1, or ERASURE (-1).
            #   Length must be divisible by 8.
) -> tuple: # tuple[bytes, list[int]]
            #   [0] bytes: reconstructed byte sequence with 0x00 in erasure positions
            #   [1] list[int]: byte-level erasure positions (for rs_decode_bytes)
    # Step 1: Group bits into bytes (8 bits per byte, MSB first).
    #   For each group of 8 bits:
    #     if ANY bit in the group is ERASURE:
    #       → byte value = 0x00 (placeholder; RS will correct)
    #       → record this byte index in erasure_byte_positions
    #     else:
    #       → reconstruct byte: sum(bit << (7-i) for i, bit in enumerate(group))
    #
    # Step 2: Return (bytes(byte_values), erasure_byte_positions).
    #
    # NOTE on bit grouping and erasure granularity:
    #   reedsolo corrects at the byte level.  If any 1 bit in an 8-bit group
    #   is an ERASURE, mark the whole byte as erased.  This is conservative
    #   but correct — reedsolo can't correct individual bits, only whole symbols.
    #   With K=7 majority voting, a single-bit erasure within a byte is very
    #   rare because the majority vote already collapsed K witnesses into one bit.
    pass
