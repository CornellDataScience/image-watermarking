# watermark/reed_solomon.py
#
# Thin wrapper around the `reedsolo` library for the encoder and decoder.
# Install: pip install reedsolo
#
# Error channel (from RESULTS_EXTENDED.md):
#   ~50% erasure rate on hard MagicBrush (regions destroyed → known missing bits)
#   ~8% flip rate among surviving pairs (descriptor ordering wrong)
#
# RS corrects 2× more erasures than random errors for the same overhead.
# At 2× overhead: can handle 50% erasures + 8% errors simultaneously.
#
# reedsolo operates at the BYTE level.  The pair pool operates at the BIT level.
# Encode path:  message bytes → rs_encode_bytes → encoded bytes → bytes_to_bits → bit list
# Decode path:  bit votes (with ERASURE sentinels) → bits_to_bytes_with_erasures
#               → (encoded bytes, erasure byte positions) → rs_decode_bytes → message bytes
#
# Constraint: len(message) + n_parity_bytes <= 255 (GF field size).
# At 2× overhead, n_parity = len(message), so max message = 127 bytes.
# Short phrases ("Cornell" = 7 bytes) are well within this limit.

RS_OVERHEAD = 2.0   # encoded = payload × this; half the encoded bits are parity
ERASURE = -1        # sentinel: a bit position where all K witnesses were erased


def rs_encode_bytes(message, overhead=RS_OVERHEAD) -> bytes:
    """
    Reed-Solomon encode a byte payload.

    Parameters
    ----------
    message : bytes  — raw payload (e.g. b"Cornell", max 127 bytes at 2× overhead)
    overhead : float — redundancy ratio; encoded length = len(message) * overhead

    Returns
    -------
    bytes of length len(message) * overhead  (systematic: message bytes first, parity appended)
    """
    from reedsolo import RSCodec
    n_parity = int(round(len(message) * (overhead - 1)))
    rsc = RSCodec(nsym=n_parity)
    return bytes(rsc.encode(message))


def rs_decode_bytes(encoded, erasure_positions, overhead=RS_OVERHEAD) -> bytes:
    """
    Reed-Solomon decode with explicit erasure positions.

    Passing erasure positions lets RS correct 2× as many erasures as random errors
    for the same overhead — critical for the erasure-dominated regime here.

    Parameters
    ----------
    encoded          : bytes   — RS-encoded data (may contain errors; some positions erased)
    erasure_positions: list[int] — byte-level indices of known erasures (0x00 placeholders)
    overhead         : float   — must match the value used in rs_encode_bytes

    Returns
    -------
    bytes — recovered original message
    """
    from reedsolo import RSCodec
    n_message = int(round(len(encoded) / overhead))
    n_parity = len(encoded) - n_message
    rsc = RSCodec(nsym=n_parity)
    decoded, _, _ = rsc.decode(bytearray(encoded), erase_pos=erasure_positions)
    return bytes(decoded)[:n_message]


def bytes_to_bits(data) -> list:
    """
    Convert bytes to a flat list of bits, MSB first within each byte.

    bytes([0b10110001]) → [1, 0, 1, 1, 0, 0, 0, 1]
    """
    bits = []
    for b in data:
        for shift in range(7, -1, -1):
            bits.append((b >> shift) & 1)
    return bits


def bits_to_bytes_with_erasures(bits) -> tuple:
    """
    Reconstruct bytes from a list of bit votes that may contain ERASURE sentinels.

    Groups bits into bytes (8 per byte, MSB first).  If any bit in a group is
    ERASURE (-1), the whole byte is marked as erased (0x00 placeholder) because
    reedsolo corrects at the byte (symbol) level, not the bit level.

    Parameters
    ----------
    bits : list[int]
        Each element is 0, 1, or ERASURE (-1).  Length must be divisible by 8.

    Returns
    -------
    tuple[bytes, list[int]]
        [0] byte sequence with 0x00 at erasure positions
        [1] list of byte-level erasure positions for rs_decode_bytes
    """
    assert len(bits) % 8 == 0, f"bits length {len(bits)} is not divisible by 8"

    byte_values = []
    erasure_byte_positions = []

    for byte_idx in range(len(bits) // 8):
        group = bits[byte_idx * 8 : byte_idx * 8 + 8]
        if any(b == ERASURE for b in group):
            byte_values.append(0x00)
            erasure_byte_positions.append(byte_idx)
        else:
            val = sum(bit << (7 - i) for i, bit in enumerate(group))
            byte_values.append(val)

    return bytes(byte_values), erasure_byte_positions
