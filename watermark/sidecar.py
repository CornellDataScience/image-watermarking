# watermark/sidecar.py
#
# The Sidecar is the ONLY file the decoder receives besides the edited image.
# It stores:
#   centroids — (cx, cy) of every before-image region referenced in the pair table
#   pairs     — for each RS-encoded bit position, the K (region_i, region_j) witnesses
#   metadata  — encode-time parameters so the decoder needs zero hardcoded values
#
# Size (7-byte message "Cornell", K=7, 2× RS):
#   112 bits × 7 pairs × 2 IDs = 1,568 bytes raw pair table
#   ~100 unique region centroids × 8 bytes = ~800 bytes
#   Compressed total: well under 1 KB
#
# Serialization: JSON → zlib compressed → magic header
#   [4 bytes: "WMK!"] [4 bytes: uint32 uncompressed len] [N bytes: zlib payload]
#   JSON chosen over msgpack for zero extra dependencies; compresses well with zlib.

import json
import zlib
import struct
from dataclasses import dataclass, field

SIDECAR_MAGIC = b"WMK!"


@dataclass
class Sidecar:
    centroids: dict  # dict[int, tuple[float, float]] — before-region ID → (cx, cy)
    pairs: list      # list[list[tuple[int, int]]]    — pairs_for_bits[j] = K (i,j) tuples
    metadata: dict = field(default_factory=dict)


def build_sidecar(centroids_all, pairs_for_bits, metadata) -> Sidecar:
    """
    Construct a Sidecar, filtering centroids down to only referenced regions.

    Parameters
    ----------
    centroids_all  : dict[int, (float, float)] — ALL region centroids from the before-image
    pairs_for_bits : list[list[tuple[int,int]]] — output of assign_pairs_to_bits
    metadata       : dict — encode-time params (n_segments, k, rs_overhead, etc.)

    Returns
    -------
    Sidecar
    """
    referenced = {
        r
        for bit_pairs in pairs_for_bits
        for (i, j) in bit_pairs
        for r in (i, j)
    }
    centroids = {r: centroids_all[r] for r in referenced}
    return Sidecar(centroids=centroids, pairs=pairs_for_bits, metadata=metadata)


def serialize_sidecar(sidecar) -> bytes:
    """
    Serialize a Sidecar to a compressed binary blob suitable for EXIF or a .wm file.
    """
    payload = {
        "centroids": {str(r): list(c) for r, c in sidecar.centroids.items()},
        "pairs": [[[i, j] for (i, j) in bit_pairs] for bit_pairs in sidecar.pairs],
        "metadata": sidecar.metadata,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    compressed = zlib.compress(raw, level=9)
    header = SIDECAR_MAGIC + struct.pack(">I", len(raw))
    return header + compressed


def deserialize_sidecar(blob) -> Sidecar:
    """
    Deserialize a compressed binary blob back to a Sidecar object.
    """
    if blob[:4] != SIDECAR_MAGIC:
        raise ValueError("not a valid sidecar: bad magic bytes (expected b'WMK!')")
    expected_len = struct.unpack(">I", blob[4:8])[0]
    raw = zlib.decompress(blob[8:])
    if len(raw) != expected_len:
        raise ValueError(f"decompressed length mismatch: got {len(raw)}, expected {expected_len}")
    payload = json.loads(raw.decode())
    centroids = {
        int(k): (float(v[0]), float(v[1]))
        for k, v in payload["centroids"].items()
    }
    pairs = [
        [(int(i), int(j)) for (i, j) in bit_pairs]
        for bit_pairs in payload["pairs"]
    ]
    return Sidecar(centroids=centroids, pairs=pairs, metadata=payload["metadata"])


def sidecar_to_file(sidecar, path) -> None:
    with open(path, "wb") as f:
        f.write(serialize_sidecar(sidecar))


def sidecar_from_file(path) -> Sidecar:
    with open(path, "rb") as f:
        return deserialize_sidecar(f.read())
