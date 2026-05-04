# watermark/sidecar_b.py
#
# SidecarB: Option 2 (Design B) sidecar format.
# Stores the full before-image region_map + per-region DWT-LL descriptor values
# so the decoder can use IoU + Hungarian matching instead of centroid proximity.
#
# Trade-off vs SidecarC (Option 3):
#   Larger file (~50-200 KB compressed vs <1 KB) but significantly more accurate
#   region matching, especially on heavy AI edits that shift region centroids.
#
# Serialization: JSON payload (descriptors, pairs, metadata) + binary region_map,
# both zlib-compressed, with a "WMB!" magic header.
#
# Format on disk:
#   [4 bytes: "WMB!"]
#   [4 bytes: uint32 json_raw_len]
#   [4 bytes: uint32 json_comp_len]
#   [4 bytes: uint32 rm_raw_len]
#   [4 bytes: uint32 rm_comp_len]
#   [json_comp_len bytes: zlib-compressed JSON]
#   [rm_comp_len bytes:   zlib-compressed uint16 region_map]

import json
import zlib
import struct
import numpy as np
from dataclasses import dataclass, field

SIDECAR_B_MAGIC = b"WMB!"


@dataclass
class SidecarB:
    region_map: np.ndarray  # (H, W) uint16 — before-image SLIC region labels
    num_regions: int         # number of unique regions in region_map
    descriptors: list        # list[float] — DWT-LL energy per before-region
    pairs: list              # list[list[tuple[int, int]]] — K (i,j) pairs per encoded bit
    metadata: dict = field(default_factory=dict)


def build_sidecar_b(region_map, num_regions, descriptors, pairs_for_bits, metadata) -> SidecarB:
    return SidecarB(
        region_map=np.asarray(region_map, dtype=np.uint16),
        num_regions=int(num_regions),
        descriptors=list(descriptors),
        pairs=pairs_for_bits,
        metadata=metadata,
    )


def serialize_sidecar_b(sidecar: SidecarB) -> bytes:
    # JSON payload: everything except the region_map array.
    payload = {
        "region_map_shape": list(sidecar.region_map.shape),
        "num_regions": sidecar.num_regions,
        "descriptors": sidecar.descriptors,
        "pairs": [[[i, j] for (i, j) in bit_pairs] for bit_pairs in sidecar.pairs],
        "metadata": sidecar.metadata,
    }
    json_raw = json.dumps(payload, separators=(",", ":")).encode()
    json_comp = zlib.compress(json_raw, level=9)

    # Region map: uint16 little-endian binary, then compressed.
    rm_raw = sidecar.region_map.astype(np.uint16).tobytes()
    rm_comp = zlib.compress(rm_raw, level=9)

    header = (
        SIDECAR_B_MAGIC
        + struct.pack(">I", len(json_raw))
        + struct.pack(">I", len(json_comp))
        + struct.pack(">I", len(rm_raw))
        + struct.pack(">I", len(rm_comp))
    )
    return header + json_comp + rm_comp


def deserialize_sidecar_b(blob: bytes) -> SidecarB:
    if blob[:4] != SIDECAR_B_MAGIC:
        raise ValueError(
            f"not a valid SidecarB: expected magic b'WMB!', got {blob[:4]!r}"
        )

    offset = 4
    json_raw_len  = struct.unpack(">I", blob[offset:offset + 4])[0]; offset += 4
    json_comp_len = struct.unpack(">I", blob[offset:offset + 4])[0]; offset += 4
    rm_raw_len    = struct.unpack(">I", blob[offset:offset + 4])[0]; offset += 4
    rm_comp_len   = struct.unpack(">I", blob[offset:offset + 4])[0]; offset += 4

    json_raw = zlib.decompress(blob[offset: offset + json_comp_len])
    if len(json_raw) != json_raw_len:
        raise ValueError(
            f"JSON decompressed length mismatch: got {len(json_raw)}, expected {json_raw_len}"
        )
    offset += json_comp_len

    rm_raw = zlib.decompress(blob[offset: offset + rm_comp_len])
    if len(rm_raw) != rm_raw_len:
        raise ValueError(
            f"region_map decompressed length mismatch: got {len(rm_raw)}, expected {rm_raw_len}"
        )

    payload = json.loads(json_raw.decode())
    region_map = np.frombuffer(rm_raw, dtype=np.uint16).reshape(payload["region_map_shape"])

    pairs = [
        [(int(i), int(j)) for (i, j) in bit_pairs]
        for bit_pairs in payload["pairs"]
    ]

    return SidecarB(
        region_map=region_map,
        num_regions=payload["num_regions"],
        descriptors=payload["descriptors"],
        pairs=pairs,
        metadata=payload["metadata"],
    )


def sidecar_b_to_file(sidecar: SidecarB, path: str) -> None:
    with open(path, "wb") as f:
        f.write(serialize_sidecar_b(sidecar))


def sidecar_b_from_file(path: str) -> SidecarB:
    with open(path, "rb") as f:
        return deserialize_sidecar_b(f.read())
