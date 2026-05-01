# watermark/__init__.py
#
# Public API for the watermarking system.
#
# Two entry points are exposed:
#
#   encode_watermark(image, message, key) -> sidecar
#       Takes a numpy image and a bytes payload, returns a Sidecar object
#       that can be serialized to ~6-12 KB and stored alongside the image.
#
#   decode_watermark(edited_image, sidecar) -> bytes
#       Takes the AI-edited image and the sidecar, recovers the original
#       bytes payload without needing the original image.
#
# Everything else in this package is internal to those two pipelines.
# External callers should only import from here.
#
# Dependency map (what calls what):
#
#   encoder.py
#     ├── pair_pool.py          (build + shuffle the candidate pair pool)
#     ├── reed_solomon.py       (RS encode the raw message bytes → bit array)
#     └── sidecar.py            (assemble + serialize the sidecar)
#
#   decoder.py
#     ├── centroid_matching.py  (nearest-centroid lookup on edited image)
#     ├── reed_solomon.py       (RS decode with explicit erasure positions)
#     └── sidecar.py            (deserialize the sidecar)
#
# Both encoder and decoder call into the existing codebase:
#   regions/approach_regions.py   → slic_superpixels()
#   descriptors/dwt_descriptor.py → compute_dwt_ll()

from watermark.encoder import encode_watermark
from watermark.decoder import decode_watermark
from watermark.sidecar import Sidecar

__all__ = ["encode_watermark", "decode_watermark", "Sidecar"]
