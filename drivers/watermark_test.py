import cv2
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from watermark.encoder import encode_watermark, EncodeParams
from watermark.decoder import decode_watermark, DecodeError
from watermark.sidecar import sidecar_to_file, sidecar_from_file
from regions.approach_regions import slic_superpixels
from descriptors.dwt_descriptor import compute_raw_dwt_ll
from watermark.centroid_matching import compute_region_centroids, match_regions_by_centroid
from watermark.pair_pool import build_pair_pool, assign_pairs_to_bits
from watermark.reed_solomon import rs_encode_bytes, bytes_to_bits, ERASURE, bits_to_bytes_with_erasures

# ── EDIT THESE ────────────────────────────────────────────────────────────────
ORIGINAL_IMAGE  = "data/dog.jpg"
ALTERED_IMAGE   = "data/dog_altered3.jpg"   # swap this to test a new edit
MESSAGE         = b"hi im a dog named bruno"
KEY             = 42
SIDECAR_PATH    = "dog_bruno.wm"

PARAMS = EncodeParams(
    k            = 7,
    rs_overhead  = 2.0,
    min_margin   = 0.05,
)
# ─────────────────────────────────────────────────────────────────────────────

def encode():
    image = cv2.imread(ORIGINAL_IMAGE)
    if image is None:
        print(f"ERROR: could not load {ORIGINAL_IMAGE}"); return

    print(f"Encoding '{MESSAGE.decode()}' into {ORIGINAL_IMAGE} ({image.shape[1]}×{image.shape[0]} px)")
    sidecar = encode_watermark(image, MESSAGE, KEY, PARAMS)
    sidecar_to_file(sidecar, SIDECAR_PATH)
    print(f"  {len(sidecar.pairs)} RS-encoded bits, {len(sidecar.centroids)} regions referenced")
    print(f"  Sidecar saved → {SIDECAR_PATH} ({os.path.getsize(SIDECAR_PATH):,} bytes)\n")


def decode():
    after = cv2.imread(ALTERED_IMAGE)
    if after is None:
        print(f"ERROR: could not load {ALTERED_IMAGE}"); return

    sidecar = sidecar_from_file(SIDECAR_PATH)
    print(f"Decoding {ALTERED_IMAGE} ({after.shape[1]}×{after.shape[0]} px)")

    # Diagnostics — run manually so we can show raw output even on RS failure
    orig  = cv2.imread(ORIGINAL_IMAGE)
    seg_o = slic_superpixels(orig)
    seg_a = slic_superpixels(after)
    d_o   = compute_raw_dwt_ll(orig,  seg_o)
    d_a   = compute_raw_dwt_ll(after, seg_a)
    c_a   = compute_region_centroids(seg_a)

    match  = match_regions_by_centroid(sidecar.centroids, c_a, centroid_threshold=40.0)
    erased = sum(1 for v in match.values() if v is None)
    total  = len(match)

    enc_bits       = bytes_to_bits(rs_encode_bytes(MESSAGE, overhead=PARAMS.rs_overhead))
    pool           = build_pair_pool(d_o, key=KEY, min_margin=PARAMS.min_margin, k=PARAMS.k)
    pairs_for_bits = assign_pairs_to_bits(pool, enc_bits, k=PARAMS.k)

    correct = wrong = erased_bits = 0
    voted_bits = []
    for exp, pairs_j in zip(enc_bits, pairs_for_bits):
        votes = []
        for (r1, r2) in pairs_j:
            s1, s2 = match.get(r1), match.get(r2)
            if s1 is None or s2 is None: continue
            votes.append(1 if d_a[s1] > d_a[s2] else 0)
        if not votes:
            erased_bits += 1
            voted_bits.append(ERASURE)
        else:
            maj = 1 if sum(votes) > len(votes) / 2 else 0
            voted_bits.append(maj)
            if maj == exp: correct += 1
            else: wrong += 1

    # Show RS decode result (or raw pre-RS bytes if RS fails)
    try:
        result = decode_watermark(after, sidecar)
        print(f"  RESULT (RS recovered): \"{result.decode('utf-8', errors='replace')}\"\n")
    except DecodeError:
        raw_bytes, _ = bits_to_bytes_with_erasures(voted_bits)
        raw_msg = raw_bytes[:len(MESSAGE)]
        printable = raw_msg.decode('utf-8', errors='replace')
        print(f"  RESULT (RS failed — raw majority vote): {repr(raw_msg)}  →  \"{printable}\"\n")

    non_erased = correct + wrong
    flip_rate  = wrong / non_erased if non_erased else 0
    byte_err   = 1 - (1 - flip_rate) ** 8
    rs_cap     = ((len(MESSAGE) * (PARAMS.rs_overhead - 1)) // 2) / (len(MESSAGE) * PARAMS.rs_overhead)

    print(f"Diagnostics:")
    print(f"  Regions found:   {total - erased}/{total} ({100*(total-erased)/total:.1f}%)")
    print(f"  Erased bits:     {erased_bits}/{len(enc_bits)}")
    print(f"  Bit flip rate:   {flip_rate*100:.2f}%  ({wrong} wrong / {non_erased} non-erased)")
    print(f"  Byte error rate: {byte_err*100:.1f}%  (RS {PARAMS.rs_overhead:.0f}x capacity: {rs_cap*100:.1f}%)")


if __name__ == "__main__":
    encode()
    decode()
