# Trial 1 — Encoding Implementation & Test Results

## What was built

Six files were implemented in the `watermark/` folder, completing the encode/decode pipeline described in `approach3_encoding_inital_explaination.md`:

| File | Role |
|---|---|
| `centroid_matching.py` | Computes region centroids; matches before→after regions by nearest Euclidean centroid |
| `reed_solomon.py` | Wraps `reedsolo` library; handles RS encode/decode and bit↔byte conversion with erasure positions |
| `pair_pool.py` | Generates all ~19,900 region pairs, filters by margin, splits by sign, shuffles with secret key |
| `sidecar.py` | Sidecar dataclass + JSON/zlib serialize/deserialize; stored as `[WMK!][len][compressed]` |
| `encoder.py` | Wires steps 1–8: SLIC → centroids → descriptors → pair pool → RS encode → assign pairs → sidecar |
| `decoder.py` | Wires steps 1–8: SLIC → centroids → descriptors → centroid match → majority vote → RS decode |

---

## Bug found during testing: wrong descriptor

**Problem:** `compute_dwt_ll` returns the *normalized LL energy fraction* (E_LL / total_energy), which for real photographs is always in [0.90, 1.00] with std ≈ 0.012. At MIN_MARGIN=0.05, only ~200 of 19,900 pairs survived — far too few to embed even a short message.

**Diagnosis:**
```
descriptor min:  0.904
descriptor max:  1.000
descriptor std:  0.012
pairs surviving margin 0.05:  211
pool_0 size: 50  →  total_capacity: 7 bits  (need 112 for "Cornell")
```

**Fix:** Added `compute_raw_dwt_ll` to `descriptors/dwt_descriptor.py`. This returns `sum(cA²) / num_region_pixels` — raw LL energy normalized by region pixel count instead of total subband energy. This matches what the design doc intended: `d[k] = E_LL = sum(cA²)`, area-normalized so ordering reflects image brightness rather than superpixel size.

```
descriptor min:  1,683
descriptor max:  111,344
descriptor std:  25,696
pairs surviving margin 0.05:  16,471  (effectively all pairs)
pool_0: 7,902  pool_1: 8,569  →  total_capacity: 1,128 bits
```

---

## Bug found: resolution mismatch

**Problem:** FragFake before/after images can have different resolutions (e.g. 375×500 → 765×1024). Sidecar centroids are in before-image pixel space. With a 2× resolution difference, centroid matching places after-image centroids at twice the coordinates, breaking nearest-centroid lookup.

**Fix:** Stored `image_height` and `image_width` in `sidecar.metadata`. The decoder resizes the edited image to the before-image dimensions before running SLIC, ensuring centroid coordinates are in the same pixel space.

---

## Test results

### Test 1 — Zero-edit baseline (same image encode and decode)
**Image:** `data/dog.jpg` (611×640)
**Message:** `b"Cornell"`

```
encode OK — sidecar pairs: 112
decode OK — Cornell  ✓
```

### Test 2 — Easy Gemini-IG replacement
**Before:** `original_tv_000000394600.jpg` (427×640)
**After:** `tv_000000394600_replacement.png` (683×1024, resized to 427×640 for decoding)
**Edit type:** Easy replacement

```
encode OK — pairs: 112
decode OK — Cornell  ✓
```

### Test 3 — Hard Gemini-IG replacement
**Before:** `original_toilet_000000375194.jpg` (375×500)
**After:** `toilet_000000375194_replacement.png` (765×1024)
**Edit type:** Hard replacement

```
encode OK — pairs: 112
DecodeError: Reed-Solomon decode failed: Too many errors to correct
  0/112 bit positions erased (0.0%)
```

**Why this fails:** Hard replacement edits change the image content enough to flip the DWT-LL ordering of most region pairs. Diagnostics showed:

```
centroid matches: 174/174  (0 erasures — all regions still findable)
effective vote flip rate: 53.83%  (basically random)
```

Because all 174 regions are still spatially present (centroid drift < 40px), they register as matched, not erased. But the region content was replaced, so descriptor orderings flip. With a 53% flip rate at K=7 witnesses, ~58% of majority votes are wrong. RS at 2× overhead can correct at most ~25% bit errors with no erasure info — 58% wrong is far beyond capacity.

---

## What the results mean

The easy edit worked and the hard edit failed — this is **expected and consistent with `RESULTS_EXTENDED.md`**. The key distinction:

- **Easy edits:** Fewer and smaller changes → most region orderings survive → RS can correct the few wrong bits → decode succeeds
- **Hard replacement:** Large content replacement → ordering of affected regions completely randomized → too many wrong votes for RS at 2× overhead

The hard replacement failure mode is different from what RS is designed for. RS handles **erasures** (known missing bits) efficiently. But here all bits have votes — they're just *wrong* votes (0% erasures, 53% errors). RS at 2× can handle ~25% random errors max.

---

## What would fix hard replacement

Three options in order of cost:

1. **Increase K (witnesses per bit):** K=11 instead of K=7. More votes → majority is more likely to be right. Helps when flip rate is driven by a minority of affected pairs.

2. **Increase RS overhead:** 4× instead of 2× gives RS enough parity to correct 50% random errors. Message doubles in encoded bits: "Cornell" → 7 bytes → 56 bytes encoded → 448 bits. Still within pair pool capacity (1,128 bits).

3. **Tighter min_margin filter:** Only use pairs with large margin (d[i] >> d[j]). High-margin pairs are more likely to survive aggressive edits because a larger change is needed to flip their ordering. The current effective min_margin is ≈0 (raw LL values range 0–111,344 so 0.05 filters nothing). Setting min_margin = 5000–10000 would concentrate pairs in the high-confidence region.

---

## State of the system after Trial 1

| Component | Status |
|---|---|
| Imports and module structure | ✓ Working |
| Encoder (all 8 steps) | ✓ Working |
| Decoder (all 8 steps) | ✓ Working |
| Zero-edit baseline | ✓ Passes |
| Easy AI edit (Gemini-IG easy) | ✓ Passes |
| Hard AI edit (Gemini-IG hard replacement) | ✗ Fails — exceeds RS capacity |
| Resolution mismatch handling | ✓ Fixed |
| Descriptor variance issue | ✓ Fixed (raw LL per pixel) |
