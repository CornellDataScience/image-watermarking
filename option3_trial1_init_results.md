# Option 3 — Trial 1 Initial Results

## Setup

**Image:** `data/dog.jpg` — white fluffy dog holding a purple Skimmer frisbee in its mouth, standing on green grass. Resolution 640×611 px.

**Message:** `b"hi im a dog named bruno"` (23 bytes)

**Encode parameters:** K=7, rs_overhead=2×, min_margin=0.05 (defaults)

**Sidecar:** `dog_bruno.wm` (12,479 bytes) — 368 RS-encoded bits, 182 referenced regions

**Key:** 42

---

## Test 1 — Frisbee recolor + ground replacement

**Altered image:** `data/dog_altered.jpg`

**Edit description:** Purple frisbee recolored to bright red. Green grass ground replaced with dark muddy dirt. Dog itself unchanged. Same resolution (640×611).

**Result: DECODE FAILED**

```
Reed-Solomon decode failed: Too many errors to correct
  0/368 bit positions erased (0.0%)
```

**Diagnostics:**

| Metric | Value |
|---|---|
| Regions found by centroid | 182/182 (100%) |
| Raw per-pair flip rate | 32.3% (5,313 / 16,471 pairs) |
| Bit flip rate after K=7 majority vote | 19.8% (73/368 bits wrong) |
| Byte error rate | ~75% |
| RS 2× correction capacity | ~24% |

**Why it failed:** DWT-LL measures coarse average brightness energy per region. The frisbee went from dark purple to bright red (large LL energy increase in that region) and the ground went from bright green grass to dark mud (large LL energy decrease across a large area). Both are exactly the regions the pair pool relied on for stable orderings. With 32.3% of pairs flipping, the RS decoder was overwhelmed — nearly 3× over its correction capacity.

**Parameter sweep attempted:** Tested K ∈ {7, 11, 15, 21} and min_margin ∈ {0, 1000, 3000, 5000, 8000, 12000, 20000} and RS overhead ∈ {2×, 3×, 4×}. No combination succeeded — the pair pool for this image (~8,000 pairs) is too small to support the high K needed to bring the bit flip rate below RS correction capacity at any viable overhead.

**Root cause:** Color replacement and ground swap are global area changes, not local object insertions. The design doc's note applies directly: *"LL is more stable than HH under AI editing — true for object insertion edits. May not hold for global edits (style transfer, color grading, night/day conversion)."*

---

## Test 2 — Red collar addition + slight zoom out

**Altered image:** `data/dog_altered2.jpg`

**Edit description:** Red collar added around the dog's neck. Image appears very slightly zoomed out (more grass visible at edges) but was output at the same 640×611 resolution. Frisbee color, ground, and dog body unchanged.

**Result: DECODE SUCCESS**

```
"hi im a dog named bruno"
```

**Diagnostics:**

| Metric | Value |
|---|---|
| Regions found by centroid | 181/182 (99.5%) |
| Bit flip rate after K=7 majority vote | 0.82% (3/368 bits wrong) |
| Byte error rate | 6.3% |
| RS 2× correction capacity | ~24% |

**Why it succeeded:** The collar is a small, localized addition confined to 1–2 superpixel regions on the dog's neck. DWT-LL energy of every other region (frisbee, grass, background, dog body) was completely unaffected. Only 3 bits out of 368 were wrong — RS at 2× corrected them trivially, using only ~25% of its available correction budget.

The slight zoom-out caused exactly 1 region to drift beyond the 40px centroid threshold → treated as a known erasure. RS handles erasures at twice the efficiency of random errors, so this added negligible burden.

**Sidecar used:** `dog_bruno.wm` — the default K=7, 2× RS encode. No parameter changes needed.

---

## Summary

| Edit type | Flip rate | Result |
|---|---|---|
| Color replacement + ground swap (large area, LL-energy-disrupting) | 32.3% raw | FAIL — exceeds RS capacity at any tested parameter |
| Small object addition (collar) + minor zoom | 0.82% after majority vote | PASS — comfortable margin |

**Key finding:** The system performs as designed for local object additions. Global color/content changes that shift the LL energy of large image areas (frisbee, background) are outside the threat model and cannot be recovered regardless of parameter tuning, given the pair pool size of a typical image (~8,000 pairs).
