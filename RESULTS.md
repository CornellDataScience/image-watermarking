# Pairwise Region Stability Evaluation — Full Results Write-Up

---

## What This Research Is

This project investigates a technique for **image watermarking using pairwise region ordering**. The core idea: split an image into regions (superpixels or clusters), compute a descriptor value for each region, and use the *relative ordering* of those descriptor values — which region has a higher value than which other region — as a way to encode hidden information in the image.

For example: if region 3 has a higher DWT energy than region 7, that inequality encodes a single bit: `(3 > 7) = 1`. A watermark is thousands of these bits packed together from all possible region pairs in the image.

For this to work as a watermark, two things must be true:
1. **The orderings must survive normal image processing** — JPEG compression, blurring, resizing, brightness adjustment. If these destroy the orderings, the watermark can't be read back.
2. **The orderings must survive real AI-based editing** — if someone uses an AI tool to alter the image (add or replace objects), the watermark should ideally still be readable.

The evaluation measures **flip rate**: what fraction of pairwise orderings reverse sign after an edit. Low flip rate = orderings are stable = watermark survives. The target is **below 10% flip rate** — above that, too many bits are wrong to decode the watermark reliably.

We also track **usable pair yield** — how many region pairs per image have a large enough margin (|d_i − d_j| > 0.05) to be reliable in the first place. This is the practical bit budget. Higher yield = more bits you can embed.

---

## What We Tested

### Segmentation methods (how to split the image into regions)
- **SLIC superpixels** — splits the image into ~200 compact, color-homogeneous superpixels. High region count = high yield.
- **k-means** — clusters pixels into 5 color groups. Very coarse — only ~5 regions per image.
- **Watershed** — edge-based flooding algorithm that finds natural region boundaries. High region count but unstable under transforms.

### Descriptor methods (what value to assign each region)
- **LBP (Local Binary Pattern)** — texture-based descriptor measuring local pixel patterns within each region. Four variants: mean, entropy, nonuniform, edge.
- **DWT (Discrete Wavelet Transform)** — frequency-based descriptor measuring energy in four frequency subbands: LL (low-frequency approximation), LH (horizontal edges), HL (vertical edges), HH (diagonal edges/high-frequency noise).

### Two evaluation modes
- **Mode A** — Synthetic transforms: 10 natural images, each subjected to 7 transforms (JPEG Q30, JPEG Q50, resize ×0.5, blur k=5, blur k=9, brightness +30, brightness −30). 70 before/after pairs total. Tests the floor: can the combo survive normal post-processing?
- **Mode B** — Real AI edits: 10 pairs from the FragFake dataset (Gemini-IG, easy difficulty, addition and replacement edits). Tests semantic robustness: does the combo survive actual content manipulation by an AI model?

---

## Major Findings

**1. LBP descriptors are completely unsuitable for this application.**
Across all segmentation methods, LBP flip rates ranged from 27% to 36% — three to four times above the 10% target. LBP is a pixel-level texture measure and is destroyed by even minor operations like JPEG compression or a brightness shift. It was eliminated from further testing.

**2. SLIC + DWT is the only viable combination.**
All four DWT subbands paired with SLIC superpixels pass the 10% flip rate target in both Mode A and Mode B, with ~10,000–13,000 usable pairs per image. This is by far the best combination of stability and bit budget.

**3. k-means passes but is practically useless.**
With only 5 regions, k-means produces at most 10 region pairs per image — a bit budget of roughly 3 bits. This is orders of magnitude too small to embed a meaningful watermark, even though the flip rate is near 0%.

**4. Watershed fails on segmentation survival.**
Watershed passes the flip rate target but only 60–70% of region pairs survive transforms and AI edits (vs. 71–88% for SLIC). Losing 30–40% of your bit budget to failed region matching makes it unreliable for real-world use.

**5. Pairwise orderings are equally stable under real AI edits as under pixel-level noise.**
This is the most significant finding: Mode B flip rates are nearly identical to Mode A flip rates for SLIC + DWT. Real AI editing (adding or replacing objects) does not destabilize the pairwise ordering structure any more than JPEG compression does. The watermark is robust to AI manipulation.

**6. `slic + dwt_hh` is the most consistent combo across all conditions.**
1.88% flip rate in Mode A, 1.78% in Mode B — essentially unchanged. The HH subband captures diagonal high-frequency energy, which is highly stable across both pixel noise and semantic edits.

---

## Detailed Results

### Mode A — Synthetic Transforms

*10 images × 7 transforms = 70 before/after pairs. Threshold: flip rate < 10%, MIN_MARGIN = 0.05.*

| Rank | Combo | Flip Rate | Yield (pairs/img) | Seg Survival |
|---|---|---|---|---|
| #1 | slic + dwt_ll | 4.43% ✓ | 12,678 | 88.05% |
| #2 | slic + dwt_lh | 3.95% ✓ | 12,607 | 88.05% |
| #3 | slic + dwt_hl | 5.41% ✓ | 12,492 | 88.05% |
| #4 | slic + dwt_hh | 1.88% ✓ | 12,181 | 88.05% |
| #5 | watershed + dwt_ll | 2.50% ✓ | 12,657 | 69.82% |
| #6 | watershed + dwt_hh | 0.30% ✓ | 12,209 | 69.82% |
| #7 | kmeans + dwt_ll | 0.00% ✓ | **8.7** | 87.71% |
| #8 | kmeans + dwt_hh | 0.00% ✓ | **8.7** | 88.71% |
| — | All LBP combos | 27–36% ✗ | 9,000–10,300 | 88% |

**Per-transform breakdown (SLIC + DWT subbands):**

The hardest transform in Mode A is **Brightness +30**, not blur or JPEG as expected. This is caused by pixel saturation clipping — adding 30 to already-bright pixels forces them to 255, destroying the relative energy structure between regions. Brightness −30 is far gentler (pixels rarely hit 0 in natural images). Blur and JPEG are handled well across all DWT subbands, with flip rates of 0.5–3%.

---

### Mode B — Real AI Edits (FragFake, Gemini-IG, Easy)

*10 before/after pairs (5 addition edits, 5 replacement edits). Threshold: flip rate < 10%, MIN_MARGIN = 0.05.*

| Rank | Combo | Flip Rate | Yield (pairs/img) | Seg Survival |
|---|---|---|---|---|
| #1 | slic + dwt_hl | 1.63% ✓ | 10,612 | 71.47% |
| #2 | slic + dwt_hh | 1.78% ✓ | 10,513 | 71.47% |
| #3 | slic + dwt_ll | 5.39% ✓ | 10,719 | 71.47% |
| #4 | slic + dwt_lh | 7.36% ✓ | 10,524 | 71.47% |
| #5 | watershed + dwt_hh | 3.60% ✓ | 10,002 | 60.65% |
| #6 | watershed + dwt_ll | 4.75% ✓ | 10,282 | 60.65% |
| #7 | kmeans + dwt_ll | 0.00% ✓ | **6.2** | 65.00% |
| #8 | kmeans + dwt_hh | 0.00% ✓ | **6.0** | 62.00% |

**Edit type breakdown (SLIC + DWT):**

`easy__addition` (adding an object) is slightly harder than `easy__replacement` (replacing one) for most combos — except `slic + dwt_hl`, which is more sensitive to replacement. The `slic + dwt_lh` combo shows the sharpest reaction to additions (9.65% flip rate), suggesting the LH subband (horizontal edge energy) is most affected when new objects with horizontal structure are inserted.

**Note on AI image resolution:** FragFake AI editors output images at a different resolution than the original (e.g., 640×480 input becomes 1024×764 output). The after image is resized to match the before image before any processing — this preserves all spatial relationships while making the comparison valid.

---

## Cross-Mode Comparison

| Combo | Mode A flip | Mode B flip | Δ | Mode A seg surv | Mode B seg surv | Yield drop |
|---|---|---|---|---|---|---|
| slic + dwt_ll | 4.43% | 5.39% | +0.96% | 88% | 71% | −16% |
| slic + dwt_lh | 3.95% | 7.36% | +3.41% | 88% | 71% | −16% |
| slic + dwt_hl | 5.41% | 1.63% | −3.78% | 88% | 71% | −16% |
| slic + dwt_hh | 1.88% | 1.78% | −0.10% | 88% | 71% | −16% |

**Flip rate gap is small.** The largest increase is `slic + dwt_lh` at +3.41 percentage points. The largest decrease is `slic + dwt_hl` at −3.78 points. For 10-image samples these differences are within noise range — the full 100-image run will determine whether they hold.

**Segmentation survival drops consistently by ~17 points** across all SLIC combos going from Mode A to Mode B. This is not random — AI editing genuinely restructures the image content enough that more regions can't be confidently matched between before and after. This is the real cost of AI editing on the watermark: not destroyed bits, but lost bits (regions that can't be compared).

**Yield loss is meaningful but not catastrophic.** SLIC combos go from ~12,600 pairs/image (Mode A) to ~10,600 pairs/image (Mode B) — a 16% reduction. Still over 10,000 usable pairs per image, which is more than sufficient for a practical watermark.

---

## Method-by-Method Verdict

### SLIC + DWT — Viable. Move forward.
The only combination that balances high yield, high segmentation survival, and low flip rate. All four DWT subbands pass in both modes. The four subbands rank by stability:

1. **dwt_hh** — most stable, most consistent across modes. Best overall.
2. **dwt_hl** — slightly lower yield, very stable in Mode B (1.63%).
3. **dwt_ll** — best yield in Mode A (12,678), moderate stability.
4. **dwt_lh** — highest Mode B sensitivity (useful for detection), higher flip variance.

### Watershed + DWT — Not viable for production use.
Passes the flip rate target but at 60–70% segmentation survival, it cannot be relied upon. Under blur specifically, survival collapses to 46–55%, meaning roughly half the watermark bits are lost whenever someone blurs the image. The flip rates look good only because the surviving pairs happen to be stable — the structural fragility of the segmentation is the problem.

### k-means + DWT — Not viable for bit budget reasons.
Near-zero flip rates are impressive, but 5 regions = 10 pairs maximum = approximately 3 bits of watermark capacity. This is not a practical scheme. Would require a fundamentally different approach (many more clusters, or hierarchical k-means) to become useful.

### All LBP combinations — Eliminated.
27–36% flip rates across all segmentation methods and all four LBP variants (mean, entropy, nonuniform, edge). LBP is a local pixel pattern descriptor computed at the individual pixel level — it is hypersensitive to any operation that changes pixel values, including JPEG artifacts, brightness shifts, and blur. It is fundamentally incompatible with this application.

---

## Conclusion

The pairwise region ordering approach to image watermarking is viable, and **SLIC + DWT is the correct foundation**. The evaluation demonstrates that region-pair orderings based on DWT subband energy are stable across both standard post-processing (JPEG, blur, resize, brightness) and real AI content editing (object addition and replacement).

The primary candidate going forward is **slic + dwt_hh**: 1.88% flip rate on synthetic transforms, 1.78% on real AI edits, ~10,500–12,200 usable pairs per image. The consistency across modes is the defining property — it is not a descriptor that happens to work on easy cases, it is genuinely stable under meaningful image manipulation.

The key limitation identified is **segmentation survival under AI editing** (88% → 71%). This is not a failure of the descriptor — the orderings that do survive are reliable. It is a limitation of SLIC's region detection when image content changes structurally. Future work could explore whether larger minimum region size, or a more semantically grounded segmentation (e.g. SAM), could close that gap.

**Next step:** Run both modes at full scale (100 COCO images for Mode A, larger FragFake sample for Mode B) to confirm these preliminary numbers are stable across a statistically meaningful sample.
