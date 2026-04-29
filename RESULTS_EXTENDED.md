# Pairwise Region Stability Evaluation — Extended Results (Multi-Editor, Hard Difficulty)

---

## What This Evaluation Adds

The first evaluation ([RESULTS.md](RESULTS.md)) established that SLIC + DWT is a viable foundation for pairwise-inequality watermarking. It ran Mode B on **10 pairs from a single editor (Gemini-IG) at easy difficulty only**. That was enough to confirm the approach works — but not enough to find where it breaks.

This evaluation runs the same pipeline on **116 before/after pairs** spanning:
- **4 AI editors**: Gemini-IG, GoT, MagicBrush, UltraEdit
- **2 difficulty levels**: easy and hard
- **2 edit types**: addition (inserting a new object) and replacement (swapping an existing object)

The purpose is to find each combo's *breaking point* — the severity level and editor at which flip rate climbs above 10% or segmentation survival collapses to the point where the watermark becomes impractical. A combo that only works against one easy editor is not a real candidate for deployment.

The stratum key in this run is `{editor}__{difficulty}__{edit_type}`, producing 16 strata. This lets us compare Gemini-IG easy vs. hard directly, addition vs. replacement within the same editor, and editor-to-editor at the same difficulty.

**Note on sample sizes:** Some buckets came up short due to the unique-pairing invariant (each original image can only appear in one bucket). The worst affected were MagicBrush/easy/replacement (2 pairs), MagicBrush/hard/replacement (3 pairs), and UltraEdit/hard/addition (4 pairs). Statistics from those three strata are directionally useful but noisy — treat them as signals, not measurements.

---

## What We Tested

Evaluation setup is identical to RESULTS.md with one structural change: the driver now caches segmentation and IoU+Hungarian matching per `(image pair, segment_func)` rather than recomputing for each combo. All four SLIC subbands share one segmentation and one IoU pass per image pair. This produces identical results to the previous approach at roughly 2–3× the speed.

### Segmentation methods
- **SLIC superpixels** — ~200 compact superpixels per image
- **k-means** — 5 color-cluster regions per image
- **Watershed** — edge-based flooding, high region count

### Descriptor methods
- **DWT subbands**: LL (low-frequency energy), LH (horizontal edges), HL (vertical edges), HH (high-frequency diagonal noise)
- LBP variants were excluded — eliminated in RESULTS.md with 27–36% flip rates

### Dataset
- 116 FragFake before/after pairs
- All 4 editors, both difficulties, both edit types
- MIN_MARGIN = 0.05 (pairs with |d_i − d_j| < 0.05 excluded from flip rate computation)
- FLIP_RATE_TARGET = 10%

---

## Major Findings

**1. All combos pass the 10% aggregate flip rate target — but the aggregate conceals a critical per-stratum failure.**
The headline numbers look fine: every combo ranks PASS. However, `slic + dwt_hh` reaches **34.07% flip rate on MagicBrush hard additions**, and `slic + dwt_hl` reaches **16.56%** on the same stratum. The aggregate masks these failures because MagicBrush hard is only a fraction of the 116 pairs. Per-stratum analysis is the only honest evaluation.

**2. `slic + dwt_hh` is not the most robust combo — it is the most brittle under heavy content addition.**
RESULTS.md concluded that `dwt_hh` was the best subband, with 1.88% flip on synthetic transforms and 1.78% on Gemini-IG easy AI edits. This evaluation reverses that conclusion. `dwt_hh` passes every stratum *except* MagicBrush hard addition (34.07%) and GoT hard addition (9.93%, just under threshold). The HH subband captures diagonal high-frequency energy — precisely the kind of signal that new AI-inserted objects introduce in large quantities. When a model adds a complex object with fine texture detail, it floods the HH subband of the affected regions and destroys the ordering relationships that were stable before.

**3. `slic + dwt_ll` is the most robust combo overall.**
It is the only subband that stays below 10% flip rate in every single one of the 16 strata, including MagicBrush hard addition (7.63%). LL captures low-frequency brightness and color energy — a global, slowly-varying signal that AI edits perturb less than high-frequency structure. `slic + dwt_ll` replaces `slic + dwt_hh` as the recommended production combo.

**4. MagicBrush hard addition is the breaking point for SLIC segmentation.**
Across all combos, the MagicBrush hard addition stratum has the lowest segmentation survival: **51.52%** with SLIC (vs. 88% on synthetic transforms and 71–75% for other editors at easy). Nearly half of all watermark bits are lost before the descriptor question is even asked. At MagicBrush hard replacement it drops further to **43.36%** — 57% of bits lost. For watershed, the hard replacement stratum collapses to **25.97%**: three in four bits lost. This is not a descriptor failure; it is the segmentation method failing to find matching regions when image content changes substantially.

**5. MagicBrush replacement edits are paradoxically the most stable for surviving pairs.**
Despite having the worst segmentation survival, every combo reports **0.00% flip rate** on both MagicBrush easy and hard replacement. The pairs that do survive keep their ordering perfectly. This tells us something about MagicBrush's editing behavior: replacement edits swap existing objects but preserve the surrounding region structure — the frequency energy of regions that survive matching is not disrupted. Addition edits, by contrast, inject new image content that directly alters region energies in the areas where new objects are placed.

**6. GoT is the least disruptive editor for this watermarking scheme.**
GoT hard addition has **84.88% segmentation survival** — nearly as high as the 88% seen on purely synthetic transforms. GoT hard replacement holds at 77.32%. Across all 16 strata, GoT produces the highest yield numbers (13,100–13,270 pairs/image for additions) and consistently low flip rates. GoT's edits appear to leave region structure largely intact. This is the opposite of MagicBrush.

**7. Hard difficulty primarily destroys bits, not orderings.**
For Gemini-IG, going from easy to hard *reduces* flip rate for addition (4.51% → 0.80%) while dropping segmentation survival (75.5% → 67.7%). For UltraEdit, hard addition flip rates are near 0% across all subbands. The pattern is consistent: harder AI edits destroy more regions (higher segmentation failure) but the regions that do survive have more stable orderings. The real cost of harder edits is **lost bits, not wrong bits**. This confirms the finding from RESULTS.md and extends it to the hard difficulty range.

**8. The margin retention metric has a numerical overflow bug for HL, LH, and HH subbands.**
The values reported for `slic + dwt_hh` margin retention (~10^69) and `slic + dwt_hl` (~10^36) are nonsense caused by floating-point overflow. The margin retention calculation divides `after_margin / before_margin` across all surviving pairs without applying the MIN_MARGIN filter. For high-frequency subbands, many region pairs have near-zero before-margins (floating-point noise from the DWT computation), producing ratios in the billions or higher. These ratios blow up the mean. The LL subband (19.1×) and k-means combos (0.84–1.0×) report reasonable values because their descriptor scales are larger and before-margins rarely approach zero. This does not affect flip rate or yield — those metrics are correct — but margin retention should be filtered by MIN_MARGIN in `evaluation_metrics.py` before it can be trusted for HL/LH/HH subbands.

---

## Detailed Results

### Aggregate Rankings

*116 pairs, all editors, easy + hard, addition + replacement. Threshold: flip rate < 10%, MIN_MARGIN = 0.05.*

| Rank | Combo | Flip Rate | Yield (pairs/img) | Seg Survival |
|---|---|---|---|---|
| #1 | slic + dwt_ll | 3.96% ✓ | 9,780 | 68.41% |
| #2 | watershed + dwt_ll | 4.01% ✓ | 9,731 | 57.28% |
| #3 | slic + dwt_hl | 4.57% ✓ | 9,713 | 68.41% |
| #4 | slic + dwt_lh | 5.27% ✓ | 9,678 | 68.41% |
| #5 | slic + dwt_hh | 6.33% ✓ | 9,503 | 68.41% |
| #6 | watershed + dwt_hh | 3.83% ✓ | 9,374 | 57.28% |
| #7 | kmeans + dwt_hh | 0.00% ✓ | **6.1** | 62.67% |
| #8 | kmeans + dwt_ll | 0.00% ✓ | **6.1** | 62.67% |

---

### Per-Stratum Breakdown: `slic + dwt_ll` (the recommended combo)

*This is the most important table. Each row is a distinct (editor, difficulty, edit_type) condition.*

| Stratum | Flip Rate | Yield | Seg Survival |
|---|---|---|---|
| Gemini-IG / easy / addition | 4.51% ✓ | 10,793 | 75.50% |
| Gemini-IG / easy / replacement | 4.61% ✓ | 9,934 | 65.74% |
| Gemini-IG / hard / addition | 0.80% ✓ | 9,739 | 67.72% |
| Gemini-IG / hard / replacement | 3.00% ✓ | 7,989 | 60.87% |
| GoT / easy / addition | 1.69% ✓ | 13,253 | 88.38% |
| GoT / easy / replacement | 4.88% ✓ | 9,065 | 66.60% |
| GoT / hard / addition | 2.20% ✓ | 13,270 | 84.88% |
| GoT / hard / replacement | 1.71% ✓ | 10,755 | 77.32% |
| MagicBrush / easy / addition | 7.95% ✓ | 9,914 | 68.33% |
| MagicBrush / easy / replacement | 0.00% ✓ | 9,202 | 59.97% |
| MagicBrush / hard / addition | 7.63% ✓ | 7,088 | 51.52% |
| MagicBrush / hard / replacement | 0.00% ✓ | 5,867 | 43.36% |
| UltraEdit / easy / addition | 1.59% ✓ | 9,374 | 62.45% |
| UltraEdit / easy / replacement | 2.74% ✓ | 6,767 | 60.21% |
| UltraEdit / hard / addition | 1.08% ✓ | 9,763 | 66.46% |
| UltraEdit / hard / replacement | 0.78% ✓ | 8,231 | 61.19% |

`slic + dwt_ll` is the only combo to pass the 10% threshold in all 16 strata.

---

### The Breaking Point: `slic + dwt_hh` per stratum

*The same table for the combo that RESULTS.md previously ranked as best.*

| Stratum | Flip Rate | Yield | Seg Survival |
|---|---|---|---|
| Gemini-IG / easy / addition | 0.30% ✓ | 10,566 | 75.50% |
| Gemini-IG / easy / replacement | 2.64% ✓ | 9,760 | 65.74% |
| Gemini-IG / hard / addition | 0.91% ✓ | 9,518 | 67.72% |
| Gemini-IG / hard / replacement | 0.00% ✓ | 7,826 | 60.87% |
| GoT / easy / addition | 0.90% ✓ | 12,808 | 88.38% |
| GoT / easy / replacement | 0.00% ✓ | 8,791 | 66.60% |
| GoT / hard / addition | **9.93%** ✓ | 12,944 | 84.88% |
| GoT / hard / replacement | 6.81% ✓ | 10,598 | 77.32% |
| MagicBrush / easy / addition | 5.14% ✓ | 9,611 | 68.33% |
| MagicBrush / easy / replacement | 0.00% ✓ | 8,846 | 59.97% |
| MagicBrush / hard / addition | **34.07%** ✗ | 6,776 | 51.52% |
| MagicBrush / hard / replacement | 0.00% ✓ | 5,534 | 43.36% |
| UltraEdit / easy / addition | 1.28% ✓ | 8,982 | 62.45% |
| UltraEdit / easy / replacement | 0.00% ✓ | 6,697 | 60.21% |
| UltraEdit / hard / addition | 0.00% ✓ | 9,348 | 66.46% |
| UltraEdit / hard / replacement | 0.00% ✓ | 7,984 | 61.19% |

`slic + dwt_hh` looks excellent on 14 of 16 strata but fails catastrophically on the 15th. This is exactly the kind of hidden fragility that single-editor, easy-only evaluation cannot detect.

---

## Contrast with RESULTS.md

### What RESULTS.md got right

- SLIC + DWT is the correct family of combos — confirmed across all 4 editors and both difficulty levels.
- k-means and watershed are not viable for opposite reasons (too few pairs vs. too few surviving regions) — confirmed and strengthened.
- The primary cost of AI editing is lost bits (segmentation failure), not wrong bits (descriptor flip) — confirmed and extended to hard difficulty.
- Segmentation survival under AI editing is the binding constraint, not flip rate — confirmed.

### What RESULTS.md got wrong (because it only tested Gemini-IG easy)

| Claim in RESULTS.md | Verdict from this evaluation |
|---|---|
| "`slic + dwt_hh` is the most consistent combo" | **Reversed.** dwt_hh fails at 34.07% on MagicBrush hard addition. dwt_ll is the most consistent. |
| "Pairwise orderings are equally stable under AI edits as pixel noise" | **Qualified.** True for Gemini-IG and GoT. False for MagicBrush hard additions. |
| "Segmentation survival drops by ~17pp going to AI edits" | **Partial.** The drop is editor-dependent: 3pp for GoT, 7–8pp for Gemini-IG/UltraEdit, 20–37pp for MagicBrush hard. |
| "slic + dwt_hh: 1.78% flip in Mode B" | **Scope too narrow.** That number is Gemini-IG easy only. The same combo is 34.07% on MagicBrush hard. |
| "Next step: scale up to 100 COCO images and larger FragFake sample" | **Superseded.** The more informative next step is understanding why MagicBrush hard addition breaks the HH subband and whether the segmentation or descriptor is to blame. |

### Subband ranking revision

RESULTS.md ranked the four subbands: `hh > hl > ll > lh`. This evaluation produces a different ranking:

| Subband | Old rank (Gemini-IG easy only) | New rank (4 editors, easy + hard) | Why changed |
|---|---|---|---|
| dwt_ll | #3 | **#1** | Survives all 16 strata including MagicBrush hard addition |
| dwt_hl | #2 | #2 | Passes all strata but reaches 16.56% on MagicBrush hard addition — close to failing |
| dwt_lh | #4 | #3 | Higher variance but stays below 10% across strata |
| dwt_hh | #1 | **#4** | Collapses to 34.07% on MagicBrush hard addition |

The lesson: high-frequency subbands (HH, HL) are more sensitive to new content being added because AI-generated objects have rich high-frequency texture detail. Low-frequency subbands (LL) measure energy at a scale coarser than individual objects and are therefore more invariant to content insertion.

---

## Method-by-Method Verdict

### SLIC + dwt_ll — Viable. Recommended for production.
Passes the 10% flip rate threshold in all 16 strata, including the hardest condition in this dataset (MagicBrush hard addition: 7.63%). Yield is ~7,088–13,270 pairs/image depending on stratum — the lower end at MagicBrush hard is still large enough for a practical watermark. This is the only combo in this evaluation that can be described as unconditionally robust across editors and difficulty levels.

### SLIC + dwt_hl — Conditionally viable.
Passes in 15 of 16 strata but reaches 16.56% on MagicBrush hard addition — 6.5 points above threshold. Acceptable as a secondary or ensemble subband but not as the sole watermark carrier. Would be appropriate in a multi-subband scheme where HL bits are treated as lower-reliability and given more ECC redundancy.

### SLIC + dwt_lh — Conditionally viable.
Peaks at 11.79% on MagicBrush hard addition — just above threshold. Same verdict as HL: usable in an ensemble but not standalone.

### SLIC + dwt_hh — Not viable as a standalone combo.
The 34.07% flip rate on MagicBrush hard addition disqualifies it from standalone use. It is excellent on 14 of 16 strata — but a deployed watermark cannot selectively avoid MagicBrush hard edits. RESULTS.md's recommendation to use dwt_hh as the primary combo was premature; that conclusion was an artifact of the limited evaluation scope.

### Watershed + DWT — Not viable.
The previous verdict is strengthened. Watershed hard replacement (MagicBrush) collapses to 25.97% segmentation survival. At 57.28% overall survival vs. 68.41% for SLIC, watershed consistently loses more bits and gains nothing in return.

### k-means + DWT — Not viable.
~6 pairs/image confirmed across 116 pairs. The 0.00% flip rates are partly an artifact of `0/0` — in many strata there are no pairs above the MIN_MARGIN threshold at all, so the flip rate denominator is zero. MagicBrush hard replacement produces zero surviving usable pairs.

---

## Known Issue: Margin Retention Overflow

The `mean_margin_retention` values for the HL, LH, and HH subbands are numerically invalid (reported values in the range 10^34 to 10^69). The cause: `evaluation_metrics.compute_metrics` computes the ratio `after_margin / before_margin` for all surviving pairs without applying the MIN_MARGIN filter. For high-frequency subbands, many pairs have near-zero before-margins from floating-point noise in the DWT, producing enormous individual ratios that blow up the mean. The fix is to apply the same `min_margin_threshold` filter to the margin retention calculation as is already applied to the flip rate. This bug does not affect flip rate, yield, or segmentation survival — those three metrics are computed correctly.

---

## Conclusion and Recommendations

### What is confirmed

The pairwise region ordering approach to image watermarking is viable. Across 116 image pairs spanning 4 AI editors, 2 difficulty levels, and 2 edit types, SLIC superpixel segmentation combined with DWT subband descriptors produces watermark encodings that survive real AI manipulation at useful bit budgets (~7,000–13,000 pairs per image).

### The production recommendation

**`slic + dwt_ll` is the correct combo for a deployed watermark.** It is the only combination in this evaluation that stays below 10% flip rate across every tested condition, including the hardest: MagicBrush hard object additions. The previous recommendation of `slic + dwt_hh` was based on easy-difficulty testing only and does not hold under harder AI edits.

### The binding constraint

**Segmentation survival under hard AI editing** remains the main limitation. Even `slic + dwt_ll` loses 48.5% of watermark bits on MagicBrush hard addition (51.5% survival) and 56.6% on MagicBrush hard replacement (43.4% survival). The surviving pairs are reliable — the problem is how many there are to begin with. An ECC layer must be designed to tolerate this: at hard edit conditions, roughly half the bits are lost as erasures (segmentation failure) and ~8% of the surviving bits are wrong (flip rate). A code that can correct 8% errors and 50% erasures simultaneously is the design target.

### Next steps

**1. Fix the margin retention overflow in `evaluation_metrics.py`.**
Apply `min_margin_threshold` to the retention ratio computation, same as it is applied to flip rate. This is a one-function fix and makes the metric trustworthy for all subbands.

**2. Investigate why MagicBrush hard addition breaks SLIC segmentation so severely.**
Segmentation survival drops to 51% on this stratum — 37 percentage points below GoT hard addition. The question is whether MagicBrush's edits are genuinely larger in spatial scale, or whether they produce images whose color/texture properties cause SLIC to produce very different superpixel boundaries. Visualizing a few before/after pairs from this stratum alongside their SLIC segmentation maps would answer this directly.

**3. Test a semantically-grounded segmentation (SAM or similar) on MagicBrush hard.**
The STABILITY_PLAN flagged this as an open question. The evidence now points to it as the highest-priority experiment: SLIC's superpixels are defined by local color homogeneity, which breaks when AI editing introduces new objects. A segmentation method that produces semantically stable regions (background, sky, furniture, etc.) that survive content changes would close the survival gap. The evaluation infrastructure is already in place — adding a new `segment_func` and running it through the same pipeline is all that is required.

**4. Design the ECC layer using the per-stratum distributions.**
The data now provides the two numbers the ECC designer needs per deployment scenario:
- For easy edits (Gemini-IG / GoT): ~70–88% survival, ~2–5% flip rate. A lightweight code can handle this.
- For hard edits (MagicBrush): ~43–52% survival, ~0–8% flip rate. This is an erasure-dominated regime — a code with strong erasure correction capacity (e.g. Reed-Solomon or LDPC with erasure decoding) and moderate error correction.
- Choose the code for the hardest condition you expect to encounter in deployment.

**5. Scale up the evaluation to 50+ pairs per stratum for statistical confidence.**
The current run has as few as 2–4 pairs in the low-count buckets. Even the full-count buckets (10 pairs) are preliminary. Running at 50 pairs per (editor, difficulty, edit_type) bucket would tighten the confidence intervals enough to draw firm quantitative conclusions. The `--limit` argument in `download_fragfake.py` and `--splits train,test` make this straightforward.