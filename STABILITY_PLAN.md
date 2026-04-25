# Stability Evaluation Plan

Goal: find the `(segmentation, descriptor)` combination that best identifies
semantically stable regions of an image — regions whose pairwise descriptor
inequalities survive real-world edits (compression, blur, AI inpainting). Those
regions are the best carriers for a watermark, because information placed there
has the highest chance of surviving downstream modification.

This document specifies the evaluation loop, justifies each design decision,
and describes the additions needed beyond the code currently in `stability/`.

---

## 1. The core idea

A watermark encoded as pairwise descriptor inequalities (region *i* has a
larger descriptor value than region *j*) only survives if two things hold after
an edit:

1. **Regions *i* and *j* still exist** (segmentation didn't dissolve them).
2. **The sign of `d_i − d_j` is preserved** (descriptor ordering didn't flip).

A `(segment_func, descriptor_func)` combo is "good" to the extent that, across
a large and varied set of before/after image pairs, many region pairs satisfy
both conditions. The evaluation loop is the process of measuring this and
ranking combos.

### Why pairwise inequalities rather than absolute descriptor values

Absolute values are fragile to global shifts — a brightness change or a JPEG
pass will move every descriptor by some amount. Pairwise sign comparisons are
invariant to monotonic transformations of the descriptor axis, so they
tolerate the kinds of perturbations that preserve semantic content. This is
the premise of the existing `compute_pairwise_signs` logic in
`stability/pairwise_stability.py` and we are keeping it.

---

## 2. The per-image evaluation loop

For a single before/after pair and a single `(segment_func, descriptor_func)`
combo:

1. Segment the *before* image → regions `{0 … n−1}` with masks `B_0 … B_{n−1}`.
2. Segment the *after* image → regions `{0 … m−1}` with masks `A_0 … A_{m−1}`.
3. Compute region correspondence (see §3).
4. Compute descriptors on both images.
5. For every pair `(i, j)` with `i < j` in the before-image:
   - If either `i` or `j` has no valid match in the after-image → tag as
     **segmentation failure**.
   - Else let `i' = match(i)`, `j' = match(j)` and compare
     `sign(d_i − d_j)` to `sign(d_{i'} − d_{j'})`.
     - If signs differ → **descriptor flip**.
     - Else → **stable pair**.
6. Record per-pair: flip vs. stable vs. seg-failure, and the margin
   `|d_i − d_j|` before and after.

This separation of "segmentation failure" from "descriptor flip" is a
deliberate change from the current implementation, which treats a missing
region as a flip. Keeping them distinct is useful downstream because it tells
us whether a bad combo is the region function's fault or the descriptor
function's fault — actionable signal for Group 1 vs. Group 2.

---

## 3. Region correspondence via IoU + Hungarian matching

### The problem with the current code

`run_stability_test` in `stability/pairwise_stability.py` assumes that region
index `i` in the before-image refers to the same physical area as region `i`
in the after-image. That assumption holds for mild pixel perturbations
(JPEG, blur) where segmentation is approximately stable, but it breaks
immediately for AI edits where objects appear, disappear, or shift enough to
reshuffle indices. Under the current assumption, high flip rates on FragFake
would be uninterpretable — we would not know whether the descriptor changed
or whether we were comparing two different physical regions.

### Why IoU

For two masks `A` and `B`,

```
IoU(A, B) = |A ∩ B| / |A ∪ B|
```

IoU captures position, shape, *and* size in a single number: 1.0 means the
two masks cover identical pixel footprints, 0.0 means no overlap at all.


### Why Hungarian assignment rather than greedy matching

Greedy matching walks through before-regions in order and assigns each one to
its best-overlapping after-region. This is order-dependent and produces
incorrect assignments when an extra after-region partially overlaps several
before-regions: the first before-region grabs the extra, and the true match
downstream gets stuck with a worse alternative.

The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) solves the
full assignment problem — it finds the mapping that maximizes the sum of
IoU across all matched pairs simultaneously. This is the textbook-correct
way to match two sets of regions by a similarity matrix and is the approach
used by e.g. multi-object trackers.

Complexity is O((max(n, m))³), which is fine for segmentations in the
low-hundreds of regions (SLIC default is ~200). If a segmentation method
produces thousands of regions we will coarsen it before matching rather
than trying to scale up Hungarian.

### Handling unequal region counts

The IoU matrix is rectangular when `n ≠ m`, which Hungarian handles natively:
the smaller side is fully assigned and the larger side has leftovers. This is
exactly the behavior we want:

- **Unmatched before-regions** → the edit destroyed that region. Any pair
  `(i, j)` involving it is tagged as a segmentation failure.
- **Unmatched after-regions** → the edit added new content (AI inpainting
  added an object). Ignored for stability purposes since there is no
  before-state to compare to.

### Low-quality matches

Hungarian will always produce an assignment, even if the best available match
for some region has IoU of 0.02. A match that weak is not a match — it means
the region was destroyed and Hungarian paired it with random leftover pixels.
We apply an IoU threshold (starting value: **0.5**) *after* solving:

- Matched pair with IoU ≥ 0.5 → real correspondence, use it.
- Matched pair with IoU < 0.5 → treat as unmatched (segmentation failure).

The 0.5 threshold is the standard detection-evaluation convention (it is what
COCO uses for its primary mAP metric). We will revisit it once we see real
distributions on FragFake — if most real matches cluster above 0.7, we can
tighten; if severe AI edits push good matches down to 0.3, we loosen.

---

## 4. Dataset-level aggregation

A single image tells us almost nothing. We run the per-image loop over many
before/after pairs and aggregate per combo.

### Metrics per `(segment_func, descriptor_func)`

Across all images in an evaluation set:

- **Mean flip rate** — of pairs that survived segmentation, what fraction had
  their sign flip? Lower is better.
- **Mean margin retention** — ratio of after-margin to before-margin,
  averaged over surviving pairs. Closer to 1.0 is better.
- **Segmentation survival rate** — of all before-pairs, what fraction had
  both regions survive the edit? Higher is better.
- **Usable pair yield** — average number of pairs per image that were both
  segmentation-survived and sign-stable. This is the practical bit budget: it
  tells us how many bits of watermark the combo can actually carry on a
  typical image.

A combo is "good" to the extent that all four look good together. Flip rate
alone is insufficient — a combo that produces only two regions per image will
have a great flip rate but a useless pair yield.

### Stratification by edit severity (the main analytical lever)

FragFake ships images with multiple severity levels baked into the directory
structure: `easy` vs `hard`, `addition` vs `replacement`, and different
editors. We compute every metric above *per stratum* rather than only
globally. This gives us, for each combo, a curve across severity levels —
essentially a robustness profile.

This is where the innovation over the existing stability code lives:

1. **Synthetic transforms (JPEG / blur / resize / brightness)** — the floor.
   These are the transforms already listed in `stability/transformations.py`
   and they represent distortions that do not change semantic content.
2. **FragFake easy** — small, localized AI edits.
3. **FragFake hard** — larger AI edits.

A combo that stays flat across all three levels is genuinely robust. A combo
that is excellent on synthetic transforms but collapses on FragFake easy is
overfit to pixel-level noise and is not actually doing anything semantic.
A combo's "breaking point" — the highest severity at which flip rate stays
below some target (e.g. 10%) — becomes a rankable number and, we believe, a
more honest measure of semantic stability than any single aggregate.

### Why this framing is new relative to prior stability code

The current `run_stability_test` measures stability against a fixed battery
of synthetic transforms on a single image. It cannot distinguish pixel
robustness from semantic robustness, and it cannot tell us at what severity a
combo breaks. Running the same loop across FragFake's graded severity levels,
with proper region correspondence, turns flip rate into a function of edit
intensity rather than a single number — and that function is what lets us
compare combos meaningfully.

---

## 5. Changes required to the current code

Summarizing what needs to be built on top of what already exists:

### New module: `stability/region_matching.py`

Pure function layer, no dependencies on descriptors or the stability driver.

- `compute_iou_matrix(seg_before, seg_after) -> np.ndarray`
  Returns an `n × m` matrix of IoU values. Implementation computes, for each
  pair, `|B_i ∩ A_k| / |B_i ∪ A_k|` using boolean mask arithmetic.
- `match_regions(seg_before, seg_after, iou_threshold=0.5) -> dict[int, int]`
  Calls `compute_iou_matrix`, solves with
  `scipy.optimize.linear_sum_assignment` (note: this maximizes by negating
  the cost), applies the threshold, and returns `{before_idx: after_idx}`
  containing only confident matches.

### Modified: `stability/pairwise_stability.py`

`run_stability_test` currently assumes index equality. It needs to accept an
optional correspondence map and apply it when looking up after-descriptors.
We also extend `PairResult` (or add a new result type) to carry a status of
`stable | descriptor_flip | segmentation_failure` so the aggregation layer
can count the categories separately.

Existing behavior (self-stability on a single image with synthetic transforms,
no correspondence needed because segmentation is near-identical) should keep
working — the correspondence argument defaults to `None` and the old code
path is preserved.

### New module: `stability/evaluation.py`

The dataset-level driver. One function:

- `evaluate_combo(pairs_iter, segment_func, descriptor_func) -> ComboReport`
  Consumes an iterator of `(before, after, metadata)` tuples (the shape
  already emitted by `stability/fragfake_loader.py`), runs the per-image
  loop with region matching, and aggregates into the four metrics in §4,
  stratified by any metadata fields the caller passes in.

### New driver: `drivers/evaluate_stability.py`

CLI wrapper that wires FragFake (via the existing `iter_fragfake_pairs`) and
the synthetic transforms together, runs every configured combo, and writes
per-combo reports. This replaces the current `drivers/test_stability.py` for
dataset-scale evaluation; `test_stability.py` stays as the single-image
smoke test.

### Unchanged

- `core/types.py`, `regions/*`, `descriptors/*`, `stability/transformations.py`,
  `stability/fragfake_loader.py` — all fine as-is.
- The `SegmentationResult` interface is sufficient for IoU computation
  because `region_map` is already a per-pixel label array, which is exactly
  the input IoU needs.

---

## 6. Open questions to resolve during implementation

These are real uncertainties, not rhetorical flourishes. We should not treat
any of the following as settled:

1. **IoU threshold** — 0.5 is the starting point because it is standard, but
   the right value is empirical and depends on how much AI edits perturb
   segmentation. Revisit after the first FragFake run.
2. **How to treat descriptor flips on pairs with tiny before-margin** — a
   pair with `|d_i − d_j| = 0.01` in the before-image is already noise; it
   will flip under any edit and should probably be excluded from the flip-rate
   numerator. Minimum-margin filter value TBD.
3. **Should we match regions only once on the original and warp masks, or
   re-segment the edited image?** Re-segmenting (current plan) is more honest
   — it measures segmentation robustness as a first-class property. Warping
   masks would measure descriptor robustness in isolation. We want the
   honest version, but this trade-off should be noted.
4. **Hungarian on very fine segmentations** — if any method produces, say,
   5000 regions, O(n³) gets uncomfortable. Plan: coarsen-before-match. But
   we have not committed to a coarsening rule yet.
5. **Does the matching layer need to be symmetric?** Matching from before→after
   is not the same as after→before when counts differ. Hungarian is
   symmetric in that the optimal assignment is unique given the cost matrix,
   but the interpretation of "unmatched" regions differs. We pick the
   before-anchored framing because the watermark is embedded in the
   before-image and must be recovered from the after-image.
6. **The runtime pipeline is not described in this document.** This plan is
   scoped to *offline evaluation* of `(segmentation, descriptor)` combos. It
   does not specify how embedding or decoding works in the deployed system.
   Worth writing down separately that: decoding runs on the altered image
   alone (no reference to the original is needed — that is the whole point
   of an inequality watermark); embedding modifies pixels inside chosen
   regions until pair signs match the target bitstring with a safety margin
   derived from this evaluation's margin-retention numbers. Resolve by
   writing a companion `PIPELINE.md` once evaluation results start landing.
7. **Error-correcting code layer.** The current framing treats each pair as
   an independent 1-bit channel. In reality we will layer an ECC (BCH,
   Reed–Solomon with erasure decoding, or LDPC) on top so that a handful of
   flipped pairs and a handful of segmentation-failed pairs don't destroy
   the payload. The stability evaluation gives the ECC designer exactly the
   two numbers they need — per-pair flip probability (for error correction)
   and per-pair segmentation-failure probability (for erasure correction).
   Resolve by: (a) once flip/seg-failure distributions are in hand, pick a
   code whose correction capacity comfortably exceeds both rates at the
   strictest FragFake stratum we care about; (b) prefer codes that handle
   erasures efficiently since seg-failures are cheap to detect at decode.
8. **Embedding-based descriptors (DINO / CLIP / SAM encoder).** Not currently
   in scope but a natural candidate once infrastructure exists. Open
   sub-questions: which model checkpoint to pin (reproducibility matters —
   the decoder must run the exact same weights); whether to crop-and-encode
   per region or to patch-pool from a single whole-image forward pass
   (patch-pooling is strongly preferred because it avoids OOD masking
   artifacts and is much faster); how to reduce the resulting 768-dim
   vector to a scalar usable by the pair-sign framework. Candidate scalar
   reductions: (a) fixed random projection onto a unit vector chosen once
   at project start (simple, reproducible, semantically arbitrary);
   (b) projection onto the first PCA direction of this image's region
   embeddings (semantically meaningful but image-dependent, so decoder must
   recompute); (c) cosine similarity to an anchor embedding such as the
   global image token (semantically meaningful and reproducible);
   (d) learned linear projection trained end-to-end to maximize pair-sign
   preservation under edits (strongest, most work, blurs into open
   question 9). Resolve by running (a) and (c) as baselines in the
   evaluation loop first and deciding based on numbers.
9. **Should we use ML to learn parts of the pipeline?** The current plan is
   entirely analytical (hand-crafted segmentations, hand-crafted
   descriptors, flip-rate metrics). ML could enter in several distinct
   places and they should not be conflated:
   - *Learned descriptor.* Train a small network mapping region crops (or
     pooled patch embeddings) to a scalar, with a loss that directly
     optimizes pair-sign agreement across before/after pairs from FragFake.
     This is metric learning with a rank-stability objective. Most
     ambitious, most likely to actually outperform hand-crafted
     descriptors, and requires FragFake-scale paired data — which we
     already have.
   - *Learned combo router.* For each image, extract cheap features
     (texture variance, region count, contrast, edit severity hint) and
     train a classifier to predict which `(segmentation, descriptor)`
     combo will produce the lowest flip rate *for that specific image*.
     Then pick the combo per-image at embedding time instead of globally.
     Moderate effort, directly useful, turns combo ranking into a local
     rather than global problem.
   - *Learned pair selector.* Given a fixed combo, predict for each
     candidate pair `(i, j)` whether its sign will survive edits, based on
     features of the two regions (size, starting margin, texture contrast,
     distance, embedding similarity). Labels come directly from FragFake
     outcomes. Most practically useful because it improves embedding-time
     pair selection without changing anything else.
   The measurement infrastructure this plan builds is a prerequisite for
   any of these — all three need the per-pair outcome data the evaluation
   loop produces. Resolve by finishing the evaluation loop first, then
   picking one (likely the learned pair selector, since it has the best
   work-to-value ratio) and prototyping it on FragFake outcomes.
10. **Dataset sizing and sampling.** The plan says "run over many images"
    without pinning down how many. Unknowns: how many FragFake images per
    stratum are enough for stable mean flip rates (probably in the low
    hundreds, but needs a convergence check — run on 50/100/200/500 and
    see when the metric stops moving); whether to subsample uniformly or
    stratify by editor type; how many synthetic-transform base images to
    pair against (probably ~500 from a standard source like COCO val).
    Resolve empirically by plotting metric stability vs. sample size on a
    preliminary run and picking the knee of the curve.
11. **Minimum-margin filter for the flip-rate numerator.** Pairs whose
    before-margin is already near zero will flip under any perturbation
    and shouldn't be counted against the combo. Open: what minimum margin
    to require, and whether the threshold should be absolute (e.g.
    `|d_i − d_j| > 0.05`) or relative (top-*k*% of pairs by margin).
    Resolve by looking at the flip-rate distribution as a function of
    before-margin on a first FragFake run — there is likely an obvious
    margin below which pairs are pure noise.

---

## 7. Summary

The current code measures stability against synthetic transforms on a single
image under the assumption that region indices are stable across edits. That
is enough to sanity-check a combo but not enough to rank combos for real-world
use.

The plan is to:

1. Add Hungarian-based IoU matching so "the same region before and after"
   actually means the same physical area.
2. Separate descriptor flips from segmentation failures so we know which
   component to blame when a combo underperforms.
3. Run the evaluation across FragFake's graded severity levels so flip rate
   becomes a function of edit intensity and we can identify each combo's
   breaking point.
4. Rank combos by their robustness profile across that severity curve, not
   by any single aggregate.

The result is an evaluation loop that can honestly say which
`(segmentation, descriptor)` combination produces region pairs whose
inequality ordering survives real edits — which is the property a
pairwise-inequality watermark actually needs.
