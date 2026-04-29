# Preliminary Encoding Scheme — Design Notes

This document captures the design thinking, open questions, doubts, and intuitions
behind turning the pairwise stability research into an actual watermarking system.
It is not a final spec — it is a record of the reasoning process, including where
things are uncertain or could break.

---

## The Core Idea (and Why It's Clever)

The pairwise ordering scheme encodes information not in pixel values but in
*relationships between regions*. If region i has higher DWT LL energy than region j,
that inequality encodes a single bit. A watermark is thousands of these inequalities
stacked together.

The appeal is that relationships are more invariant than absolute values. JPEG
compression, blur, and brightness shifts all change pixel values, but they tend to
preserve which region has more texture energy than which other region. The stability
research confirms this empirically: across 116 before/after AI-edited pairs, the
orderings that survive region matching are wrong only ~4% of the time for dwt_ll.

---

## The First Design Question: Modify or Select?

### Modifying pixels to enforce orderings

The naive approach: for each bit you want to embed, pick a pair (i, j) and then
nudge pixel values in the image until the ordering matches. Want to encode bit 1?
Make d_i > d_j. Want to encode bit 0? Make d_i < d_j.

**Why this is problematic:**
- Any modification, however small, risks creating an ordering that barely satisfies
  the constraint. A pair where d_i is only slightly larger than d_j is exactly the
  kind of near-zero-margin pair that the stability research shows are fragile.
- You're introducing artificial structure into the image. A skilled adversary with
  access to the watermarking scheme could potentially detect or remove it.
- The modification itself might be detectable as an artifact.

### Selecting pairs by natural ordering (the better approach)

Instead of modifying the image, compute descriptors first, then pick pairs whose
*existing* natural ordering already matches the bit you want to encode.

Want to encode bit 1 at position k? Find a pair where d_i > d_j naturally.
Want to encode bit 0? Find a pair where d_i < d_j naturally.

**Why this is better:**
- No image modification at all. The watermark is read from structure that was
  already there.
- Naturally large-margin pairs (|d_i - d_j| >> 0.05) are far more stable than
  anything you could create by nudging pixels. You're picking from the strongest
  natural orderings, not manufacturing fragile ones.
- The stability data you collected is directly measuring this: the orderings of
  naturally-large-margin pairs under real AI editing. It applies directly.

**The catch (and it is real):**
- The pair selection depends on the original image's descriptor values. The decoder
  needs to know which pairs were selected. But without the original image, how does
  it know?
- You cannot embed a truly arbitrary message without either (a) modifying the image,
  (b) storing a sidecar alongside the image, or (c) restricting the "message" to
  whatever the image naturally encodes under a given key.
- This is the central tension that drives the three design options below.

---

## The Three Design Options

The key insight is that the design space is defined by a single question:
**what information does the decoder have access to?**

### Design A — Key only, no sidecar (fingerprinting / detection)

**Decoder has:** a secret key seed and a registered fingerprint stored server-side.

The key defines a fixed set of P pairs (e.g. P=1000), chosen independently of
image content. At registration, the encoder reads the natural orderings of those
P pairs on the original image — this P-bit string is the fingerprint, stored in a
database. At verification, the decoder reads the same P pairs on the received image
and compares against the registered fingerprint. The flip rate tells you how much
the image has changed.

**The key's role here:** determines *which pairs* to look at. Like a stencil —
the key is the stencil, the image's natural orderings are what shows through it.
Different keys produce different fingerprints of the same image. Without the key,
an attacker can't know which pairs to check.

**What this can do:**
- Authenticate: prove an image came from a registered source.
- Quantify damage: the flip rate between registered and received fingerprint
  measures how much the image was altered.

**What this cannot do:**
- Embed an arbitrary message. The fingerprint is derived from the image, not chosen.
- Reliably detect AI editing specifically. The extended results show that flip rates
  under AI editing (0–8% for dwt_ll) overlap heavily with flip rates under JPEG
  compression and blur. You can't distinguish them from flip rate alone.

**Where Design A breaks:**
- Segmentation survival is editor-dependent (GoT: ~85%, MagicBrush hard: ~43–51%).
  If too many regions are lost, the fingerprint comparison becomes noisy.
- If the attacker knows the key, they can read the fingerprint and compute what
  modifications would destroy it while keeping the image visually similar.
- It provides no information about *who* watermarked the image or *what* they wanted
  to say — just whether the image is authentic.

---

### Design B — Key + full sidecar (arbitrary message, maximum decoder info)

**Decoder has:** the key seed and a sidecar file stored alongside the image.

The sidecar contains:
- The original segmentation (region_map as a compressed array)
- The original descriptor values (N floats, one per region)
- The message length in bits

At encode time, the encoder selects pairs from pool[0] (pairs where d_i < d_j) and
pool[1] (pairs where d_i > d_j) to match the message bits. The key shuffles the
pools so the assignment is unpredictable. At decode time, the decoder re-derives
the exact same assignment from the stored original descriptors + the key, then uses
match_regions() to map original region indices to received-image region indices,
then takes a majority vote.

**The key's role here:** scrambles the mapping from bit positions to pair indices.
Without the key, even a decoder with the full sidecar cannot determine which pair
encodes which bit. The pools (which pairs belong to pool[0] vs pool[1]) are readable
from the stored descriptor values without the key — but the assignment of bits to
pairs requires the key.

**What this can do:**
- Embed a truly arbitrary message (author ID, timestamp, content hash).
- Recover the message from an AI-edited image, using the ECC layer to correct
  for lost pairs (erasures) and flipped pairs (errors).

**What this cannot do:**
- Work if the sidecar is lost or stripped. Robustness of the sidecar is a separate
  infrastructure problem (store it in a database, embed it in EXIF, etc.).

**Where Design B breaks:**
- If both the image and the sidecar are modified together by an adversary who knows
  the scheme, they could potentially forge a different message.
- The sidecar is a few hundred KB (mostly the region_map). This is fine for
  archival/provenance systems but awkward for social media sharing.

---

### Design C — Key + tiny sidecar (hybrid, middle ground)

**Decoder has:** the key seed and a ~5KB sidecar (original descriptor values + region
centroids only, not the full region_map).

Instead of storing the full region_map for IoU matching, store only the N region
centroid coordinates (N × 2 floats, ~3KB for N=200) and the N original descriptor
values (~1.6KB). At decode time, use nearest-centroid matching instead of
IoU+Hungarian to map original regions to received-image regions.

**The key's role:** same as Design B.

**Trade-off:** Centroid matching is less accurate than IoU — regions that shift
spatially under AI editing may match to the wrong centroid. This slightly increases
effective erasure rate compared to Design B. But the sidecar fits in EXIF metadata
(which survives most image processing pipelines) rather than requiring a separate
file.

**Where Design C breaks:**
- Under heavy AI editing (MagicBrush hard), region centroids shift or disappear.
  Centroid matching degrades faster than IoU matching in these cases.
- If the image is significantly resized or cropped, centroid positions become
  meaningless without a spatial normalization step.

---

## What the Extended Results Tell Us About the Design Choices

### dwt_ll, not dwt_hh

RESULTS.md concluded dwt_hh was the best descriptor. RESULTS_EXTENDED.md reverses
this. dwt_hh collapses to 34.07% flip rate on MagicBrush hard addition — the HH
subband captures diagonal high-frequency energy, which is exactly what AI-generated
objects introduce. dwt_ll captures coarse low-frequency brightness energy, which
is too global for individual object insertions to disrupt significantly.

**Intuition for why this makes sense in hindsight:** when an AI model inserts a new
object into an image, it changes the fine texture detail of the affected region but
leaves the broad illumination structure of the whole image largely intact. LL
measures the latter. HH measures the former.

**Where this intuition can break:** if an AI edit involves global illumination
changes (e.g. changing from day to night, or dramatic color grading), LL would be
more affected than HH. The extended results don't test this scenario. Any edit that
is more of a global transformation than a local object insertion could flip the
subband ranking again.

### The binding constraint is erasures, not errors

Across all 16 strata, the pattern is consistent: hard AI edits destroy regions
(high segmentation failure rate) but the surviving regions keep their orderings
(low flip rate). The worst case is MagicBrush hard:

- Segmentation survival: 43–52%
- Flip rate among survivors: 0–8%

This means roughly half the bits are lost as *erasures* (you know they're gone —
match_regions() returns None) and of the remaining bits, ~8% are wrong (errors you
don't know about). This is an erasure-dominated regime.

**Why this matters for ECC design:** Reed-Solomon and related codes can correct
twice as many erasures as errors, because erasures come with location information.
Your pipeline already produces this location information — PairResult.status ==
"segmentation_failure" is the erasure flag. Passing these as known erasure positions
to an RS decoder rather than guessing them is the right move.

The ECC design target: handle 50% erasures + 8% errors simultaneously. This is
achievable with RS at reasonable redundancy — roughly 2× overhead on the message.

### The flip rate gap between editors is informative but not actionable for detection

GoT barely disturbs segmentation (85% survival at hard). MagicBrush hard drops it
to 43–52%. But you can't use this gap for AI detection because:
1. You don't know which editor was used.
2. High survival could also come from synthetic transforms.
3. Low survival could come from extreme JPEG compression, heavy blur, or cropping.

The flip rate signal specifically is not a reliable detector of AI editing. The
segmentation survival signal is more discriminative but has significant overlap
with legitimate heavy post-processing.

---

## Open Questions

### 1. Why does MagicBrush hard addition break SLIC so severely?

Segmentation survival at MagicBrush hard addition is 51% — 37 percentage points
below GoT hard at the same difficulty. The current hypothesis is that MagicBrush
produces larger or more structurally disruptive edits than GoT, causing SLIC's
local-color-homogeneity superpixels to form completely different boundaries in the
affected regions. But we haven't confirmed this. Visualizing a few before/after pairs
with their SLIC maps overlaid would directly answer whether the superpixel structure
shifts locally (near the edit) or globally (across the whole image).

### 2. Would semantically-grounded segmentation (SAM) help?

SLIC segments by local color similarity. SAM segments by object identity. The
hypothesis is that SAM would produce regions (sky, table, chair) that survive AI
object insertions better than SLIC's color-based patches, because the *semantic*
structure of the background is preserved even when new objects are added. This is
the highest-priority open experiment given the current results. The evaluation
infrastructure already supports plugging in a new segment_func — adding SAM is
mostly a dependency and interface question.

**Where this could break:** SAM on AI-edited images might produce different semantic
regions for the same object if the AI edit changes surrounding context. Also SAM
is much slower than SLIC, which matters for real-time watermark verification.

### 3. What K is needed for majority voting under 50% erasure?

With K=5 and 50% erasure rate, the expected number of surviving votes per bit is
2.5 — right at the majority boundary, with high variance. K=11 is the rough threshold
for majority voting to reliably handle 50% erasure. But at K=11, the pair budget
needed is 11× message_length_bits pairs per pool direction, which for a 100-byte
message is 8,800 pairs per pool — still achievable given ~9,000 usable pairs total
at MagicBrush hard, but barely. Reed-Solomon operating on the full message is likely
more bit-efficient than per-bit K-majority voting.

### 4. What happens under adversarial attacks?

The stability research tests survival under well-intentioned AI editing. An adversary
who knows the watermarking scheme exists (but not the key) would try to remove or
forge the watermark deliberately. The natural attacks are:
- Re-segmenting the image and slightly perturbing regions to flip many orderings
- Applying aggressive operations (strong blur, high JPEG compression) to destroy
  segmentation
- Regenerating the image with a diffusion model conditioned on the original

None of these are tested in the current evaluation. The current results only bound
performance against non-adversarial AI editing. Adversarial robustness is a
separate research question.

### 5. Can you distinguish which AI editor was used?

GoT, Gemini-IG, UltraEdit, and MagicBrush produce characteristically different
segmentation survival rates. GoT barely disrupts SLIC (85% survival even at hard);
MagicBrush hard destroys it (43%). This per-editor fingerprint might be exploitable
for forensic identification of which tool was used — but the current sample sizes
(as few as 2–4 pairs per stratum for some editors) are too small to make this claim.
Worth revisiting at 50+ pairs per stratum.

### 6. What is the practical message length limit?

Rough calculation for dwt_ll, MagicBrush hard (worst case):
- Usable pairs per image: ~5,867–7,088 (from the per-stratum table)
- Split roughly 50/50 by natural ordering: ~2,900–3,500 pairs per pool direction
- At K=5 majority vote: ~580–700 bits per pool direction → ~580 bits total capacity
- At K=3: ~960 bits → ~120 bytes
- With Reed-Solomon (2× overhead): ~60 bytes of actual message at K=3

That is tight. 60 bytes is enough for a UUID (16 bytes) + Unix timestamp (8 bytes)
+ a short hash (32 bytes) — sufficient for provenance, not sufficient for rich
metadata. Increasing the region count (SLIC n_segments > 200) would expand this,
but at the cost of smaller regions that may be less stable individually.

---

## Where the Intuition Can Break

**"High margin = high stability"** — generally true, but margin is computed on the
original image. Under AI editing, the relationship between original margin and
post-edit stability is not guaranteed. A pair with high margin before editing could
still flip if the edit specifically affects one of the two regions strongly. The
margin is predictive on average but not deterministic per pair.

**"LL is more stable than HH under AI editing"** — true for object insertion edits.
May not hold for global edits (style transfer, color grading, night/day conversion).
The extended results only cover object addition and replacement. Broader edit types
could reverse the subband ranking again.

**"Lost bits are erasures you know about"** — true for segmentation failure, where
match_regions() returns None and you know the pair is unreadable. But if a region
matches to the *wrong* region due to a near-threshold IoU match, you get a silent
error (wrong bit, unknown) rather than a known erasure. The iou_threshold=0.5 parameter
controls this trade-off: higher threshold means fewer false matches (more erasures,
fewer silent errors) at the cost of more bits lost. The current value of 0.5 was
chosen as the COCO detection standard, not optimized for this application.

**"The message is recoverable from a single edited image"** — the whole scheme
assumes the encoder runs once on the original and the decoder runs once on the
received image. If the image is edited multiple times in sequence (AI edit → JPEG
compress → resize), the cumulative damage to segmentation survival compounds. The
current evaluation only tests single-step edits. Multi-step processing pipelines
(common in social media) are untested.

---

## Summary of Current Best Guess

If forced to design the system today given the available evidence:

- **Descriptor:** dwt_ll — the only combo unconditionally robust across all 16 strata
- **Segmentation:** SLIC with n_segments=200 — adequate for now; SAM is the upgrade
- **Encoding:** select pairs by natural ordering (no pixel modification)
- **Pool selection:** sort by margin descending, key-shuffle within top-M candidates
- **Fault tolerance:** Reed-Solomon ECC on the full message, designed for 50% erasures
  and 8% errors; pass segmentation_failure pairs as known erasure positions
- **Sidecar:** store original region_map + descriptor values (Design B) for accuracy;
  consider Design C (centroids only) if EXIF-only storage is required
- **Message capacity:** ~60–120 bytes at worst case (MagicBrush hard), ~300–500 bytes
  at typical case (Gemini-IG / GoT / UltraEdit)
- **Open before building:** fix the margin retention overflow in evaluation_metrics.py;
  run SAM segmentation on MagicBrush hard pairs to see if survival improves

The approach is viable. The segmentation survival constraint is the binding limit,
not the descriptor or encoding logic. Every engineering decision from here forward
should be evaluated against the MagicBrush hard condition — if it works there, it
works everywhere tested so far.
