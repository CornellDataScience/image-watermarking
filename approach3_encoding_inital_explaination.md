# Approach 3 — Pairwise Region Ordering Watermark: Algorithm and Justification

This document describes the full encoding and decoding algorithm for the pairwise region ordering watermarking scheme, followed by a detailed justification of every design choice against the alternatives — including relevant prior literature.

---

## The Core Idea

A watermark is encoded not in pixel values but in **relationships between image regions**. If region i has higher DWT LL energy than region j, that inequality encodes a single bit: `sign(d[i] - d[j]) = 1`. A watermark is hundreds of these inequalities stacked together, each backed by K independent witnesses (pairs) for fault tolerance.

The sidecar — the only extra file the decoder receives — contains two things:
1. The spatial centroid of each region referenced in the pair table
2. The pair assignment table: for each encoded bit position, the K pairs of region IDs that encode it

No original image. No full region map. No descriptor values. Just spatial anchors (~2 KB) and pair indices (~9 KB), compressible to ~6–8 KB total.

---

## Phase 0 — Shared Setup

Choose a **secret key** (random seed) and fix parameters:

| Parameter | Value | Role |
|---|---|---|
| `n_segments` | 200 | SLIC superpixel count |
| `compactness` | 20 | SLIC spatial regularization |
| `MIN_MARGIN` | 0.05 | Minimum `\|d[i] - d[j]\|` for a pair to be used |
| `K` | 7 (easy) / 11 (hard) | Witnesses per encoded bit |
| RS overhead | 2× | Reed-Solomon redundancy ratio |
| Centroid threshold | ~40px | Max drift before treating a region as an erasure |

The key's role is narrow: it shuffles the pool of candidate pairs before assignment, so without the key an attacker with the sidecar cannot determine which pairs correspond to which bit positions.

---

## Phase 1 — Encode

Runs once on the original image. Produces the sidecar.

### Step 1 — Segment the image

Run SLIC superpixels on the original image in LAB color space:

```python
slic_superpixels(image)  # n_segments=200, compactness=20
```

Each pixel is assigned a label 0–199. The result is a `SegmentationResult` with `region_map` (H×W integer array) and `num_regions`.

### Step 2 — Compute region centroids

For each region k, find all pixels where `region_map == k` and take the mean of their (x, y) coordinates. Store `C[k] = (cx, cy)`. A subset of these will go into the sidecar.

### Step 3 — Compute descriptors

For each region k:
1. Extract the bounding box of its pixels
2. Fill non-region pixels within the box with the region's per-channel mean (prevents DWT boundary artifacts)
3. Apply a single-level Haar DWT to the grayscale crop
4. Record `d[k] = E_LL = sum(cA²)` — the low-frequency approximation energy

This is `compute_dwt_ll()`. Each region gets one scalar.

### Step 4 — Build the pair pool

Generate all pairs (i, j) with i < j over 200 regions (~19,900 pairs). For each pair:

```
margin(i, j) = |d[i] - d[j]|
```

Discard pairs where `margin < MIN_MARGIN`. Sort the remainder by margin descending. Based on empirical results, this yields ~7,000–13,000 usable pairs per image depending on image content.

Split into two directional pools:
- `pool[1]` — pairs where `d[i] > d[j]` naturally (encodes bit 1)
- `pool[0]` — pairs where `d[i] < d[j]` naturally (encodes bit 0)

Shuffle each pool independently using the secret key.

### Step 5 — RS-encode the message

Take the raw payload (e.g. 60 bytes = 480 bits). Apply Reed-Solomon at 2× overhead → 960 encoded bits. These 960 bits are what get embedded. The extra 480 bits are redundancy that allows the decoder to correct errors and erasures.

### Step 6 — Assign pairs to bits

For each of the 960 encoded bit positions j:
- If `encoded[j] == 1`: pop the next K pairs from `pool[1]`
- If `encoded[j] == 0`: pop the next K pairs from `pool[0]`

Record `pairs[j] = [(r1, r2), ..., (rK1, rK2)]`.

Because the pools were sorted by margin descending, earlier bit positions get the highest-margin (most stable) pairs. This means the bits with the most redundancy also have the best individual pair quality.

### Step 7 — Write the sidecar

Collect `U` = all unique region IDs appearing in any pair. The sidecar contains:

```
{
  "centroids": { r: (cx, cy)  for r in U },
  "pairs":     [ K_pairs_for_bit_0, K_pairs_for_bit_1, ..., K_pairs_for_bit_959 ]
}
```

Serialized and compressed: ~6–12 KB. Fits in a JPEG EXIF tag (64 KB limit).

**Sidecar size breakdown (60-byte message, K=5, 2× RS):**

| Component | Calculation | Size |
|---|---|---|
| Pair table | 960 bits × 5 pairs × 2 region IDs × 1 byte | 9,600 bytes |
| Centroids | ~300 unique regions × 8 bytes (two float32) | 2,400 bytes |
| Compressed total | | ~6–8 KB |

---

## Phase 2 — Decode

Runs on the AI-edited image plus the sidecar. The original image is not available.

### Step 1 — Segment the edited image

Run `slic_superpixels()` on the edited image with the same parameters. This produces new region IDs 0–199. These IDs are **not** the same as the original — SLIC renumbers from scratch. Region 2 in the after-image is not the same spatial area as region 2 in the before-image.

### Step 2 — Compute after-image centroids and descriptors

For every after-image region s:
- Compute centroid `D[s] = (cx, cy)`
- Compute `e[s] = dwt_ll` energy

### Step 3 — Establish region correspondence

For each original region r referenced in the sidecar:

```
match[r] = argmin_s  ||C[r] - D[s]||₂
```

If the nearest distance exceeds the threshold:

```
match[r] = None   →  erasure (known missing bit)
```

This is the key step. Nearest-centroid lookup is what lets the decoder find the correct after-image region without access to the original image. It works because SLIC centroids are spatially stable: even when the region's exact boundary changes, its center of mass stays roughly in the same place for edits that don't relocate or delete the underlying content.

Setting `match[r] = None` when the drift is too large converts bad matches into explicit erasures rather than silent wrong votes. Erasures are far more recoverable by RS than undetected errors.

### Step 4 — Majority vote per bit

For each encoded bit position j:

```
votes = []
for (r1, r2) in pairs[j]:
    s1, s2 = match[r1], match[r2]
    if s1 is None or s2 is None:
        continue          # erasure — skip entirely
    votes.append(1 if e[s1] > e[s2] else 0)

if len(votes) == 0:
    decoded[j] = ERASURE
else:
    decoded[j] = majority(votes)
```

The `continue` on missing regions is critical — a missing region should never cast a vote, because the nearest-centroid match returned `None` specifically because we can't trust it.

### Step 5 — RS decode

Pass `decoded` (960 values, some ERASURE) and the list of ERASURE positions to the Reed-Solomon decoder. It:
- Uses erasure positions to recover bits without guessing
- Corrects the remaining ~8% of non-erasure bit flips

Output: the original 480-bit payload.

---

## Two Fault Tolerance Layers, Handling Different Failure Modes

The majority vote and RS operate at different levels and handle distinct failure modes.

**Layer 1 — Majority voting (per-bit):**
K witnesses vote on each individual bit. Handles localized region damage: if 3 of K=7 pairs are erased, 4 surviving pairs still give a clean majority. Handles occasional wrong-region centroid matches: one wrong vote is overridden by the correct majority.

**Layer 2 — Reed-Solomon (across bits):**
Operates on the full 960-bit decoded string. Handles the ~8% descriptor flip rate (wrong bits you don't know are wrong). Handles the ERASURE positions (known-missing bits) at twice the efficiency of random errors, because you pass their positions explicitly. The empirical numbers that drive the RS design target: 50% erasures + 8% errors simultaneously (MagicBrush hard, worst case).

```
60-byte payload
      ↓  RS encode (2×)
960 encoded bits
      ↓  K pairs per bit selected by margin + key
sidecar written
      ↓  [AI editing happens]
      ↓  nearest-centroid matching
      ↓  majority vote per bit
960 decoded bits (~8% wrong, some ERASURE)
      ↓  RS decode with erasure positions
60-byte payload recovered
```

---

## Why This Algorithm, Over Everything Else

### Traditional coefficient-value methods fail against the actual attack model

**LSB and DCT/DWT coefficient modification (Cox et al. 1997, Barni et al. 2001):**
These embed a bit by nudging an absolute coefficient value — a pixel LSB, a DCT coefficient, a DWT coefficient. The bit is in the *value*, not a *relationship*. When an AI editor regenerates a region of the image, it synthesizes new pixel values from scratch via a diffusion or GAN process. The absolute coefficient values change unconditionally. The embedded bit is gone.

**Spread spectrum watermarking (Cox et al. 1997):**
Distributes watermark energy across many DCT coefficients like a CDMA signal. Very robust to JPEG and Gaussian noise. Against AI editing: the editor is not adding bounded zero-mean noise to existing coefficients — it is replacing pixel content entirely in the affected region, destroying the spread signal.

**QIM — Quantization Index Modulation (Chen & Wornell 2001):**
Provably near-capacity for Gaussian channels. Embeds bits by forcing coefficients onto interleaved quantization lattices. AI editing shifts coefficients far beyond any quantization bin in the affected regions, destroying lattice assignments. The Gaussian channel assumption does not model semantic content replacement.

**HiDDeN / StegaStamp / RivaGAN (Zhu et al. 2018, Tancik et al. 2020, Zhang et al. 2019):**
Train a neural encoder-decoder pair end-to-end. Robust to JPEG, noise, small geometric transforms because those were in the training distribution. The fundamental failure: an AI editor does not apply a learned inverse of the encoder's noise pattern — it synthesizes entirely new pixel content. HiDDeN was designed before diffusion-based editing existed as a practical concern. Robustness to semantic content replacement was never part of the training objective.

**Tree-Ring / Stable Signature / Gaussian Shading (Wen et al. 2023, Fernandez et al. 2023):**
Work by injecting watermarks into the latent space or noise vector of a diffusion model. Only applicable if *you generated the image* using a model you control. This project must watermark arbitrary photographs before AI editing. These methods are categorically inapplicable.

**Perceptual hashing (pHash — Zauner 2010):**
Conceptually the closest precursor. pHash records the sign of DCT coefficients relative to the image mean — an ordinal representation. But: it produces a fixed-length fingerprint, not an embeddable arbitrary message. It has no per-bit redundancy, no ECC, and no spatial locality. It is a single global hash, not thousands of independently-recoverable region-pair bits.

---

### Why ordinal relationships are inherently more robust than cardinal values

Ordinal measurement (which of two values is larger) is fundamentally more robust than cardinal measurement (what the actual values are). An ordinal relationship only flips if the perturbation is large enough and specifically asymmetric enough to make one value *cross over* another.

For an AI edit that is spatially local — which object insertion is, by definition — the crossing-over argument reduces to:
- Pairs where **both** regions fall outside the edited area: completely unaffected. The descriptor values of unedited regions are unchanged, so their ordering is unchanged.
- Pairs where **one** region is inside and one is outside: may flip if the inside region's LL energy changes enough to cross the outside region's value. Magnitude-dependent.
- Pairs where **both** regions are inside the edited area: most at risk, but LL's coarse scale means even a heavily-edited region's brightness average often survives.

This is exactly the pattern in the empirical results: 43–88% segmentation survival (regions inside the edit are often not even findable), and among the survivors, 0–8% flip rate. The pairs that survive are the ones outside the edited area. Outside the edited area, ordering relationships are essentially untouched.

---

### Why SLIC at n=200 (not watershed, k-means, SAM, or fixed grids)

**k-means at k=5:** 5 regions = 10 pairs = ~3 bits of capacity. Eliminated on bit budget alone.

**Watershed:** Produces ~200 regions but segmentation survival collapses to 25.97% on MagicBrush hard replacement — three in four bits lost before the descriptor question is even asked. Watershed boundaries are defined by local image gradients, which AI-inserted objects disrupt severely. Overall survival is 57.28% vs 68.41% for SLIC. No upside compensates.

**Fixed grids:** No content awareness. A fixed 32×32 patch will straddle semantic boundaries, mixing distinct objects into one descriptor. SLIC produces compact, color-homogeneous regions with perceptually meaningful boundaries in LAB space.

**SAM:** Would produce semantically stable regions ("sky", "table", "wall") that might survive content changes better than SLIC's color patches. This is the highest-priority open experiment. It has not been empirically measured yet — SLIC is used because it has confirmed numbers across 116 pairs and 4 editors. The evaluation infrastructure is already pluggable: `segment_func` is a parameter, so SAM is a drop-in.

**SLIC at n=200 in LAB:** Empirically achieves 43–88% segmentation survival across all 16 tested strata. Compact regions with compactness=20 produce spatially well-defined centroids, which is what makes the sidecar's nearest-centroid lookup reliable. No other tested segmentation method achieves this across the full range of editors and difficulty levels.

---

### Why DWT LL (not HH, HL, LH, not LBP)

**LBP:** Local Binary Patterns measure fine pixel-level texture. Destroyed by JPEG compression and brightness shifts — 27–36% flip rate on purely synthetic transforms. Eliminated in RESULTS.md on the basis of basic post-processing robustness alone.

**DWT HH:** Captures diagonal high-frequency detail — fine texture, edges, noise. This is exactly the signal that AI-generated objects introduce. A newly synthesized object has rich high-frequency structure. MagicBrush inserting a detailed object floods the HH subband of affected regions. Result: **34.07% flip rate** on MagicBrush hard addition — 3.4× the 10% threshold. RESULTS.md incorrectly ranked HH as best because it only tested Gemini-IG easy, where HH happened to be stable. RESULTS_EXTENDED.md reversed this conclusion.

**DWT HL and LH:** Capture vertical and horizontal edge energy respectively. More stable than HH, but HL reaches 16.56% on MagicBrush hard addition — still above threshold as a standalone. Conditionally viable in an ensemble, not standalone.

**DWT LL:** Captures the coarse low-frequency approximation of each region — essentially the average brightness and color distribution at a scale much larger than individual texture features. LL energy of a region changes only when the overall illumination structure of that region changes. AI object insertion modifies fine detail but leaves the broad illumination structure of background regions intact. Result: **0–8% flip rate across all 16 strata**, including MagicBrush hard addition (7.63%). The only subband with no stratum failures.

The intuition is backed by the physics of AI editing: diffusion models add fine-grained detail to inserted objects. That detail is in the HH/HL/LH frequency bands. The LL band captures the global average, which is too coarse to be significantly disrupted by a local object insertion.

---

### Why majority voting + Reed-Solomon as two separate ECC layers

The failure mode is **erasure-dominated, not error-dominated**. At MagicBrush hard replacement: 57% of bits lost as known erasures, ~0% of surviving bits wrong. A standard error-correcting code designed for random errors wastes correction capacity on a problem (errors) that barely exists while failing at the actual problem (erasures).

Reed-Solomon corrects twice as many erasures as random errors for the same code overhead — this is the fundamental property of erasure coding, and it applies directly here because the segmentation failure detection gives explicit erasure positions. The `segmentation_failure` status already exists in `PairResult`; the equivalent in the decoder is `match[r] = None`. Passing these positions to the RS decoder — rather than treating them as random errors to guess — is the correct model for this channel.

Majority voting is the per-bit pre-filter that RS never sees. Its role is to collapse K witnesses into a single reliable bit before handing to RS. It handles two things RS cannot: (1) it reduces the flip rate RS has to handle (from ~8% raw to lower after majority filtering); (2) it provides redundancy against the small fraction of cases where centroid matching returns the wrong region silently (a near-threshold centroid match that goes in the wrong direction). These are undetectable as erasures and become random errors — majority voting limits their influence to a single wrong vote out of K.

The two layers are complementary. Remove majority voting and RS faces higher raw error rates. Remove RS and majority voting alone cannot correct across-bit systematic errors. Together they handle the erasure-dominated channel this system operates in.

---

### Why no pixel modification

Not modifying the image means the watermark is **invisible by construction**, not by hiding a detectable signal. Any pixel-space watermark — including all deep learning methods — produces structural artifacts in pixel values or frequency coefficients. A skilled adversary with a forensic tool can in principle detect that something was embedded, and a resampling or diffusion operation can destroy it.

The ordering relationships read at encode time were already present in the image. The sidecar simply records which ones to look at and where to find the relevant regions. There is no foreign structure to detect or remove.

An adversary who does not know the key cannot determine which region pairs carry the watermark. An adversary who does know the key could attempt to flip the orderings — but flipping a high-margin pair requires a visible, targeted modification to one of its regions, which introduces its own detectable artifact and is spatially constrained.

---

### The literature gap this fills

The overwhelming majority of watermarking research — DCT/DWT methods, spread spectrum, QIM, deep learning — was designed and evaluated against *statistical transforms*: JPEG compression, Gaussian noise, blur, rotation. This was the realistic attack model until ~2022.

AI content editors (MagicBrush, Gemini-IG, GoT, UltraEdit) represent a qualitatively different attack model: they do not apply a mathematical transform to existing pixel values — they semantically regenerate regions of the image. No prior published watermarking method was designed or evaluated against this model.

This project is among the first to be empirically evaluated against semantic AI editing as the primary threat, across 4 editors, 2 difficulty levels, and 2 edit types (116 image pairs). The finding that *ordinal relationships between coarse-scale region descriptors survive AI editing better than any cardinal signal* is not derived from prior literature — it is derived from the evaluation. That is what makes this approach the correct choice for the specific threat model we care about: it is the only approach with empirical evidence that it works against it.

---

## Summary of Design Choices

| Decision | Choice | Reason |
|---|---|---|
| Descriptor type | Ordinal (ordering) | More robust than cardinal values; only flips when one value crosses another |
| Segmentation | SLIC n=200, LAB space | Empirically best survival (43–88%) across all 16 tested strata |
| Descriptor | DWT LL energy | Only subband below 10% flip rate in all 16 strata; coarser than AI edit scale |
| Region finding at decode | Nearest-centroid lookup | Tiny sidecar (~2 KB centroids), no original image needed |
| Failed region handling | Explicit erasure (match=None) | RS decodes erasures 2× more efficiently than random errors |
| Per-bit redundancy | Majority voting over K pairs | Handles localized region loss before RS sees the bit |
| Cross-bit redundancy | Reed-Solomon 2× overhead | Designed for erasure-dominated regime (50% erasures, 8% errors) |
| Pixel modification | None | Invisible by construction; no embedded noise pattern to detect or destroy |
| Sidecar contents | Centroids + pair table only | ~6–12 KB; EXIF-compatible; no original image or descriptor values |
