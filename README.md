# Pairwise Ordinal Image Watermarking

A watermark that survives AI image editing — **without modifying a single pixel**.

Instead of hiding data in pixel values, this system encodes a message in the *relative ordering*
of frequency-energy descriptors between pairs of image regions. If region `i` has higher DWT-LL
energy than region `j`, that inequality carries a bit: `sign(d[i] - d[j]) = 1`. The encoder never
changes the image; it *selects* pairs whose natural ordering already spells out the message and
records those selections in a small **sidecar file**.

The decoder needs only the edited image plus the sidecar — never the original.

---

## Status

**Working end-to-end for messages under ~60 bytes.**

| | |
|---|---|
| Encode / decode pipeline | Working (two sidecar variants) |
| Zero-edit round-trip | Passes |
| Light AI edits (object addition, small local changes) | Passes |
| Heavy AI edits (large-area replacement, recolor, global color changes) | Fails — outside the threat model |
| Practical payload | ~60 bytes at default parameters (hard ceiling 70 B on a 640×611 image, 127 B from Reed-Solomon) |
| Sidecar size | 24 KB (centroid variant) / 43 KB (region-map variant) for a 59-byte message on a 640×611 image |

60 bytes is enough for provenance data — a UUID (16 B) + Unix timestamp (8 B) + a short hash
(32 B) — which is the intended use case. It is not enough for rich metadata.

---

## Why this approach

Traditional watermarks (LSB, DCT/DWT coefficient embedding, spread spectrum) hide information in
*pixel or coefficient values*. AI editors like MagicBrush, Gemini-IG, GoT, and UltraEdit do not
perturb those values — they **regenerate image content semantically**, which destroys value-based
watermarks completely.

Ordinal relationships between *coarse* region descriptors are far more durable. The DWT-LL subband
measures low-frequency brightness energy, which changes slowly and globally; AI edits mostly rewrite
fine texture (the HH subband) in localized areas. So "region A is brighter than region B" tends to
survive an edit that "the 3rd DWT coefficient of block 47 equals 0.183" does not.

This was validated before the encoder was built — see [RESULTS.md](RESULTS.md) and
[RESULTS_EXTENDED.md](RESULTS_EXTENDED.md).

---

## Quick start

```bash
pip install -r requirements.txt
```

### Interactive demo (Gradio)

```bash
python demo/app.py
```

Two tabs:

- **Encode** — upload an image, type a message, pick a method, tune `k` / RS overhead, download the
  `.wm` sidecar.
- **Decode** — upload the *altered* image and the `.wm` sidecar. The method is auto-detected from the
  sidecar's magic bytes. Optionally upload the original image for bit-level diagnostics (flip rate,
  byte error rate, RS capacity headroom).

### From Python

```python
import cv2
from watermark.encoder import encode_watermark, EncodeParams
from watermark.decoder import decode_watermark, DecodeError
from watermark.sidecar import sidecar_to_file, sidecar_from_file

image = cv2.imread("minimal_data_testing/dog.jpg")          # BGR, as cv2 returns

# Encode — produces a sidecar; the image is untouched.
sidecar = encode_watermark(image, b"hi im a dog named bruno", key=42,
                           params=EncodeParams(k=7, rs_overhead=2.0))
sidecar_to_file(sidecar, "dog_bruno.wm")

# Decode — original image NOT needed, secret key NOT needed.
edited = cv2.imread("minimal_data_testing/dog_altered2.jpg")
try:
    print(decode_watermark(edited, sidecar_from_file("dog_bruno.wm")).decode())
except DecodeError as e:
    print("edit exceeded error-correction capacity:", e)
```

### Batch driver with diagnostics

[drivers/watermark_test.py](drivers/watermark_test.py) encodes and decodes one image pair and prints
region-match counts, bit flip rate, byte error rate, and RS headroom. Edit the constants at the top
(`ORIGINAL_IMAGE`, `ALTERED_IMAGE`, `MESSAGE`, `KEY`, `PARAMS`) and run:

```bash
python drivers/watermark_test.py
```

---

## How it works

### Encode ([watermark/encoder.py](watermark/encoder.py))

| Step | What happens | Module |
|---|---|---|
| 1 | SLIC superpixels — 200 segments, compactness 20, LAB space | [regions/approach_regions.py](regions/approach_regions.py) |
| 2 | Region centroids `(cx, cy)` for every region | [watermark/centroid_matching.py](watermark/centroid_matching.py) |
| 3 | Descriptor `d[k] = sum(cA²) / n_pixels` — raw Haar DWT-LL energy per pixel | [descriptors/dwt_descriptor.py](descriptors/dwt_descriptor.py) |
| 4 | Build all `C(n,2)` region pairs (~16,500 for the 182 regions SLIC actually produces on the demo image), drop those below `min_margin`, sort by margin descending, split into `pool_1` (`d[i] > d[j]`) and `pool_0`, shuffle each with the secret key | [watermark/pair_pool.py](watermark/pair_pool.py) |
| 5 | Reed-Solomon encode the message at `rs_overhead` (default 2×), expand to a bit array | [watermark/reed_solomon.py](watermark/reed_solomon.py) |
| 6 | Assign `k` pairs from the matching pool to each encoded bit — popping from the front, so the highest-margin (most stable) pairs get used first | [watermark/pair_pool.py](watermark/pair_pool.py) |
| 7 | Write the sidecar: centroids of referenced regions, the pair table, and all encode parameters | [watermark/sidecar.py](watermark/sidecar.py) |

The **secret key** is consumed entirely at encode time. It shuffles the pools so that an attacker
holding the sidecar cannot tell which pairs map to which bit positions. The decoder never needs it.

### Decode ([watermark/decoder.py](watermark/decoder.py))

| Step | What happens |
|---|---|
| 1 | Read every parameter from `sidecar.metadata` — nothing is hardcoded |
| 2 | Resize the edited image back to the original dimensions (AI editors routinely change resolution; sidecar centroids live in original pixel space) |
| 3 | SLIC on the edited image with the same parameters, then descriptors and centroids |
| 4 | Match before-regions → after-regions. Unmatched regions become **erasures** |
| 5 | Per bit: each of the `k` witness pairs votes `1` if `d[s1] > d[s2]`. Witnesses whose region is erased abstain rather than vote wrong. All `k` erased → the bit is an erasure |
| 6 | Group bits into bytes; any byte containing an erased bit is flagged as a byte-level erasure |
| 7 | Reed-Solomon decode with explicit erasure positions, then slice to `message_length` |

### Two fault-tolerance layers

They handle genuinely different failure modes, which is why both exist:

- **Majority voting over `k` witnesses** fixes *wrong* bits — a few region pairs whose ordering
  flipped because the edit touched them.
- **Reed-Solomon** fixes *missing* bits — regions the edit destroyed outright. RS corrects twice as
  many erasures as random errors for the same parity, which matters because heavy AI edits are an
  erasure-dominated regime (~50% of regions lost on MagicBrush hard, but the survivors are reliable).

---

## The two sidecar variants

Both are produced by the same pipeline and differ only in how the decoder re-identifies regions.
The demo auto-detects which one it was handed by reading the 4-byte magic header.

| | **Option 3 — centroid matching** | **Option 2 — IoU matching** |
|---|---|---|
| Modules | `encoder.py` / `decoder.py` / `sidecar.py` | `encoder_b.py` / `decoder_b.py` / `sidecar_b.py` |
| Magic bytes | `WMK!` | `WMB!` |
| Sidecar holds | Referenced region centroids + pair table | Full before-image region map + per-region descriptors + pair table |
| Region matching | Nearest Euclidean centroid, erasure past `centroid_threshold` (40 px) | IoU + Hungarian assignment, erasure below `iou_threshold` |
| Size | Compact — 12 KB for a 23-byte message, 24 KB for 59 bytes | Larger — also stores an `H×W` uint16 map; 43 KB for the same 59-byte message |
| Trade-off | Small and fast; centroids drift under heavy edits | More accurate matching when content shifts, at a real size cost |

Option 3's centroid matching does **not** enforce a 1-to-1 correspondence — two before-regions can
map to the same after-region near an edit boundary, producing a wrong vote rather than an erasure.
Majority voting at `k=7` absorbs a small number of these.

---

## Parameters and capacity

`EncodeParams` in [watermark/encoder.py](watermark/encoder.py):

| Parameter | Default | Role |
|---|---|---|
| `n_segments` | 200 | SLIC superpixel count |
| `compactness` | 20 | SLIC spatial regularization |
| `min_margin` | 0.05 | Minimum `\|d[i] - d[j]\|` for a pair to be usable |
| `k` | 7 | Witnesses per bit — must be odd to avoid ties |
| `rs_overhead` | 2.0 | Reed-Solomon redundancy; encoded bytes = message bytes × this |
| `centroid_threshold` | 40.0 | Max centroid drift in px before a region is treated as erased |
| `iou_threshold` (Option 2) | 0.5 (demo uses 0.25) | Min IoU for a valid region match |

Capacity is `min(len(pool_1), len(pool_0)) // k` bits. Measured on `minimal_data_testing/dog.jpg`
(640×611, 182 SLIC regions, `pool_1` = 8,569 / `pool_0` = 7,902):

| `k` | Capacity | Max message at 2× RS |
|---|---|---|
| 7 | 1,128 bits | **70 bytes** |
| 11 | 718 bits | 44 bytes |
| 15 | 526 bits | 32 bytes |

**This is where the ~60-byte working limit comes from.** Raising `k` buys robustness against
flipped orderings but costs capacity linearly. Reed-Solomon imposes a second, independent ceiling:
`len(message) + parity ≤ 255` (GF(256) field size), so at 2× overhead no message can exceed 127
bytes regardless of image size.

Exceeding capacity raises a `ValueError` at encode time with the exact numbers, not a silent
failure.

One gotcha worth knowing: `min_margin = 0.05` is inherited from the stability evaluation, which used
the *normalized* LL energy fraction (always in [0.90, 1.00]). `compute_raw_dwt_ll` returns raw
per-pixel energy in the range ~1,700–111,000, so at 0.05 the filter discards nothing. To actually
concentrate the pool on high-confidence pairs, `min_margin` needs to be in the thousands — see
[trial1_encoding.md](trial1_encoding.md), where sweeping it up to 20,000 was one of the three
attempted fixes for hard edits.

---

## What the evaluation established

### Stability study — which segmentation × descriptor combination to build on

116 FragFake before/after pairs across 4 AI editors (Gemini-IG, GoT, MagicBrush, UltraEdit),
2 difficulty levels, 2 edit types. Full write-ups: [RESULTS.md](RESULTS.md) (first pass, 10 pairs)
and [RESULTS_EXTENDED.md](RESULTS_EXTENDED.md) (116 pairs, per-stratum).

- **SLIC + DWT-LL is the production combination.** It is the only combo that stays under the 10%
  flip-rate target in all 16 strata, including the worst one (MagicBrush hard addition, 7.63%).
- **DWT-HH looked best and isn't.** It won on easy edits (1.78% flip rate) but hits **34.07%** on
  MagicBrush hard additions — new AI-inserted objects flood the high-frequency subband. This is the
  finding that reversed the earlier recommendation, and it only showed up in per-stratum analysis;
  the aggregate hid it.
- **LBP descriptors are unusable** — 27–36% flip rates everywhere, destroyed by JPEG or a brightness
  shift alone.
- **k-means is unusable for a different reason** — near-zero flip rate, but ~6 usable pairs per image
  (about 3 bits of capacity).
- **Watershed loses too many bits** — 57% segmentation survival vs. 68% for SLIC, collapsing to 26%
  on MagicBrush hard replacement.
- **The binding constraint is erasures, not errors.** Harder edits destroy more regions rather than
  corrupting the survivors' orderings. This is exactly what drove the Reed-Solomon-with-erasures
  design.

### End-to-end encode/decode trials

A **59-byte** message (`"hi im a dog named bruno and i live in ithaca new york 14850"`) encoded into
`minimal_data_testing/dog.jpg` at defaults (`k=7`, 2× RS, 944 encoded bits), then decoded from each
AI-edited variant in `minimal_data_testing/`. Both sidecar variants were run on the same encode and
agreed on every case:

| Edit | Output size | Result |
|---|---|---|
| None (baseline round-trip) | 640×611 | **Recovered** |
| Red collar added, slight zoom-out (`dog_altered2.jpg`) | 640×611 | **Recovered** |
| Tail dyed pink + red collar added, upscaled (`dog_altered4.jpg`) | 1056×992 | **Recovered** |
| Tail dyed pink + teal collar, reframed and upscaled (`dog_altered3.jpg`) | 1056×992 | Failed |
| Frisbee recolored red + grass replaced with mud (`dog_altered.jpg 22-53-45-067.jpg`) | 640×611 | Failed |
| Whole image re-rendered as a marker illustration (`dog_altered_chatgpt_marker.jpg`) | 1284×1225 | Failed |
| Colors inverted (`dog_altered_inverted_colors.jpg`) | 1284×1225 | Failed |
| Scene fully regenerated — dog indoors on a couch with a flamingo (`dog_altered_flamingo.png`) | 1284×1225 | Failed |

Three out of eight is the honest headline, and the split is exactly the designed one: **localized
additions survive, including a 1.7× upscale; whole-image regeneration and global color changes do
not.** The two collar edits differ only in how much of the frame the editor re-drew — `dog_altered4`
kept the background, `dog_altered3` reframed it — and that is the whole margin between success and
failure.

Earlier per-edit diagnostics with a shorter 23-byte payload, from
[option3_trial1_init_results.md](option3_trial1_init_results.md) and
[trial1_encoding.md](trial1_encoding.md):

| Edit | Bit flip rate after `k=7` vote | Result |
|---|---|---|
| Red collar added + slight zoom | 0.82% (3/368 bits) | Recovered — 181/182 regions found, ~25% of the RS budget used |
| Easy Gemini-IG replacement | low | Recovered |
| Frisbee recolored + ground swapped | 19.8% (32.3% raw) | Failed — ~3× over RS capacity at every tested `k`, `min_margin`, and overhead |
| Hard Gemini-IG replacement | 53.8% — effectively random | Failed |

When most regions still register as spatially present (centroid drift under threshold) but their
content has been rewritten, the decoder gets confident *wrong* votes instead of erasures — the one
failure mode Reed-Solomon cannot exploit.

---

## Limitations and threat model

- **In scope:** localized object additions, small edits, resolution changes, JPEG compression, blur,
  mild brightness shifts, easy-difficulty AI edits.
- **Out of scope:** global color grading, style transfer, day/night conversion, large-area
  replacement, colour inversion. These shift the LL energy of large image areas, which is precisely
  the signal the scheme rests on. No parameter tuning recovers them.
- **Brightness `+30` is harder than blur**, because clipping at 255 flattens the energy structure
  between bright regions; `-30` is much gentler.
- **Multi-step pipelines are untested.** All evaluation covers a single edit. Real-world chains
  (AI edit → re-compress → resize, as social platforms do) compound the damage.
- **Silent errors near the matching threshold.** A region that matches the *wrong* region just above
  the IoU/centroid cutoff produces a wrong bit with no erasure flag. Raising the threshold trades
  silent errors for known erasures — usually a good trade, since RS handles erasures at 2× efficiency.
- **Not adversarially tested.** An attacker who holds the sidecar and knows the scheme could target
  the referenced regions directly. The key protects pair→bit assignment, not the region list.
- **Known bug** (documented, not yet fixed): `mean_margin_retention` in
  [stability/evaluation_metrics.py](stability/evaluation_metrics.py) overflows for the HL/LH/HH
  subbands because the ratio is computed without the `MIN_MARGIN` filter. Flip rate, yield, and
  segmentation survival are unaffected.

---

## Repository layout

```
watermark/          the encode/decode system
  encoder.py          Option 3 encoder (centroid sidecar)
  decoder.py          Option 3 decoder
  sidecar.py          Sidecar dataclass, JSON+zlib serialization ("WMK!")
  encoder_b.py        Option 2 encoder (region-map sidecar)
  decoder_b.py        Option 2 decoder (IoU + Hungarian matching)
  sidecar_b.py        SidecarB format ("WMB!")
  pair_pool.py        pair generation, margin filter, key shuffle, bit assignment
  reed_solomon.py     RS wrapper + bit/byte conversion with erasure positions
  centroid_matching.py  region centroids, nearest-centroid matching
  __init__.py         public API: encode_watermark, decode_watermark, Sidecar

regions/            segmentation approaches — slic_superpixels() is the one in use
descriptors/        DWT and LBP descriptors — compute_raw_dwt_ll() is the one in use
core/types.py       SegmentationResult (region_map, num_regions)

stability/          the evaluation harness
  pairwise_stability.py   per-image flip-rate loop
  region_matching.py      IoU matrix + Hungarian assignment
  evaluation_metrics.py   PairResult, EvaluationMetrics, aggregation, stratification
  fragfake_loader.py      FragFake before/after pair iterator
  transformations.py      synthetic attacks (JPEG, resize, blur, brightness)

drivers/            entry points (see below)
demo/app.py         Gradio UI for encode + decode
test/               pytest suites for descriptors, regions, and the loader
minimal_data_testing/   the dog image plus seven AI-edited variants
```

`dog_bruno.wm` in the repo root is a committed sample sidecar — `b"hi im a dog named bruno"`
encoded into `dog.jpg` at `k=11`.

Note: `data/` is gitignored — datasets are downloaded, not committed. The demo images live in
`minimal_data_testing/`.

### Drivers

| Script | Purpose |
|---|---|
| [drivers/watermark_test.py](drivers/watermark_test.py) | Encode + decode one pair with full diagnostics |
| [drivers/run_pairwise_stability.py](drivers/run_pairwise_stability.py) | The evaluation harness. Config block at line 176: `MODE` `'A'` = synthetic transforms, `'B'` = FragFake; `COMBO_MATRIX` selects which segmentation × descriptor combos to rank |
| [drivers/download_fragfake.py](drivers/download_fragfake.py) | Fetch FragFake from HuggingFace into the layout the loader expects |
| [drivers/download_coco_sample.py](drivers/download_coco_sample.py) | Fetch COCO val2017 images for Mode A |
| [drivers/explore_stability_dog_fragfake.py](drivers/explore_stability_dog_fragfake.py) | Single-pair stability inspection. **Currently broken** — imports `descriptors.mock_descriptors`, which no longer exists |
| [drivers/run_pipeline.py](drivers/run_pipeline.py) | Original integration stub against the `segment_image` / `compute_descriptors` interface contract |

```bash
# Datasets
python drivers/download_coco_sample.py --limit 100 --output-dir data/
python drivers/download_fragfake.py --output-dir data/fragfake --limit 10
python drivers/download_fragfake.py --editors Gemini-IG --difficulties easy \
       --edit-types addition --limit 5 --dry-run

# Ranked stability report (edit MODE / COMBO_MATRIX at the top of the file first)
python drivers/run_pairwise_stability.py
```

### Tests

```bash
python -m pytest test/test_LBPdescriptor.py test/test_fragfake_loader.py test/test_descriptors.py -v
```

12 tests, all passing. Two files are not pytest suites and are excluded above:
`test/test_regions.py` holds visual checks whose functions take a `segment_func` and an image as
arguments (pytest tries to collect them and errors on the missing fixtures), and
`test/test_DWTdescriptor.py` is a standalone descriptor-robustness script run as
`python test/test_DWTdescriptor.py <image>`.

---

## Documentation index

| Document | What's in it |
|---|---|
| [approach3_encoding_inital_explaination.md](approach3_encoding_inital_explaination.md) | The full algorithm spec plus a justification of every design choice against alternatives and prior literature. **Start here.** |
| [preliminary_encoding_scheme.md](preliminary_encoding_scheme.md) | Design exploration: modify-vs-select, the three sidecar designs (A/B/C), open questions, and where the intuition can break |
| [RESULTS.md](RESULTS.md) | First stability evaluation — 10 images × 7 transforms, 10 FragFake pairs |
| [RESULTS_EXTENDED.md](RESULTS_EXTENDED.md) | 116-pair evaluation across 4 editors × 2 difficulties × 2 edit types, per-stratum |
| [STABILITY_PLAN.md](STABILITY_PLAN.md) | The evaluation methodology that RESULTS was built to execute |
| [trial1_encoding.md](trial1_encoding.md) | First working encode/decode run — including the two bugs found (wrong descriptor, resolution mismatch) |
| [option3_trial1_init_results.md](option3_trial1_init_results.md) | Per-edit encode/decode results on the dog image |
| [REGIONS_INTERFACE.md](REGIONS_INTERFACE.md) | Contract between segmentation and descriptor modules |
| [FRAGFAKE_INTERFACE.md](FRAGFAKE_INTERFACE.md) | On-disk layout and manifest format for the FragFake dataset |

---

## Where to take it next

Drawn from the open questions in the docs, in rough priority order:

1. **Semantically grounded segmentation (SAM or similar).** SLIC's superpixels are defined by local
   color homogeneity, which is exactly what AI editing changes. Segmentation survival is the binding
   constraint (51% on MagicBrush hard addition), and the evaluation harness already accepts a new
   `segment_func` with no other changes.
2. **More regions for more capacity.** `n_segments > 200` expands the pair pool and pushes the
   60-byte ceiling up, at the cost of smaller and individually less stable regions.
3. **Multi-subband ensembles.** Use LL as the primary carrier with HL/LH bits at lower reliability
   and heavier ECC.
4. **Fix the margin-retention overflow** in `evaluation_metrics.py` (one-function fix).
5. **Scale the evaluation to 50+ pairs per stratum** — several buckets currently have only 2–4 pairs.

---

## Team

Built by the image watermarking team at [Cornell Data Science](https://github.com/CornellDataScience).

| Member | Role |
|---|---|
| Monisha Bommu | Tech Lead |
| Duru Alalyi | Project Manager |
| Jay Talwar | Member |
| Andy Do | Member |
| Agrim Jaimini | Member |
| Pete Olhava | Member |
| Aydan Gerber | Member |
