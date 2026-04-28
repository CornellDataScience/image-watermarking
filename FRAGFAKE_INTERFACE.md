# FragFake Data Interface Contract

## Purpose

This document defines the data interface between the **FragFake Downloader** and any
**Loader Consumer** (stability evaluation, descriptor development, ad-hoc inspection).
The goal is to allow the downloader and consumers to be implemented independently while
ensuring compatibility when integrated.

The downloader produces an **on-disk directory** mirroring the FragFake dataset layout,
together with a **JSONL manifest** of before/after pairs. The loader consumes that
directory and yields `FragFakePair` objects.

---

# 1. On-disk Layout

All consumers assume the output root has the following structure:

```
<root>/
├── manifest.jsonl
├── Gemini-IG/
│   ├── easy/
│   │   ├── addition/
│   │   │   ├── original/
│   │   │   │   └── original_<class>_<id>.jpg
│   │   │   └── <class>_<id>_addition.png
│   │   └── replacement/
│   │       ├── original/
│   │       │   └── original_<class>_<id>.jpg
│   │       └── <class>_<id>_replacement.png
│   └── hard/
│       ├── addition/
│       │   ├── original/
│       │   └── ...
│       └── replacement/
│           ├── original/
│           └── ...
├── GoT/
│   └── ...          # same substructure
├── MagicBrush/
│   └── ...          # same substructure
└── UltraEdit/
    └── ...          # same substructure
```

Rules:

* `<root>` is configurable; the default is `data/fragfake/`.
* Each `(editor, difficulty, edit_type)` bucket is **self-contained**: its `original/`
  subfolder holds only the originals paired into that bucket, and the edited images
  sit alongside that `original/` folder.
* There is **no** top-level `original/` folder. The same `<class>_<id>` original never
  appears in two buckets — every original is assigned to exactly one bucket. See §6 for
  the rationale.
* File extensions are preserved from the source dataset: `.jpg` for originals, `.png`
  for edits.
* The `<class>_<id>` substring (e.g. `laptop_000000412289`) is the pairing key linking
  an edit back to its original.

---

# 2. Manifest Format

The downloader must emit `<root>/manifest.jsonl`: one JSON object per line, one line per
before/after pair.

Schema:

```json
{
  "image_id":    "<class>_<id>",
  "editor":      "Gemini-IG | GoT | MagicBrush | UltraEdit",
  "difficulty":  "easy | hard",
  "edit_type":   "addition | replacement",
  "before_path": "<Editor>/<difficulty>/<edit_type>/original/original_<class>_<id>.jpg",
  "after_path":  "<Editor>/<difficulty>/<edit_type>/<class>_<id>_<edit_type>.png"
}
```

Example line:

```json
{"image_id": "laptop_000000412289", "editor": "Gemini-IG", "difficulty": "easy", "edit_type": "addition", "before_path": "Gemini-IG/easy/addition/original/original_laptop_000000412289.jpg", "after_path": "Gemini-IG/easy/addition/laptop_000000412289_addition.png"}
```

Rules:

### Rule 1 — Paths Relative to Root

`before_path` and `after_path` must be **relative** to `<root>`. Consumers resolve them
by joining against `<root>`. Absolute paths are invalid.

### Rule 2 — Both Sides Must Exist

A manifest line is emitted only if **both** `before_path` and `after_path` resolve to
files on disk with non-zero size at the time the manifest is written. Missing or empty
files must be omitted entirely, not emitted as null entries.

### Rule 3 — One Line per Pair

Each before/after pair occupies exactly one line.

### Rule 4 — Unique Pairing

Every `image_id` appears in **at most one** manifest line. An original is assigned to
exactly one `(editor, difficulty, edit_type)` bucket and is never reused across buckets,
so its `before_path` is unique across the manifest. See §6 for the statistical motivation.

---

# 3. FragFakePair Dataclass

The loader must yield instances of this dataclass:

```python
@dataclass
class FragFakePair:
    before: np.ndarray      # (H, W, 3), uint8, BGR
    after: np.ndarray       # (H, W, 3), uint8, BGR
    image_id: str           # e.g. "laptop_000000412289"
    editor: str             # Gemini-IG | GoT | MagicBrush | UltraEdit
    difficulty: str         # easy | hard
    edit_type: str          # addition | replacement
```

Field guarantees:

* `before` and `after` are decoded with `cv2.imread` — **BGR channel order**, `uint8`
  dtype, shape `(H, W, 3)`.
* `before.shape` and `after.shape` are *not* required to match. AI edits can change
  image dimensions. Consumers that require matched sizes must resize explicitly.
* `image_id`, `editor`, `difficulty`, `edit_type` are copied verbatim from the manifest
  row.

---

# 4. Loader Function Signature

The loader module must expose a single public function with the following signature:

```python
def iter_fragfake_pairs(
    root: str | Path = "data/fragfake",
    editors: list[str] | None = None,
    difficulties: list[str] | None = None,
    edit_types: list[str] | None = None,
) -> Iterator[FragFakePair]:
    """
    Read <root>/manifest.jsonl, filter by the provided arguments
    (None on any axis means no filter on that axis), load both
    images with cv2.imread, and yield FragFakePair.

    A manifest row whose before or after image fails to load
    (cv2.imread returns None) is skipped with warnings.warn
    rather than raising.
    """
```

Semantics:

* The function is a **generator** — consumers iterate lazily and never need to hold the
  full dataset in memory.
* `None` on a filter axis disables filtering for that axis; a `list[str]` restricts
  output to rows whose tag is in the list.
* Filtering is applied **before** image loading, so filtered rows do not incur
  `cv2.imread` cost.
* `cv2.imread` failure (missing file, corrupt file, unsupported format) triggers
  `warnings.warn` and `continue`, never an exception.

---

# 5. Metadata Axes

The four tag fields and their allowed values:

### editor

```
Gemini-IG | GoT | MagicBrush | UltraEdit
```

Identifies which AI editor produced the after-image. These are the four editors shipped
in FragFake.

### difficulty

```
easy | hard
```

Coarse edit-severity label from FragFake. `easy` edits are smaller and more localized;
`hard` edits are larger or more disruptive.

### edit_type

```
addition | replacement
```

Whether the edit **added** new content to the image or **replaced** existing content.

### image_id

The `<class>_<id>` substring extracted from the source filenames (e.g.
`laptop_000000412289`). This is the pairing key that links an edit back to its original
and is stable across rows.

---

# 6. Rationale

The on-disk layout mirrors FragFake's native directory structure rather than flattening
everything into a single bag of pairs. Keeping `editor`, `difficulty`, and `edit_type`
visible in the path serves two purposes:

* **Stratified evaluation** — downstream stability metrics must be reported per editor
  and per severity level (see `STABILITY_PLAN.md` §4). A layout that preserves these
  axes lets consumers walk a stratum directly (e.g. `Gemini-IG/easy/`) without parsing
  the manifest.
* **Human inspection** — engineers debugging a failure mode can `ls` into a single
  `<Editor>/<difficulty>/<edit_type>/` folder and compare edits visually without
  extra tooling.

The manifest is the authoritative index for programmatic access; the folder structure
is a convenience for humans and for stratum-level stratified sampling.

### Unique pairing across buckets

Every original photo is assigned to **exactly one** bucket. The downloader enforces
this with a deterministic seeded greedy walk over alphabetically-sorted buckets: each
bucket claims the next N unassigned originals from its candidate pool, and an original
already claimed by an earlier bucket is skipped. This invariant matters for downstream
analysis: cross-bucket comparisons (e.g. comparing Gemini-IG hard-addition flip rates
against MagicBrush easy-replacement flip rates) operate on **disjoint** underlying image
populations, so per-bucket stability statistics are statistically independent. No
original photo's idiosyncrasies — unusual textures, lighting, framing — contaminate
two measurements at once. As a side effect, each bucket folder's nested `original/`
subfolder contains only the originals actually paired into that bucket, making bucket
folders fully self-contained on disk.

---

# 7. Testing Stability on a Specific Image

To run the stability pipeline on one FragFake pair, use the driver:

```bash
python drivers/explore_stability_fragfake.py --image-id <image_id>
```

For example:

```bash
python drivers/explore_stability_fragfake.py --image-id snowboard_000000096638
```

Omitting `--image-id` uses the first pair the loader yields.

**Listing available image IDs:**

```bash
python -c "import json; [print(json.loads(l)['image_id']) for l in open('data/fragfake/manifest.jsonl')]"
```

Or to see IDs for a specific bucket:

```bash
python -c "
import json
for l in open('data/fragfake/manifest.jsonl'):
    r = json.loads(l)
    if r['editor'] == 'Gemini-IG' and r['difficulty'] == 'easy' and r['edit_type'] == 'addition':
        print(r['image_id'])
"
```

**What the driver prints:** a stability report over all region pairs in the before-image, showing flip rates and margins under the AI edit. High flip rates are expected at this stage — region matching across before/after is not yet implemented (see `STABILITY_PLAN.md` §3).

---

# Summary

Downloader guarantees:

```
HuggingFace rows → on-disk layout + manifest.jsonl
```

Loader guarantees:

```
on-disk layout + manifest.jsonl → Iterator[FragFakePair]
```

As long as the **manifest schema and path conventions are respected**, the downloader
and any consumer of `iter_fragfake_pairs` remain compatible.