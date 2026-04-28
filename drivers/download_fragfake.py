"""
download_fragfake.py
--------------------
Downloads the FragFake dataset (Vincent-HKUSTGZ/FragFake) from HuggingFace
and lays it out on disk, emitting a manifest.jsonl of before/after pairs.

See FRAGFAKE_INTERFACE.md for the on-disk layout and manifest schema.

=============================================================================
HuggingFace access strategy
=============================================================================
Running `HfApi().list_repo_files(repo_id='Vincent-HKUSTGZ/FragFake',
repo_type='dataset')` shows the repo ships images as **individual files in an
`Image/` tree**, not packed inside parquet shards. Layout:

    Image/original/original_<class>_<id>.jpg
    Image/<Editor>/<difficulty>/<edit_type>/<class>_<id>_<edit_type>.png

So we use strategy (a): download individual image files with `hf_hub_download`.
We do **not** use `datasets.load_dataset` (forbidden — materializes the whole
25 GB blob) or parquet streaming (no parquet here).

The pair structure comes from `VLM_Dataset/<Editor>/<difficulty>/
<Editor>_<difficulty>_<split>_conversation.json` — a list where each entry has
a single-image `images` field and a VQA `messages` field. Item paths are
either `Image/original/...` or `Image/<Editor>/.../<edit_type>/...`. The
`<class>_<id>` substring is the pairing key.

Counts for reference (probed 2026-04 on the public snapshot):
  - test splits: exactly 200 items = 100 originals + 100 edits per
    (editor, difficulty)
  - train splits: much larger; easy/train ~4.2k items, hard/train ~3.2k

HuggingFace's snapshot also contains two extra editors (Flux, step1xedit) that
the brief did not name. This script only supports the four named editors by
default; extending `KNOWN_EDITORS` below is the single-line change to pick up
the others.

TODO: edited-object labels could be extracted from the VQA messages[].content
later if a stability metric wants them.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "Vincent-HKUSTGZ/FragFake"
REPO_TYPE = "dataset"

KNOWN_EDITORS = ["Gemini-IG", "GoT", "MagicBrush", "UltraEdit"]
KNOWN_DIFFICULTIES = ["easy", "hard"]
KNOWN_SPLITS = ["train", "test"]
KNOWN_EDIT_TYPES = ["addition", "replacement"]

DEFAULT_LIMIT = 10  # pairs per (editor, difficulty, edit_type) bucket
DEFAULT_SEED = 42

# Match "Image/original/original_<class>_<id>.<ext>" and capture <class>_<id>
ORIGINAL_RE = re.compile(r"^Image/original/original_(?P<iid>.+)\.(?:jpg|jpeg|png)$")
# Match "Image/<Editor>/<difficulty>/<edit_type>/<class>_<id>_<edit_type>.<ext>"
EDIT_RE = re.compile(
    r"^Image/(?P<editor>[^/]+)/(?P<difficulty>easy|hard)/"
    r"(?P<edit_type>addition|replacement)/"
    r"(?P<iid>.+)_(?P=edit_type)\.(?:jpg|jpeg|png)$"
)


def _parse_csv_arg(raw: str | None, allowed: list[str], flag: str) -> list[str]:
    """Parse a CSV CLI flag against an allowed whitelist. Empty/None → allowed."""
    if raw is None:
        return list(allowed)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    bad = [v for v in values if v not in allowed]
    if bad:
        raise SystemExit(
            f"{flag}: unknown values {bad!r}. Allowed: {allowed}"
        )
    return values


def _conversation_filename(editor: str, difficulty: str, split: str) -> str:
    """HF path to the conversation JSON for a given (editor, difficulty, split)."""
    return (
        f"VLM_Dataset/{editor}/{difficulty}/"
        f"{editor}_{difficulty}_{split}_conversation.json"
    )


def _target_original_path(
    output_dir: Path, editor: str, difficulty: str, edit_type: str,
    image_id: str, src_ext: str = "jpg",
) -> Path:
    """
    Path where an original image is stored *for a particular bucket*.

    Under the new layout each (editor, difficulty, edit_type) bucket is
    self-contained: its `original/` subfolder holds only the originals
    paired into that bucket. Unique-pairing (see FRAGFAKE_INTERFACE.md §6)
    guarantees the same `<class>_<id>` original never appears in two
    buckets, so this path is unambiguous per (image_id, bucket).
    """
    return (
        output_dir / editor / difficulty / edit_type / "original" /
        f"original_{image_id}.{src_ext}"
    )


def _target_edit_path(
    output_dir: Path, editor: str, difficulty: str, edit_type: str,
    image_id: str, src_ext: str,
) -> Path:
    return (
        output_dir / editor / difficulty / edit_type /
        f"{image_id}_{edit_type}.{src_ext}"
    )


def _download_file(src_repo_path: str, dst_path: Path, dry_run: bool) -> int:
    """
    Download one file from the HF repo to dst_path if missing or empty.
    Returns the number of bytes on disk after the call (0 on dry-run).
    """
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return dst_path.stat().st_size

    if dry_run:
        return 0

    local = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, filename=src_repo_path
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # hf_hub_download returns a cache path; copy (not symlink) so data/fragfake
    # is self-contained.
    data = Path(local).read_bytes()
    dst_path.write_bytes(data)
    return dst_path.stat().st_size


def _load_conversation(editor: str, difficulty: str, split: str) -> list[dict]:
    """Download and parse one conversation JSON."""
    fname = _conversation_filename(editor, difficulty, split)
    local = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=fname)
    with open(local) as f:
        return json.load(f)


def _enumerate_bucket_candidates(
    editor: str, difficulty: str, edit_type: str, splits: list[str],
) -> list[dict]:
    """
    For one (editor, difficulty, edit_type) bucket, gather every pair the
    HuggingFace conversation JSONs offer.

    A "candidate" is an `image_id` whose original AND a matching edit
    (same editor/difficulty/edit_type) both appear in the conversation
    JSON. Returns a list of dicts holding pairing metadata + HF source
    paths so the caller can later download the chosen subset.

    Returns [] (with a stderr warning) if the conversation JSON for any
    requested split is missing — handles the Gemini-IG/hard and GoT/hard
    404 case where the upstream dataset has no data for that combination.
    """
    candidates_by_id: dict[str, dict] = {}
    for split in splits:
        try:
            conv = _load_conversation(editor, difficulty, split)
        except Exception as e:
            print(
                f"  WARN: cannot load conversation "
                f"{editor}/{difficulty}/{split}: {e}",
                file=sys.stderr,
            )
            continue

        originals: dict[str, tuple[str, str]] = {}  # iid -> (src, ext)
        edits: dict[str, tuple[str, str]] = {}      # iid -> (src, ext)
        for item in conv:
            path = item["images"][0]
            kind = _classify_path(path)
            if kind is None:
                continue
            tag, meta = kind
            if tag == "original":
                originals[meta["image_id"]] = (path, meta["ext"])
            else:
                if (
                    meta["editor"] != editor
                    or meta["difficulty"] != difficulty
                    or meta["edit_type"] != edit_type
                ):
                    continue
                edits[meta["image_id"]] = (path, meta["ext"])

        for iid, (edit_src, edit_ext) in edits.items():
            if iid not in originals:
                continue
            if iid in candidates_by_id:
                # Already discovered in an earlier split; keep first occurrence.
                continue
            orig_src, orig_ext = originals[iid]
            candidates_by_id[iid] = {
                "image_id": iid,
                "editor": editor,
                "difficulty": difficulty,
                "edit_type": edit_type,
                "original_src": orig_src,
                "original_ext": orig_ext,
                "edit_src": edit_src,
                "edit_ext": edit_ext,
            }

    return list(candidates_by_id.values())


def _assign_buckets(
    candidates_by_bucket: dict[tuple[str, str, str], list[dict]],
    limit: int,
    seed: int,
) -> tuple[dict[tuple[str, str, str], list[dict]], list[str]]:
    """
    Greedy unique-pairing assignment.

    Walks buckets in alphabetical (editor, difficulty, edit_type) order,
    deterministically shuffles each bucket's candidate pool with a single
    seeded RNG, then takes the first `limit` candidates whose `image_id`
    has not already been claimed by an earlier bucket. The single shared
    `claimed` set enforces the invariant that every original belongs to
    exactly one bucket.

    Returns (assignment, warnings) where `assignment` maps bucket tuple →
    chosen list[Candidate] (len ≤ limit) and `warnings` is a list of
    human-readable shortfall messages.
    """
    rng = random.Random(seed)
    claimed: set[str] = set()
    assignment: dict[tuple[str, str, str], list[dict]] = {}
    shortfalls: list[str] = []

    for bucket in sorted(candidates_by_bucket.keys()):
        pool = list(candidates_by_bucket[bucket])
        rng.shuffle(pool)
        chosen: list[dict] = []
        for cand in pool:
            if cand["image_id"] in claimed:
                continue
            chosen.append(cand)
            claimed.add(cand["image_id"])
            if len(chosen) >= limit:
                break
        assignment[bucket] = chosen
        if len(chosen) < limit:
            ed, df, et = bucket
            shortfalls.append(
                f"  WARN: bucket {ed}/{df}/{et} got {len(chosen)}/{limit} "
                f"pairs (pool={len(pool)}; "
                f"{len(pool) - len(chosen)} candidates were already claimed "
                f"by earlier buckets or pool was too small)"
            )

    return assignment, shortfalls


def _classify_path(path: str) -> tuple[str, dict] | None:
    """
    Classify an image path from a conversation JSON.

    Returns ("original", {"image_id": ..., "ext": ...}) or
            ("edit",     {"editor": ..., "difficulty": ..., "edit_type": ...,
                           "image_id": ..., "ext": ...})
    Returns None if the path does not match either shape.
    """
    m = ORIGINAL_RE.match(path)
    if m:
        ext = path.rsplit(".", 1)[-1]
        return "original", {"image_id": m.group("iid"), "ext": ext}
    m = EDIT_RE.match(path)
    if m:
        ext = path.rsplit(".", 1)[-1]
        return "edit", {
            "editor": m.group("editor"),
            "difficulty": m.group("difficulty"),
            "edit_type": m.group("edit_type"),
            "image_id": m.group("iid"),
            "ext": ext,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download FragFake from HuggingFace into a bucket-self-contained "
            "on-disk layout with unique-pairing of originals across buckets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python drivers/download_fragfake.py --dry-run\n"
            "  python drivers/download_fragfake.py --editors Gemini-IG "
            "--difficulties easy --edit-types addition --limit 5\n"
            "  python drivers/download_fragfake.py --output-dir data/fragfake\n"
        ),
    )
    parser.add_argument(
        "--editors", default=None,
        help=f"CSV subset of {KNOWN_EDITORS}. Default: all four.",
    )
    parser.add_argument(
        "--difficulties", default=None,
        help=f"CSV subset of {KNOWN_DIFFICULTIES}. Default: both.",
    )
    parser.add_argument(
        "--edit-types", dest="edit_types", default=None,
        help=f"CSV subset of {KNOWN_EDIT_TYPES}. Default: both.",
    )
    parser.add_argument(
        "--splits", default="test",
        help=(
            f"CSV subset of {KNOWN_SPLITS} — which split(s) to scan when "
            "enumerating the candidate pool for each bucket. Default: test."
        ),
    )
    parser.add_argument(
        "--output-dir", default="data/fragfake",
        help="Destination root. Default: data/fragfake",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List intended downloads without fetching.",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=(
            f"Pairs per (editor, difficulty, edit_type) bucket. "
            f"Default: {DEFAULT_LIMIT}."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, metavar="S",
        help=(
            f"Seed for the deterministic shuffle that drives bucket "
            f"assignment. Default: {DEFAULT_SEED}."
        ),
    )
    args = parser.parse_args()

    editors = _parse_csv_arg(args.editors, KNOWN_EDITORS, "--editors")
    difficulties = _parse_csv_arg(args.difficulties, KNOWN_DIFFICULTIES, "--difficulties")
    edit_types = _parse_csv_arg(args.edit_types, KNOWN_EDIT_TYPES, "--edit-types")
    splits = _parse_csv_arg(args.splits, KNOWN_SPLITS, "--splits")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: enumerate candidates per bucket ----
    # For every requested (editor, difficulty, edit_type) bucket, enumerate
    # the (original, edit) pairs FragFake offers. No selection happens yet.

    print(
        f"Enumerating candidates for editors={editors}, "
        f"difficulties={difficulties}, edit_types={edit_types}, "
        f"splits={splits} ..."
    )

    candidates_by_bucket: dict[tuple[str, str, str], list[dict]] = {}
    for editor in editors:
        for difficulty in difficulties:
            for edit_type in edit_types:
                bucket = (editor, difficulty, edit_type)
                cands = _enumerate_bucket_candidates(
                    editor, difficulty, edit_type, splits,
                )
                candidates_by_bucket[bucket] = cands

    pool_total = sum(len(v) for v in candidates_by_bucket.values())
    print(
        f"  pools built: {len(candidates_by_bucket)} buckets, "
        f"{pool_total} candidate pairs total (before unique-pairing assignment)"
    )

    # ---- Phase 1b: greedy unique-pairing assignment ----
    assignment, shortfall_warnings = _assign_buckets(
        candidates_by_bucket, limit=args.limit, seed=args.seed,
    )
    for w in shortfall_warnings:
        print(w, file=sys.stderr)

    # Build download jobs from the assignment. Each candidate produces one
    # original-download job (to its bucket-local original/ subfolder) plus
    # one edit-download job. The unique-pairing invariant guarantees no
    # two candidates share an image_id, so destination paths can never
    # collide.
    jobs: dict[Path, str] = {}
    edit_meta: dict[Path, dict] = {}  # edit dst -> candidate
    per_combo_counts: Counter = Counter()
    assigned_image_ids: set[str] = set()
    for bucket, chosen in assignment.items():
        ed, df, et = bucket
        for cand in chosen:
            iid = cand["image_id"]
            assigned_image_ids.add(iid)
            orig_dst = _target_original_path(
                output_dir, ed, df, et, iid, cand["original_ext"],
            )
            edit_dst = _target_edit_path(
                output_dir, ed, df, et, iid, cand["edit_ext"],
            )
            jobs[orig_dst] = cand["original_src"]
            jobs[edit_dst] = cand["edit_src"]
            edit_meta[edit_dst] = cand
            per_combo_counts[bucket] += 1

    n_pairs = len(edit_meta)
    print(
        f"  assigned {n_pairs} unique pairs across "
        f"{sum(1 for v in assignment.values() if v)} non-empty buckets "
        f"= {len(jobs)} files to fetch"
    )

    if args.dry_run:
        print("\n[dry-run] sample of 10 intended downloads:")
        for i, (dst, src) in enumerate(list(jobs.items())[:10]):
            print(f"  {src}  ->  {dst}")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        print("\n[dry-run] per-bucket assigned pair counts:")
        for bucket, n in sorted(per_combo_counts.items()):
            ed, df, et = bucket
            print(f"  {ed}/{df}/{et}: {n}")
        print("\n[dry-run] exiting without downloading or writing manifest.")
        return

    # ---- Phase 2: download ----
    total_bytes = 0
    skipped = 0
    for dst, src in tqdm(jobs.items(), desc="downloading", unit="file"):
        try:
            before_size = dst.stat().st_size if dst.exists() else -1
            size = _download_file(src, dst, dry_run=False)
            if before_size > 0 and size == before_size:
                skipped += 1
            total_bytes += size
        except Exception as e:
            print(f"  WARN: failed to download {src}: {e}", file=sys.stderr)

    # ---- Phase 3: write manifest ----
    # Emit one JSONL line per (edit) whose before+after both exist with non-zero size.
    manifest_path = output_dir / "manifest.jsonl"
    emitted = 0
    per_combo_emitted: Counter = Counter()
    with manifest_path.open("w") as f:
        for edit_dst, cand in edit_meta.items():
            ed = cand["editor"]
            df = cand["difficulty"]
            et = cand["edit_type"]
            iid = cand["image_id"]
            before = _target_original_path(
                output_dir, ed, df, et, iid, cand["original_ext"],
            )
            after = edit_dst
            if not (before.exists() and before.stat().st_size > 0):
                continue
            if not (after.exists() and after.stat().st_size > 0):
                continue
            record = {
                "image_id": iid,
                "editor": ed,
                "difficulty": df,
                "edit_type": et,
                "before_path": str(before.relative_to(output_dir)),
                "after_path": str(after.relative_to(output_dir)),
            }
            f.write(json.dumps(record) + "\n")
            emitted += 1
            per_combo_emitted[(ed, df, et)] += 1

    # ---- Phase 4: summary ----
    originals_on_disk = sum(
        1 for cand in edit_meta.values()
        if _target_original_path(
            output_dir, cand["editor"], cand["difficulty"], cand["edit_type"],
            cand["image_id"], cand["original_ext"],
        ).exists()
    )
    print("\n=== Download summary ===")
    print(f"  output root:       {output_dir}")
    print(f"  originals on disk: {originals_on_disk}")
    print(f"  edits on disk:     {sum(1 for d in edit_meta if d.exists())}")
    print(f"  total bytes:       {total_bytes:,}")
    print(f"  skipped (already present): {skipped}")
    print(f"  manifest lines emitted:    {emitted}  (at {manifest_path})")
    print(f"  unique image_ids assigned: {len(assigned_image_ids)}  "
          f"(== {emitted} ⇔ no original was reused across buckets)")
    print(f"  per (editor, difficulty, edit_type):")
    for bucket, n in sorted(per_combo_emitted.items()):
        ed, df, et = bucket
        marker = "" if n >= args.limit else f"   [shortfall: {n}/{args.limit}]"
        print(f"    {ed}/{df}/{et}: {n}{marker}")
    if shortfall_warnings:
        print(f"  buckets with shortfalls: {len(shortfall_warnings)} "
              f"(see WARN lines above)")


if __name__ == "__main__":
    main()