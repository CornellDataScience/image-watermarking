"""
download_coco_sample.py
-----------------------
Downloads a sample of N COCO val2017 images to data/ for Mode A evaluation.

Two-step process:
  1. Download the COCO 2017 annotation zip (~241 MB, one-time) to read the
     official image filename list. The list is cached to data/.coco_image_list.json
     so subsequent runs skip this heavyweight step entirely.
  2. Download the first --limit images (or a shuffled subset) from
     images.cocodataset.org/val2017/ into --output-dir.

The downloaded images are picked up automatically by Mode A in
drivers/run_pairwise_stability.py (IMAGE_DIR = "data/", MAX_IMAGES = 100).

Usage:
  python drivers/download_coco_sample.py                  # 100 images → data/
  python drivers/download_coco_sample.py --limit 50       # 50 images
  python drivers/download_coco_sample.py --shuffle        # random sample
  python drivers/download_coco_sample.py --dry-run        # list without fetching
  python drivers/download_coco_sample.py --keep-zip       # keep annotation zip
"""

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


ANNOTATIONS_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ANNOTATIONS_JSON_INNER = "annotations/instances_val2017.json"
IMAGES_BASE_URL = "http://images.cocodataset.org/val2017/"

DEFAULT_LIMIT = 100
DEFAULT_OUTPUT_DIR = "data"
CACHE_FILENAME = ".coco_image_list.json"


def _download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Stream url to dest with a tqdm progress bar."""
    req = urllib.request.urlopen(url)
    total = int(req.headers.get("Content-Length", 0)) or None
    with open(dest, "wb") as f, tqdm(
        desc=desc,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        file=sys.stdout,
    ) as bar:
        while True:
            chunk = req.read(65536)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))


def _fetch_image_list(output_dir: Path, keep_zip: bool, dry_run: bool) -> list[dict]:
    """
    Return the COCO val2017 image metadata list (one dict per image).
    Uses a cached JSON if available; otherwise downloads the annotation zip.
    Each dict has at least {"file_name": "000000XXXXXX.jpg", "coco_url": "..."}.
    """
    cache_path = output_dir / CACHE_FILENAME
    if cache_path.exists():
        print(f"  Image list cache found: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    if dry_run:
        print("[dry-run] Would download COCO annotation zip to read image list.")
        return []

    zip_path = output_dir / "_annotations_val2017_temp.zip"
    print(f"Downloading COCO val2017 annotation zip (~241 MB) …")
    _download_with_progress(ANNOTATIONS_ZIP_URL, zip_path, desc="annotations.zip")

    print(f"  Extracting image list from {ANNOTATIONS_JSON_INNER} …")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ANNOTATIONS_JSON_INNER) as f:
            data = json.load(f)
    images: list[dict] = data["images"]

    # Cache just what we need (filename + URL) to keep it small.
    slim = [{"file_name": im["file_name"], "coco_url": im["coco_url"]} for im in images]
    with open(cache_path, "w") as f:
        json.dump(slim, f)
    print(f"  Cached image list ({len(slim)} entries) → {cache_path}")

    if not keep_zip:
        zip_path.unlink()
        print(f"  Deleted annotation zip.")

    return slim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a COCO val2017 image sample for Mode A evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, metavar="N",
        help=f"Number of images to download. Default: {DEFAULT_LIMIT}.",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Directory to save images. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Random sample instead of first N by image ID.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S",
        help="RNG seed for --shuffle. Default: 42.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without downloading anything.",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the annotation zip after caching the image list.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in output_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    print(f"Output dir: {output_dir}  ({len(existing)} images already present)")

    # Step 1: get full image list (from cache or annotation zip)
    all_images = _fetch_image_list(output_dir, keep_zip=args.keep_zip, dry_run=args.dry_run)

    if args.dry_run:
        print(f"[dry-run] Would select {args.limit} images "
              f"({'shuffled' if args.shuffle else 'first N by ID'}) "
              f"and download missing ones to {output_dir}/")
        return

    if not all_images:
        print("No image list available. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: select subset
    pool = list(all_images)
    if args.shuffle:
        random.Random(args.seed).shuffle(pool)
    selected = pool[:args.limit]

    need = [img for img in selected if img["file_name"] not in existing]
    print(f"\nSelected {len(selected)} images  ({len(selected) - len(need)} already present, {len(need)} to download)")

    # Step 3: download
    downloaded = 0
    failed = 0
    for img_meta in tqdm(need, desc="downloading", unit="img", file=sys.stdout):
        filename = img_meta["file_name"]
        url = IMAGES_BASE_URL + filename
        dest = output_dir / filename
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
        except Exception as e:
            print(f"\n  WARN: failed {filename}: {e}", file=sys.stderr)
            if dest.exists():
                dest.unlink()
            failed += 1

    total_on_disk = len(list(output_dir.glob("*.jpg")))
    print(f"\n=== COCO download summary ===")
    print(f"  output dir:       {output_dir}")
    print(f"  downloaded now:   {downloaded}")
    print(f"  skipped (cached): {len(selected) - len(need)}")
    print(f"  failed:           {failed}")
    print(f"  .jpg files in dir: {total_on_disk}")
    if failed:
        print(f"  Re-run to retry failed downloads (skips already-present files).")


if __name__ == "__main__":
    main()
