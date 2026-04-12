"""
Selective downloader for the FragFake dataset.

FragFake (https://huggingface.co/datasets/Vincent-HKUSTGZ/FragFake) ships
the entire image tree (~25 GB across 8+ editors). For stage-3 stability
testing we only need a subset: one editor + one difficulty + one operation,
plus the matching original COCO images.

Run from the repo root:
    python drivers/download_fragfake.py
    python drivers/download_fragfake.py --editor MagicBrush --difficulty easy --operation addition
    python drivers/download_fragfake.py --dest data/fragfake

Default subset: MagicBrush / easy / addition.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Vincent-HKUSTGZ/FragFake"


def build_allow_patterns(editor: str, difficulty: str, operation: str) -> list[str]:
    """
    Build the allow_patterns list passed to snapshot_download.

    We need:
    - The edited images for the chosen editor/difficulty/operation
    - All original COCO images (these are shared across all editors)
    """
    return [
        f"Image/{editor}/{difficulty}/{operation}/**",
        "Image/original/**",
    ]


def download(editor: str, difficulty: str, operation: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    patterns = build_allow_patterns(editor, difficulty, operation)

    print(f"Downloading FragFake subset to {dest}")
    print(f"  editor:     {editor}")
    print(f"  difficulty: {difficulty}")
    print(f"  operation:  {operation}")
    print(f"  patterns:   {patterns}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
    )

    edited_dir = dest / "Image" / editor / difficulty / operation
    original_dir = dest / "Image" / "original"

    n_edited = sum(1 for _ in edited_dir.glob("*")) if edited_dir.exists() else 0
    n_original = sum(1 for _ in original_dir.glob("*")) if original_dir.exists() else 0

    print(f"\nDone.")
    print(f"  edited images:   {n_edited} in {edited_dir}")
    print(f"  original images: {n_original} in {original_dir}")

    if n_edited == 0:
        print("\nWarning: no edited images found. Check that the editor/difficulty/operation "
              "combination exists on the HuggingFace repo.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Download a subset of FragFake.")
    parser.add_argument("--editor", default="MagicBrush",
                        help="Editor name (e.g. MagicBrush, GoT, UltraEdit, Gemini-IG, Flux).")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "hard"])
    parser.add_argument("--operation", default="addition", choices=["addition", "replacement"])
    parser.add_argument("--dest", default="data/fragfake", type=Path,
                        help="Destination directory (relative to repo root).")
    args = parser.parse_args()

    download(args.editor, args.difficulty, args.operation, args.dest)


if __name__ == "__main__":
    main()
