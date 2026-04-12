"""
Loader for FragFake (original, edited) image pairs.

The FragFake dataset stores edited images under
    Image/<editor>/<difficulty>/<operation>/<object>_<cocoID>_<operation>.jpg
and the matching originals under
    Image/original/<object>_<cocoID>.jpg

(The HuggingFace dataset card describes a slightly different naming scheme
with .png extensions and an "original_" prefix; the actual files on disk
use .jpg and no prefix. Confirmed empirically against MagicBrush easy.)

This module provides:
- derive_original_path: pure filename rewriter, edited path -> original path
- iter_fragfake_pairs:  generator yielding (orig_bgr, edited_bgr, metadata)

The loader is intentionally a pure data-access layer. It does not import
from regions/, descriptors/, or pairwise_stability — it just hands out
image pairs so the stability test can consume them.
"""

import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# Edited filename pattern: <object>_<cocoID>_<operation>.png
# cocoID is a long zero-padded integer; object can contain underscores.
# We anchor on the trailing _<operation>.<ext> and the cocoID being all digits.
_EDITED_RE = re.compile(r"^(?P<object>.+)_(?P<coco_id>\d+)_(?P<operation>addition|replacement)\.[a-zA-Z0-9]+$")


def derive_original_path(edited_path: Path, dataset_root: Path) -> Path:
    """
    Given the path to an edited image, return the path to the matching original.

    Example
    -------
    edited:   data/fragfake/Image/MagicBrush/easy/addition/airplane_000000126413_addition.jpg
    original: data/fragfake/Image/original/airplane_000000126413.jpg

    The original is always a .jpg under Image/original/.
    """
    name = edited_path.name
    m = _EDITED_RE.match(name)
    if not m:
        raise ValueError(f"Edited filename does not match expected pattern: {name}")
    obj = m.group("object")
    coco_id = m.group("coco_id")
    return dataset_root / "Image" / "original" / f"{obj}_{coco_id}.jpg"


def iter_fragfake_pairs(
    root: Path,
    editor: str = "MagicBrush",
    difficulty: str = "easy",
    operation: str = "addition",
    limit: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """
    Iterate over (original, edited, metadata) tuples for one FragFake subset.

    Parameters
    ----------
    root        : path to the dataset root (e.g. data/fragfake)
    editor      : editor name (MagicBrush, GoT, UltraEdit, Gemini-IG, Flux, ...)
    difficulty  : "easy" or "hard"
    operation   : "addition" or "replacement"
    limit       : if set, stop after yielding this many pairs

    Yields
    ------
    (orig_bgr, edited_bgr, metadata) where metadata is a dict with:
        coco_id, object, editor, operation, original_path, edited_path

    Pairs whose original is missing on disk are skipped (with a warning),
    so the caller never receives an unloadable pair.
    """
    root = Path(root)
    edited_dir = root / "Image" / editor / difficulty / operation
    if not edited_dir.exists():
        raise FileNotFoundError(
            f"Edited image directory not found: {edited_dir}. "
            f"Did you run drivers/download_fragfake.py with matching arguments?"
        )

    yielded = 0
    for edited_path in sorted(edited_dir.iterdir()):
        if not edited_path.is_file():
            continue
        try:
            original_path = derive_original_path(edited_path, root)
        except ValueError as e:
            print(f"  skip (bad filename): {edited_path.name}: {e}")
            continue

        if not original_path.exists():
            print(f"  skip (no original): {edited_path.name} -> {original_path.name}")
            continue

        orig_img = cv2.imread(str(original_path))
        edited_img = cv2.imread(str(edited_path))
        if orig_img is None or edited_img is None:
            print(f"  skip (cv2.imread failed): {edited_path.name}")
            continue

        m = _EDITED_RE.match(edited_path.name)
        metadata = {
            "coco_id": m.group("coco_id"),
            "object": m.group("object"),
            "editor": editor,
            "operation": operation,
            "original_path": str(original_path),
            "edited_path": str(edited_path),
        }

        yield orig_img, edited_img, metadata
        yielded += 1
        if limit is not None and yielded >= limit:
            return
