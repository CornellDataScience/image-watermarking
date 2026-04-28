"""
fragfake_loader.py
------------------
Loader for the on-disk FragFake layout produced by
drivers/download_fragfake.py.

Reads <root>/manifest.jsonl, filters by editor / difficulty / edit_type,
decodes before and after images with cv2.imread (BGR, uint8), and yields
FragFakePair instances.

No network access and no huggingface_hub import lives here; this module
is fully offline so tests can run against tmp_path fixtures.

See FRAGFAKE_INTERFACE.md for the full producer/consumer contract.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FragFakePair:
    """
    A single before/after image pair from FragFake.

    Fields
    ------
    before : np.ndarray
        The original image, shape (H, W, 3), dtype uint8, BGR channel order
        (as returned by cv2.imread).
    after : np.ndarray
        The edited image, shape (H', W', 3), dtype uint8, BGR. Dimensions
        are not guaranteed to match `before` — AI edits can change image
        size.
    image_id : str
        The <class>_<id> pairing key extracted from the source filenames,
        e.g. "laptop_000000412289".
    editor : str
        Which AI editor produced the after-image. One of
        {"Gemini-IG", "GoT", "MagicBrush", "UltraEdit"}.
    difficulty : str
        FragFake's coarse severity tag. One of {"easy", "hard"}.
    edit_type : str
        Whether the edit added or replaced content. One of
        {"addition", "replacement"}.
    """

    before: np.ndarray
    after: np.ndarray
    image_id: str
    editor: str
    difficulty: str
    edit_type: str


def iter_fragfake_pairs(
    root: str | Path = "data/fragfake",
    editors: list[str] | None = None,
    difficulties: list[str] | None = None,
    edit_types: list[str] | None = None,
) -> Iterator[FragFakePair]:
    """
    Yield FragFakePair objects from an on-disk FragFake layout.

    Parameters
    ----------
    root : str | Path
        The output root containing manifest.jsonl and the image folders.
        Defaults to "data/fragfake".
    editors : list[str] | None
        If provided, only yield pairs whose editor tag is in the list.
        None means no filter on this axis.
    difficulties : list[str] | None
        If provided, only yield pairs whose difficulty tag is in the list.
        None means no filter on this axis.
    edit_types : list[str] | None
        If provided, only yield pairs whose edit_type tag is in the list.
        None means no filter on this axis.

    Yields
    ------
    FragFakePair
        One pair per matching manifest line whose before and after images
        both decode successfully. Pairs whose before or after fails to load
        (cv2.imread returns None) are skipped with warnings.warn.

    Notes
    -----
    Filtering is applied before image loading, so excluded rows do not
    incur cv2.imread cost.
    """
    root = Path(root)
    manifest_path = root / "manifest.jsonl"

    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if editors is not None and record["editor"] not in editors:
                continue
            if difficulties is not None and record["difficulty"] not in difficulties:
                continue
            if edit_types is not None and record["edit_type"] not in edit_types:
                continue

            before_path = root / record["before_path"]
            after_path = root / record["after_path"]

            before = cv2.imread(str(before_path))
            if before is None:
                warnings.warn(
                    f"skipping {record['image_id']}: before image failed to load "
                    f"({before_path})"
                )
                continue

            after = cv2.imread(str(after_path))
            if after is None:
                warnings.warn(
                    f"skipping {record['image_id']}: after image failed to load "
                    f"({after_path})"
                )
                continue

            yield FragFakePair(
                before=before,
                after=after,
                image_id=record["image_id"],
                editor=record["editor"],
                difficulty=record["difficulty"],
                edit_type=record["edit_type"],
            )