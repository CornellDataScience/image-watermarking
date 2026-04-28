"""
test_fragfake_loader.py
-----------------------
Pytest suite for stability.fragfake_loader.iter_fragfake_pairs.

Run with:
    python -m pytest tests/test_fragfake_loader.py -v

All tests use tmp_path fixtures only — no real FragFake data, no network.
"""

import json

import cv2
import numpy as np
import pytest

from stability.fragfake_loader import FragFakePair, iter_fragfake_pairs


IMG_SHAPE = (16, 16, 3)


def _write_blank(path):
    """Write a 16x16x3 uint8 zeros PNG/JPG to disk (BGR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros(IMG_SHAPE, dtype=np.uint8)
    assert cv2.imwrite(str(path), img), f"cv2.imwrite failed for {path}"


def _build_fake_dataset(root):
    """
    Populate `root` with 3 fake pairs spanning 2 editors, 2 difficulties,
    2 edit_types. Returns the list of manifest records that were written.
    """
    records = [
        {
            "image_id": "laptop_000000412289",
            "editor": "Gemini-IG",
            "difficulty": "easy",
            "edit_type": "addition",
            "before_path": "Gemini-IG/easy/addition/original/original_laptop_000000412289.jpg",
            "after_path": "Gemini-IG/easy/addition/laptop_000000412289_addition.png",
        },
        {
            "image_id": "cat_000000123456",
            "editor": "Gemini-IG",
            "difficulty": "hard",
            "edit_type": "replacement",
            "before_path": "Gemini-IG/hard/replacement/original/original_cat_000000123456.jpg",
            "after_path": "Gemini-IG/hard/replacement/cat_000000123456_replacement.png",
        },
        {
            "image_id": "dog_000000999999",
            "editor": "MagicBrush",
            "difficulty": "easy",
            "edit_type": "addition",
            "before_path": "MagicBrush/easy/addition/original/original_dog_000000999999.jpg",
            "after_path": "MagicBrush/easy/addition/dog_000000999999_addition.png",
        },
    ]

    for rec in records:
        _write_blank(root / rec["before_path"])
        _write_blank(root / rec["after_path"])

    manifest = root / "manifest.jsonl"
    with manifest.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return records


def test_yields_pairs_with_metadata(tmp_path):
    _build_fake_dataset(tmp_path)

    pairs = list(iter_fragfake_pairs(tmp_path))

    assert len(pairs) == 3
    for p in pairs:
        assert isinstance(p, FragFakePair)
        assert p.before.shape == IMG_SHAPE
        assert p.after.shape == IMG_SHAPE
        assert p.before.dtype == np.uint8
        assert p.after.dtype == np.uint8

    ids = {p.image_id for p in pairs}
    assert ids == {
        "laptop_000000412289",
        "cat_000000123456",
        "dog_000000999999",
    }

    by_id = {p.image_id: p for p in pairs}
    assert by_id["laptop_000000412289"].editor == "Gemini-IG"
    assert by_id["laptop_000000412289"].difficulty == "easy"
    assert by_id["laptop_000000412289"].edit_type == "addition"
    assert by_id["cat_000000123456"].difficulty == "hard"
    assert by_id["cat_000000123456"].edit_type == "replacement"
    assert by_id["dog_000000999999"].editor == "MagicBrush"


def test_editor_filter(tmp_path):
    _build_fake_dataset(tmp_path)

    pairs = list(iter_fragfake_pairs(tmp_path, editors=["Gemini-IG"]))

    assert len(pairs) == 2
    assert all(p.editor == "Gemini-IG" for p in pairs)


def test_difficulty_filter(tmp_path):
    _build_fake_dataset(tmp_path)

    pairs = list(iter_fragfake_pairs(tmp_path, difficulties=["easy"]))

    assert len(pairs) == 2
    assert all(p.difficulty == "easy" for p in pairs)


def test_edit_type_filter(tmp_path):
    _build_fake_dataset(tmp_path)

    pairs = list(iter_fragfake_pairs(tmp_path, edit_types=["addition"]))

    assert len(pairs) == 2
    assert all(p.edit_type == "addition" for p in pairs)


def test_combined_filters(tmp_path):
    _build_fake_dataset(tmp_path)

    pairs = list(
        iter_fragfake_pairs(
            tmp_path,
            editors=["Gemini-IG"],
            difficulties=["easy"],
            edit_types=["addition"],
        )
    )

    assert len(pairs) == 1
    assert pairs[0].image_id == "laptop_000000412289"


def test_missing_file_warns_and_skips(tmp_path):
    records = _build_fake_dataset(tmp_path)

    bad_record = {
        "image_id": "ghost_000000000001",
        "editor": "Gemini-IG",
        "difficulty": "easy",
        "edit_type": "addition",
        "before_path": "Gemini-IG/easy/addition/original/original_ghost_000000000001.jpg",
        "after_path": "Gemini-IG/easy/addition/ghost_000000000001_addition.png",
    }
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("a") as f:
        f.write(json.dumps(bad_record) + "\n")

    with pytest.warns(UserWarning):
        pairs = list(iter_fragfake_pairs(tmp_path))

    assert len(pairs) == len(records)
    ids = {p.image_id for p in pairs}
    assert "ghost_000000000001" not in ids