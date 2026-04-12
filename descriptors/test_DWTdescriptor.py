"""
test_DWTdescriptor.py
---------------------
Robustness test for the DWT energy descriptor.

For each of 4 common image attacks, slides a 64×64 grid over both the
original and attacked image, computes cosine similarity between matching
patch descriptors, and reports how many patches survived (sim >= 0.95).

Usage
-----
    python descriptors/test_DWTdescriptor.py data/dog.jpg

Dependencies: pywt, numpy, Pillow, scikit-image
"""

import argparse
import io
import tempfile
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# dwt_descriptor.py lives in the same directory; Python adds the script's
# own directory to sys.path when run directly, so this import works without
# any path manipulation.
from dwt_descriptor import dwt_descriptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_SIZE = 64
SIM_THRESHOLD = 0.95


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0  # both zero vectors — treat as identical
    return float(np.dot(a, b) / denom)


def pil_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def compare_images(original: Image.Image, attacked: Image.Image) -> dict:
    """
    Slide a 64×64 grid over both images, run dwt_descriptor on each patch
    pair, and return aggregated similarity stats.
    """
    orig_arr = pil_to_array(original)
    atk_arr  = pil_to_array(attacked)

    h, w = orig_arr.shape[:2]
    similarities = []

    for row in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for col in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch_orig = orig_arr[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            patch_atk  = atk_arr [row:row + PATCH_SIZE, col:col + PATCH_SIZE]

            fp_orig = dwt_descriptor(patch_orig)
            fp_atk  = dwt_descriptor(patch_atk)

            similarities.append(cosine_similarity(fp_orig, fp_atk))

    sims = np.array(similarities)
    survived = int(np.sum(sims >= SIM_THRESHOLD))
    damaged  = len(sims) - survived

    return {
        "total":    len(sims),
        "survived": survived,
        "damaged":  damaged,
        "avg_sim":  float(sims.mean()),
        "min_sim":  float(sims.min()),
    }


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def attack_jpeg(img: Image.Image) -> Image.Image:
    """Save as JPEG at quality 75 and reload."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, format="JPEG", quality=75)
        return Image.open(f.name).copy()  # copy so file can be closed/deleted


def attack_blur(img: Image.Image) -> Image.Image:
    """Gaussian blur with radius 1."""
    return img.filter(ImageFilter.GaussianBlur(radius=1))


def attack_resize(img: Image.Image) -> Image.Image:
    """Downsample to 256×256 then back to 512×512 (LANCZOS both ways)."""
    small = img.resize((256, 256), Image.LANCZOS)
    return small.resize((512, 512), Image.LANCZOS)


def attack_brightness(img: Image.Image) -> Image.Image:
    """Increase brightness by 10%."""
    return ImageEnhance.Brightness(img).enhance(1.10)


ATTACKS = [
    ("JPEG quality 75",              attack_jpeg),
    ("Gaussian blur (radius 1)",     attack_blur),
    ("Resize 512→256→512",           attack_resize),
    ("Brightness +10%",              attack_brightness),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness test for the DWT energy descriptor.",
        epilog="Example:  python descriptors/test_DWTdescriptor.py data/dog.jpg",
    )
    parser.add_argument("image", help="Path to the test image.")
    args = parser.parse_args()

    # Load and normalise to a 512×512 RGB baseline.
    original = Image.open(args.image).convert("RGB").resize((512, 512), Image.LANCZOS)
    print(f"Image  : {args.image}  →  resized to 512×512")
    print(f"Grid   : {512 // PATCH_SIZE}×{512 // PATCH_SIZE} = {(512 // PATCH_SIZE) ** 2} patches  "
          f"(threshold: cosine sim ≥ {SIM_THRESHOLD})\n")

    col_w = 26
    header = (
        f"{'Attack':<{col_w}}"
        f"{'Total':>7}  "
        f"{'Survived':>9}  "
        f"{'Damaged':>8}  "
        f"{'Avg sim':>8}  "
        f"{'Min sim':>8}"
    )
    print(header)
    print("─" * len(header))

    verdicts: list[tuple[str, bool]] = []

    for name, fn in ATTACKS:
        attacked = fn(original)
        stats = compare_images(original, attacked)

        survived_pct = stats["survived"] / stats["total"] * 100
        passed = stats["avg_sim"] >= SIM_THRESHOLD

        print(
            f"{name:<{col_w}}"
            f"{stats['total']:>7}  "
            f"{stats['survived']:>8} ({survived_pct:4.0f}%)  "
            f"{stats['damaged']:>8}  "
            f"{stats['avg_sim']:>8.4f}  "
            f"{stats['min_sim']:>8.4f}"
        )
        verdicts.append((name, passed))

    # ---- Overall verdict ----------------------------------------------------
    print("\n" + "─" * len(header))
    print("Verdict:\n")
    for name, passed in verdicts:
        status = "SURVIVED  ✓" if passed else "STRUGGLED ✗"
        print(f"  {status}  {name}")

    n_passed = sum(p for _, p in verdicts)
    print(f"\n{n_passed}/{len(verdicts)} attacks survived (avg sim >= {SIM_THRESHOLD})")


if __name__ == "__main__":
    main()
