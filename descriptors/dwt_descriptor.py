"""
dwt_descriptor.py
-----------------
Implements a 4-dimensional DWT energy descriptor for image regions,
then slides a 64×64 window across the entire image (no overlap) and
saves every patch fingerprint to fingerprints.json.

Dependencies: pywt, numpy, Pillow, scikit-image
"""

import json
import argparse
import numpy as np
import pywt
from pathlib import Path
from PIL import Image
from skimage import color
from skimage import data as skdata


# ---------------------------------------------------------------------------
# Core descriptor function
# ---------------------------------------------------------------------------

def dwt_descriptor(region: np.ndarray) -> np.ndarray:
    """
    Compute a normalized 4-element DWT energy descriptor for an image region.

    Parameters
    ----------
    region : np.ndarray
        H×W×3 uint8 (or float) RGB patch.

    Returns
    -------
    np.ndarray, shape (4,)
        Normalized energy vector [E_LL, E_LH, E_HL, E_HH] summing to 1.
        Subband order matches pywt convention: LL, LH (horizontal edges),
        HL (vertical edges), HH (diagonal / texture noise).
    """

    # Step 1 — Convert to grayscale.
    # DWT operates on a single-channel 2-D signal.  rgb2gray applies the
    # standard luminance weights (0.2126 R + 0.7152 G + 0.0722 B) so the
    # result reflects perceived brightness rather than a raw average.
    if region.ndim == 3:
        gray = color.rgb2gray(region.astype(np.float64))
    else:
        # Already single-channel; just normalise to [0, 1] if needed.
        gray = region.astype(np.float64)
        if gray.max() > 1.0:
            gray /= 255.0

    # Step 2 — Apply a single-level 2-D Discrete Wavelet Transform (Haar).
    # pywt.dwt2 returns:
    #   cA  — approximation coefficients (LL subband): low-pass in both dims,
    #          captures the coarse, average structure of the region.
    #   (cH, cV, cD) — detail coefficients:
    #     cH (LH): low-pass horizontal × high-pass vertical → horizontal edges
    #     cV (HL): high-pass horizontal × low-pass vertical → vertical edges
    #     cD (HH): high-pass in both dims → diagonal edges / fine texture
    # Haar is the simplest wavelet (box filter); it's fast and gives clean
    # energy separation between smooth and sharp content.
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')

    # Step 3 — Compute energy (sum of squared coefficients) for each subband.
    # Energy is a natural summary statistic for wavelet coefficients:
    # large coefficients (= strong response) contribute quadratically,
    # so a single high-contrast edge dominates over many weak ones.
    E_LL = float(np.sum(cA ** 2))   # coarse / low-frequency energy
    E_LH = float(np.sum(cH ** 2))   # horizontal-edge energy
    E_HL = float(np.sum(cV ** 2))   # vertical-edge energy
    E_HH = float(np.sum(cD ** 2))   # diagonal / texture energy

    energies = np.array([E_LL, E_LH, E_HL, E_HH], dtype=np.float64)

    # Step 4 — Normalise by total energy so the vector sums to 1.
    # This makes the descriptor invariant to overall brightness: a dark patch
    # and a bright patch of the same texture pattern will get the same vector.
    total = energies.sum()
    if total > 0:
        energies /= total

    return energies  # shape (4,), values in [0, 1], sum == 1


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _crop(image: np.ndarray, row: int, col: int, size: int = 64) -> np.ndarray:
    """Return a size×size RGB patch starting at (row, col)."""
    return image[row: row + size, col: col + size]


def main() -> None:
    # ---- Parse arguments -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="DWT region descriptor demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3.14 dwt_descriptor.py photo.jpg\n"
            "  python3.14 dwt_descriptor.py photo.jpg --resize 512\n"
            "  python3.14 dwt_descriptor.py img1.jpg img2.jpg --resize 512\n"
        ),
    )
    parser.add_argument(
        "images", nargs="*",
        help="Path(s) to image file(s). Omit to use the built-in test image.",
    )
    parser.add_argument(
        "--resize", type=int, default=None, metavar="N",
        help=(
            "Resize every image to N×N before patching. "
            "Use this when comparing images of different resolutions so that "
            "a 64×64 patch always covers the same proportional area of each image. "
            "Example: --resize 512"
        ),
    )
    args = parser.parse_args()

    # ---- Load image(s) -------------------------------------------------------
    images: list[tuple[str, np.ndarray]] = []

    if args.images:
        for path in args.images:
            # Pillow handles JPEG, PNG, TIFF, BMP, WEBP, etc.
            # Convert to RGB for a consistent 3-channel uint8 array regardless
            # of the source format (RGBA, palette, greyscale, etc.).
            pil_img = Image.open(path).convert("RGB")

            if args.resize:
                # Resize to a square canvas so patches cover the same
                # proportional region across images of different resolutions.
                # LANCZOS gives the best quality when downsampling.
                pil_img = pil_img.resize((args.resize, args.resize), Image.LANCZOS)

            images.append((path, np.array(pil_img)))
    else:
        # No file supplied — use the bundled scikit-image test image.
        img = skdata.astronaut()
        if args.resize:
            pil_img = Image.fromarray(img).resize((args.resize, args.resize), Image.LANCZOS)
            img = np.array(pil_img)
        images.append(("built-in astronaut", img))
        print("No image path provided — using built-in astronaut test image.")
        print("Usage: python3.14 dwt_descriptor.py <image> [image2 ...] [--resize N]\n")

    PATCH_SIZE = 64
    all_fingerprints: list[dict] = []

    # ---- Run descriptor on each image ----------------------------------------
    for img_path, image in images:
        print(f"{'─' * 60}")
        print(f"Image : {img_path}")
        print(f"Shape : {image.shape}  dtype: {image.dtype}")
        if args.resize:
            print(f"Resized to {args.resize}×{args.resize} before patching")

        h, w = image.shape[:2]

        # ---- Slide a 64×64 window across the image with no overlap ----------
        # Only full patches are kept; any remainder at the right/bottom edge
        # is skipped so every patch is exactly PATCH_SIZE×PATCH_SIZE.
        image_fingerprints: list[dict] = []
        for row in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for col in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = _crop(image, row=row, col=col, size=PATCH_SIZE)
                fp = dwt_descriptor(patch)
                image_fingerprints.append({
                    "image": img_path,
                    "row": row,
                    "col": col,
                    "fingerprint": fp.tolist(),
                })

        all_fingerprints.extend(image_fingerprints)
        print(f"Patches: {len(image_fingerprints)} ({h // PATCH_SIZE} rows × {w // PATCH_SIZE} cols)\n")

        # ---- Summary table: LL energy across all patches --------------------
        ll_values = np.array([p["fingerprint"][0] for p in image_fingerprints])
        print(f"{'Statistic':<12} {'LL energy':>12}")
        print("-" * 26)
        print(f"{'min':<12} {ll_values.min():>12.4f}")
        print(f"{'max':<12} {ll_values.max():>12.4f}")
        print(f"{'mean':<12} {ll_values.mean():>12.4f}")
        print()

    # ---- Save all fingerprints to JSON --------------------------------------
    out_path = Path("fingerprints.json")
    out_path.write_text(json.dumps(all_fingerprints, indent=2))
    print(f"Saved {len(all_fingerprints)} fingerprints → {out_path.resolve()}")


def compute_dwt_descriptor(image, seg, component=0) -> list[float]:
    """
    Region-level DWT descriptor compatible with the evaluation interface.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3)  uint8
    seg   : SegmentationResult
    component : int
        Which DWT energy subband to use as the scalar per region.
        0=E_LL, 1=E_LH, 2=E_HL, 3=E_HH. Default 0 (E_LL).

    Returns
    -------
    list[float] of length seg.num_regions
    """
    descriptors = []

    for k in range(seg.num_regions):
        mask = seg.region_map == k

        # Guard: watershed can produce label IDs with zero pixels assigned
        # when transforms change the gradient landscape and labels become
        # non-contiguous. rows/cols would be empty and .min() would crash.
        if not mask.any():
            descriptors.append(0.0)
            continue

        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1

        crop = image[r0:r1, c0:c1].astype(np.float64)
        mask_crop = mask[r0:r1, c0:c1]

        # Guard: pywt.dwt2 needs at least 2 pixels in each dimension.
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            descriptors.append(0.0)
            continue

        # Fill non-region pixels with the region's per-channel mean so the
        # mask boundary creates a smooth transition rather than a hard zero edge.
        for ch in range(crop.shape[2]):
            channel_mean = crop[:, :, ch][mask_crop].mean()
            crop[:, :, ch][~mask_crop] = channel_mean

        energy_vec = dwt_descriptor(crop)
        descriptors.append(float(energy_vec[component]))

    return descriptors


def compute_dwt_ll(image, seg) -> list[float]:
    return compute_dwt_descriptor(image, seg, component=0)

def compute_dwt_lh(image, seg) -> list[float]:
    return compute_dwt_descriptor(image, seg, component=1)

def compute_dwt_hl(image, seg) -> list[float]:
    return compute_dwt_descriptor(image, seg, component=2)

def compute_dwt_hh(image, seg) -> list[float]:
    return compute_dwt_descriptor(image, seg, component=3)


if __name__ == "__main__":
    main()
