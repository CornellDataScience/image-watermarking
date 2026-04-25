import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from regions.mock_regions import mock_segment_image
from regions.approach_regions import slic_superpixels, otsu_threshold
from descriptors.mock_descriptors import mock_compute_descriptors
from stability.transformations import get_default_transformations
from stability.pairwise_stability import run_stability_test, select_stable_pairs


OUT_DIR = "exploring_stability_visuals"


def slugify(name):
    return name.lower().replace(" ", "_").replace("+", "plus").replace("=", "")


def save_aggregate_heatmap(result, out_path):
    n = result.num_regions
    hm = np.full((n, n), np.nan)
    for pr in result.pair_results:
        hm[pr.i, pr.j] = pr.flip_rate
        hm[pr.j, pr.i] = pr.flip_rate

    plt.figure(figsize=(8, 7))
    plt.imshow(hm, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(label='Flip Rate (avg across all transforms)')
    plt.title(f"Aggregate Flip Rate — {result.segment_func_name}")
    plt.xlabel("Region j")
    plt.ylabel("Region i")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_overlay(image, segmentation, stable_pairs, name, out_path):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    stable_region_ids = set()
    for pr in stable_pairs:
        stable_region_ids.add(pr.i)
        stable_region_ids.add(pr.j)

    overlay = img_rgb.copy().astype(np.float32)
    mask_stable = np.isin(segmentation.region_map, list(stable_region_ids))
    overlay[~mask_stable] *= 0.3
    green_tint = np.zeros_like(img_rgb, dtype=np.float32)
    green_tint[mask_stable, 1] = 40
    overlay = np.clip(overlay + green_tint, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Stable Regions — {name} ({len(stable_pairs)} pairs, {len(stable_region_ids)} regions)")
    axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Stable Regions (green)"); axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_per_transform_heatmaps(result, transform_names, out_path):
    """Grid of NxN heatmaps — one per transform. Cell = 1 if that pair flipped under this transform, else 0."""
    n = result.num_regions
    T = len(transform_names)

    flip_maps = [np.full((n, n), np.nan) for _ in range(T)]
    for pr in result.pair_results:
        orig = pr.signs[0]
        for t in range(T):
            flipped = 1.0 if pr.signs[t + 1] != orig else 0.0
            flip_maps[t][pr.i, pr.j] = flipped
            flip_maps[t][pr.j, pr.i] = flipped

    cols = min(4, T)
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for t, (tname, fm) in enumerate(zip(transform_names, flip_maps)):
        ax = axes[t]
        ax.imshow(fm, cmap='RdYlGn_r', vmin=0, vmax=1)
        flipped_count = int(np.nansum(fm) / 2)
        ax.set_title(f"{tname}\n{flipped_count} flipped", fontsize=10)
        ax.set_xlabel("j"); ax.set_ylabel("i")
    for k in range(T, len(axes)):
        axes[k].axis('off')
    fig.suptitle(f"Per-Transform Flip Maps — {result.segment_func_name}", y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_per_transform_breakdown(result, transform_names, out_path):
    """Bar chart: # pairs that flipped under each individual transform."""
    T = len(transform_names)
    flipped_counts = np.zeros(T, dtype=int)
    for pr in result.pair_results:
        orig = pr.signs[0]
        for t in range(T):
            if pr.signs[t + 1] != orig:
                flipped_counts[t] += 1

    total = len(result.pair_results)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(T), flipped_counts, color='steelblue')
    ax.set_xticks(range(T))
    ax.set_xticklabels(transform_names, rotation=30, ha='right')
    ax.set_ylabel(f"# pairs flipped (of {total})")
    ax.set_title(f"Per-Transform Flipped Pair Counts — {result.segment_func_name}")
    for bar, count in zip(bars, flipped_counts):
        pct = 100 * count / total if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8)
    ax.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    img = cv2.imread("data/dog.jpg")
    assert img is not None, "Could not load data/dog.jpg"

    os.makedirs(OUT_DIR, exist_ok=True)

    transforms = get_default_transformations()
    transform_names = [t[0] for t in transforms]

    configs = [
        ("Mock Grid 4x4", lambda im: mock_segment_image(im, grid_size=4)),
        ("SLIC Superpixels", slic_superpixels),
        ("Otsu Threshold", otsu_threshold),
    ]

    for name, seg_func in configs:
        slug = slugify(name)
        print(f"\n========== {name} ==========")

        result = run_stability_test(img, seg_func, mock_compute_descriptors, transforms, segment_func_name=name)
        stable = select_stable_pairs(result, max_flip_rate=0.0, min_margin=5.0)
        print(f"  {result.num_regions} regions, {len(result.pair_results)} pairs total, {len(stable)} stable")

        save_aggregate_heatmap(result, f"{OUT_DIR}/heatmap_{slug}.png")
        seg = seg_func(img)
        save_overlay(img, seg, stable, name, f"{OUT_DIR}/overlay_{slug}.png")
        save_per_transform_heatmaps(result, transform_names, f"{OUT_DIR}/per_transform_heatmaps_{slug}.png")
        save_per_transform_breakdown(result, transform_names, f"{OUT_DIR}/per_transform_breakdown_{slug}.png")
        print(f"  Saved 4 PNGs → {OUT_DIR}/*_{slug}.png")

    print(f"\nAll outputs in ./{OUT_DIR}/")
    print(f"To clean up:  rm drivers/exploring_stability_visuals.py && rm -rf {OUT_DIR}")
