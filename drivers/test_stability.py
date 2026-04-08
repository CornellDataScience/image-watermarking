import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.types import SegmentationResult
from regions.mock_regions import mock_segment_image
from regions.approach_regions import slic_superpixels, slic_plus_kmeans, otsu_threshold
from descriptors.mock_descriptors import mock_compute_descriptors
from stability.transformations import get_default_transformations
from stability.pairwise_stability import (
    run_stability_test, select_stable_pairs, print_stability_report
)


def test_stability_basic(segment_func, descriptor_func, img, name):
    """Run stability test and print report."""
    print(f"\nRunning stability test for {name}...")

    transforms = get_default_transformations()
    result = run_stability_test(img, segment_func, descriptor_func, transforms, segment_func_name=name)

    assert len(result.pair_results) > 0, "No pairs produced"
    assert result.num_transforms == len(transforms)

    print_stability_report(result)
    print(f"Stability test passed for {name}")

    return result


def test_pair_selection(segment_func, descriptor_func, img, name):
    """Run stability + pair selection, verify selected pairs meet criteria."""
    print(f"\nRunning pair selection test for {name}...")

    transforms = get_default_transformations()
    result = run_stability_test(img, segment_func, descriptor_func, transforms, segment_func_name=name)

    stable = select_stable_pairs(result, max_flip_rate=0.0, min_margin=5.0)

    for pr in stable:
        assert pr.flip_rate == 0.0, f"Pair ({pr.i},{pr.j}) has flip_rate {pr.flip_rate}"
        assert pr.mean_margin >= 5.0, f"Pair ({pr.i},{pr.j}) has margin {pr.mean_margin}"

    # Verify sorted by margin descending
    for k in range(len(stable) - 1):
        assert stable[k].mean_margin >= stable[k + 1].mean_margin

    print(f"  Selected {len(stable)} stable pairs out of {len(result.pair_results)} total")
    if stable:
        print(f"  Best pair: ({stable[0].i},{stable[0].j}) margin={stable[0].mean_margin:.2f}")
        print(f"  Worst selected: ({stable[-1].i},{stable[-1].j}) margin={stable[-1].mean_margin:.2f}")

    print(f"Pair selection test passed for {name}")
    return stable


def visualize_flip_rate_heatmap(stability_result):
    """Display NxN heatmap where cell (i,j) = flip rate of that pair."""
    n = stability_result.num_regions
    heatmap = np.full((n, n), np.nan)

    for pr in stability_result.pair_results:
        heatmap[pr.i, pr.j] = pr.flip_rate
        heatmap[pr.j, pr.i] = pr.flip_rate

    plt.figure(figsize=(8, 7))
    plt.imshow(heatmap, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar(label='Flip Rate')
    plt.title(f"Pairwise Flip Rate Heatmap — {stability_result.segment_func_name}")
    plt.xlabel("Region j")
    plt.ylabel("Region i")
    plt.tight_layout()
    plt.show()


def visualize_stable_pairs_on_image(image, segmentation, stable_pairs, name=""):
    """Overlay image with stable regions highlighted."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Identify which regions appear in stable pairs
    stable_region_ids = set()
    for pr in stable_pairs:
        stable_region_ids.add(pr.i)
        stable_region_ids.add(pr.j)

    # Create overlay: green for stable regions, dim everything else
    overlay = img_rgb.copy().astype(np.float32)
    mask_stable = np.isin(segmentation.region_map, list(stable_region_ids))
    overlay[~mask_stable] *= 0.3  # dim unstable regions

    # Green tint on stable regions
    green_tint = np.zeros_like(img_rgb, dtype=np.float32)
    green_tint[mask_stable, 1] = 40
    overlay = np.clip(overlay + green_tint, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Stable Regions — {name} ({len(stable_pairs)} pairs, {len(stable_region_ids)} regions)")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Stable Regions (green)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img = cv2.imread("data/dog.jpg")
    assert img is not None, "Could not load data/dog.jpg"

    # Mock grid 4x4 — fast, deterministic, good for visualization
    print("\n========== Mock Grid 4x4 ==========")
    segment_mock_4x4 = lambda img: mock_segment_image(img, grid_size=4)
    result_mock = test_stability_basic(segment_mock_4x4, mock_compute_descriptors, img, "Mock Grid 4x4")
    stable_mock = test_pair_selection(segment_mock_4x4, mock_compute_descriptors, img, "Mock Grid 4x4")

    # Real segmentation methods
    for func, name in [
        (slic_superpixels, "SLIC Superpixels"),
        (otsu_threshold, "Otsu Threshold"),
    ]:
        print(f"\n========== {name} ==========")
        test_stability_basic(func, mock_compute_descriptors, img, name)
        test_pair_selection(func, mock_compute_descriptors, img, name)

    # Visualization for mock grid
    print("\n========== Visualization ==========")
    visualize_flip_rate_heatmap(result_mock)
    seg_mock = mock_segment_image(img, grid_size=4)
    visualize_stable_pairs_on_image(img, seg_mock, stable_mock, "Mock Grid 4x4")
