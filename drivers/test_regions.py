#TODO: Create tests (MULTIPLE) to check independently if your approach works; Group 1 - Regions

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.types import SegmentationResult
from regions.approach_regions import slic_superpixels, slic_plus_kmeans, meanshift, k_means, sift_features, harris_corners, otsu_threshold, adaptive_threshold, region_growing, watershed_segmentation, felzenszwalb_segmentation
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import cv2
import numpy as np

from regions.approach_regions import slic_superpixels

def test_segment_image(segment_func, img): 
    print("Running basic functionality test...")

    result = segment_func(img)

    assert result.region_map is not None
    assert result.num_regions > 0

    print("Basic functionality test passed")

def test_region_map_shape(segment_func, img):
    print("Running shape consistency test...")

    result = segment_func(img)

    assert result.region_map.shape[:2] == img.shape[:2]

    print("Shape consistency test passed")

def test_labels_valid(segment_func, img):
    print("Running label validity test...")

    result = segment_func(img)

    region_map = result.region_map

    assert region_map.min() == 0
    assert region_map.max() == result.num_regions - 1

    print("Label validity test passed")

def test_deterministic(segment_func, img):
    print("Running determinism test...")

    result1 = segment_func(img)
    result2 = segment_func(img)

    assert (result1.region_map == result2.region_map).all()

    print("Determinism test passed")

def test_visualize(segment_func, img, name):
    print(f"Running visualization test for {name}...")
    result = segment_func(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    colors = np.random.randint(0, 255, (result.num_regions, 3), dtype=np.uint8)
    colored = colors[result.region_map]
    vis = mark_boundaries(img_rgb, result.region_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(name, fontsize=14)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(vis)
    axes[1].set_title(f"Boundaries ({result.num_regions} regions)")
    axes[1].axis("off")

    axes[2].imshow(colored)
    axes[2].set_title("Colored Regions")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def test_visualize_sift(img):
    print("Running SIFT visualization test...")

    result = k_means(img, k=3)
    region_map = result.region_map
    num_regions = result.num_regions
    sift_img, keypoints, descriptors = sift_features(img)
    harris_img = harris_corners(img)

    pixel_vals = img.reshape((-1, 3)).astype(np.float32)

    _, labels, centers = cv2.kmeans(
    pixel_vals,
    num_regions,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
    10,
    cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()].reshape(img.shape)

    print(f"Number of regions: {num_regions}")
    print(f"SIFT keypoints detected: {len(keypoints)}")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmented Image (k={num_regions})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Keypoints")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corners")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def all_tests(segment_func, img, name):
    print(f"\n=== Testing {name} ===")
    test_segment_image(segment_func, img)
    test_region_map_shape(segment_func, img)
    test_labels_valid(segment_func, img)
    # test_deterministic(segment_func, img)
    test_visualize(segment_func, img, name)

if __name__ == "__main__":
    img = cv2.imread("data/dog.jpg")

    for func, name in [
        (slic_superpixels, "SLIC Superpixels"),
        (slic_plus_kmeans, "SLIC + KMeans"),
        (meanshift,        "Mean Shift"),
        (otsu_threshold, "OTSU Threshold"),
        (adaptive_threshold, "Adaptive Threshold"),
        (watershed_segmentation, "Watershed Segmentation"),
        (region_growing, "Region growing"),
        (felzenszwalb_segmentation, "Felzenszwalb (Graph based) segmentation")
    ]:
        all_tests(func, img, name)

    test_visualize_sift(img)

#continue writing more tests for regions below: