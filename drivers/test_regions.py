#TODO: Create tests (MULTIPLE) to check independently if your approach works; Group 1 - Regions

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.types import SegmentationResult
from regions.approach_regions import slic_superpixels, slic_plus_kmeans, meanshift
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import cv2
import numpy as np

from regions.approach_regions import slic_superpixels

def test_segment_image(segment_func, image): 
    print("Running basic functionality test...")

    img = cv2.imread(image)
    result = segment_func(img)

    assert result.region_map is not None
    assert result.num_regions > 0

    print("Basic functionality test passed")

def test_region_map_shape(segment_func, image):
    print("Running shape consistency test...")

    img = cv2.imread(image)
    result = segment_func(img)

    assert result.region_map.shape[:2] == img.shape[:2]

    print("Shape consistency test passed")

def test_labels_valid(segment_func, image):
    print("Running label validity test...")

    img = cv2.imread(image)
    result = segment_func(img)

    region_map = result.region_map

    assert region_map.min() == 0
    assert region_map.max() == result.num_regions - 1

    print("Label validity test passed")

def test_deterministic(segment_func, image):
    print("Running determinism test...")
    
    img = cv2.imread(image)

    result1 = segment_func(img)
    result2 = segment_func(img)

    assert (result1.region_map == result2.region_map).all()

    print("Determinism test passed")

def test_visualize_segmentation(segment_func, image):
    print("Running visualization test...")

    img = cv2.imread(image)
    result = segment_func(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    vis = mark_boundaries(img_rgb, result.region_map)

    plt.imshow(vis)
    plt.title(f"Segments: {result.num_regions}")
    plt.axis("off")
    plt.show()

def test_visualize_regions_colored(segment_func, image):
    img = cv2.imread(image)
    result = segment_func(img)

    region_map = result.region_map
    h, w = region_map.shape

    colors = np.random.randint(0, 255, (result.num_regions, 3), dtype=np.uint8)
    colored = colors[region_map]

    plt.imshow(colored)
    plt.title("Colored Regions")
    plt.axis("off")
    plt.show()

def all_tests(segment_func, image):
    test_segment_image(segment_func, image)
    test_region_map_shape(segment_func, image)
    test_labels_valid(segment_func, image)
    # test_deterministic(segment_func, image)
    test_visualize_segmentation(segment_func, image)
    test_visualize_regions_colored(segment_func, image)

all_tests(slic_superpixels,"data/dog.jpg")
all_tests(slic_plus_kmeans,"data/dog.jpg")
all_tests(meanshift,"data/dog.jpg")


#continue writing more tests for regions below: