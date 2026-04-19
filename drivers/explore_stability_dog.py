import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from regions.approach_regions import slic_superpixels
from descriptors.mock_descriptors import mock_compute_descriptors
from regions.mock_regions import mock_segment_image
from stability.transformations import jpeg_compress
from stability.pairwise_stability import (
    run_stability_test, select_stable_pairs, print_stability_report
)
from regions.approach_regions import slic_superpixels
from stability.transformations import jpeg_compress


img = cv2.imread("data/dog.jpg")

transforms = [("JPEG Q50", lambda im: jpeg_compress(im, 50))]

result = run_stability_test(
    img,
    segment_func=slic_superpixels,
    descriptor_func=mock_compute_descriptors,
    transformations=transforms,
    segment_func_name="SLIC + mean-gray",
)

print_stability_report(result)
stable = select_stable_pairs(result, max_flip_rate=0.0, min_margin=5.0)
print(f"{len(stable)} stable pairs survived JPEG Q50")

seg_before = slic_superpixels(img)
seg_after = slic_superpixels(jpeg_compress(img, 50))
print(f"before regions: {seg_before.num_regions}, after: {seg_after.num_regions}")

