#TODO: insert logic here (group 1 - regions):

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage import color
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.types import SegmentationResult

def slic_superpixels(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)
    labels = slic(image_lab, n_segments=200, compactness=20, start_label=0)
    region_map = labels.astype(int)
    num_regions = int(region_map.max() + 1)

    return SegmentationResult(region_map, num_regions)

def slic_plus_kmeans(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)
    labels = slic(image_lab, n_segments=200, compactness=20, start_label=0)
    num_regions = int(labels.max() + 1)

    region_features = []
    for i in range(num_regions):
        mask = labels == i

        avg_color = image_rgb[mask].mean(axis=0)

        y_coords, x_coords = np.where(mask)
        cx = x_coords.mean() / image.shape[1]
        cy = y_coords.mean() / image.shape[0]
        position_weight = 3

        region_features.append([
            avg_color[0], avg_color[1], avg_color[2],
            cx * position_weight, cy * position_weight
        ])

    region_features = np.array(region_features)

    kmeans = KMeans(n_clusters=4)
    region_groups = kmeans.fit_predict(region_features)

    merged_labels = np.zeros_like(labels)
    for i in range(num_regions):
        merged_labels[labels == i] = region_groups[i]

    region_map = merged_labels.astype(int)
    num_regions_final = int(region_map.max() + 1)

    return SegmentationResult(region_map, num_regions_final)

def meanshift(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)

    h, w = image.shape[:2]
    pixels = image_lab.reshape(-1, 3)

    bandwidth = estimate_bandwidth(pixels, quantile=0.1, n_samples=500, random_state=42)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(pixels)

    labels = meanshift.labels_.reshape(h, w)
    num_regions = int(labels.max() + 1)

    region_features = []
    for i in range(num_regions):
        mask = labels == i

        avg_color = image_rgb[mask].mean(axis=0)

        y_coords, x_coords = np.where(mask)
        cx = x_coords.mean() / w
        cy = y_coords.mean() / h
        position_weight = 3

        region_features.append([
            avg_color[0], avg_color[1], avg_color[2],
            cx * position_weight, cy * position_weight
        ])

    region_features = np.array(region_features)

    region_map = labels.astype(int)
    num_regions_final = int(region_map.max() + 1)

    return SegmentationResult(region_map, num_regions_final)
 
def k_means(image, k=3):
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixel_vals,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    region_map = labels.reshape(image.shape[:2])
    num_regions = k

    return SegmentationResult(region_map, num_regions)

def otsu_threshold(image):
    """
    Applies Otsu's thresholding.
    Returns:
        binary_image: thresholded image
        region_map: 0/1 labels
        num_regions: 2
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    region_map = (binary > 0).astype(int)
    num_regions = 2

    return SegmentationResult(region_map, num_regions)


def adaptive_threshold(image):
    """
    Applies adaptive thresholding.
    Returns:
        binary_image: thresholded image
        region_map: 0/1 labels
        num_regions: 2
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,   
        2     
    )

    region_map = (binary > 0).astype(int)
    num_regions = 2

    return SegmentationResult(region_map, num_regions)

def watershed_segmentation(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_rgb, markers)

    region_map = markers.copy()
    region_map[region_map == -1] = 0
    region_map = region_map.astype(int)

    num_regions = int(region_map.max() + 1)

    return SegmentationResult(region_map, num_regions)

def region_growing(image, threshold=50):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image_gray.shape

    region_map = -np.ones((h, w), dtype=int)
    current_label = 0

    def get_neighbors(y, x):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx

    for y in range(h):
        for x in range(w):
            if region_map[y, x] != -1:
                continue

            seed_value = image_gray[y, x]
            queue = [(y, x)]
            region_map[y, x] = current_label

            while queue:
                cy, cx = queue.pop(0)

                for ny, nx in get_neighbors(cy, cx):
                    if region_map[ny, nx] == -1:
                        if abs(int(image_gray[ny, nx]) - int(seed_value)) < threshold:
                            region_map[ny, nx] = current_label
                            queue.append((ny, nx))

            current_label += 1

    num_regions = int(region_map.max() + 1)

    return SegmentationResult(region_map, num_regions)

def sift_features(image):
    """
    Detects SIFT keypoints and descriptors.
    Returns:
        image with keypoints drawn
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    img_keypoints = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return img_keypoints, keypoints, descriptors

def harris_corners(image):
    """
    Detects corners using Harris Corner Detection.
    Returns:
        image with corners marked in red
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    result = image.copy()
    result[dst > 0.01 * dst.max()] = [0, 0, 255]

    return result
