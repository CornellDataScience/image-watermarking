#TODO: insert logic here (group 1 - regions):

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage import color
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

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
