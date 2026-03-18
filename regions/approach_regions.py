#TODO: insert logic here (group 1 - regions):

# from core.types import SegmentationResult
import cv2
import matplotlib.pyplot as plt
import numpy as np

# def segment_image(image):

#     # algorithm here
#     region_map = ...
#     num_regions = ...

#     return SegmentationResult(region_map, num_regions)

# image_processing_pipeline.py


def segment_image(image, k=3):
    """
    Segments image into k regions using K-means clustering.
    Returns:
        segmented_image: color-segmented output
        region_map: label per pixel
        num_regions: number of clusters (k)
    """
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

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)

    region_map = labels.reshape(image.shape[:2])
    num_regions = k

    return segmented_image, region_map, num_regions


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


def main():
    img = cv2.imread("image.png")

    if img is None:
        print("Error: Could not load image.")
        return

    segmented_img, region_map, num_regions = segment_image(img, k=3)

    sift_img, keypoints, descriptors = sift_features(img)

    harris_img = harris_corners(img)

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


if __name__ == "__main__":
    main()
