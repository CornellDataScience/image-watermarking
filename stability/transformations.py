import cv2
import numpy as np


def jpeg_compress(image, quality=50):
    """Simulate JPEG compression by encoding and decoding in memory."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def resize_down_up(image, scale=0.5):
    """Downsample by scale, then upsample back to original size."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def gaussian_blur(image, ksize=5):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def brightness_shift(image, delta=30):
    """Add a constant to all pixel values, clamped to [0, 255]."""
    return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def get_default_transformations():
    """
    Return a standard battery of (name, transform_func) tuples.
    """
    return [
        ("JPEG Q30",    lambda img: jpeg_compress(img, 30)),
        ("JPEG Q50",    lambda img: jpeg_compress(img, 50)),
        ("Resize 0.5",  lambda img: resize_down_up(img, 0.5)),
        ("Blur k=5",    lambda img: gaussian_blur(img, 5)),
        ("Blur k=9",    lambda img: gaussian_blur(img, 9)),
        ("Bright +30",  lambda img: brightness_shift(img, 30)),
        ("Bright -30",  lambda img: brightness_shift(img, -30)),
    ]
