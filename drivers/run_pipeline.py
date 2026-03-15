from regions.approach_regions import segment_image
from descriptors.approach_descriptors import compute_descriptors
from core.types import SegmentationResult


image = "insert test image" #TODO: replace with your image loading function (integration) - groups can ignore

segmentation = segment_image(image)

descriptors = compute_descriptors(image, segmentation)

print(descriptors)