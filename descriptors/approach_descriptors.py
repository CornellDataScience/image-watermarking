#TODO: insert logic here (group 2 - descriptors):

from core.types import SegmentationResult

def compute_descriptors(image, segmentation):
    descriptors = []

    for r in range(segmentation.num_regions):
        # compute descriptor
        descriptors.append(0) #whatever your descriptor is for that region

    return descriptors