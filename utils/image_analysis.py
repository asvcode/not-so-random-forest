"""This file contains functions for image analysis
"""
# import miscellaneous modules
import numpy as np


def extract_patches(image, boxes, scores, score_threshold):
    """extracts patches from image given bounding boxes

    Args:
        image : image represented as np array
        boxes : list of bounding box coordinates
        scores : list of scores for each bounding box
        score_threshold : threshold used for suppressing bounding boxes

    Returns:
        patches : image patches circumscribed by bounding boxes

    """
    patches = []
    for i, b in enumerate(boxes):
        if scores[i] < score_threshold:
            continue
        b = np.array(b).astype(int)
        patches.append(image[b[1]:b[3], b[0]:b[2], :])
    return patches
