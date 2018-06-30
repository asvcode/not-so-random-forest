# function to extract bounded objects given image and bounding boxes

import numpy as np

def extract_objects(image, boxes, scores, score_threshold):
    objects = []
    for i, b in enumerate(boxes):
        if scores[i] < score_threshold:
            continue
        b = np.array(b).astype(int)
        objects.append(unscaled_image[b[1]:b[3], b[0]:b[2], :])
    return objects
