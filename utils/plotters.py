"""This file contains functions for plotting
"""

# import keras_retinanet modules
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box

# import custom modules
from utils.image_analysis import extract_patches

# import miscellaneous modules
import pickle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize_inference(image_path):
    """visualizes trees and species information in image

    Args:
        image_path : path to image

    Returns:
        patches : image patches circumscribed by bounding boxes

    """
    unscaled_image = np.asarray(Image.open(image_path))

    # convert grayscale image to a 3-channel image
    if np.ndim(unscaled_image) == 2 or unscaled_image.shape[2] == 1:
            unscaled_image = np.repeat(unscaled_image[:, :, np.newaxis],
                                       3, axis=2)

    # drop alpha channel
    unscaled_image = unscaled_image[:, :, :3]

    # bgr to rgb
    image = unscaled_image
    prefix = image_path.split('.')[0]
    inference = pickle.load(open(prefix+'_inference.p', 'rb'))

    boxes = inference['boxes']
    scores = inference['scores']
    score_threshold = inference['score_threshold']

    # extract tree patches
    tree_patches = extract_patches(image, boxes, scores, score_threshold)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    for box in boxes:
        box = box.astype(int)
        draw_box(draw, box, color=[0, 255, 255], thickness=20)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw[:, :, ::-1])

    fig, axes = plt.subplots(nrows=1, ncols=len(tree_patches),
                             figsize=(20, 10))
    if len(tree_patches) > 1:
        for i, ax in enumerate(axes):
            ax.imshow(tree_patches[i])
            ax.axis('off')
    else:
        axes.imshow(tree_patches[0])
        axes.axis('off')

    print('Species probabilities: ')
    print(inference['species_probabilities'])
