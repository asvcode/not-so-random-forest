#!/usr/bin/env python

# import miscellaneous modules
from PIL import Image
import argparse
import sys
import glob
import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import imghdr
import os


# import keras
from keras import backend
from keras.preprocessing import image
from keras.models import load_model

# import keras_retinanet modules
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, \
 resize_image

# import custom modules
sys.path.append("..")
from utils.image_analysis import extract_patches


def get_session():
    """returns a tensorflow session

    Args:
        None
    Returns:
        tf.Session : tensorflow session

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def object_detector(file, detector_model):
    """runs object detector on images to detect trees

    Args:
        file : path to image
        detector_model: tree detector model
    Returns:
        detector_output : tuple contanining detector output

    """

    # set constants
    SCORE_THRESHOLD = 0.5

    image = read_image_bgr(file)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    unscaled_image = np.asarray(Image.open(file))

    # convert grayscale image to a 3-channel image
    if np.ndim(unscaled_image) == 2 or unscaled_image.shape[2] == 1:
        unscaled_image = np.repeat(unscaled_image[:, :, np.newaxis], 3, axis=2)

    # drop alpha channel
    unscaled_image = unscaled_image[:, :, :3]

    # bgr to rgb
    unscaled_image[:, :, ::-1].copy()

    # run detector on image
    start = time.time()
    boxes, scores, labels = detector_model.predict_on_batch(
                                                np.expand_dims(image, axis=0))
    print("detection time: ", time.time() - start)

    # extract tree patches from image
    boxes /= scale
    tree_patches = extract_patches(unscaled_image, boxes[0],
                                   scores[0], score_threshold=SCORE_THRESHOLD)
    valid_boxes = [box for i, box in enumerate(
        boxes[0]) if scores[0][i] > SCORE_THRESHOLD]
    valid_scores = [score for i, score in enumerate(
        scores[0]) if scores[0][i] > SCORE_THRESHOLD]
    detector_output = (tree_patches, unscaled_image,
                       valid_boxes, valid_scores, SCORE_THRESHOLD)
    return detector_output


def species_predictor(tree_patches, classifier_model):
    """runs object detector on images to detect trees

    Args:
        tree_patches : list of tree patches
        classifier_model : species classifier model
    Returns:
        species_probabilities : np array with species species_probabilities

    """

    resized_tree_patches = []
    for tree_patch in tree_patches:
        resized_tree_patch = cv2.resize(tree_patch, (224, 224))
        img = image.img_to_array(resized_tree_patch)
        img = np.expand_dims(img, axis=0)
        resized_tree_patches.append(img)
    if resized_tree_patches:
        resized_tree_patches = np.vstack(resized_tree_patches)

        start = time.time()
        species_probabilities = classifier_model.predict_classes(resized_tree_patches)
        print("classification time: ", time.time() - start)
    else:
        species_probabilities = []

    return(species_probabilities)


def tree_extractor(image_folder, detector, classifier):
    """loads and passes images to detector and classifier models,
        writes inference results to disk

    Args:
        image_folder : path to folder containing images
        detector : path to detector model
        classifier: path to classifier model
    Returns:
        None

    """
    # get tensorflow session
    backend.tensorflow_backend.set_session(get_session())

    detector_model = models.load_model(detector, backbone_name='resnet50')
    classifier_model = load_model(filepath=classifier)
    #class_indices = pickle.load(open(classifier[:-3] +
                                     #'_class_indices.p', 'rb'))
    class_indices = []
    files = glob.glob(image_folder + '/*')

    for file in files:
        if not imghdr.what(file):
            continue
        print(file)

        # running object detector on image to detect trees
        (tree_patches, unscaled_image, valid_boxes, valid_scores,
         score_threshold) = object_detector(file, detector_model)

        # running species classifier on detected trees
        species_probabilities = species_predictor(tree_patches,
                                                  classifier_model)
        prefix = os.path.basename(file).split('.')[0]

        # write inference results to disk
        with open(image_folder + '/' + prefix +
                  '_inference.p', 'wb') as handle:
            pickle.dump({"species": class_indices,
                         "species_probabilities": species_probabilities,
                         "boxes": valid_boxes, "scores": valid_scores,
                         "score_threshold": score_threshold}, handle)


# argument parser
def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Script for running inference on images.')
    parser.add_argument(
        '--dir', help='Folder containing images to run inference on.',
        default='./sample_images/')
    parser.add_argument(
        '--detector', help='Model for tree detection.')
    parser.add_argument(
        '--classifier', help='Model for species classification.')
    return parser.parse_args(args)


def main(args=None):
    args = sys.argv[1:]
    args = parse_args(args)

    # extract trees
    tree_extractor(image_folder=args.dir, detector=args.detector,
                   classifier=args.classifier)


if __name__ == '__main__':
    main()
