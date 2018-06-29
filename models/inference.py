#!/usr/bin/env python

# import miscellaneous modules
import argparse
import os
import sys
import glob
import pandas as pd
import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import imghdr

# import keras
import keras
from keras import backend
from keras.preprocessing import image
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# function to extract bounded objects given image and bounding boxes
def extract_bounded_objects(unscaled_image, boxes, scores, score_threshold):
    objects = []
    for i, b in enumerate(boxes[0]):
        if scores[0][i] < score_threshold:
            continue
        b = np.array(b).astype(int)
        objects.append(unscaled_image[b[1]:b[3], b[0]:b[2], :])
    return objects

# function to predict bounding boxes on objects in an image
def object_detector(file, detector_model):

    score_threshold = 0.5


    image = read_image_bgr(file)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    unscaled_image = np.asarray(Image.open(file))

    # convert grayscale image to a 3-channel image
    if np.ndim(unscaled_image) == 2 or unscaled_image.shape[2] == 1:
        unscaled_image = np.repeat(unscaled_image[:, :, np.newaxis], 3, axis=2)

    # drop alpha channel
    unscaled_image = unscaled_image[:, :, :3]


    unscaled_image[:, :, ::-1].copy()
    start = time.time()
    boxes, scores, labels = detector_model.predict_on_batch(np.expand_dims(image, axis=0))
    print("detection time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    print(unscaled_image.shape)
    tree_images = extract_bounded_objects(unscaled_image, boxes, scores, score_threshold=score_threshold)
    valid_boxes = [box for i, box in enumerate(boxes[0]) if scores[0][i] > score_threshold]

    return (tree_images, unscaled_image, valid_boxes, score_threshold)


# function to predict species given tree images
def species_predictor(tree_images, classifier_model):


    resized_tree_images = []
    for tree_image in tree_images:
        resized_tree_image = cv2.resize(tree_image, (224, 224))
        img = image.img_to_array(resized_tree_image)
        img = np.expand_dims(img, axis=0)
        resized_tree_images.append(img)
    if resized_tree_images:
        resized_tree_images = np.vstack(resized_tree_images)

        start = time.time()
        species_probabilities = classifier_model.predict(resized_tree_images)
        print("classification time: ", time.time() - start)
    else:
        species_probabilities = []

    return(species_probabilities)

# function to direct calls to detector and classifier models and write inference results to disk
def tree_extractor(image_folder, detector='resnet50_csv_50_epochs_inference',
 classifier='date_palm_species_classifier_ResNet50_imagenet_1_epochs_unfrozen'):


    backend.tensorflow_backend.set_session(get_session())
    detector_model = models.load_model('./saved_models/tree_detectors/' + detector + '.h5', backbone_name='resnet50')

    # custom fix for weird error during inferncing with MobileNet
    #with CustomObjectScope({'relu6': keras.applications.mobilenet.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.mobilenet.layers.DepthwiseConv2D}):
        #classifier_model = load_model(filepath='./saved_models/species_classifiers/' + classifier+'.h5')

    classifier_model = load_model(filepath='./saved_models/species_classifiers/' + classifier+'.h5')
    classifier_history = pickle.load(open('./saved_models/species_classifiers/' + classifier + '_history.p', 'rb'))
    class_indices = pickle.load(open('./saved_models/species_classifiers/' + classifier + '_class_indices.p', 'rb'))

    files = glob.glob(image_folder + '/*')

    for file in files:
        if not imghdr.what(file):
            continue
        print(file)

        # running object detector on image to detect trees
        (tree_images, unscaled_image, valid_boxes, score_threshold) = object_detector(file, detector_model)

        # classifying detected trees
        species_probabilities = species_predictor(tree_images, classifier_model)

        prefix = file.split('.')[0]
        prefix1 = prefix.split('/')[0]
        prefix2 = prefix.split('/')[1]

        with open(image_folder + '/' + prefix2 + '_inference.p', 'wb') as handle:
            pickle.dump({"tree_images": tree_images, "species": class_indices, "species_probabilities": species_probabilities, "image": unscaled_image, "boxes": valid_boxes, "score_threshold": score_threshold}, handle)



def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running inference on images.')
    parser.add_argument('--dir', help='Folder containing images to run inference on.')
    parser.add_argument('--detector', help='Model for tree detection.', default='resnet50_csv_50_epochs_inference')
    parser.add_argument('--classifier', help ='Model for species classification.', default='date_palm_species_classifier_ResNet50_imagenet_1_epochs_unfrozen')
    return parser.parse_args(args)


def main(args=None):

    args = sys.argv[1:]
    args = parse_args(args)

    # extract trees
    tree_extractor(image_folder=args.dir, detector=args.detector, classifier=args.classifier)

if __name__ == '__main__':
    main()
