#!/usr/bin/env python

#import miscellaneous modules
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
from keras import backend
from keras.preprocessing import image
from keras.models import load_model

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
def extract_bounded_objects(unscaled_image, boxes, scores, score_threshold=0.5):
    objects = []
    for i, b in enumerate(boxes[0]):
        if scores[0][i] < score_threshold:
            continue
        b = np.array(b).astype(int)
        objects.append(unscaled_image[b[1]:b[3], b[0]:b[2], :])
    return objects

def object_detector(file):

    score_threshold = 0.5
    prefix = file.split('.')[0]
    extension = file.split('.')[1]
    prefix1 = prefix.split('/')[0]
    prefix2 = prefix.split('/')[1]

    image = read_image_bgr(file)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    unscaled_image = np.asarray(Image.open(file))
    unscaled_image[:, :, ::-1].copy()
    boxes, scores, labels = detector_model.predict_on_batch(np.expand_dims(image, axis=0))
    print("detection time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    tree_images = extract_bounded_objects(unscaled_image, boxes, scores, score_threshold=0.5)
    valid_boxes = [box for i, box in enumerate(boxes[0]) if scores[0][i] > 0.5]

    return (tree_images, unscaled_image, valid_boxes, score_threshold)


def species_classifier(tree_images, classifier):
    classifier_model = load_model(filepath='inceptionv3_imagenet_unfrozen_20_epochs.h5')

    resized_tree_images = []
    for tree_image in tree_images:
        resized_tree_image.append(cv2.resize(tree_image, (224, 224)))


def tree_extractor(image_folder, detector='resnet50_csv_50_inference.h5',
 classifier='inceptionv3_imagenet_unfrozen_20_epochs.h5'):


    backend.tensorflow_backend.set_session(get_session())
    detector_model = models.load_model('./saved_models/' + detector, backbone_name='resnet50')

    files = glob.glob(image_folder + '/')

    for file in files:
        if not imghdr.what(file):
            continue
        print(file)

        # runnin object detector on file to detect trees
        (tree_images, unscaled_image, valid_boxes, score_threshold) = object_detector(file)

        #classifying detected trees
        species = species_classifier(tree_images)

        with open(image_folder + '/' + prefix2 + '_inference.p', 'wb') as handle:
            pickle.dump({"tree_images": tree_images, "species": species, "image": unscaled_image, "boxes": valid_boxes, "score_threshold": score_threshold}, handle)



def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running inference on images.')
    parser.add_argument('--dir', help='Folder containing images to run inference on.')
    parser.add_argument('--detector', help='Model for tree detection.', default='resnet50_csv_50_inference.h5')
    parser.add_argument('--classifier', help ='Model for species classification.', default='inceptionv3_imagenet_unfrozen_20_epochs.h5')
    return parser.parse_args(args)


def main(args=None):


    args = sys.argv[1:]
    args = parse_args(args)
    # extract trees
    tree_extractor(image_folder=args.dir, detector=args.detector, classifier=arg.classifier)

if __name__ == '__main__':
    main()
