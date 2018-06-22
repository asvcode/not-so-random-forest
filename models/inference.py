#!/usr/bin/env python

import argparse
import os
import sys
import glob
import pandas as pd
import pickle

# import keras
from keras import backend

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from PIL import Image

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def extract_bounded_objects(unscaled_image, boxes, scores, score_threshold=0.5):
    objects = []
    for i, b in enumerate(boxes[0]):
        if scores[0][i] < score_threshold:
            continue
        b = np.array(b).astype(int)
        objects.append(unscaled_image[b[1]:b[3], b[0]:b[2], :])
    return objects

def tree_extractor(image_folder, detector='resnet50_csv_50_inference.h5', score_threshold=0.5):
    backend.tensorflow_backend.set_session(get_session())
    model = models.load_model('./saved_models/' + detector, backbone_name='resnet50')
    types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG')
    files = []
    for type in types:
        files.extend(glob.glob(image_folder + '/' + type))

    for file in files:
        print(file)

        prefix = file.split('.')[0]
        extension = file.split('.')[1]
        prefix1 = prefix.split('/')[0]
        prefix2 = prefix.split('/')[1]

        image = read_image_bgr(file)
        image = preprocess_image(image)
        image, scale = resize_image(image)

        unscaled_image = np.asarray(Image.open(file))
        unscaled_image[:, :, ::-1].copy()

        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("detection time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        tree_images = extract_bounded_objects(unscaled_image, boxes, scores, 0.5)
        valid_boxes = [box for i, box in enumerate(boxes[0]) if scores[0][i] > 0.5]

        with open(image_folder + '/' + prefix2 + '_inference.p', 'wb') as handle:
            pickle.dump({"tree_images": tree_images, "image": unscaled_image, "boxes": valid_boxes, "score_threshold": score_threshold}, handle)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running inference on images.')
    parser.add_argument('--dir', help='Folder containing images to run inference on.')
    parser.add_argument('--detector', help='Model for tree detection.', default='resnet50_csv_50_inference.h5')
    #parser.add_argument('--classifier', help ='Model for species classification.')
    return parser.parse_args(args)


def main(args=None):


    args = sys.argv[1:]
    args = parse_args(args)
    # extract trees
    tree_extractor(image_folder=args.dir, detector=args.detector)

if __name__ == '__main__':
    main()
