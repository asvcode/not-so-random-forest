#!/usr/bin/env python

import argparse
import os
import sys
import glob
import pandas as pd

# import keras
import keras


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

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def tree_extractor():
    keras.backend.tensorflow_backend.set_session(get_session())
    model = models.load_model(model_path, backbone_name='resnet50')

    types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', )

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running inference on images.')
    parser.add_argument('folder_in', help='Folder containing images to run inference on.')
    parser.add_argument('--detector', help='Model for tree detection.')
    parser.add_argument('--classifier', help ='Model for species classification.')

    return parser.parse_args(args)


def main(args=None):
    # set the modified tf session as backend in keras


    # parse arguments

    if args is None:
        args = sys.argv[:1]
    args = parse_args(args)
