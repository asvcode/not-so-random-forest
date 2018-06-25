from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy as np
import os
import keras
import glob
from keras.utils import np_utils
from keras.layers import Input, Dense, concatenate, Flatten, Dropout, multiply, BatchNormalization, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import TruncatedNormal, he_normal
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from random import shuffle

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
