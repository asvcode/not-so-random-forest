#!/usr/bin/env python

# import keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.layers import Input, Dense, concatenate, Flatten, Dropout, multiply, BatchNormalization, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import TruncatedNormal, he_normal
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

# import miscellaneous modules
import sys
import numpy as np
import glob
import pickle
import argparse
import os
import cv2
import time
from PIL import Image


def train_classifier(image_folder, classifier, freeze_backbone=False, save_prefix='None', epochs='20'):

    # data generators
    image_generator = ImageDataGenerator(rescale=1./255)
    batch_size = 5
    epochs = int(epochs)

    train_generator = image_generator.flow_from_directory(image_folder + '/train/',
                                                           batch_size=batch_size, target_size=(224, 224), class_mode='categorical')

    validation_generator = image_generator.flow_from_directory(image_folder + '/validate',
                                                           batch_size=batch_size, target_size=(224, 224), class_mode='categorical')

    # instantiate model
    if classifier == 'InceptionV3':
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif classifier == 'VGG16':
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif classifier == 'VGG19':
        base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif classifier == 'ResNet50':
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif classifier == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif classifier == 'MobileNet':
        base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif classifier == 'Simple':
        base_model = Sequential()
        base_model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
        base_model.add(Activation('relu'))
        base_model.add(MaxPooling2D(pool_size=(2, 2)))

        base_model.add(Conv2D(32, (3, 3)))
        base_model.add(Activation('relu'))
        base_model.add(MaxPooling2D(pool_size=(2, 2)))

        base_model.add(Conv2D(64, (3, 3)))
        base_model.add(Activation('relu'))
        base_model.add(MaxPooling2D(pool_size=(2, 2)))


    # freeze backbone layers in base model
    if freeze_backbone:
        print('All base layers have been frozen.')
        for i, layer in enumerate(base_model.layers):
            layer.trainable = False
    else:
        print('Warning: All base layers have been unfrozen.')
        for i, layer in enumerate(base_model.layers):
            layer.trainable = True

    # attach custom layer on top of base model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(len(train_generator.class_indices.keys()), activation="softmax", name="y")(x)
    model = Model(inputs=base_model.input, outputs=output)
    adam = Adam(lr=0.0001)
    sgd = SGD(lr=0.0001)
    #print(model.summary())

    # compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # fit model
    if freeze_backbone:
        model_name = 'species_classifier_' + classifier + '_imagenet_' + str(epochs) + '_epochs_frozen'
    else:
        model_name = 'species_classifier_' + classifier + '_imagenet_' + str(epochs) + '_epochs_unfrozen'

    if save_prefix != 'None':
        model_name = save_prefix + '_' + model_name

    filepath = 'saved_models/species_classifiers/' + model_name + '.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit_generator(train_generator, \
    samples_per_epoch=len(train_generator.filenames)//batch_size, \
    epochs=epochs,validation_data=validation_generator, \
    validation_steps=len(validation_generator.filenames)//batch_size, callbacks=callbacks_list, verbose=0)



    #model.save('saved_models/species_classifiers/' + model_name + '.h5')

    pickle.dump(history.history, open('saved_models/species_classifiers/' + model_name + '_history.p', 'wb'))

    pickle.dump(train_generator.class_indices, open('saved_models/species_classifiers/' + model_name + '_class_indices.p', 'wb'))

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for training species classifier.')
    parser.add_argument('--dir', help='Folder containing training and validation images.', default='../data/classification/')
    parser.add_argument('--classifier', help ='Model for species classification.', default='InceptionV3')
    parser.add_argument('--freeze_backbone', help='Switch to freeze layers in base model', action='store_true')
    parser.add_argument('--save_prefix', help= 'Prefix to add to name of the file that will be saved', default='None')
    parser.add_argument('--epochs', default='20')
    return parser.parse_args(args)


def main(args=None):

    args = sys.argv[1:]
    args = parse_args(args)

    # classify species
    train_classifier(image_folder=args.dir, classifier=args.classifier, freeze_backbone=args.freeze_backbone, save_prefix=args.save_prefix, epochs=args.epochs)

if __name__ == '__main__':
    main()
