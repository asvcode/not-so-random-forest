#!/usr/bin/env python

# import keras modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

# import miscellaneous modules
import sys
import pickle
import argparse

# set constants
IMAGE_SHAPE = (224, 224, 3)


def custom_cnn():
    """constructs a custom CNN

    Args:
        None
    Returns:
        model : custom CNN

    """
    # set constants
    FILTER_SHAPE = (3, 3)
    POOL_SIZE = (2, 2)

    # add layers
    model = Sequential()
    model.add(Conv2D(32, FILTER_SHAPE, input_shape=IMAGE_SHAPE,
                     name='conv2d_1'))
    model.add(Activation('relu', name='activation_1'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE, name='max_pooling2d_1'))

    model.add(Conv2D(32, FILTER_SHAPE, name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE, name='max_pooling2d_2'))

    model.add(Conv2D(64, FILTER_SHAPE, name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(MaxPooling2D(pool_size=POOL_SIZE, name='max_pooling2d_3'))

    return model


def attach_top(base_model, n_classes):
    """add classification layers on top of base_model

    Args:
        base_model : base keras model
        n_classes : number of classes
    Returns:
        output : full model

    """
    base_model.add(Flatten())
    base_model.add(Dense(64))
    base_model.add(Activation('relu', name='activation_t1'))
    base_model.add(Dropout(0.5))
    base_model.add(Dense(1))
    base_model.add(Activation('sigmoid', name='activation_t2'))
    # base_model.add(Dense(n_classes))
    # base_model.add(Activation('softmax', name='activation_t2'))

    return base_model


def train_classifier(image_dir, classifier, save_dir, epochs, batch_size,
                     freeze_backbone=False, suffix=''):
    """train species classifier on image data

    Args:
        image_dir : image directory
        classifier : string denoting species classifier model
        freeze_backbone : boolean for freezing backbone of classifier
        suffix : desired prefix to the model name
        epochs : number of epochs
        batch_size : batch size
    Returns:
        None

    """
    # data generators
    image_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True)
    train_generator = image_generator.flow_from_directory(
        image_dir + '/train/', batch_size=batch_size,
        target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        shuffle=True, class_mode='binary')
    validation_generator = image_generator.flow_from_directory(
        image_dir + '/validate/', batch_size=batch_size,
        target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        shuffle=True, class_mode='binary')

    # instantiate model
    if classifier == 'InceptionV3':
        base_model = InceptionV3(weights="imagenet", include_top=False,
                                 input_shape=IMAGE_SHAPE)
    elif classifier == 'ResNet50':
        base_model = ResNet50(weights="imagenet", include_top=False,
                              input_shape=IMAGE_SHAPE)
    elif classifier == 'Custom':
        base_model = custom_cnn()

    # freeze backbone layers in base model
    if freeze_backbone:
        print('All base layers have been frozen.')
        for i, layer in enumerate(base_model.layers):
            layer.trainable = False
    else:
        print('Warning: All base layers have been unfrozen.')
        for i, layer in enumerate(base_model.layers):
            layer.trainable = True

    # attach custom classification layers on top of base model
    model = attach_top(base_model, len(train_generator.class_indices.keys()))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer='rmsprop')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames)//batch_size,
        epochs=epochs, validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames)//batch_size)

    # save model, history and class indices to file
    model_name = classifier + '_species_classifier'
    if suffix:
        model_name = model_name + '_' + suffix
    filepath = save_dir + model_name + '.h5'
    model.save(filepath)
    #pickle.dump(history.history, open(
     #save_dir + model_name +
     #'_history.p', 'wb'))
    #pickle.dump(train_generator.class_indices, open(
        #save_dir + model_name +
        #'_class_indices.p', 'wb'))
    print('Model saved to ' + filepath)


# argument parser
def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Script for training species classifier.')
    parser.add_argument(
        '--image_dir',
        help='Folder containing training and validation images.',
        default='../data/classification/')
    parser.add_argument(
        '--save_dir', help='Folder to save trained model.',
        default='saved_models/species_classifiers/')
    parser.add_argument(
        '--classifier', help='Model for species classification.',
        default='ResNet50')
    parser.add_argument(
        '--freeze_backbone', help='Switch to freeze layers in base model.',
        action='store_true')
    parser.add_argument(
        '--suffix',
        help='Suffix to add to name of the file that will be saved.',
        default='')
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train.', type=int, default=30)
    parser.add_argument(
        '--batch_size',
        help='Batch size.', type=int, default=5)
    return parser.parse_args(args)


def main(args=None):
    args = sys.argv[1:]
    args = parse_args(args)

    # pass arguments to classifier
    train_classifier(image_dir=args.image_dir, classifier=args.classifier,
                     save_dir=args.save_dir, epochs=args.epochs,
                     batch_size=args.batch_size,
                     freeze_backbone=args.freeze_backbone, suffix=args.suffix)


if __name__ == '__main__':
    main()
