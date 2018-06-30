#!/usr/bin/env python

# import keras modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.optimizers import Adam

# import miscellaneous modules
import sys
import pickle
import argparse


def custom_cnn():
    """constructs a custom CNN

    Args:
        None
    Returns:
        model : CNN with n_layers

    """
    # set constants
    IMAGE_SHAPE = (224, 224, 3)
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
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(activation="softmax", name="y")(x)

    return output


def train_classifier(image_dir, classifier, freeze_backbone=False, suffix='',
                     epochs=20):
    """train species classifier on image data

    Args:
        image_dir : image directory
        classifier : string denoting species classifier model
        freeze_backbone : boolean for freezing backbone of classifier
        suffix : desired prefix to the model name
        epochs : number of epochs

    Returns:
        None

    """
    # data generators
    image_generator = ImageDataGenerator(rescale=1./255)
    batch_size = 20

    train_generator = image_generator.flow_from_directory(
        image_dir + '/train/', batch_size=batch_size,
        target_size=(224, 224), class_mode='categorical')
    validation_generator = image_generator.flow_from_directory(
        image_dir + '/validate/', batch_size=batch_size,
        target_size=(224, 224), class_mode='categorical')

    # instantiate model
    if classifier == 'InceptionV3':
        base_model = InceptionV3(weights="imagenet", include_top=False,
                                 input_shape=(224, 224, 3))
    elif classifier == 'ResNet50':
        base_model = ResNet50(weights="imagenet", include_top=False,
                              input_shape=(224, 224, 3))
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
    output = attach_top(base_model, len(train_generator.class_indices.keys()))

    model = Model(inputs=base_model.input, outputs=output)

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=Adam(lr=0.0001, clipvalue=0.5))

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames)//batch_size,
        epochs=epochs, validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames)//batch_size)

    # save model, history and class indices to file
    model_name = classifier + '_' + str(epochs) + 'epochs'
    if suffix:
        model_name = model_name + '_' + suffix
    filepath = 'saved_models/species_classifiers/' + model_name + '.h5'
    model.save(filepath + '.h5')
    pickle.dump(history.history, open(
     'saved_models/species_classifiers/' + model_name +
     '_history.p', 'wb'))
    pickle.dump(train_generator.class_indices, open(
        'saved_models/species_classifiers/' + model_name +
        '_class_indices.p', 'wb'))


# argument parser
def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Script for training species classifier.')
    parser.add_argument(
        '--dir', help='Folder containing training and validation images.',
        default='../data/classification/')
    parser.add_argument(
        '--classifier', help='Model for species classification.',
        default='InceptionV3')
    parser.add_argument(
        '--freeze_backbone', help='Switch to freeze layers in base model',
        action='store_true')
    parser.add_argument(
        '--suffix',
        help='suffix to add to name of the file that will be saved',
        default='')
    parser.add_argument('--epochs', default='20')
    return parser.parse_args(args)


def main(args=None):
    args = sys.argv[1:]
    args = parse_args(args)

    # pass arguments to classifier
    train_classifier(image_dir=args.dir, classifier=args.classifier,
                     freeze_backbone=args.freeze_backbone,
                     suffix=args.suffix, epochs=int(args.epochs))


if __name__ == '__main__':
    main()
