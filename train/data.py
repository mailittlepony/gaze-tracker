#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

IMG_SIZE = (64, 64)


def create_data_generators(data_dir, batch_size=32, validation_split=0.2, seed=42):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=10,
        brightness_range=[0.7, 1.3],
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen


def compute_class_weights(train_gen, boost_left=False, boost_up=False):
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(weights))

    if boost_left and 1 in class_weight_dict:
        class_weight_dict[1] *= 1.5
    if boost_up and 4 in class_weight_dict:
        class_weight_dict[4] *= 1.5

    return class_weight_dict

