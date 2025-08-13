#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow_addons as tfa 


def get_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen


def train_model(model, train_gen, val_gen, class_weights=None, epochs=30, save_path='model.keras'):
    if not any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers):
        x = model.output
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(len(train_gen.class_indices), activation='softmax')(x)
        model = tf.keras.Model(inputs=model.input, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    val_gen.reset()
    y_pred_prob = model.predict(val_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.classes
    class_labels = list(train_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix - Gaze Direction")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title("Training History")
    plt.show()

    return history


