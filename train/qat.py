#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, val_gen):
    val_gen.reset()
    y_pred_prob = model.predict(val_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

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


def apply_quant_aware_training(model, train_gen, val_gen, epochs=12, batch_size=32):
    def annotate_layer(layer):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) and layer.name != 'predictions':
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=annotate_layer
    )

    q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    q_aware_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),  
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    q_aware_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=max(1, train_gen.samples // batch_size),
        validation_steps=max(1, val_gen.samples // batch_size)
    )

    print("\nEvaluation after QAT fine-tuning:")
    evaluate_model(q_aware_model, val_gen)

    return q_aware_model


def convert_to_tflite(qat_model, val_gen, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i, (images, _) in enumerate(val_gen):
            if i >= 300:  
                break
            yield [images.astype(np.float32)]

    converter.representative_dataset = representative_dataset

    tflite_quant_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"Quantized TFLite model saved to: {output_path}")

