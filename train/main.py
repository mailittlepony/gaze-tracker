#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import argparse
import os
from data import create_data_generators, compute_class_weights
from model import build_functional_model
from train import train_model
from qat import apply_quant_aware_training, convert_to_tflite
from tensorflow.keras.models import load_model


def main(args):
    train_gen, val_gen = create_data_generators(args.data_dir, args.batch)
    class_weights = compute_class_weights(train_gen)

    model = build_functional_model(num_classes=len(train_gen.class_indices))

    train_model(model, train_gen, val_gen, class_weights, args.epochs, args.save_path)

    if args.qat:
        print("Starting Quantization Aware Training (QAT)...")
        model = load_model(args.save_path, compile=False)
        q_aware_model = apply_quant_aware_training(model, train_gen, val_gen, epochs=5, batch_size=args.batch)
        convert_to_tflite(q_aware_model, val_gen, args.qat_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gaze direction classification model with optional QAT.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset folder with subfolders for each class")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save-path", type=str, default="models/gaze_model.keras", help="Path to save the trained model")
    parser.add_argument("--qat", action="store_true", help="Enable Quantization Aware Training (QAT)")
    parser.add_argument("--qat-output", type=str, default="models/gaze_model_qat_int8.tflite", help="Output path for quantized TFLite model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    main(args)


