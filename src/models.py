#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import cv2
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

from .config import TFLITE_MODEL_PATH, FACE_REC_MODEL_PATH


gaze_interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
gaze_interpreter.allocate_tensors()
gaze_input, gaze_output = gaze_interpreter.get_input_details(), gaze_interpreter.get_output_details()

face_interpreter = Interpreter(model_path=FACE_REC_MODEL_PATH)
face_interpreter.allocate_tensors()
face_input, face_output = face_interpreter.get_input_details(), face_interpreter.get_output_details()


def predict_gaze(input_data):
    gaze_interpreter.set_tensor(gaze_input[0]['index'], input_data)
    gaze_interpreter.invoke()
    return gaze_interpreter.get_tensor(gaze_output[0]['index'])


def get_face_embedding(aligned_bgr):
    img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) * 0.0078125
    img = np.expand_dims(img, axis=0)
    face_interpreter.set_tensor(face_input[0]['index'], img)
    face_interpreter.invoke()
    emb = face_interpreter.get_tensor(face_output[0]['index'])[0]
    return emb / (np.linalg.norm(emb) + 1e-10)

