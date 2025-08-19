#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

IMG_SIZE = (64, 64)
CLASS_NAMES = ['down', 'left', 'right', 'straight', 'up']

# Models
TFLITE_MODEL_PATH = "models/gaze_model_qat_int8.tflite"
FACE_REC_MODEL_PATH = "models/FaceMobileNet_Float32.tflite"

# Eye state
SMOOTHING_FRAMES = 15
LEFT_EYE_LMK = [362, 385, 387, 263, 373, 380]
BLINK_THRESH = 0.22
BLINK_CONSEC_FRAMES = 2
LONG_CLOSE_FRAMES = 8

# Face lock
INITIAL_LOCK_FRAMES = 5
MAX_EMBEDDINGS = 15
MATCH_SIM_THRESHOLD = 0.55
UPDATE_SIM_THRESHOLD = 0.70

# Template (112x112)
TEMPLATE_5PTS = [
    [38.2946, 51.6963], 
    [73.5318, 51.5014], 
    [56.0252, 71.7366], 
    [41.5493, 92.3655], 
    [70.7299, 92.2041]
]
OUT_SIZE = (112, 112)

# Face Mesh landmarks
LMK_ID_5PTS = [33, 263, 1, 61, 291]

