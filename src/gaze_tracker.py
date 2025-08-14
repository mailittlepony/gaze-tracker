#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

TFLITE_MODEL_PATH = "gaze_model_qat_int8.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (64, 64)
CLASS_NAMES = ['down', 'left', 'right', 'straight', 'up']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SMOOTHING_FRAMES = 10
pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
BLINK_THRESH = 0.22
BLINK_CONSEC_FRAMES = 2
blink_counter = 0
blinking = False

def crop_left_eye(frame, landmarks):
    h, w, _ = frame.shape
    left_eye_indices = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]
    x_coords = [int(landmarks[i].x * w) for i in left_eye_indices]
    y_coords = [int(landmarks[i].y * h) for i in left_eye_indices]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    width = x_max - x_min
    height = y_max - y_min
    box_size = max(width, height)

    pad_top = int(0.5 * box_size)
    pad_bottom = int(0.2 * box_size)
    pad_side = int(0.3 * box_size)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    x_min_new = max(cx - box_size // 2 - pad_side, 0)
    x_max_new = min(cx + box_size // 2 + pad_side, w)
    y_min_new = max(cy - box_size // 2 - pad_top, 0)
    y_max_new = min(cy + box_size // 2 + pad_bottom, h)

    eye_img = frame[y_min_new:y_max_new, x_min_new:x_max_new]
    return eye_img, (x_min_new, y_min_new, x_max_new, y_max_new)

def preprocess_eye(eye_img):
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = cv2.resize(eye_img, IMG_SIZE)
    eye_img = eye_img.astype(np.float32) / 255.0
    return np.expand_dims(eye_img, axis=0)

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gaze_label = ""  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        ear_left = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
        if ear_left < BLINK_THRESH:
            blink_counter += 1
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blinking = True
        else:
            blink_counter = 0
            blinking = False

        eye_img, (x_min, y_min, x_max, y_max) = crop_left_eye(frame, landmarks)

        if blinking:
            gaze_label = "blinking"
        else:
            input_data = preprocess_eye(eye_img)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            gaze_idx = np.argmax(output_data)
            confidence = output_data[0][gaze_idx]

            pred_buffer.append((gaze_idx, confidence))

            if len(pred_buffer) == SMOOTHING_FRAMES:
                smoothed_idx = max(set([p[0] for p in pred_buffer]),
                                   key=[p[0] for p in pred_buffer].count)
                gaze_label = CLASS_NAMES[smoothed_idx]
            else:
                gaze_label = CLASS_NAMES[gaze_idx]  

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label_text = gaze_label
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (30, 30 - th - 5), (30 + tw + 5, 30 + 5), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (32, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Gaze Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()

