#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import cv2
import mediapipe as mp
from collections import deque
import logging

from config import *
from utils import align_face, crop_left_eye, preprocess_eye, eye_aspect_ratio
from models import predict_gaze, get_face_embedding
from tracking import update_embeddings

logging.basicConfig(level=logging.INFO)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5, refine_landmarks=True, 
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

pred_buffer = deque(maxlen=SMOOTHING_FRAMES)
blink_counter, eye_closed_frames = 0, 0
blinking = False

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        gaze_label, tracked_face_landmarks, tracking_active = "", None, False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    aligned = align_face(frame, face_landmarks.landmark)
                except Exception:
                    continue
                new_embedding = get_face_embedding(aligned)
                status = update_embeddings(new_embedding)

                if status in ["lock_init", "match", "update"]:
                    tracked_face_landmarks = face_landmarks.landmark
                    tracking_active = True
                    logging.info(f"[{status.upper()}] Face recognized")
                    break

        if tracking_active and tracked_face_landmarks:
            h, w, _ = frame.shape
            ear_left = eye_aspect_ratio(tracked_face_landmarks, LEFT_EYE_LMK, w, h)
            if ear_left < BLINK_THRESH:
                blink_counter += 1
                eye_closed_frames += 1
                if blink_counter >= BLINK_CONSEC_FRAMES: blinking = True
            else:
                blink_counter, eye_closed_frames, blinking = 0, 0, False

            eye_img, (x_min, y_min, x_max, y_max) = crop_left_eye(frame, tracked_face_landmarks)

            if eye_closed_frames >= LONG_CLOSE_FRAMES:
                gaze_label = "close"
            elif blinking:
                gaze_label = "blinking"
            else:
                input_data = preprocess_eye(eye_img)
                if input_data is not None:
                    output_data = predict_gaze(input_data)
                    gaze_idx = output_data.argmax()
                    pred_buffer.append((gaze_idx, output_data[0][gaze_idx]))
                    if len(pred_buffer) == SMOOTHING_FRAMES:
                        smoothed_idx = max(set([p[0] for p in pred_buffer]),
                                           key=[p[0] for p in pred_buffer].count)
                        gaze_label = CLASS_NAMES[smoothed_idx]
                    else:
                        gaze_label = CLASS_NAMES[gaze_idx]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, gaze_label, (32, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        cv2.imshow("Gaze Detection + Face Lock", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()

