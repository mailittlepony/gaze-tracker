#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import cv2
import mediapipe as mp
from collections import deque

from gaze_tracker.src.utils import align_face, crop_left_eye, preprocess_eye, eye_aspect_ratio
from gaze_tracker.src.models import predict_gaze, get_face_embedding, load_models
from gaze_tracker.src.tracking import update_embeddings
from gaze_tracker.src.config import *


class GazeTracker:
    def __init__(self, smoothing=SMOOTHING_FRAMES, enable_tracking=False, model_dir="models"):
        self.enable_tracking = enable_tracking

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pred_buffer = deque(maxlen=smoothing)
        self.blink_counter = 0
        self.eye_closed_frames = 0
        self.blinking = False

        self.tracked_face_landmarks = None
        self.tracking_active = False
        load_models(model_dir)


    def get_eye_state(self, frame):
        """Returns current eye state: 'down', 'left', 'right', 'straight', 'up', 'blinking', 'closed', 'no_face'."""
        gaze_label = ""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self.tracked_face_landmarks = None
            self.tracking_active = False
            self.blink_counter = 0
            self.eye_closed_frames = 0
            self.blinking = False
            self.pred_buffer.clear()
            return "no_face"

        if self.enable_tracking:
            matched = False
            for face_landmarks in results.multi_face_landmarks:
                try:
                    aligned = align_face(frame, face_landmarks.landmark)
                    new_embedding = get_face_embedding(aligned)
                    status = update_embeddings(new_embedding)
                    if status in ["lock_init", "match", "update"]:
                        self.tracked_face_landmarks = face_landmarks.landmark
                        self.tracking_active = True
                        matched = True
                        break
                except Exception:
                    continue

            if not matched:
                self.tracked_face_landmarks = None
                self.tracking_active = False
        else:
            self.tracked_face_landmarks = results.multi_face_landmarks[0].landmark
            self.tracking_active = True

        if self.tracking_active and self.tracked_face_landmarks:
            h, w, _ = frame.shape
            ear_left = eye_aspect_ratio(self.tracked_face_landmarks, LEFT_EYE_LMK, w, h)
            if ear_left < BLINK_THRESH:
                self.blink_counter += 1
                self.eye_closed_frames += 1
                if self.blink_counter >= BLINK_CONSEC_FRAMES:
                    self.blinking = True
            else:
                self.blink_counter = 0
                self.eye_closed_frames = 0
                self.blinking = False

            eye_img, _ = crop_left_eye(frame, self.tracked_face_landmarks)

            if self.eye_closed_frames >= LONG_CLOSE_FRAMES:
                gaze_label = "closed"
            elif self.blinking:
                gaze_label = "blinking"
            else:
                input_data = preprocess_eye(eye_img)
                if input_data is not None:
                    output_data = predict_gaze(input_data)
                    gaze_idx = output_data.argmax()
                    self.pred_buffer.append((gaze_idx, output_data[0][gaze_idx]))
                    if len(self.pred_buffer) == self.pred_buffer.maxlen:
                        smoothed_idx = max(set([p[0] for p in self.pred_buffer]),
                                           key=[p[0] for p in self.pred_buffer].count)
                        gaze_label = CLASS_NAMES[smoothed_idx]
                    else:
                        gaze_label = CLASS_NAMES[gaze_idx]

        return gaze_label


    def draw_bbox(self, frame, gaze_label):
        """Draws bounding box and gaze label for debugging."""
        if self.tracking_active and self.tracked_face_landmarks:
            _, (x_min, y_min, x_max, y_max) = crop_left_eye(frame, self.tracked_face_landmarks)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, gaze_label, (32, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        return frame


    def get_landmarks(self):
        """Return landmarks from the last processed frame.
        Call get_eye_state(frame) or process_frame(frame) first."""
        return self.tracked_face_landmarks


    def select_landmarks(self, indices):
        """
        Returns only the selected landmarks from the last tracked face.
        indices: list of integers
        """
        if self.tracked_face_landmarks is None:
            return None
        return [self.tracked_face_landmarks[i] for i in indices if 0 <= i < len(self.tracked_face_landmarks)]

