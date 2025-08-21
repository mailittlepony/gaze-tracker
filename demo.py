#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import sys
import os
sys.path.insert(0, os.path.abspath(".."))
# Do not put this in your script if outside the repository

import logging
import cv2
from gaze_tracker import GazeTracker

logging.basicConfig(level=logging.INFO)

tracker = GazeTracker(enable_tracking=True, model_dir="models")
# model_dir="gaze_tracker/models" in your script

# Example mouth indices from Mediapipe face mesh
MOUTH_LMK = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13
]

# Example of drawing function but draw any visuals you want
def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=2, thickness=-1):
    """Draws given landmarks on a frame."""
    if landmarks:
        h, w, _ = frame.shape
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coords to pixels
            cv2.circle(frame, (x, y), radius, color, thickness)
    return frame

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state = tracker.get_eye_state(frame)
        print("Eye state:", state)

        # Draw any selected landmark or draw all landmarks with get_landmarks
        mouth_landmarks = tracker.select_landmarks(MOUTH_LMK)
        frame = draw_landmarks(frame, mouth_landmarks, color=(0, 0, 255), radius=2)

        frame = tracker.draw_bbox(frame, state)

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()


