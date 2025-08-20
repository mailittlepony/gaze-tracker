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

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state = tracker.get_eye_state(frame)
        print("Eye state:", state)

        frame = tracker.draw_preview(frame, state)
        cv2.imshow("Gaze Preview", frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()


