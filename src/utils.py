#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import cv2
import numpy as np
from .config import TEMPLATE_5PTS, OUT_SIZE, LMK_ID_5PTS, IMG_SIZE


def umeyama(src, dst, estimate_scale=True):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - mean_src, dst - mean_dst
    cov = (dst_demean.T @ src_demean) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = (U @ Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    if estimate_scale:
        var_src = (src_demean ** 2).sum() / src.shape[0]  
        scale = S.sum() / var_src
    else:
        scale = 1.0
    t = mean_dst.T - scale * (R @ mean_src.T)
    M = np.zeros((2, 3))
    M[:, :2] = scale * R
    M[:, 2] = t
    return M.astype(np.float32)


def get_facemesh_pts(landmarks, frame_w, frame_h):
    pts = [[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in LMK_ID_5PTS]
    return np.array(pts, dtype=np.float32)


def align_face(frame, landmarks):
    h, w, _ = frame.shape
    src5 = get_facemesh_pts(landmarks, w, h)
    M = umeyama(src5, np.array(TEMPLATE_5PTS, dtype=np.float32), True)
    return cv2.warpAffine(frame, M, OUT_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_left_eye(frame, landmarks):
    h, w, _ = frame.shape
    indices = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]

    x_coords = np.array([landmarks[i].x * w for i in indices])
    y_coords = np.array([landmarks[i].y * h for i in indices])

    left_corner = np.array([landmarks[362].x * w, landmarks[362].y * h])
    right_corner = np.array([landmarks[263].x * w, landmarks[263].y * h])
    cx, cy = np.mean(x_coords), np.mean(y_coords)
    angle = np.degrees(np.arctan2(right_corner[1] - left_corner[1], right_corner[0] - left_corner[0]))

    cos_a = np.cos(-np.radians(angle))
    sin_a = np.sin(-np.radians(angle))
    x_rot = cos_a * (x_coords - cx) - sin_a * (y_coords - cy) + cx
    y_rot = sin_a * (x_coords - cx) + cos_a * (y_coords - cy) + cy

    x_min, x_max = int(np.min(x_rot)), int(np.max(x_rot))
    y_min, y_max = int(np.min(y_rot)), int(np.max(y_rot))
    box_size = max(x_max - x_min, y_max - y_min)
    
    pad_top, pad_bottom, pad_side = int(0.5 * box_size), int(0.2 * box_size), int(0.3 * box_size)
    x_min_new, x_max_new = max(int(cx - box_size // 2 - pad_side), 0), min(int(cx + box_size // 2 + pad_side), w)
    y_min_new, y_max_new = max(int(cy - box_size // 2 - pad_top), 0), min(int(cy + box_size // 2 + pad_bottom), h)

    eye_img = frame[y_min_new:y_max_new, x_min_new:x_max_new]
    return eye_img, (x_min_new, y_min_new, x_max_new, y_max_new)


def preprocess_eye(eye_img):
    if eye_img is None or eye_img.size == 0:
        return None
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = cv2.resize(eye_img, IMG_SIZE).astype(np.float32) / 255.0
    return np.expand_dims(eye_img, axis=0)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

