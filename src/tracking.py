#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.local>
#
# Distributed under terms of the MIT license.

import numpy as np
from .config import INITIAL_LOCK_FRAMES, MAX_EMBEDDINGS, MATCH_SIM_THRESHOLD, UPDATE_SIM_THRESHOLD

locked_face_embeddings = []


def cosine_similarity(e1, e2):
    return float(np.dot(e1, e2))


def update_embeddings(new_embedding):
    global locked_face_embeddings
    if len(locked_face_embeddings) < INITIAL_LOCK_FRAMES:
        locked_face_embeddings.append(new_embedding)
        return "lock_init"

    sims = [cosine_similarity(e, new_embedding) for e in locked_face_embeddings]
    max_sim = max(sims) if sims else -1.0

    if max_sim >= MATCH_SIM_THRESHOLD:
        if max_sim >= UPDATE_SIM_THRESHOLD and len(locked_face_embeddings) < MAX_EMBEDDINGS:
            if all(cosine_similarity(new_embedding, e) < 0.995 for e in locked_face_embeddings):
                locked_face_embeddings.append(new_embedding)
                return "update"
        return "match"
    return "reject"

