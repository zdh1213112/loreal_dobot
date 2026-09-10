"""Geometry for a camera-down offset approach, in metres and User coordinates."""
import numpy as np


def plan_offset(target, camera_rotation, length, finger_span=0.060, clearance=0.020):
    values = np.array([length, finger_span, clearance])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Offset geometry dimensions must be finite and positive")
    # Project RGB camera +Y into the horizontal User plane.
    direction = np.asarray(camera_rotation, dtype=float)[:, 1].copy()
    direction[2] = 0
    norm = np.linalg.norm(direction)
    if norm < 0.2:
        raise ValueError("Image-down direction has no reliable table projection")
    direction /= norm
    distance = length / 2 + finger_span / 2 + clearance
    return np.asarray(target, dtype=float) + direction * distance
