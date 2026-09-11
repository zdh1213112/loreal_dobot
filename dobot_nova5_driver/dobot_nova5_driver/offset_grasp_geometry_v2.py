"""Geometry for a camera-down offset approach, in metres and User coordinates."""
import numpy as np


def camera_rotation_at_target_tcp(
    current_tcp_rotation,
    current_camera_rotation,
    target_tcp_rotation,
):
    """Predict camera attitude after rotating the TCP to its target attitude."""

    current_tcp = np.asarray(current_tcp_rotation, dtype=float)
    current_camera = np.asarray(current_camera_rotation, dtype=float)
    target_tcp = np.asarray(target_tcp_rotation, dtype=float)
    if any(rotation.shape != (3, 3) for rotation in (
        current_tcp, current_camera, target_tcp
    )):
        raise ValueError("TCP and camera rotations must be 3x3 matrices")
    if not all(np.all(np.isfinite(rotation)) for rotation in (
        current_tcp, current_camera, target_tcp
    )):
        raise ValueError("TCP and camera rotations must be finite")

    # R_tcp_to_camera is fixed by the mounted tool/camera geometry. Applying
    # the target TCP attitude predicts the camera attitude without physically
    # stopping for a separate orientation-only motion first.
    tcp_to_camera = current_tcp.T @ current_camera
    return target_tcp @ tcp_to_camera


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
