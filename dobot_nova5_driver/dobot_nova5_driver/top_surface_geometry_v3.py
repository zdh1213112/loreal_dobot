"""Pure geometry helpers for selecting and orienting a box top surface."""

from __future__ import annotations

import cv2
import numpy as np


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    x_axis = rotation[:, 0] / (np.linalg.norm(rotation[:, 0]) + 1e-12)
    y_axis = rotation[:, 1] - np.dot(rotation[:, 1], x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def align_axis_sign_to_reference(
    rotation: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    """Remove the physically equivalent 180-degree in-plane sign flip."""

    aligned = orthonormalize(rotation)
    if reference is None:
        return aligned
    reference_axes = orthonormalize(reference)
    if float(np.dot(aligned[:, 0], reference_axes[:, 0])) < 0.0:
        aligned[:, 0] = -aligned[:, 0]
        aligned[:, 1] = -aligned[:, 1]
    return aligned


def normalized_plane(plane: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a unit plane normal and consistently scaled offset."""

    raw = np.asarray(plane[:3], dtype=np.float64)
    norm = float(np.linalg.norm(raw)) + 1e-12
    return raw / norm, float(plane[3]) / norm


def select_top_plane_candidate(
    plane_candidates,
    table_normal: np.ndarray | None,
    *,
    table_normal_alignment: float = 0.90,
    minimum_camera_facing_cos: float = 0.55,
):
    """Choose the physical top instead of the largest visible object face."""

    if table_normal is not None:
        aligned = []
        for count, plane in plane_candidates:
            candidate_normal, _ = normalized_plane(plane)
            alignment = abs(float(np.dot(candidate_normal, table_normal)))
            if alignment >= float(table_normal_alignment):
                aligned.append((count, plane, alignment))
        if not aligned:
            return None
        count, plane, _ = max(aligned, key=lambda item: item[0])
        return count, plane, "table-aligned-object-plane"

    facing = []
    for count, plane in plane_candidates:
        candidate_normal, _ = normalized_plane(plane)
        camera_facing = abs(float(candidate_normal[2]))
        if camera_facing >= float(minimum_camera_facing_cos):
            facing.append((camera_facing, count, plane))
    if not facing:
        return None
    _, count, plane = max(facing, key=lambda item: (item[0], item[1]))
    return count, plane, "camera-facing-object-plane"


def fit_top_surface_axes(
    top_surface_points: np.ndarray,
    normal: np.ndarray,
    reference: np.ndarray | None,
    *,
    minimum_points: int = 20,
    near_square_axis_ratio: float = 1.25,
) -> np.ndarray | None:
    """Fit physical box-edge axes from metric 3-D top-plane points only.

    A rectangle edge is an undirected line, so ``x_axis`` and ``-x_axis`` are
    geometrically equivalent.  Keep the legacy camera-frame convention on the
    first valid frame (X axis points toward non-positive camera Y), then use the
    previous frame as the sign reference.  Without that first-frame convention
    an otherwise identical grasp can command an unnecessary 180-degree wrist
    rotation.
    """

    points = np.asarray(top_surface_points, dtype=np.float64)
    if len(points) < int(minimum_points):
        return None
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_u = seed - float(np.dot(seed, normal)) * normal
    if float(np.linalg.norm(basis_u)) < 0.2:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis_u = seed - float(np.dot(seed, normal)) * normal
    basis_u /= np.linalg.norm(basis_u) + 1e-12
    basis_v = np.cross(normal, basis_u)
    basis_v /= np.linalg.norm(basis_v) + 1e-12

    center = np.median(points, axis=0)
    centered = points - center
    local_points = np.column_stack((centered @ basis_u, centered @ basis_v)).astype(
        np.float32
    )
    rectangle = cv2.boxPoints(cv2.minAreaRect(local_points))
    edge_a_2d = rectangle[1] - rectangle[0]
    edge_b_2d = rectangle[2] - rectangle[1]
    length_a = float(np.linalg.norm(edge_a_2d))
    length_b = float(np.linalg.norm(edge_b_2d))
    if min(length_a, length_b) < 1e-4:
        return None

    axis_a = basis_u * float(edge_a_2d[0]) + basis_v * float(edge_a_2d[1])
    axis_b = basis_u * float(edge_b_2d[0]) + basis_v * float(edge_b_2d[1])
    axis_a /= np.linalg.norm(axis_a) + 1e-12
    axis_b /= np.linalg.norm(axis_b) + 1e-12
    if length_a >= length_b:
        long_axis, short_axis = axis_a, axis_b
    else:
        long_axis, short_axis = axis_b, axis_a

    x_axis = long_axis
    aspect_ratio = max(length_a, length_b) / max(1e-9, min(length_a, length_b))
    if reference is not None and aspect_ratio <= float(near_square_axis_ratio):
        reference_axes = orthonormalize(reference)
        # minAreaRect can exchange two near-equal edge labels by 90 degrees.
        # Preserve edge identity so filtering never averages orthogonal frames
        # into a diagonal grasp direction.
        if abs(float(np.dot(short_axis, reference_axes[:, 0]))) > abs(
            float(np.dot(long_axis, reference_axes[:, 0]))
        ):
            x_axis = short_axis
    x_axis -= float(np.dot(x_axis, normal)) * normal
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    if reference is None and float(x_axis[1]) > 0.0:
        # Preserve the direction convention used by the original 2-D OBB
        # implementation.  This does not alter the fitted edge line or gripper
        # opening; it only selects the non-flipped representative of that line.
        x_axis = -x_axis
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    return align_axis_sign_to_reference(
        np.column_stack((x_axis, y_axis, normal)),
        reference,
    )
