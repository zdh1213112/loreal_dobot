"""Fast target-relative 3-D clearance checks for the cosmetic-box handoff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandoffClearanceResult:
    """Result of one full-scene point-cloud check above a selected box."""

    clear: bool
    candidate_point_count: int
    occupied_voxel_count: int
    largest_cluster_point_count: int
    candidate_mask: np.ndarray


@dataclass(frozen=True)
class GripperSideClearanceResult:
    """Occupancy of the two finger descent corridors beside a target."""

    clear: bool
    negative_side_clear: bool
    positive_side_clear: bool
    negative_candidate_point_count: int
    positive_candidate_point_count: int
    negative_largest_cluster_point_count: int
    positive_largest_cluster_point_count: int
    negative_target_excluded_point_count: int
    positive_target_excluded_point_count: int
    candidate_mask: np.ndarray
    # Target-local occupied voxels for each side.  These are exposed so the
    # vision state machine can require spatial persistence across independent
    # LIVE clouds without relying on frame-to-frame point indices.
    negative_candidate_voxel_indices: np.ndarray
    positive_candidate_voxel_indices: np.ndarray


_VOXEL_NEIGHBORS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _largest_connected_voxel_cluster_points(
    points: np.ndarray,
    voxel_size_m: float,
) -> tuple[int, int]:
    """Return occupied-voxel count and largest 26-connected cluster size."""

    occupied_voxels, largest_cluster_points, _ = _connected_voxel_cluster_stats(
        points,
        voxel_size_m,
    )
    return occupied_voxels, largest_cluster_points


def _connected_voxel_cluster_stats(
    points: np.ndarray,
    voxel_size_m: float,
) -> tuple[int, int, np.ndarray]:
    """Return occupancy, largest-cluster points, and its voxel indices."""

    if len(points) == 0:
        return 0, 0, np.empty((0, 3), dtype=np.int64)
    voxel_indices = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique_voxels, point_counts = np.unique(
        voxel_indices,
        axis=0,
        return_counts=True,
    )
    voxel_weights = {
        tuple(int(value) for value in voxel): int(count)
        for voxel, count in zip(unique_voxels, point_counts)
    }
    unvisited = set(voxel_weights)
    largest_cluster_points = 0
    largest_cluster_voxels: set[tuple[int, int, int]] = set()
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        cluster_points = 0
        cluster_voxels = {seed}
        while stack:
            voxel = stack.pop()
            cluster_points += voxel_weights[voxel]
            for dx, dy, dz in _VOXEL_NEIGHBORS:
                neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
                    cluster_voxels.add(neighbor)
        if cluster_points > largest_cluster_points:
            largest_cluster_points = cluster_points
            largest_cluster_voxels = cluster_voxels
    return (
        len(unique_voxels),
        largest_cluster_points,
        np.asarray(sorted(largest_cluster_voxels), dtype=np.int64).reshape(-1, 3),
    )


def measure_voxel_overlap(
    previous_voxels: np.ndarray | None,
    current_voxels: np.ndarray,
    *,
    min_overlap_voxels: int = 1,
    min_overlap_ratio: float = 0.0,
) -> tuple[int, float, bool]:
    """Measure spatial overlap between two occupied-voxel sets.

    The ratio is normalized by the smaller set, allowing a fixed obstacle
    whose visible sampling changes between FFS frames to remain confirmable.
    ``previous_voxels=None`` (or an empty set) never confirms persistence.
    """

    if int(min_overlap_voxels) < 0:
        raise ValueError("min_overlap_voxels must be non-negative")
    if not 0.0 <= float(min_overlap_ratio) <= 1.0:
        raise ValueError("min_overlap_ratio must be in [0, 1]")
    current = np.asarray(current_voxels, dtype=np.int64)
    if current.ndim != 2 or current.shape[1:] != (3,):
        raise ValueError("current_voxels must have shape (N, 3)")
    if previous_voxels is None:
        return 0, 0.0, False
    previous = np.asarray(previous_voxels, dtype=np.int64)
    if previous.ndim != 2 or previous.shape[1:] != (3,):
        raise ValueError("previous_voxels must have shape (N, 3)")
    if len(previous) == 0 or len(current) == 0:
        return 0, 0.0, False
    previous_set = {tuple(int(value) for value in voxel) for voxel in previous}
    current_set = {tuple(int(value) for value in voxel) for voxel in current}
    overlap = len(previous_set & current_set)
    ratio = overlap / float(max(1, min(len(previous_set), len(current_set))))
    confirmed = (
        overlap > 0
        and overlap >= int(min_overlap_voxels)
        and ratio >= float(min_overlap_ratio)
    )
    return int(overlap), float(ratio), bool(confirmed)


def evaluate_target_overhead_clearance(
    points_3d: np.ndarray,
    box_center: np.ndarray,
    box_extent: np.ndarray,
    box_rotation: np.ndarray,
    *,
    xy_margin_m: float,
    vertical_gap_m: float,
    check_height_m: float,
    voxel_size_m: float,
    min_obstacle_points: int,
) -> HandoffClearanceResult:
    """Check a target-relative prism above the box using the full scene cloud.

    ``box_rotation`` contains the local box axes in its columns.  Its third
    axis points from the top surface toward the camera, so positive local Z is
    the space above the box.  Points on the box and tabletop are excluded by
    starting the prism above ``box_extent[2] / 2 + vertical_gap_m``.
    """

    points = np.asarray(points_3d, dtype=np.float64)
    center = np.asarray(box_center, dtype=np.float64).reshape(-1)
    extent = np.asarray(box_extent, dtype=np.float64).reshape(-1)
    rotation = np.asarray(box_rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")
    if center.shape != (3,):
        raise ValueError("box_center must contain 3 values")
    if extent.shape != (3,) or np.any(extent <= 0.0):
        raise ValueError("box_extent must contain 3 positive values")
    if rotation.shape != (3, 3):
        raise ValueError("box_rotation must have shape (3, 3)")
    if xy_margin_m < 0.0 or vertical_gap_m < 0.0 or check_height_m <= 0.0:
        raise ValueError("clearance dimensions must be non-negative and height positive")
    if voxel_size_m <= 0.0 or min_obstacle_points <= 0:
        raise ValueError("voxel_size_m and min_obstacle_points must be positive")

    finite = np.all(np.isfinite(points), axis=1)
    local_points = np.zeros_like(points)
    local_points[finite] = (points[finite] - center) @ rotation
    half_length = 0.5 * extent[0] + float(xy_margin_m)
    half_width = 0.5 * extent[1] + float(xy_margin_m)
    lower_z = 0.5 * extent[2] + float(vertical_gap_m)
    upper_z = lower_z + float(check_height_m)
    candidate_mask = (
        finite
        & (np.abs(local_points[:, 0]) <= half_length)
        & (np.abs(local_points[:, 1]) <= half_width)
        & (local_points[:, 2] >= lower_z)
        & (local_points[:, 2] <= upper_z)
    )
    candidates = points[candidate_mask]
    occupied_voxels, largest_cluster_points = _largest_connected_voxel_cluster_points(
        candidates,
        voxel_size_m,
    )
    blocked = (
        len(candidates) >= int(min_obstacle_points)
        and largest_cluster_points >= int(min_obstacle_points)
    )
    return HandoffClearanceResult(
        clear=not blocked,
        candidate_point_count=int(len(candidates)),
        occupied_voxel_count=int(occupied_voxels),
        largest_cluster_point_count=int(largest_cluster_points),
        candidate_mask=candidate_mask,
    )


def evaluate_gripper_side_clearance(
    points_3d: np.ndarray,
    box_center: np.ndarray,
    box_extent: np.ndarray,
    box_rotation: np.ndarray,
    *,
    target_point_mask: np.ndarray | None = None,
    finger_span_m: float,
    target_exclusion_m: float,
    side_check_depth_m: float,
    grasp_below_center_fraction: float,
    vertical_margin_above_m: float,
    voxel_size_m: float,
    min_obstacle_points: int,
) -> GripperSideClearanceResult:
    """Check the two vertical paths used by the open gripper fingers.

    The target's local Y axis is the gripper closing direction.  Two compact
    prisms immediately outside the target's ``-Y`` and ``+Y`` faces model the
    space swept by the fingers during vertical descent.  The broad tabletop is
    excluded by stopping the check at the configured grasp depth instead of
    extending it to the bottom face.

    ``target_point_mask`` identifies points belonging to the selected SAM
    target.  Those points are excluded even if a temporarily underestimated
    fitted width places the target's own side surface inside a finger corridor.
    All non-target scene points remain eligible obstacle evidence.

    A side is blocked only by a sufficiently large connected point cluster, so
    isolated stereo noise cannot authorize a false collision stop.  Either
    blocked side makes the complete descent unsafe.
    """

    points = np.asarray(points_3d, dtype=np.float64)
    center = np.asarray(box_center, dtype=np.float64).reshape(-1)
    extent = np.asarray(box_extent, dtype=np.float64).reshape(-1)
    rotation = np.asarray(box_rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")
    if center.shape != (3,):
        raise ValueError("box_center must contain 3 values")
    if extent.shape != (3,) or np.any(extent <= 0.0):
        raise ValueError("box_extent must contain 3 positive values")
    if rotation.shape != (3, 3):
        raise ValueError("box_rotation must have shape (3, 3)")
    if target_point_mask is None:
        target_points = np.zeros(len(points), dtype=bool)
    else:
        target_points = np.asarray(target_point_mask, dtype=bool)
        if target_points.shape != (len(points),):
            raise ValueError("target_point_mask must have shape (N,)")
    if finger_span_m <= 0.0 or side_check_depth_m <= 0.0:
        raise ValueError("finger_span_m and side_check_depth_m must be positive")
    if target_exclusion_m < 0.0 or vertical_margin_above_m < 0.0:
        raise ValueError("clearance margins must be non-negative")
    if not 0.0 <= grasp_below_center_fraction <= 0.5:
        raise ValueError("grasp_below_center_fraction must be in [0, 0.5]")
    if voxel_size_m <= 0.0 or min_obstacle_points <= 0:
        raise ValueError("voxel_size_m and min_obstacle_points must be positive")

    finite = np.all(np.isfinite(points), axis=1)
    local_points = np.zeros_like(points)
    local_points[finite] = (points[finite] - center) @ rotation
    half_length, half_width, half_height = 0.5 * extent
    corridor_half_length = min(half_length, 0.5 * float(finger_span_m))
    lower_height = -float(grasp_below_center_fraction) * extent[2]
    upper_height = half_height + float(vertical_margin_above_m)
    common = (
        finite
        & (np.abs(local_points[:, 0]) <= corridor_half_length)
        & (local_points[:, 2] >= lower_height)
        & (local_points[:, 2] <= upper_height)
    )

    side_masks = []
    candidate_counts = []
    largest_clusters = []
    target_excluded_counts = []
    candidate_voxels = []
    side_clear = []
    for sign in (-1.0, 1.0):
        outward = sign * local_points[:, 1] - half_width
        raw_side_mask = (
            common
            & (outward >= float(target_exclusion_m))
            & (
                outward
                <= float(target_exclusion_m) + float(side_check_depth_m)
            )
        )
        target_excluded_mask = raw_side_mask & target_points
        side_mask = raw_side_mask & ~target_points
        candidates = points[side_mask]
        # Keep the existing world-frame cluster safety test, and additionally
        # expose all side candidates in target-local voxels for persistence.
        # The global point-cloud sampling order is not stable between FFS
        # inferences, so persistence must not rely on point indices or on a
        # small coincidental overlap of only one sub-cluster.
        local_candidates = local_points[side_mask]
        candidate_voxels.append(
            np.unique(
                np.floor(local_candidates / float(voxel_size_m)).astype(np.int64),
                axis=0,
            )
            if len(local_candidates)
            else np.empty((0, 3), dtype=np.int64)
        )
        _, largest_cluster = _largest_connected_voxel_cluster_points(
            candidates,
            voxel_size_m,
        )
        blocked = (
            len(candidates) >= int(min_obstacle_points)
            and largest_cluster >= int(min_obstacle_points)
        )
        side_masks.append(side_mask)
        candidate_counts.append(int(len(candidates)))
        largest_clusters.append(int(largest_cluster))
        target_excluded_counts.append(int(np.count_nonzero(target_excluded_mask)))
        side_clear.append(not blocked)

    return GripperSideClearanceResult(
        clear=bool(side_clear[0] and side_clear[1]),
        negative_side_clear=bool(side_clear[0]),
        positive_side_clear=bool(side_clear[1]),
        negative_candidate_point_count=candidate_counts[0],
        positive_candidate_point_count=candidate_counts[1],
        negative_largest_cluster_point_count=largest_clusters[0],
        positive_largest_cluster_point_count=largest_clusters[1],
        negative_target_excluded_point_count=target_excluded_counts[0],
        positive_target_excluded_point_count=target_excluded_counts[1],
        candidate_mask=side_masks[0] | side_masks[1],
        negative_candidate_voxel_indices=candidate_voxels[0],
        positive_candidate_voxel_indices=candidate_voxels[1],
    )


def evaluate_camera_right_handoff_clearance(
    points_3d: np.ndarray,
    box_center: np.ndarray,
    box_extent: np.ndarray,
    box_rotation: np.ndarray,
    *,
    right_extension_m: float,
    side_margin_m: float,
    vertical_gap_m: float,
    check_height_m: float,
    voxel_size_m: float,
    min_obstacle_points: int,
) -> HandoffClearanceResult:
    """Check the target core plus compact D405 image-right handoff corridor.

    The direction is camera ``+X`` projected onto the detected box top plane,
    so rotating the cosmetic box does not rotate the known 102 approach side.
    The protected footprint includes the target itself and a one-sided
    extension from its right edge.  There is no corresponding left extension.
    """

    points = np.asarray(points_3d, dtype=np.float64)
    center = np.asarray(box_center, dtype=np.float64).reshape(-1)
    extent = np.asarray(box_extent, dtype=np.float64).reshape(-1)
    rotation = np.asarray(box_rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")
    if center.shape != (3,):
        raise ValueError("box_center must contain 3 values")
    if extent.shape != (3,) or np.any(extent <= 0.0):
        raise ValueError("box_extent must contain 3 positive values")
    if rotation.shape != (3, 3):
        raise ValueError("box_rotation must have shape (3, 3)")
    if right_extension_m <= 0.0 or side_margin_m < 0.0:
        raise ValueError("right_extension_m must be positive and side_margin_m non-negative")
    if vertical_gap_m < 0.0 or check_height_m <= 0.0:
        raise ValueError("clearance height dimensions must be non-negative and height positive")
    if voxel_size_m <= 0.0 or min_obstacle_points <= 0:
        raise ValueError("voxel_size_m and min_obstacle_points must be positive")

    normal = rotation[:, 2]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("box_rotation has an invalid surface-normal axis")
    normal = normal / normal_norm

    camera_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right_axis = camera_right - float(np.dot(camera_right, normal)) * normal
    right_norm = float(np.linalg.norm(right_axis))
    if right_norm <= 1e-6:
        raise ValueError("box top plane is parallel to the D405 image plane")
    right_axis /= right_norm
    if float(np.dot(right_axis, camera_right)) < 0.0:
        right_axis = -right_axis
    side_axis = np.cross(normal, right_axis)
    side_axis /= float(np.linalg.norm(side_axis)) + 1e-12

    # Project the rotated target footprint onto the camera-right corridor axes.
    local_x_axis = rotation[:, 0]
    local_y_axis = rotation[:, 1]
    target_half_right = 0.5 * (
        extent[0] * abs(float(np.dot(local_x_axis, right_axis)))
        + extent[1] * abs(float(np.dot(local_y_axis, right_axis)))
    )
    target_half_side = 0.5 * (
        extent[0] * abs(float(np.dot(local_x_axis, side_axis)))
        + extent[1] * abs(float(np.dot(local_y_axis, side_axis)))
    )

    finite = np.all(np.isfinite(points), axis=1)
    relative = np.zeros_like(points)
    relative[finite] = points[finite] - center
    right_values = relative @ right_axis
    side_values = relative @ side_axis
    height_values = relative @ normal
    lower_height = 0.5 * extent[2] + float(vertical_gap_m)
    upper_height = lower_height + float(check_height_m)
    over_target = (
        (np.abs(right_values) <= target_half_right)
        & (np.abs(side_values) <= target_half_side)
    )
    in_right_corridor = (
        (right_values >= target_half_right)
        & (right_values <= target_half_right + float(right_extension_m))
        & (np.abs(side_values) <= target_half_side + float(side_margin_m))
    )
    candidate_mask = (
        finite
        & (over_target | in_right_corridor)
        & (height_values >= lower_height)
        & (height_values <= upper_height)
    )
    candidates = points[candidate_mask]
    occupied_voxels, largest_cluster_points = _largest_connected_voxel_cluster_points(
        candidates,
        voxel_size_m,
    )
    blocked = (
        len(candidates) >= int(min_obstacle_points)
        and largest_cluster_points >= int(min_obstacle_points)
    )
    return HandoffClearanceResult(
        clear=not blocked,
        candidate_point_count=int(len(candidates)),
        occupied_voxel_count=int(occupied_voxels),
        largest_cluster_point_count=int(largest_cluster_points),
        candidate_mask=candidate_mask,
    )
