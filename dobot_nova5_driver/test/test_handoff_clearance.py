import numpy as np

from dobot_nova5_driver.handoff_clearance import (
    evaluate_camera_right_handoff_clearance,
    evaluate_gripper_side_clearance,
    evaluate_target_overhead_clearance,
    measure_voxel_overlap,
)


DEFAULTS = {
    "xy_margin_m": 0.05,
    "vertical_gap_m": 0.02,
    "check_height_m": 0.30,
    "voxel_size_m": 0.01,
    "min_obstacle_points": 8,
}


def evaluate(points, rotation=None):
    return evaluate_target_overhead_clearance(
        np.asarray(points, dtype=np.float64),
        box_center=np.array([0.0, 0.0, 0.05]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3) if rotation is None else rotation,
        **DEFAULTS,
    )


def test_box_top_and_table_points_do_not_block_clearance():
    top_points = [[x, y, 0.10] for x in (-0.04, 0.0, 0.04) for y in (-0.02, 0.0, 0.02)]
    table_points = [[x, y, 0.0] for x in (-0.08, 0.0, 0.08) for y in (-0.04, 0.0, 0.04)]

    result = evaluate(top_points + table_points)

    assert result.clear
    assert result.candidate_point_count == 0


def test_connected_points_above_target_block_clearance():
    obstacle = [
        [0.001 * index, 0.0, 0.16 + 0.001 * (index % 3)]
        for index in range(12)
    ]

    result = evaluate(obstacle)

    assert not result.clear
    assert result.candidate_point_count == 12
    assert result.largest_cluster_point_count == 12


def test_sparse_noise_does_not_form_a_blocking_cluster():
    noise = [
        [-0.08, -0.04, 0.15],
        [-0.04, 0.04, 0.20],
        [0.00, -0.04, 0.25],
        [0.04, 0.04, 0.30],
        [0.08, -0.04, 0.35],
        [-0.08, 0.04, 0.38],
        [0.08, 0.04, 0.18],
        [0.00, 0.04, 0.36],
    ]

    result = evaluate(noise)

    assert result.clear
    assert result.candidate_point_count == 8
    assert result.largest_cluster_point_count < 8


def test_clearance_prism_follows_rotated_target_axes():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    obstacle = [
        [0.0, 0.001 * index, 0.16]
        for index in range(10)
    ]

    result = evaluate(obstacle, rotation=rotation)

    assert not result.clear


def test_zero_xy_margin_ignores_points_just_outside_target_footprint():
    outside_target = [
        [0.051 + 0.0001 * index, 0.0, 0.16]
        for index in range(12)
    ]
    parameters = dict(DEFAULTS)
    parameters["xy_margin_m"] = 0.0

    result = evaluate_target_overhead_clearance(
        np.asarray(outside_target, dtype=np.float64),
        box_center=np.array([0.0, 0.0, 0.05]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3),
        **parameters,
    )

    assert result.clear
    assert result.candidate_point_count == 0


RIGHT_CORRIDOR_DEFAULTS = {
    "right_extension_m": 0.08,
    "side_margin_m": 0.03,
    "vertical_gap_m": 0.02,
    "check_height_m": 0.18,
    "voxel_size_m": 0.01,
    "min_obstacle_points": 8,
}


def evaluate_right_corridor(points, rotation=None):
    return evaluate_camera_right_handoff_clearance(
        np.asarray(points, dtype=np.float64),
        box_center=np.array([0.0, 0.0, 0.05]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3) if rotation is None else rotation,
        **RIGHT_CORRIDOR_DEFAULTS,
    )


def test_camera_right_corridor_blocks_connected_approach_points():
    obstacle = [
        [0.06 + 0.001 * index, 0.0, 0.16]
        for index in range(12)
    ]

    result = evaluate_right_corridor(obstacle)

    assert not result.clear
    assert result.candidate_point_count == 12


def test_camera_right_corridor_keeps_target_core_protected():
    above_target = [[0.0, 0.0, 0.16 + 0.001 * index] for index in range(12)]

    result = evaluate_right_corridor(above_target)

    assert not result.clear
    assert result.candidate_point_count == 12


def test_camera_right_corridor_ignores_external_points_on_left():
    left_side = [[-0.06 - 0.001 * index, 0.0, 0.16] for index in range(12)]

    result = evaluate_right_corridor(left_side)

    assert result.clear
    assert result.candidate_point_count == 0


def test_camera_right_corridor_stops_at_configured_eight_centimeters():
    beyond_corridor = [
        [0.135 + 0.001 * index, 0.0, 0.16]
        for index in range(12)
    ]

    result = evaluate_right_corridor(beyond_corridor)

    assert result.clear
    assert result.candidate_point_count == 0


def test_camera_right_direction_does_not_rotate_with_target():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    right_side = [[0.04 + 0.001 * index, 0.0, 0.16] for index in range(12)]

    result = evaluate_right_corridor(right_side, rotation=rotation)

    assert not result.clear


SIDE_CLEARANCE_DEFAULTS = {
    "finger_span_m": 0.060,
    "target_exclusion_m": 0.004,
    "side_check_depth_m": 0.030,
    "grasp_below_center_fraction": 0.25,
    "vertical_margin_above_m": 0.080,
    "voxel_size_m": 0.010,
    "min_obstacle_points": 20,
}


def evaluate_finger_sides(points, rotation=None, target_point_mask=None):
    return evaluate_gripper_side_clearance(
        np.asarray(points, dtype=np.float64),
        box_center=np.array([0.0, 0.0, 0.05]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3) if rotation is None else rotation,
        target_point_mask=target_point_mask,
        **SIDE_CLEARANCE_DEFAULTS,
    )


def test_gripper_side_clearance_ignores_target_and_table_surfaces():
    target_sides = [
        [x, sign * 0.03, z]
        for sign in (-1.0, 1.0)
        for x in (-0.02, 0.0, 0.02)
        for z in (0.04, 0.07, 0.10)
    ]
    table = [
        [x, y, 0.0]
        for x in (-0.02, 0.0, 0.02)
        for y in (-0.05, 0.0, 0.05)
    ]

    result = evaluate_finger_sides(target_sides + table)

    assert result.clear
    assert result.negative_candidate_point_count == 0
    assert result.positive_candidate_point_count == 0


def test_gripper_side_clearance_blocks_if_either_finger_path_is_occupied():
    negative_obstacle = [
        [0.001 * (index % 4), -0.045, 0.06 + 0.001 * (index // 4)]
        for index in range(24)
    ]

    result = evaluate_finger_sides(negative_obstacle)

    assert not result.clear
    assert not result.negative_side_clear
    assert result.positive_side_clear
    assert result.negative_largest_cluster_point_count == 24


def test_gripper_side_clearance_reports_both_occupied_finger_paths():
    obstacles = [
        [0.001 * (index % 4), sign * 0.045, 0.06 + 0.001 * (index // 4)]
        for sign in (-1.0, 1.0)
        for index in range(24)
    ]

    result = evaluate_finger_sides(obstacles)

    assert not result.clear
    assert not result.negative_side_clear
    assert not result.positive_side_clear
    assert result.negative_candidate_point_count == 24
    assert result.positive_candidate_point_count == 24


def test_gripper_side_clearance_only_checks_centered_finger_span():
    obstacle_beside_long_box_end = [
        [0.040 + 0.001 * (index % 4), 0.045, 0.06 + 0.001 * (index // 4)]
        for index in range(24)
    ]

    result = evaluate_finger_sides(obstacle_beside_long_box_end)

    assert result.clear
    assert result.positive_candidate_point_count == 0


def test_gripper_side_clearance_excludes_selected_target_mask_leaking_outside_fit():
    leaked_target_side = [
        [0.001 * (index % 4), 0.045, 0.06 + 0.001 * (index // 4)]
        for index in range(24)
    ]

    result = evaluate_finger_sides(
        leaked_target_side,
        target_point_mask=np.ones(len(leaked_target_side), dtype=bool),
    )

    assert result.clear
    assert result.positive_candidate_point_count == 0
    assert result.positive_largest_cluster_point_count == 0
    assert result.positive_target_excluded_point_count == 24


def test_gripper_side_clearance_still_blocks_non_target_points_after_exclusion():
    leaked_target_side = [
        [0.001 * (index % 4), 0.045, 0.06 + 0.001 * (index // 4)]
        for index in range(24)
    ]
    negative_obstacle = [
        [0.001 * (index % 4), -0.045, 0.06 + 0.001 * (index // 4)]
        for index in range(24)
    ]
    points = leaked_target_side + negative_obstacle
    target_mask = np.array(
        [True] * len(leaked_target_side) + [False] * len(negative_obstacle),
        dtype=bool,
    )

    result = evaluate_finger_sides(points, target_point_mask=target_mask)

    assert not result.clear
    assert not result.negative_side_clear
    assert result.positive_side_clear
    assert result.negative_candidate_point_count == 24
    assert result.positive_candidate_point_count == 0
    assert result.positive_target_excluded_point_count == 24


def test_gripper_side_clearance_rejects_target_mask_with_wrong_length():
    points = [[0.0, 0.045, 0.06]]

    with np.testing.assert_raises_regex(ValueError, "target_point_mask"):
        evaluate_finger_sides(
            points,
            target_point_mask=np.array([True, False], dtype=bool),
        )


def test_gripper_side_clearance_exposes_target_local_candidate_voxels():
    obstacle = [
        [0.001 * (index % 4), 0.045, 0.06 + 0.001 * (index // 4)]
        for index in range(24)
    ]

    result = evaluate_finger_sides(obstacle)

    assert not result.positive_side_clear
    assert result.positive_candidate_voxel_indices.ndim == 2
    assert result.positive_candidate_voxel_indices.shape[1] == 3
    assert len(result.positive_candidate_voxel_indices) > 0


def test_voxel_overlap_requires_same_spatial_region_not_just_two_hits():
    previous = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    same_region = np.array([[0, 0, 0], [1, 0, 0], [9, 9, 9]], dtype=np.int64)
    different_region = np.array([[6, 6, 6], [7, 6, 6], [8, 6, 6]], dtype=np.int64)

    overlap, ratio, confirmed = measure_voxel_overlap(
        previous,
        same_region,
        min_overlap_voxels=2,
        min_overlap_ratio=0.5,
    )
    assert overlap == 2
    assert ratio == 2 / 3
    assert confirmed

    overlap, ratio, confirmed = measure_voxel_overlap(
        previous,
        different_region,
        min_overlap_voxels=1,
        min_overlap_ratio=0.1,
    )
    assert overlap == 0
    assert ratio == 0.0
    assert not confirmed


def test_voxel_overlap_first_or_empty_sample_never_confirms():
    current = np.array([[1, 2, 3]], dtype=np.int64)

    assert measure_voxel_overlap(None, current, min_overlap_voxels=0)[2] is False
    assert measure_voxel_overlap(
        np.empty((0, 3), dtype=np.int64),
        current,
        min_overlap_voxels=0,
    )[2] is False


def test_voxel_overlap_rejects_small_coincidental_cluster_with_runtime_thresholds():
    previous = np.array(
        [[index, 0, 0] for index in range(6)],
        dtype=np.int64,
    )
    current = previous.copy()

    overlap, ratio, confirmed = measure_voxel_overlap(
        previous,
        current,
        min_overlap_voxels=8,
        min_overlap_ratio=0.50,
    )

    assert overlap == 6
    assert ratio == 1.0
    assert not confirmed
