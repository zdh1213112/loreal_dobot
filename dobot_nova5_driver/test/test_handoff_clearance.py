import numpy as np

from dobot_nova5_driver.handoff_clearance import (
    evaluate_camera_right_handoff_clearance,
    evaluate_target_overhead_clearance,
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
