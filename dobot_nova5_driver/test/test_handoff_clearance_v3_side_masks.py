import numpy as np

from dobot_nova5_driver.handoff_clearance_v3 import (
    evaluate_gripper_side_clearance,
    find_support_plane_points,
    find_unsupported_near_depth_points,
)


def test_side_candidate_masks_identify_the_logged_side():
    points = np.asarray(
        [
            [0.001 * (index % 4), sign * 0.045, 0.06 + 0.001 * (index // 4)]
            for sign in (-1.0, 1.0)
            for index in range(24)
        ],
        dtype=np.float64,
    )
    result = evaluate_gripper_side_clearance(
        points,
        box_center=np.array([0.0, 0.0, 0.05]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3),
        finger_span_m=0.060,
        target_exclusion_m=0.010,
        side_check_depth_m=0.030,
        grasp_below_center_fraction=0.25,
        vertical_margin_above_m=0.080,
        voxel_size_m=0.010,
        min_obstacle_points=20,
    )

    assert result.negative_candidate_mask[:24].all()
    assert not result.negative_candidate_mask[24:].any()
    assert not result.positive_candidate_mask[:24].any()
    assert result.positive_candidate_mask[24:].all()


def test_stable_farther_sensor_depth_rejects_ffs_near_ghost():
    points = np.array([[0.0, 0.0, 0.48], [0.0, 0.0, 0.495]])
    rejected = find_unsupported_near_depth_points(
        points,
        np.array([True, True]),
        np.full((40, 40), 0.50),
        fx=100.0,
        fy=100.0,
        cx=20.0,
        cy=20.0,
    )
    assert rejected.tolist() == [True, False]


def test_missing_or_mixed_sensor_depth_keeps_obstacle():
    points = np.array([[0.0, 0.0, 0.48]])
    missing = np.zeros((40, 40))
    mixed = np.full((40, 40), 0.50)
    mixed[19, 19] = 0.45
    for depth in (missing, mixed):
        rejected = find_unsupported_near_depth_points(
            points,
            np.array([True]),
            depth,
            fx=100.0,
            fy=100.0,
            cx=20.0,
            cy=20.0,
        )
        assert not rejected[0]


def test_rejected_ghost_does_not_enter_finger_corridor():
    points = np.array(
        [[0.001 * (index % 4), 0.045, 0.06 + 0.001 * (index // 4)]
         for index in range(24)]
    )
    result = evaluate_gripper_side_clearance(
        points,
        box_center=np.array([0.0, 0.0, 0.05]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3),
        valid_point_mask=np.zeros(len(points), dtype=bool),
        finger_span_m=0.060,
        target_exclusion_m=0.010,
        side_check_depth_m=0.030,
        grasp_below_center_fraction=0.25,
        vertical_margin_above_m=0.080,
        voxel_size_m=0.010,
        min_obstacle_points=20,
    )
    assert result.clear
    assert result.positive_candidate_point_count == 0


def test_support_plane_filter_removes_only_surface_points():
    points = np.array([
        [0.0, 0.045, 0.001],
        [0.0, 0.045, 0.003],
        [0.0, 0.045, 0.011],
    ])
    rejected = find_support_plane_points(
        points,
        np.array([True, True, True]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        max_distance_m=0.003,
    )
    assert rejected.tolist() == [True, True, False]


def test_support_plane_filter_respects_candidate_mask_and_tilt():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.10, 0.0, 0.010],
        [0.10, 0.0, 0.020],
    ])
    rejected = find_support_plane_points(
        points,
        np.array([False, True, True]),
        np.array([-0.1, 0.0, 1.0, 0.0]),
        max_distance_m=0.003,
    )
    assert rejected.tolist() == [False, True, False]


def test_support_plane_recheck_keeps_raised_obstacle_blocking():
    surface = np.array([
        [0.001 * (index % 4), 0.045, 0.040]
        for index in range(24)
    ])
    raised = np.array([
        [0.001 * (index % 4), 0.045, 0.055]
        for index in range(24)
    ])
    points = np.concatenate((surface, raised))
    options = dict(
        box_center=np.array([0.0, 0.0, 0.06]),
        box_extent=np.array([0.10, 0.06, 0.10]),
        box_rotation=np.eye(3),
        finger_span_m=0.060,
        target_exclusion_m=0.010,
        side_check_depth_m=0.030,
        grasp_below_center_fraction=0.25,
        vertical_margin_above_m=0.080,
        voxel_size_m=0.010,
        min_obstacle_points=20,
    )
    initial = evaluate_gripper_side_clearance(points, **options)
    surface_mask = find_support_plane_points(
        points,
        initial.candidate_mask,
        np.array([0.0, 0.0, 1.0, -0.040]),
        max_distance_m=0.003,
    )
    rechecked = evaluate_gripper_side_clearance(
        points, valid_point_mask=~surface_mask, **options
    )
    assert surface_mask.tolist() == [True] * 24 + [False] * 24
    assert not rechecked.clear
    assert rechecked.positive_candidate_point_count == 24
