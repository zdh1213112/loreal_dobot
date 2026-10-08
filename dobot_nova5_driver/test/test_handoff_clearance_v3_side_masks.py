import numpy as np

from dobot_nova5_driver.handoff_clearance_v3 import (
    evaluate_gripper_side_clearance,
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
