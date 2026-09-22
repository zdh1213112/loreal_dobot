import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciPyRot

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 import (
    CosmeticBoxSingleArmNode,
    PregraspObservation,
    RecoverableGraspError,
    select_nearest_square_grasp_orientation,
)


@contextmanager
def stage(_name):
    yield


def pose_with_local_yaw(reference, yaw_deg):
    rotation = SciPyRot.from_euler(
        "xyz", [reference.rx, reference.ry, reference.rz], degrees=True
    ) * SciPyRot.from_euler("z", yaw_deg, degrees=True)
    rx, ry, rz = rotation.as_euler("xyz", degrees=True)
    return TcpPose(
        reference.x,
        reference.y,
        reference.z,
        float(rx),
        float(ry),
        float(rz),
    )


def test_near_square_grasp_selects_short_edge_when_it_is_nearer():
    target = TcpPose(0.55, -0.15, 0.13, 178.0, 2.0, -4.0)
    current = pose_with_local_yaw(target, 82.0)

    selected, alignment, long_deg, short_deg, selected_deg = (
        select_nearest_square_grasp_orientation(target, current)
    )

    assert alignment == "short"
    assert long_deg == pytest.approx(82.0)
    assert short_deg == pytest.approx(8.0)
    assert selected_deg == pytest.approx(8.0)
    target_y = SciPyRot.from_euler(
        "xyz", [target.rx, target.ry, target.rz], degrees=True
    ).as_matrix()[:, 1]
    selected_y = SciPyRot.from_euler(
        "xyz", [selected.rx, selected.ry, selected.rz], degrees=True
    ).as_matrix()[:, 1]
    assert abs(float(np.dot(target_y, selected_y))) < 1e-6


def test_near_square_grasp_treats_180deg_long_edge_as_equivalent():
    target = TcpPose(0.55, -0.15, 0.13, 178.0, 2.0, -4.0)
    current = pose_with_local_yaw(target, 172.0)

    _selected, alignment, long_deg, short_deg, selected_deg = (
        select_nearest_square_grasp_orientation(target, current)
    )

    assert alignment == "long"
    assert long_deg == pytest.approx(8.0)
    assert short_deg == pytest.approx(82.0)
    assert selected_deg == pytest.approx(8.0)


def test_turntable_bottom_branch_routes_before_legacy_close_and_transfer():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "offset_grasp_enabled": True,
        "grasp_z_offset_m": -0.004,
        "grasp_z_offset_limit_m": 0.010,
        "minimum_safe_tcp_z_m": 0.129,
        "turntable_enabled": True,
        "turntable_height_safety_enabled": False,
        "bottom_barcode_recovery_enabled": True,
        "dh_max_opening_m": 0.095,
        "user_index": 0,
        "command_tool_index": 1,
        "top_surface_barcode_enabled": True,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._motion_profile = lambda: {}
    node._apply_grasp_z_safety = lambda pose, *_args: pose
    node._publish_status = lambda _message: None
    node._timed_stage = stage
    node._require_cycle_active = lambda _label: None
    node._wait_for_secondary_y_clearance = lambda _label: None
    node._set_top_surface_barcode_window = lambda _enabled: None
    node._current_top_surface_barcode = lambda: ""
    node._execute_offset_entry = lambda *_args, **_kwargs: None
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 0
    pose = TcpPose(0.56, -0.15, 0.134, 178.0, 2.0, -4.0)
    node._current_command_pose = lambda: pose
    node.controller = SimpleNamespace(
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        current_joint=lambda: [0.0] * 6,
    )
    actions = []
    node.gripper = SimpleNamespace(
        read_position=lambda: 0.6,
        set_position=lambda *_args, **_kwargs: actions.append("preshape"),
        close=lambda *_args, **_kwargs: actions.append("initial_close"),
    )
    node._execute_turntable_bottom_center_recovery = (
        lambda *_args: actions.append("bottom_center_recovery")
    )
    node._mark_turntable_material_removed = lambda: actions.append("removed")
    node._place_as_top_barcode_box = lambda: actions.append("top_place")

    node._execute_one_cycle(pose, 0.055, 0.038, 0.121)

    assert actions == ["preshape", "bottom_center_recovery", "removed", "top_place"]


def test_preconfirmed_side_barcode_uses_direct_centre_hover_and_vertical_descent():
    class StopAfterApproach(RuntimeError):
        pass

    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "offset_grasp_enabled": True,
        "grasp_z_offset_m": 0.0,
        "grasp_z_offset_limit_m": 0.010,
        "minimum_safe_tcp_z_m": 0.129,
        "turntable_enabled": True,
        "turntable_height_safety_enabled": False,
        "dh_max_opening_m": 0.095,
        "dh_timeout_s": 3.0,
        "dh_grasp_force": 50,
        "user_index": 0,
        "command_tool_index": 1,
        "side_barcode_direct_hover_clearance_m": 0.120,
        "pregrasp_min_hover_clearance_m": 0.030,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._motion_profile = lambda: {
        "joint_speed": 20,
        "joint_pose_acc": 20,
        "linear_speed": 20,
        "linear_acc": 20,
    }
    node._apply_grasp_z_safety = lambda pose, *_args: pose
    node._publish_status = lambda _message: None
    node._timed_stage = stage
    node._require_cycle_active = lambda _label: None
    node._wait_for_secondary_y_clearance = lambda _label: None
    node._cycle_cancel_requested = lambda: False
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 0

    startup = TcpPose(0.40, 0.10, 0.320, 179.0, 0.0, -90.0)
    target = TcpPose(0.56, -0.15, 0.134, 178.0, 2.0, -4.0)
    feedback = [startup]
    motions = []

    def move_joint_tcp(pose, **_kwargs):
        feedback[0] = pose
        motions.append(("centre_hover", pose))

    def move_linear_tcp(pose, **_kwargs):
        feedback[0] = pose
        motions.append(("vertical_descent", pose))

    node._current_command_pose = lambda: feedback[0]
    node.controller = SimpleNamespace(
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        current_joint=lambda: [0.0] * 6,
        move_joint_tcp=move_joint_tcp,
        move_linear_tcp=move_linear_tcp,
    )
    node._revalidate_target_at_hover = lambda target_pose, *_args, **_kwargs: target_pose
    node._execute_offset_entry = lambda *_args, **_kwargs: pytest.fail(
        "preconfirmed side barcode must not use the offset entry"
    )
    node._wait_for_top_surface_barcode = lambda: pytest.fail(
        "preconfirmed side barcode must not wait for top-surface scanning"
    )
    node._current_top_surface_barcode = lambda: pytest.fail(
        "preconfirmed side barcode must not read a top-surface result"
    )
    barcode_windows = []
    node._set_top_surface_barcode_window = barcode_windows.append
    node.gripper = SimpleNamespace(
        read_position=lambda: 0.6,
        set_position=lambda *_args, **_kwargs: None,
        wait_until_stopped=lambda **_kwargs: None,
        set_force=lambda *_args: None,
        close=lambda **_kwargs: (_ for _ in ()).throw(StopAfterApproach()),
    )

    with pytest.raises(StopAfterApproach):
        node._execute_one_cycle(
            target,
            width_m=0.055,
            height_m=0.038,
            length_m=0.121,
            turntable_side_barcode="barcode_detected",
        )

    assert [name for name, _pose in motions] == [
        "centre_hover",
        "vertical_descent",
    ]
    hover = motions[0][1]
    assert (hover.x, hover.y, hover.z) == pytest.approx(
        (target.x, target.y, target.z + 0.120)
    )
    assert motions[1][1] == target
    assert barcode_windows == [False, False]


def make_sequence_node(*, initial_j6=0.0, j6_limit=355.0):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "bottom_center_first_rz_delta_deg": 40.0,
        "bottom_center_first_tool_rx_delta_deg": 70.0,
        "bottom_center_release_tool_rx_delta_deg": 70.0,
        "grasp_lift_m": 0.060,
        "bottom_flip_lift_m": 0.160,
        "bottom_flip_j6_half_turn_deg": 180.0,
        "bottom_flip_post_turn_descent_m": 0.120,
        "dh_timeout_s": 3.0,
        "dh_grasp_force": 50,
        "barcode_flip_safe_joint_limit_deg": j6_limit,
        "barcode_flip_jog_tolerance_deg": 1.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._timed_stage = stage
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    events = []
    pose = [TcpPose(0.56, -0.15, 0.134, 178.0, 2.0, -4.0)]
    joints = [[0.0, 0.0, 0.0, 0.0, 0.0, initial_j6]]
    position = [0.6]

    def set_position(value, **_kwargs):
        position[0] = value
        events.append(("open", round(value, 3)))

    def close(**_kwargs):
        position[0] = 0.35
        events.append(("close", round(pose[0].z, 3)))

    def rotate(xyz, ry_delta_deg, rz_delta_deg, linear_tcp=False):
        assert linear_tcp
        assert xyz[:2] == pytest.approx([pose[0].x, pose[0].y])
        events.append(
            (
                "rotation",
                ry_delta_deg,
                rz_delta_deg,
                round(pose[0].z, 3),
                round(pose[0].x, 4),
            )
        )

    def move_z(target_z, _motion, _label, *, lifting, target_xy=None):
        target_x, target_y = (
            (pose[0].x, pose[0].y) if target_xy is None else target_xy
        )
        pose[0] = TcpPose(
            target_x, target_y, target_z,
            pose[0].rx, pose[0].ry, pose[0].rz,
        )
        events.append(("z", round(target_z, 3), lifting, round(pose[0].x, 4)))

    def move_joint(target, **_kwargs):
        joints[0] = target
        pose[0] = TcpPose(
            pose[0].x + 0.0037,
            pose[0].y,
            pose[0].z,
            pose[0].rx,
            pose[0].ry,
            pose[0].rz,
        )
        events.append(("j6", round(target[5], 1)))

    def snap_to_nearest_face(_motion):
        joints[0][5] = 0.0
        pose[0] = TcpPose(
            pose[0].x + 0.001,
            pose[0].y,
            pose[0].z,
            pose[0].rx,
            pose[0].ry,
            pose[0].rz,
        )
        events.append(("snap", 0.0))

    def tool_rx(delta_deg, _label):
        events.append(
            ("tool_rx", delta_deg, round(pose[0].z, 3), round(pose[0].x, 4))
        )

    node._current_command_pose = lambda: pose[0]
    node._move_to_user_xyz_with_rotation = rotate
    node._bottom_center_linear_z = move_z
    node._snap_bottom_center_to_nearest_face = snap_to_nearest_face
    node._bottom_center_tool_rx = tool_rx
    node._wait_for_bottom_center_tracked_pose = lambda: TcpPose(
        0.572, -0.160, 0.134, 0.0, 0.0, 0.0
    )
    node._confirm_grasp_before_lift = (
        lambda *_args: events.append(("confirm", round(pose[0].z, 3)))
    )
    node._validate_grasp_feedback = lambda *_args: None
    node._execute_turntable_safe_departure_lift = (
        lambda *_args: events.append(("safe_departure", round(pose[0].z, 3)))
    )
    node.gripper = SimpleNamespace(
        read_position=lambda: position[0],
        set_position=set_position,
        wait_until_stopped=lambda **_kwargs: None,
        set_force=lambda *_args: None,
        close=close,
    )
    node.controller = SimpleNamespace(current_joint=lambda: joints[0], move_joint=move_joint)
    return node, pose[0], events


def test_bottom_center_sequence_accepts_j6_xy_shift_and_returns_to_absolute_grasp_z():
    node, grasp_pose, events = make_sequence_node(initial_j6=-20.0)
    motion = {"barcode_alignment_speed": 20, "barcode_alignment_acc": 20}

    node._execute_turntable_bottom_center_recovery(
        grasp_pose, 0.6, 0.057, 0.095, motion
    )

    assert events == [
        ("close", 0.134),
        ("confirm", 0.134),
        ("z", 0.194, True, 0.56),
        ("snap", 0.0),
        ("rotation", 0.0, 40.0, 0.194, 0.561),
        ("z", 0.134, False, 0.561),
        ("open", 0.6),
        ("tool_rx", 70.0, 0.134, 0.561),
        ("close", 0.134),
        ("confirm", 0.134),
        ("z", 0.294, True, 0.561),
        ("j6", 180.0),
        ("z", 0.174, False, 0.5647),
        ("open", 0.6),
        ("tool_rx", 70.0, 0.174, 0.5647),
        ("z", 0.134, False, 0.572),
        ("close", 0.134),
        ("confirm", 0.134),
        ("safe_departure", 0.134),
    ]


def test_bottom_center_linear_z_preserves_shifted_feedback_xy():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "jog_tolerance_m": 0.002,
        "minimum_safe_tcp_z_m": 0.129,
        "user_index": 0,
        "command_tool_index": 1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    pose = [TcpPose(0.5637, -0.15, 0.294, 178.0, 2.0, -4.0)]
    commanded = []

    def move_linear_tcp(target, **_kwargs):
        commanded.append(target)
        pose[0] = target

    node._current_command_pose = lambda: pose[0]
    node.controller = SimpleNamespace(
        current_joint=lambda: [0.0] * 6,
        inverse_kinematics=lambda target, **_kwargs: commanded.append(target),
        move_linear_tcp=move_linear_tcp,
    )

    node._bottom_center_linear_z(
        0.194,
        {"linear_speed": 20, "linear_acc": 20},
        "bottom recovery second descent",
        lifting=False,
    )

    assert len(commanded) == 2
    assert commanded[0].x == pytest.approx(0.5637)
    assert commanded[1].x == pytest.approx(0.5637)
    assert commanded[1].y == pytest.approx(-0.15)
    assert commanded[1].z == pytest.approx(0.194)


def test_bottom_center_linear_descent_uses_latest_tracked_xy():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "jog_tolerance_m": 0.002,
        "minimum_safe_tcp_z_m": 0.129,
        "user_index": 0,
        "command_tool_index": 1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    pose = [TcpPose(0.5647, -0.15, 0.174, 178.0, 2.0, -4.0)]

    def move_linear_tcp(target, **_kwargs):
        pose[0] = target

    node._current_command_pose = lambda: pose[0]
    node.controller = SimpleNamespace(
        current_joint=lambda: [0.0] * 6,
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        move_linear_tcp=move_linear_tcp,
    )

    node._bottom_center_linear_z(
        0.134,
        {"linear_speed": 20, "linear_acc": 20},
        "tracked final descent",
        lifting=False,
        target_xy=(0.572, -0.160),
    )

    assert (pose[0].x, pose[0].y, pose[0].z) == pytest.approx(
        (0.572, -0.160, 0.134)
    )


def test_bottom_center_wait_requires_pose_published_after_wait_started():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "bottom_center_tracking_timeout_s": 0.5,
        "pregrasp_pose_max_age_s": 0.65,
        "pregrasp_max_correction_m": 0.050,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 10
    node.latest_pregrasp_pose = TcpPose(0.50, -0.10, 0.134, 0.0, 0.0, 0.0)
    node.latest_pregrasp_pose_received_at = time.monotonic()
    node._current_command_pose = lambda: TcpPose(
        0.5647, -0.15, 0.174, 178.0, 2.0, -4.0
    )

    expected = TcpPose(0.572, -0.160, 0.134, 0.0, 0.0, 0.0)

    def publish_fresh_pose():
        time.sleep(0.03)
        with node.data_lock:
            node.pregrasp_pose_count += 1
            node.latest_pregrasp_pose = expected
            node.latest_pregrasp_pose_received_at = time.monotonic()

    publisher = threading.Thread(target=publish_fresh_pose)
    publisher.start()
    result = node._wait_for_bottom_center_tracked_pose()
    publisher.join()

    assert result == expected


def test_bottom_center_tool_rx_uses_current_local_x_axis():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "user_index": 0,
        "command_tool_index": 1,
        "jog_tolerance_m": 0.002,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._motion_profile = lambda: {"post_scan_speed": 20, "post_scan_acc": 20}
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    start = TcpPose(0.56, -0.15, 0.134, 35.0, -20.0, 80.0)
    pose = [start]
    targets = []

    def move_linear_tcp(target, **_kwargs):
        targets.append(target)
        pose[0] = target

    node._current_command_pose = lambda: pose[0]
    node.controller = SimpleNamespace(
        current_joint=lambda: [0.0] * 6,
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        move_linear_tcp=move_linear_tcp,
    )

    node._bottom_center_tool_rx(70.0, "test Tool-Rx rotation")

    assert len(targets) == 1
    expected = (
        SciPyRot.from_euler("xyz", [start.rx, start.ry, start.rz], degrees=True)
        * SciPyRot.from_euler("x", 70.0, degrees=True)
    ).as_matrix()
    actual = SciPyRot.from_euler(
        "xyz", [targets[0].rx, targets[0].ry, targets[0].rz], degrees=True
    ).as_matrix()
    assert actual == pytest.approx(expected)
    assert (targets[0].x, targets[0].y, targets[0].z) == pytest.approx(
        (start.x, start.y, start.z)
    )


def test_bottom_center_face_snap_uses_nearest_90_degree_anchor_at_short_lift():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "barcode_flip_watch_joint_index": 5,
        "d435_side_face_reference_joint_deg": 0.0,
        "barcode_flip_step_deg": -90.0,
        "barcode_flip_safe_joint_limit_deg": 355.0,
        "barcode_flip_jog_tolerance_deg": 1.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    node._is_barcode_flip_joint_safe = lambda *_args: True
    joints = [[0.0, 0.0, 0.0, 0.0, 0.0, -20.0]]

    def move_joint(target, **_kwargs):
        joints[0] = target

    node.controller = SimpleNamespace(
        current_joint=lambda: joints[0],
        move_joint=move_joint,
    )

    node._snap_bottom_center_to_nearest_face(
        {"barcode_alignment_speed": 20, "barcode_alignment_acc": 20}
    )

    assert joints[0][5] == pytest.approx(0.0)


def test_bottom_center_sequence_blocks_j6_limit_before_turn():
    node, grasp_pose, events = make_sequence_node(initial_j6=200.0, j6_limit=150.0)
    motion = {"barcode_alignment_speed": 20, "barcode_alignment_acc": 20}

    with pytest.raises(RuntimeError, match="J6 limit"):
        node._execute_turntable_bottom_center_recovery(
            grasp_pose, 0.6, 0.057, 0.095, motion
        )

    assert not any(event[0] == "j6" for event in events)


def make_v3_pregrasp_revalidation_node(*, near_hover_count):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "pregrasp_position_tolerance_m": 0.007,
        "pregrasp_angle_tolerance_deg": 8.0,
        "pregrasp_max_correction_m": 0.050,
        "pregrasp_max_correction_angle_deg": 30.0,
        "pregrasp_min_hover_clearance_m": 0.030,
        "pregrasp_unconfirmed_shift_reject_m": 0.020,
        "pregrasp_position_consensus_samples": 2,
        "pregrasp_use_live_pose_for_descent": True,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    target = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    jumped = TcpPose(0.447, -0.2, 0.1, -88.0, 2.0, -42.0)
    observation = PregraspObservation(1, jumped, 100.6, 1.0, 0.0)
    node._fresh_pregrasp_observations = lambda _previous_count: ([observation], 0.01)
    node._estimate_pregrasp_target = lambda *_args: {
        "pose": target,
        "position_mode": "initial-stable-protected",
        "orientation_mode": "initial-stable-protected",
        "position_delta_m": 0.0,
        "angle_delta_deg": 0.0,
        "velocity_xy_mps": [0.0, 0.0],
        "motion_displacement_m": 0.0,
        "observation_count": 1,
        "near_hover_count": near_hover_count,
        "consensus_count": 0,
    }
    node._current_command_pose = lambda: TcpPose(
        target.x, target.y, target.z + 0.10, target.rx, target.ry, target.rz
    )
    logger = MagicMock()
    node.get_logger = lambda: logger
    return node, target, logger


def test_v3_pregrasp_motion_only_jump_keeps_initial_stable_target():
    node, target, logger = make_v3_pregrasp_revalidation_node(near_hover_count=0)

    result = node._revalidate_target_at_hover(
        target,
        {"joint_speed": 100, "joint_pose_acc": 100},
        0.0,
        0.01,
        0,
        100.7,
    )

    assert result == target
    assert "no near-hover sample" in logger.warning.call_args.args[0]


def test_v3_pregrasp_near_hover_jump_still_refuses_descent():
    node, target, _logger = make_v3_pregrasp_revalidation_node(near_hover_count=1)

    with pytest.raises(RecoverableGraspError, match="refusing descent"):
        node._revalidate_target_at_hover(
            target,
            {"joint_speed": 100, "joint_pose_acc": 100},
            0.0,
            0.01,
            0,
            100.7,
        )
