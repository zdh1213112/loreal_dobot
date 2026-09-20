import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 import (
    CosmeticBoxSingleArmNode,
)


@contextmanager
def stage(_name):
    yield


def test_turntable_bottom_branch_skips_initial_close_and_transfer():
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


def make_sequence_node(*, initial_j6=0.0, j6_limit=355.0):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "bottom_flip_user_ry_target_deg": -45.0,
        "bottom_flip_lift_m": 0.160,
        "bottom_flip_j6_half_turn_deg": 180.0,
        "bottom_flip_post_turn_descent_m": 0.100,
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
        assert linear_tcp and rz_delta_deg == 0.0
        assert xyz[:2] == [0.56, -0.15]
        events.append(("ry", ry_delta_deg, round(pose[0].z, 3)))

    def move_z(grasp_pose, target_z, _motion, _label, *, lifting):
        assert (grasp_pose.x, grasp_pose.y) == (0.56, -0.15)
        pose[0] = TcpPose(
            grasp_pose.x, grasp_pose.y, target_z,
            pose[0].rx, pose[0].ry, pose[0].rz,
        )
        events.append(("z", round(target_z, 3), lifting))

    def move_joint(target, **_kwargs):
        joints[0] = target
        events.append(("j6", round(target[5], 1)))

    node._current_command_pose = lambda: pose[0]
    node._move_to_user_xyz_with_rotation = rotate
    node._bottom_center_linear_z = move_z
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


def test_bottom_center_sequence_uses_relative_turn_and_original_absolute_grasp_z():
    node, grasp_pose, events = make_sequence_node(initial_j6=-20.0)
    motion = {"barcode_alignment_speed": 20, "barcode_alignment_acc": 20}

    node._execute_turntable_bottom_center_recovery(
        grasp_pose, 0.6, 0.057, 0.095, motion
    )

    assert events == [
        ("open", 0.6),
        ("ry", -45.0, 0.134),
        ("close", 0.134),
        ("confirm", 0.134),
        ("z", 0.294, True),
        ("j6", 160.0),
        ("z", 0.194, False),
        ("open", 0.6),
        ("ry", 45.0, 0.194),
        ("z", 0.134, False),
        ("close", 0.134),
        ("confirm", 0.134),
        ("safe_departure", 0.134),
    ]


def test_bottom_center_sequence_blocks_j6_limit_before_turn():
    node, grasp_pose, events = make_sequence_node(initial_j6=200.0)
    motion = {"barcode_alignment_speed": 20, "barcode_alignment_acc": 20}

    with pytest.raises(RuntimeError, match="J6 limit"):
        node._execute_turntable_bottom_center_recovery(
            grasp_pose, 0.6, 0.057, 0.095, motion
        )

    assert not any(event[0] == "j6" for event in events)
