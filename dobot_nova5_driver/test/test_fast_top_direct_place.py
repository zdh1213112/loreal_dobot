import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3_fast import (
    CosmeticBoxSingleArmNode,
)


@contextmanager
def stage(_name):
    yield


@pytest.mark.parametrize(
    ("box_height_m", "clearance_z_m"),
    [(0.038, 0.260), (0.120, 0.287)],
)
def test_confirmed_top_barcode_clears_table_then_rises_toward_place(
    box_height_m, clearance_z_m
):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "turntable_enabled": True,
        "offset_grasp_enabled": True,
        "grasp_z_offset_m": 0.0,
        "grasp_z_offset_limit_m": 0.010,
        "minimum_safe_tcp_z_m": 0.100,
        "turntable_height_safety_enabled": False,
        "turntable_surface_z_m": 0.164,
        "turntable_surface_clearance_m": 0.003,
        "turntable_tcp_below_target_m": 0.0,
        "fast_side_camera_clearance_z_m": 0.260,
        "fast_side_rise_after_camera_clearance_m": 0.005,
        "grasp_lift_m": 0.060,
        "dh_max_opening_m": 0.095,
        "dh_grasp_force": 50,
        "place_release_clearance_m": 0.005,
        "dh_timeout_s": 3.0,
        "user_index": 0,
        "command_tool_index": 1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._motion_profile = lambda: {}
    node._level_turntable_grasp_pose = lambda pose: pose
    node._apply_grasp_z_safety = lambda pose, *_args: pose
    node._publish_status = lambda _message: None
    node._timed_stage = stage
    node._require_cycle_active = lambda _label: None
    node._wait_for_secondary_y_clearance = lambda _label: None
    node._set_top_surface_barcode_window = lambda _active: None
    node._current_top_surface_barcode = lambda: "barcode_detected"
    node._execute_offset_entry = lambda *_args, **_kwargs: None
    node._top_surface_barcode_place_xyz = lambda: [0.531, 0.328, 0.105]
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 0
    events = []
    grasp_pose = TcpPose(0.56, -0.15, 0.134, 180.0, 0.0, 180.0)
    node._current_command_pose = lambda: grasp_pose
    node.controller = SimpleNamespace(
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        current_joint=lambda: [0.0] * 6,
    )
    node._confirm_grasp_before_lift = lambda *_args: events.append("confirmed")
    node._move_to_user_xyz_with_rotation = (
        lambda xyz, **kwargs: events.append(("movl", xyz, kwargs))
    )
    node._mark_turntable_material_removed = lambda: events.append("removed")
    node._validate_grasp_feedback = lambda *_args: events.append("place_checked")
    node._execute_turntable_safe_departure_lift = (
        lambda *_args: events.append("safe_lift")
    )
    node.gripper = SimpleNamespace(
        read_position=lambda: 0.6,
        set_position=lambda value, **_kwargs: events.append(("open", value)),
        set_force=lambda *_args: None,
        close=lambda **_kwargs: events.append("closed"),
        wait_until_stopped=lambda **_kwargs: None,
    )

    node._execute_one_cycle(grasp_pose, 0.055, box_height_m, 0.121)

    assert [event[0] if isinstance(event, tuple) else event for event in events[1:8]] == [
        "closed", "confirmed", "movl", "movl", "removed", "movl", "place_checked",
    ]
    assert events[3][1] == pytest.approx(
        [grasp_pose.x, grasp_pose.y, clearance_z_m]
    )
    assert events[4][1] == pytest.approx(
        [0.531, 0.328, clearance_z_m + 0.005]
    )
    assert events[6][1] == pytest.approx([0.531, 0.328, 0.105])
    for event in (events[3], events[4], events[6]):
        assert event[2] == {
            "ry_delta_deg": 0.0,
            "rz_delta_deg": 0.0,
            "linear_tcp": True,
        }
    assert events[-1][0] == "open"
    assert "safe_lift" not in events
