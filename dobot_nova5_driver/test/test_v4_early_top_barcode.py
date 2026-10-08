import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v4 import (
    CosmeticBoxSingleArmNode,
)


@pytest.mark.parametrize("top_found_at", ["stopped", "rotating"])
def test_early_top_barcode_stops_turntable_and_preserves_top_result(top_found_at):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "turntable_enabled": True,
        "turntable_stationary_barcode_check_s": 0.025,
        "turntable_scan_timeout_s": 0.20,
        "turntable_mid_scan_restart_s": 0.10,
        "turntable_settle_s": 0.0,
        "d405_early_top_barcode_enabled": True,
        "top_surface_barcode_enabled": True,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.turntable_lock = threading.RLock()
    node.turntable_scan_cancel = threading.Event()
    node.turntable_state = "STOPPED"
    node.turntable_barcode_value = ""
    node.early_top_at_startup = True
    node.turntable_early_top_barcode = ""
    node.running = True
    node.shutting_down = False
    node._wait_for_secondary_y_clearance = lambda *_args, **_kwargs: None
    node._wait_for_turntable_camera_ready = lambda **_kwargs: None
    node._set_turntable_barcode_window = lambda _enabled: None
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    node._notify_d405_post_stop_revalidate = MagicMock()
    top_window = []
    toggles = []
    observed_top = ""

    def set_top_window(enabled):
        nonlocal observed_top
        top_window.append(enabled)
        if enabled and top_found_at == "stopped":
            observed_top = "barcode_detected"

    def toggle(expected, result, purpose):
        nonlocal observed_top
        assert node.turntable_state == expected
        node.turntable_state = result
        toggles.append(purpose)
        if purpose == "start" and top_found_at == "rotating":
            observed_top = "barcode_detected"
        return time.monotonic()

    node._set_top_surface_barcode_window = set_top_window
    node._current_top_surface_barcode = lambda: observed_top
    node._toggle_turntable = toggle
    node._restart_turntable_back_to_back = lambda: pytest.fail("scan continued after top hit")

    side_result = node._scan_turntable_for_side_barcode(require_cycle_active=False)

    assert side_result == ""
    assert node.turntable_early_top_barcode == "barcode_detected"
    assert top_window == [True, False]
    assert toggles == ([] if top_found_at == "stopped" else ["start", "stop"])
    node._notify_d405_post_stop_revalidate.assert_called_once_with()


def test_early_top_mode_can_enable_d405_standby_without_old_pretracking_option():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "turntable_enabled": True,
        "top_surface_barcode_enabled": True,
        "d405_pretracking_enabled": False,
        "d405_early_top_barcode_enabled": True,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.d405_pretracking_publisher = SimpleNamespace(publish=MagicMock())
    node._publish_status = lambda _message: None

    node._set_d405_pretracking(True)

    assert node.d405_pretracking_armed is True
    assert node.d405_pretracking_publisher.publish.call_args.args[0].data is True


def test_single_cycle_arms_early_top_at_startup_then_uses_post_stop_target():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.worker = None
    node.running = True
    node.shutting_down = False
    node.cycle_enabled = False
    node.action_lock = threading.RLock()
    node.turntable_lock = threading.RLock()
    node.secondary_safety_lock = threading.RLock()
    node.secondary_protective_stop_latched = threading.Event()
    node.secondary_auto_resume_requested = threading.Event()
    values = {
        "turntable_enabled": True,
        "top_surface_barcode_enabled": True,
        "d405_early_top_barcode_enabled": True,
        "single_cycle_grasp_retry_limit": 0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._publish_status = lambda _message: None
    node._timed_stage = lambda _name: nullcontext()
    node._begin_cycle_timing = MagicMock()
    node._finish_cycle_timing = MagicMock()
    node._wait_for_secondary_y_clearance = lambda _label: None
    node._require_current_material_event = lambda *_args: None
    node._move_startup_and_open = MagicMock()
    node._set_d405_pretracking = MagicMock()
    node._request_vision_target = MagicMock()
    node._set_top_surface_barcode_window = MagicMock()
    node.last_accepted_aspect_ratio = 2.0
    target = TcpPose(0.50, -0.18, 0.19, 180.0, 0.0, 0.0)
    node._wait_for_pretracked_target_after_stop = MagicMock(
        return_value=(target, 0.07, 0.05, 0.18)
    )
    node._execute_one_cycle = MagicMock()

    def wait_for_material():
        node._move_startup_and_open.assert_called_once()
        node._set_d405_pretracking.assert_called_with(True)
        assert node.early_top_at_startup is True
        return 13, "", "barcode_detected"

    node._wait_for_scanned_turntable_material = wait_for_material

    node.execute_single_cycle()

    node._request_vision_target.assert_not_called()
    node._wait_for_pretracked_target_after_stop.assert_called_once_with(
        require_cycle_enabled=True
    )
    assert node._execute_one_cycle.call_args.kwargs["turntable_top_barcode"] == "barcode_detected"
    assert node.early_top_at_startup is False
    assert node._set_d405_pretracking.call_args_list[-1].args == (False,)


def test_preconfirmed_top_goes_to_center_hover_and_top_placement_without_offset():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "turntable_enabled": True,
        "near_square_grasp_aspect_ratio": 1.3,
        "offset_grasp_enabled": True,
        "grasp_z_offset_m": 0.0,
        "grasp_z_offset_limit_m": 0.01,
        "minimum_safe_tcp_z_m": 0.10,
        "turntable_height_safety_enabled": False,
        "dh_max_opening_m": 0.095,
        "dh_timeout_s": 3.0,
        "user_index": 0,
        "command_tool_index": 1,
        "dh_grasp_force": 50,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._motion_profile = lambda: {"linear_speed": 50, "linear_acc": 50}
    node._level_turntable_grasp_pose = lambda pose: pose
    node._apply_grasp_z_safety = lambda pose, *_args: pose
    node._publish_status = lambda _message: None
    node._timed_stage = lambda _name: nullcontext()
    node._require_cycle_active = lambda _label: None
    node._wait_for_secondary_y_clearance = lambda _label: None
    node._require_turntable_top_down = lambda *_args: None
    node._set_top_surface_barcode_window = MagicMock()
    node._current_top_surface_barcode = MagicMock()
    node._execute_offset_entry = MagicMock(side_effect=AssertionError("offset entered"))
    node._place_as_top_barcode_box = MagicMock()
    node._confirm_grasp_before_lift = MagicMock()
    node._revalidate_target_at_hover = lambda target, *_args, **_kwargs: target
    node._cycle_cancel_requested = lambda: False
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 0
    current = [TcpPose(0.50, -0.18, 0.30, 180.0, 0.0, 0.0)]
    node._current_command_pose = lambda: current[0]
    node.controller = SimpleNamespace(
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        current_joint=lambda: [0.0] * 6,
        move_linear_tcp=lambda pose, **_kwargs: current.__setitem__(0, pose),
    )
    node._move_turntable_center_hover = MagicMock(
        side_effect=lambda pose, *_args: current.__setitem__(
            0, TcpPose(pose.x, pose.y, 0.30, pose.rx, pose.ry, pose.rz)
        )
    )
    node.gripper = SimpleNamespace(
        read_position=lambda: 0.5,
        set_position=lambda *_args, **_kwargs: None,
        wait_until_stopped=lambda **_kwargs: None,
        set_force=lambda *_args: None,
        close=lambda **_kwargs: None,
    )
    target = TcpPose(0.50, -0.18, 0.19, 180.0, 0.0, 0.0)

    node._execute_one_cycle(
        target, 0.07, 0.05, 0.18,
        aspect_ratio=2.4,
        turntable_top_barcode="barcode_detected",
    )

    node._execute_offset_entry.assert_not_called()
    node._move_turntable_center_hover.assert_called_once()
    node._current_top_surface_barcode.assert_not_called()
    node._place_as_top_barcode_box.assert_called_once_with(
        turntable_box_height_m=0.05
    )
