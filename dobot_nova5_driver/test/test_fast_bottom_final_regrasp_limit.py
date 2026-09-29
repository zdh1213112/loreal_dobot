import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3_fast import (
    CosmeticBoxSingleArmNode,
)


@pytest.mark.parametrize(
    ("shift_m", "accepted"),
    [(0.0542, True), (0.0601, False)],
)
def test_final_bottom_regrasp_has_its_own_60mm_limit(shift_m, accepted):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "bottom_center_tracking_timeout_s": 0.2,
        "pregrasp_pose_max_age_s": 0.65,
        "pregrasp_max_correction_m": 0.050,
        "bottom_center_final_regrasp_max_xy_shift_m": 0.060,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _label: None
    node._publish_status = lambda _message: None
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 0
    node.latest_pregrasp_pose = None
    node.latest_pregrasp_pose_received_at = 0.0
    node._current_command_pose = lambda: TcpPose(0.5, -0.1, 0.17, 0, 0, 0)
    target = TcpPose(0.5 + shift_m, -0.1, 0.13, 0, 0, 0)

    def publish_fresh_pose():
        time.sleep(0.01)
        with node.data_lock:
            node.pregrasp_pose_count += 1
            node.latest_pregrasp_pose = target
            node.latest_pregrasp_pose_received_at = time.monotonic()

    publisher = threading.Thread(target=publish_fresh_pose)
    publisher.start()
    try:
        if accepted:
            assert node._wait_for_bottom_center_tracked_pose() == target
        else:
            with pytest.raises(RuntimeError, match="limit=60.0mm"):
                node._wait_for_bottom_center_tracked_pose()
    finally:
        publisher.join()


def test_long_box_shifts_before_first_tool_rx_only_above_150mm():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "bottom_center_long_box_length_threshold_m": 0.150,
        "bottom_center_gripper_cavity_half_length_m": 0.080,
        "bottom_center_long_box_offset_margin_m": 0.010,
        "bottom_center_first_tool_rx_delta_deg": 70.0,
        "user_index": 0,
        "command_tool_index": 1,
        "dh_timeout_s": 3.0,
        "jog_tolerance_m": 0.002,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _label: None
    node._cycle_cancel_requested = lambda: False
    node._publish_status = lambda _message: None
    events = []
    pose = [TcpPose(0.50, -0.10, 0.134, 180.0, 0.0, 180.0)]

    @contextmanager
    def stage(name):
        events.append(name)
        yield

    def move_linear_tcp(target, **_kwargs):
        pose[0] = target
        events.append("shift")

    node._timed_stage = stage
    node._current_command_pose = lambda: pose[0]
    node.controller = SimpleNamespace(
        current_joint=lambda: [0.0] * 6,
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        move_linear_tcp=move_linear_tcp,
    )
    node.gripper = SimpleNamespace(
        read_position=lambda: 0.6,
        set_position=lambda value, **_kwargs: events.append(("open", value)),
        wait_until_stopped=lambda **_kwargs: None,
    )
    motion = {"linear_speed": 20, "linear_acc": 20}

    assert node._offset_bottom_long_box_before_first_tool_rx(0.150, motion) == 0.0
    assert events == []
    assert node._offset_bottom_long_box_before_first_tool_rx(0.200, motion) == pytest.approx(0.030)
    assert (pose[0].x, pose[0].y) == pytest.approx((0.50, -0.070))
    assert events == [
        "bottom_center_long_box_full_open",
        ("open", 1.0),
        "bottom_center_long_box_short_end_shift",
        "shift",
    ]
