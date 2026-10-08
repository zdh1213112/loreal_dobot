import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dobot_nova5_driver.controller_v3 import TcpPose
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v4 import (
    CosmeticBoxSingleArmNode,
)


def test_178mm_bottom_box_shifts_39mm_toward_nearest_short_end():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "bottom_center_long_box_length_threshold_m": 0.150,
        "bottom_center_gripper_cavity_half_length_m": 0.080,
        "bottom_center_long_box_offset_margin_m": 0.030,
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
    node._timed_stage = lambda _name: nullcontext()
    initial = TcpPose(0.50, -0.10, 0.134, 180.0, 0.0, 180.0)
    current = [initial]
    node._current_command_pose = lambda: current[0]
    moves = []

    def move_linear_tcp(target, **_kwargs):
        moves.append(target)
        current[0] = target

    node.controller = SimpleNamespace(
        current_joint=lambda: [0.0] * 6,
        inverse_kinematics=lambda *_args, **_kwargs: [0.0] * 6,
        move_linear_tcp=move_linear_tcp,
    )
    openings = []
    node.gripper = SimpleNamespace(
        read_position=lambda: 0.6,
        set_position=lambda value, **_kwargs: openings.append(value),
        wait_until_stopped=lambda **_kwargs: None,
    )
    motion = {"linear_speed": 20, "linear_acc": 20}

    assert node._offset_bottom_long_box_before_first_tool_rx(0.150, motion) == 0.0
    assert moves == []

    offset = node._offset_bottom_long_box_before_first_tool_rx(0.178, motion)

    assert offset == pytest.approx(0.039)
    assert openings == [1.0]
    assert len(moves) == 1
    assert math.hypot(moves[0].x - initial.x, moves[0].y - initial.y) == pytest.approx(0.039)
    assert math.hypot(moves[0].x, moves[0].y) < math.hypot(initial.x, initial.y)
    assert moves[0].z == initial.z
