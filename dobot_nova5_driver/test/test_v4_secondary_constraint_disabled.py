"""V4 may still observe 102 placement without enforcing the V5 safety gate."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v4 import (
    CosmeticBoxSingleArmNode,
)


def test_feedback_loss_does_not_stop_v4_when_only_auto_place_is_enabled():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.running = True
    node.cycle_enabled = True
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_protective_stop_latched = threading.Event()
    node.secondary_retreat_active = threading.Event()
    node.controller = SimpleNamespace(robot_mode=7)
    node.get_parameter = lambda name: SimpleNamespace(value={
        "secondary_motion_monitor_poll_s": 0.01,
        "secondary_collision_check_enabled": False,
        "turntable_auto_place_from_secondary_tcp": True,
    }[name])
    node._latch_secondary_protective_stop = MagicMock()

    def fail_feedback(**_kwargs):
        node.secondary_safety_shutdown.set()
        raise RuntimeError("102 feedback missing")

    node._read_secondary_y_clearance = fail_feedback
    node._secondary_motion_safety_loop()

    node._latch_secondary_protective_stop.assert_not_called()
