import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v4 import (
    CosmeticBoxSingleArmNode,
)


@pytest.mark.parametrize(
    ("turntable_enabled", "early_top_enabled"),
    [(True, False), (False, True)],
)
def test_fresh_d405_request_when_early_top_is_not_applicable(
    turntable_enabled, early_top_enabled
):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.action_lock = threading.RLock()
    node.turntable_lock = threading.RLock()
    node.running = True
    node.cycle_enabled = True
    values = {
        "turntable_enabled": turntable_enabled,
        "top_surface_barcode_enabled": True,
        "d405_pretracking_enabled": False,
        "d405_early_top_barcode_enabled": early_top_enabled,
        "vision_retry_delay_s": 0.1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._begin_cycle_timing = MagicMock()
    node._finish_cycle_timing = MagicMock()
    node._timed_stage = lambda _name: nullcontext()
    node._publish_status = MagicMock()
    node._wait_for_secondary_y_clearance = MagicMock()
    node._move_startup_and_open = MagicMock()
    node._set_d405_pretracking = MagicMock()
    node._wait_for_scanned_turntable_material = lambda: (13, "barcode_detected", "")
    node._require_current_material_event = MagicMock()
    node._wait_for_pretracked_target_after_stop = MagicMock()

    def request_fresh_detection():
        node.cycle_enabled = False
        return None

    node._request_vision_target = MagicMock(side_effect=request_fresh_detection)

    node._cycle_worker()

    node._request_vision_target.assert_called_once_with()
    node._wait_for_pretracked_target_after_stop.assert_not_called()
    node._set_d405_pretracking.assert_not_called()
