"""Safety checks for the D405 loss recovery after a bottom-face release."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dobot_nova5_driver import nova5_cosmetic_box_single_arm_cycle_v4 as v4
from dobot_nova5_driver import nova5_cosmetic_box_single_arm_cycle_v5 as v5


@pytest.mark.parametrize("module", [v4, v5])
def test_bottom_regrasp_target_timeout_has_its_own_recovery_signal(module):
    node = module.CosmeticBoxSingleArmNode.__new__(module.CosmeticBoxSingleArmNode)
    node.data_lock = threading.RLock()
    node.pregrasp_pose_count = 0
    node.latest_pregrasp_pose = None
    node.latest_pregrasp_pose_received_at = 0.0
    node.get_parameter = lambda name: SimpleNamespace(value={
        "bottom_center_tracking_timeout_s": 0.1,
        "pregrasp_pose_max_age_s": 0.1,
    }[name])
    node._require_cycle_active = MagicMock()

    with pytest.raises(module.BottomRegraspTargetLost, match="No fresh D405"):
        node._wait_for_bottom_center_tracked_pose()


@pytest.mark.parametrize("module", [v4, v5])
def test_synthetic_place_done_cannot_replace_a_real_new_material_event(module):
    node = module.CosmeticBoxSingleArmNode.__new__(module.CosmeticBoxSingleArmNode)
    node.turntable_condition = threading.Condition()
    node.turntable_place_done_count = 12
    node.turntable_scan_in_progress = False
    node.turntable_state = "STOPPED"

    result = node._accept_turntable_place_done(
        "automatic bottom-regrasp D405-loss retry",
        expected_current_event_number=11,
    )

    assert result is None
    assert node.turntable_place_done_count == 12
    assert node.turntable_scan_in_progress is False


@pytest.mark.parametrize("module", [v4, v5])
def test_bottom_regrasp_recovery_returns_home_before_rescan(module):
    node = module.CosmeticBoxSingleArmNode.__new__(module.CosmeticBoxSingleArmNode)
    events = []
    node.turntable_condition = threading.Condition()
    node.turntable_place_done_count = 9
    node._publish_status = MagicMock()
    node._wait_for_secondary_y_clearance = lambda *args, **kwargs: events.append("clear")
    node._timed_stage = lambda _stage: nullcontext()
    node._prepare_robot_for_startup_recovery = lambda: events.append("prepare")
    node.get_parameter = lambda _name: SimpleNamespace(value=0.260)
    node._current_command_pose = lambda: SimpleNamespace(z=0.220)
    node._motion_profile = lambda: {}
    node._bottom_center_linear_z = lambda *args, **kwargs: events.append("vertical")
    node._move_startup_and_open = lambda **kwargs: events.append("home")
    node._release_active_material_event = lambda _event: events.append("release_event")
    node._stop_turntable_if_running = lambda _reason: events.append("stop_table")
    node._accept_turntable_place_done = lambda _source, **kwargs: (
        events.append("rescan") or 10
    )
    if module is v5:
        node._safe_startup_return = lambda: nullcontext()
        node._set_secondary_pick_guard = lambda _active: events.append("release_guard")

    assert node._recover_bottom_regrasp_target_loss(9) == 10
    assert events.index("vertical") < events.index("home")
    assert events.index("home") < events.index("rescan")
    if module is v5:
        assert events.index("home") < events.index("release_guard")


@pytest.mark.parametrize("module", [v4, v5])
def test_exhausted_retry_returns_home_without_starting_another_scan(module):
    node = module.CosmeticBoxSingleArmNode.__new__(module.CosmeticBoxSingleArmNode)
    node.turntable_condition = threading.Condition()
    node.turntable_place_done_count = 9
    node._publish_status = MagicMock()
    node._wait_for_secondary_y_clearance = MagicMock()
    node._timed_stage = lambda _stage: nullcontext()
    node._prepare_robot_for_startup_recovery = MagicMock()
    node.get_parameter = lambda _name: SimpleNamespace(value=0.260)
    node._current_command_pose = lambda: SimpleNamespace(z=0.220)
    node._motion_profile = lambda: {}
    node._bottom_center_linear_z = MagicMock()
    node._move_startup_and_open = MagicMock()
    node._release_active_material_event = MagicMock()
    node._stop_turntable_if_running = MagicMock()
    node._accept_turntable_place_done = MagicMock()
    if module is v5:
        node._safe_startup_return = lambda: nullcontext()
        node._set_secondary_pick_guard = MagicMock()

    assert node._recover_bottom_regrasp_target_loss(9, rescan=False) is None
    node._move_startup_and_open.assert_called_once_with(require_cycle_active=True)
    node._accept_turntable_place_done.assert_not_called()
