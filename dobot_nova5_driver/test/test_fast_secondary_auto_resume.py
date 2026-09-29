import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3_fast import (
    CosmeticBoxSingleArmNode,
)


def make_stopped_node(*, auto_resume=True):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.secondary_safety_lock = threading.Lock()
    node.secondary_protective_stop_latched = threading.Event()
    node.secondary_protective_stop_latched.set()
    node.secondary_retreat_active = threading.Event()
    node.secondary_retreat_attempted = threading.Event()
    node.secondary_recovery_clear_samples = 0
    node.secondary_auto_resume_requested = threading.Event()
    if auto_resume:
        node.secondary_auto_resume_requested.set()
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_resume_thread = None
    node.running = True
    node.cycle_enabled = False
    node.worker = None
    node.controller = SimpleNamespace(robot_mode=5)
    node._secondary_interlock_distances = lambda: (0.160, 0.142, 0.180)
    node._publish_status = MagicMock()
    node._schedule_continuous_restart_after_secondary_clearance = MagicMock()
    return node


def test_warning_stop_resumes_only_after_180mm_and_idle():
    node = make_stopped_node()

    assert not node._recover_secondary_protective_stop_if_clear(0.179, 5)
    assert not node._recover_secondary_protective_stop_if_clear(0.181, 7)
    assert not node._recover_secondary_protective_stop_if_clear(0.180, 5)
    assert node.secondary_protective_stop_latched.is_set()
    node._schedule_continuous_restart_after_secondary_clearance.assert_not_called()

    assert node._recover_secondary_protective_stop_if_clear(0.180, 5)
    assert not node.secondary_protective_stop_latched.is_set()
    assert node.secondary_auto_resume_requested.is_set()
    node._schedule_continuous_restart_after_secondary_clearance.assert_called_once_with()


def test_interrupted_continuous_motion_keeps_resume_request_until_clear():
    node = make_stopped_node(auto_resume=False)
    node.secondary_protective_stop_latched.clear()
    node.cycle_enabled = True
    node.worker = SimpleNamespace(is_alive=lambda: True)
    node.controller.stop_motion = MagicMock()
    node.get_logger = lambda: MagicMock()

    node._latch_secondary_protective_stop("Y gap below limit", robot_mode=7)

    assert not node.cycle_enabled
    assert node.secondary_protective_stop_latched.is_set()
    assert node.secondary_auto_resume_requested.is_set()
    node.controller.stop_motion.assert_called_once_with()
    assert not node._recover_secondary_protective_stop_if_clear(0.180, 5)
    assert node._recover_secondary_protective_stop_if_clear(0.181, 5)
    node._schedule_continuous_restart_after_secondary_clearance.assert_called_once_with()


def test_operator_stop_prevents_warning_stop_auto_resume():
    node = make_stopped_node()

    node.stop_continuous_cycle()

    assert not node.secondary_auto_resume_requested.is_set()
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    assert node._recover_secondary_protective_stop_if_clear(0.200, 5)
    assert not node.secondary_protective_stop_latched.is_set()
    node._schedule_continuous_restart_after_secondary_clearance.assert_not_called()
    assert not node.cycle_enabled


def test_recovered_warning_stop_starts_fresh_continuous_worker():
    node = make_stopped_node()
    node.secondary_protective_stop_latched.clear()
    node._ensure_worker = MagicMock()

    node._resume_continuous_after_secondary_clearance()

    assert node.cycle_enabled
    assert not node.secondary_auto_resume_requested.is_set()
    node._ensure_worker.assert_called_once_with()


@pytest.mark.parametrize(
    ("auto_resume", "expected_outcome"),
    [(True, "safety_pause"), (False, "fault")],
)
def test_cancelled_worker_reports_pause_only_when_restart_is_pending(
    auto_resume, expected_outcome
):
    node = make_stopped_node(auto_resume=auto_resume)
    node.cycle_enabled = True
    node.action_lock = threading.RLock()
    node._begin_cycle_timing = MagicMock()
    node._timed_stage = lambda _stage: nullcontext()
    node._wait_for_scanned_turntable_material = MagicMock(
        side_effect=RuntimeError("motion cancelled by Stop")
    )
    node._set_top_surface_barcode_window = MagicMock()
    node._finish_cycle_timing = MagicMock()
    logger = MagicMock()
    node.get_logger = lambda: logger

    node._cycle_worker()

    assert not node.cycle_enabled
    node._finish_cycle_timing.assert_called_once_with(expected_outcome)
    if auto_resume:
        logger.warning.assert_called_once()
        logger.fatal.assert_not_called()
    else:
        logger.fatal.assert_called_once()
