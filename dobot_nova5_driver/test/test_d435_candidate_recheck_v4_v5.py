"""Fake-clock turntable checks: no hardware, ROS spinning or camera startup."""

import importlib
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dobot_nova5_driver import turntable_barcode_recheck as recheck


@pytest.fixture(params=[4, 5], ids=['V4', 'V5'])
def rig(request, monkeypatch):
    module = importlib.import_module(
        f'dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v{request.param}')
    clock = SimpleNamespace(now=100., on_sleep=lambda: None)
    clock.monotonic = lambda: clock.now
    clock.time = lambda: clock.now + 900.

    def sleep(seconds):
        clock.now += seconds
        clock.on_sleep()

    clock.sleep = sleep
    monkeypatch.setattr(module, 'time', clock)
    monkeypatch.setattr(recheck, 'time', clock)
    node = module.CosmeticBoxSingleArmNode.__new__(module.CosmeticBoxSingleArmNode)
    node.turntable_lock = threading.RLock()
    node.turntable_scan_cancel = threading.Event()
    node.turntable_state = 'RUNNING'
    node.turntable_barcode_window_active = True
    node.turntable_barcode_value = ''
    node.turntable_barcode_result_count = 0
    node.turntable_candidate = None
    node.turntable_candidate_since = 0.
    node.turntable_candidate_window_epoch = 999.
    node.turntable_candidate_last_frame_epoch = 0.
    node.turntable_candidate_rechecks = 0
    node.early_top_at_startup = False
    node.running, node.shutting_down = True, False
    node._publish_status = Mock()
    node._require_cycle_active = Mock()
    node._wait_for_secondary_y_clearance = Mock()
    node._wait_for_turntable_camera_ready = Mock()
    node._set_turntable_barcode_window = Mock()
    node._set_turntable_barcode_window.side_effect = lambda active: setattr(
        node, 'turntable_barcode_window_active', active)
    node._set_top_surface_barcode_window = Mock()
    node._set_turntable_barcode_window(False)
    node._notify_d405_post_stop_revalidate = Mock()
    node.toggles = []

    def toggle(expected, result, reason):
        assert node.turntable_state == expected
        node.toggles.append(reason)
        clock.now += .4 if result == 'STOPPED' else .1
        node.turntable_state = result
        return clock.now

    node._toggle_turntable = toggle
    params = dict(turntable_candidate_max_age_s=.4, turntable_candidate_grace_s=.12,
                  turntable_candidate_max_rechecks=2, turntable_candidate_recheck_s=.35,
                  turntable_enabled=True, turntable_stationary_barcode_check_s=0.,
                  turntable_scan_timeout_s=.5, turntable_mid_scan_restart_s=0.,
                  turntable_settle_s=0., d405_early_top_barcode_enabled=False)
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])

    def proposal(**kwargs):
        node.turntable_barcode_window_active = True
        payload = dict(pending=True, captured_at=clock.time(), generation=1,
                       source='scanner_unverified', rect=[20, 20, 120, 80])
        payload.update(kwargs)
        node._turntable_barcode_candidate_callback(SimpleNamespace(data=json.dumps(payload)))

    return SimpleNamespace(node=node, clock=clock, params=params, proposal=proposal)


def test_fresh_candidate_has_grace_for_the_normal_confirmation_path(rig):
    rig.proposal()
    assert not rig.node._recheck_turntable_barcode_candidate(
        require_cycle_active=True, top_reader=lambda: '')
    assert rig.node.toggles == []


def test_weak_candidate_is_rechecked_then_remaining_search_resumes(rig):
    rig.proposal()
    rig.clock.now += .13
    assert rig.node._recheck_turntable_barcode_candidate(
        require_cycle_active=True, top_reader=lambda: '')
    assert rig.node.turntable_state == 'RUNNING'
    assert rig.node.toggles == ['candidate recheck stop', 'candidate recheck resume']
    assert rig.node.turntable_barcode_value == ''
    rig.node._wait_for_secondary_y_clearance.assert_called_once()


@pytest.mark.parametrize('source', ['side', 'top'])
def test_confirmation_during_recheck_leaves_table_stopped(rig, source):
    rig.proposal()
    rig.clock.now += .13
    top = ['']

    def confirm():
        if source == 'side':
            rig.node.turntable_barcode_value = 'barcode_detected'
        else:
            top[0] = 'barcode_detected'

    rig.clock.on_sleep = confirm
    assert rig.node._recheck_turntable_barcode_candidate(
        require_cycle_active=True, top_reader=lambda: top[0])
    assert rig.node.turntable_state == 'STOPPED'
    assert rig.node.toggles == ['candidate recheck stop']


@pytest.mark.parametrize('failure', ['operator_stop', 'new_material', 'shutdown'])
def test_cancellation_during_recheck_never_restarts_table(rig, failure):
    rig.proposal()
    rig.clock.now += .13

    def cancel():
        if failure == 'shutdown':
            rig.node.shutting_down = True
        else:
            rig.node.turntable_scan_cancel.set()

    rig.clock.on_sleep = cancel
    with pytest.raises(RuntimeError):
        rig.node._recheck_turntable_barcode_candidate(require_cycle_active=True, top_reader=lambda: '')
    assert rig.node.turntable_state == 'STOPPED'
    assert rig.node.toggles == ['candidate recheck stop']


def test_stale_old_window_and_malformed_candidates_cannot_pause(rig):
    rig.proposal(captured_at=998.)
    assert rig.node.turntable_candidate is None
    rig.proposal(captured_at=rig.clock.time() + 3)
    assert rig.node.turntable_candidate is None
    rig.proposal(rect=[0, 0, -10, 20])
    assert rig.node.turntable_candidate is None
    rig.proposal()
    rig.clock.now += .5
    assert not rig.node._recheck_turntable_barcode_candidate(
        require_cycle_active=True, top_reader=lambda: '')


def test_brief_missing_frame_keeps_hint_but_roi_invalidation_discards_it(rig):
    rig.proposal()
    original = rig.node.turntable_candidate
    rig.clock.now += .03
    rig.proposal(pending=False)
    assert rig.node.turntable_candidate is original
    rig.proposal(pending=False, invalidated=True)
    assert rig.node.turntable_candidate is None


def test_recheck_limit_and_disabled_option_leave_rotation_running(rig):
    rig.proposal()
    rig.clock.now += .13
    rig.params['turntable_candidate_max_rechecks'] = 0
    assert not rig.node._recheck_turntable_barcode_candidate(
        require_cycle_active=True, top_reader=lambda: '')
    rig.params['turntable_candidate_max_rechecks'] = 2
    rig.node.turntable_candidate_rechecks = 2
    assert not rig.node._recheck_turntable_barcode_candidate(
        require_cycle_active=True, top_reader=lambda: '')
    assert rig.node.toggles == []


@pytest.mark.parametrize('midpoint', [False, True])
def test_full_scan_excludes_candidate_pause_from_rotation_budget(rig, midpoint):
    node, clock = rig.node, rig.clock
    node.turntable_state = 'STOPPED'
    started = clock.now
    if midpoint:
        rig.params['turntable_mid_scan_restart_s'] = .2
        rig.params['turntable_mid_scan_low_hold_ms'] = 100

        def midpoint_restart():
            node.toggles.append('midpoint restart')
            clock.now += .3

        node._restart_turntable_back_to_back = midpoint_restart

    def propose_while_rotating():
        if node.turntable_state == 'RUNNING':
            rig.proposal()

    clock.on_sleep = propose_while_rotating
    assert node._scan_turntable_for_side_barcode(require_cycle_active=False) == ''
    assert node.turntable_candidate_rechecks == 2
    assert clock.now - started >= .1 + .5 + 2 * (.4 + .35 + .1) + .4 + (.3 if midpoint else 0)
    assert node.turntable_state == 'STOPPED'
    node._notify_d405_post_stop_revalidate.assert_called_once()


def test_full_scan_success_during_candidate_pause_notifies_post_stop_d405(rig):
    node = rig.node
    node.turntable_state = 'STOPPED'

    def deliver():
        if node.turntable_state == 'RUNNING':
            rig.proposal()
        elif node.turntable_state == 'STOPPED' and node.turntable_candidate_rechecks:
            node.turntable_barcode_value = 'barcode_detected'

    rig.clock.on_sleep = deliver
    assert node._scan_turntable_for_side_barcode(require_cycle_active=False) == 'barcode_detected'
    assert node.toggles == ['start', 'candidate recheck stop']
    assert node.turntable_state == 'STOPPED'
    node._notify_d405_post_stop_revalidate.assert_called_once()
