"""Offline regressions for stale retry targets and short eye-in-hand tracking gaps."""

import ast
import importlib
import logging
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.now

    def time(self):
        return self.now + 900.0

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        self.on_sleep()


@pytest.fixture(params=[4, 5], ids=['V4', 'V5'])
def rig(request, monkeypatch):
    version = request.param
    mod = importlib.import_module(
        f'dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v{version}'
    )
    clock = Clock()
    monkeypatch.setattr(mod, 'time', clock)
    node = mod.CosmeticBoxSingleArmNode.__new__(mod.CosmeticBoxSingleArmNode)
    node.data_lock = threading.Lock()
    node.running = node.cycle_enabled = True
    node.pregrasp_pose_count = node.pose_count = 0
    node.pose_samples = deque(maxlen=20)
    node.pregrasp_observations = deque(maxlen=40)
    node.handoff_state_count = 1
    node.latest_handoff_state = 'CLEAR'
    node.latest_handoff_clear = True
    node.last_turntable_stop_monotonic = 90.0
    node.last_turntable_stop_epoch = 990.0
    node.last_turntable_stop_pregrasp_count = 0
    node.last_turntable_stop_handoff_count = 0
    node.d405_reacquire_request_epoch = 0.0
    node.d405_reacquire_applied_epoch = 0.0
    node.d405_reacquire_capture_epoch = 0.0
    node.d405_reacquire_pregrasp_count = 0
    node.d405_reacquire_handoff_count = 0
    node.latest_width_m = .08
    node.latest_length_m = .144
    node.latest_height_m = .049
    node.latest_aspect_ratio = 3.0
    node.vision_result_count = 0
    node.latest_vision_result = ''
    node.width_count = node.length_count = node.height_count = node.aspect_ratio_count = 0
    values = dict(
        pregrasp_pose_max_age_s=.65,
        pregrasp_fresh_wait_timeout_s=.6,
        pregrasp_min_hover_clearance_m=.02,
        d405_post_stop_samples=2,
        d405_post_stop_wait_timeout_s=.4,
        d405_pretracking_fallback_request=False,
        vision_samples=2, vision_timeout_s=.4,
        vision_position_stability_m=.01,
        vision_angle_stability_deg=10.,
        min_box_height_m=.005, max_box_height_m=.15,
        min_box_length_m=.02, max_box_length_m=.3,
        vision_pose_max_time_skew_s=.1,
        turntable_enabled=True, top_surface_barcode_enabled=True,
        d405_pretracking_enabled=False, d405_early_top_barcode_enabled=True,
    )
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = MagicMock()
    node._publish_status = MagicMock()
    node._require_cycle_active = MagicMock()
    node.d405_pretracking_publisher = MagicMock()
    node.d405_post_stop_revalidate_publisher = MagicMock()
    node.trigger_publisher = MagicMock()
    node.controller = MagicMock()
    pose = mod.TcpPose(.52, -.114, .19, 180., 0., 0.)

    def sample(age=0., frame_age=None):
        node.pregrasp_pose_count += 1
        item = mod.PregraspObservation(
            count=node.pregrasp_pose_count, pose=pose,
            received_at_monotonic=clock.monotonic() - age,
            frame_time_s=clock.time() - (age if frame_age is None else frame_age),
            tcp_linear_speed_mps=0.,
        )
        node.pregrasp_observations.append(item)
        return item

    def ack(request_epoch=None, capture_epoch=None):
        node._vision_result_callback(SimpleNamespace(data=(
            f'acquisition_started:{clock.time() if request_epoch is None else request_epoch:.9f}:'
            f'{clock.time() if capture_epoch is None else capture_epoch:.9f}'
        )))

    return SimpleNamespace(node=node, mod=mod, clock=clock, values=values,
                           pose=pose, sample=sample, ack=ack, version=version)


@pytest.mark.parametrize('receipt_age,frame_age', [(2.618, 2.618), (.01, 2.618)])
def test_post_stop_rejects_old_cache_even_if_after_original_stop(rig, receipt_age, frame_age):
    rig.sample(receipt_age, frame_age)
    rig.sample(receipt_age, frame_age)
    assert rig.node._wait_for_pretracked_target_after_stop() is None
    assert not hasattr(rig.node, 'last_accepted_target')


def test_post_stop_fresh_fast_path_has_no_added_wait(rig):
    rig.sample(.02)
    rig.sample(.01)
    assert rig.node._wait_for_pretracked_target_after_stop()[0] == rig.pose
    assert rig.clock.sleeps == []


def test_retry_waits_for_camera_reset_and_new_samples_before_departure(rig):
    rig.sample(.02)
    rig.sample(.01)
    rig.node._set_d405_pretracking(True)
    request_epoch = rig.clock.time()
    # Even newly delivered frames cannot authorize a departure before reset ACK.
    rig.sample()
    rig.sample()
    assert not rig.node._d405_frame_after_reacquire(request_epoch + .01)
    assert rig.node.d405_pretracking_publisher.publish.call_args.args[0].data is True
    sent = False

    def deliver():
        nonlocal sent
        if rig.clock.now >= 100.1 and not sent:
            sent = True
            rig.ack(request_epoch + .001, rig.clock.time() - .003)
            rig.node.handoff_state_count += 1
            rig.sample(.002)
            rig.sample(.001)

    rig.clock.on_sleep = deliver
    assert rig.node._wait_for_pretracked_target_after_stop()[0] == rig.pose
    assert rig.clock.now >= 100.1
    assert rig.node.vision_result_count == 0  # ACK is not an empty/success result.


def test_delayed_old_ack_does_not_satisfy_new_retry(rig):
    rig.ack(999., 999.1)
    rig.node._mark_d405_reacquisition()
    rig.ack(999., 999.9)
    rig.sample()
    rig.sample()
    assert rig.node._wait_for_pretracked_target_after_stop() is None


def test_handoff_must_be_fresh_and_clear_after_rearm(rig):
    rig.node._set_d405_pretracking(True)
    rig.ack()
    rig.clock.now += .01
    rig.sample(.002)
    rig.sample(.001)
    # Old CLEAR state is not sufficient after a new camera session.
    assert rig.node._wait_for_pretracked_target_after_stop() is None


@pytest.mark.parametrize('callback', ['_vision_pose_callback', '_pregrasp_pose_callback'])
def test_reset_during_pose_transform_rejects_inflight_old_pose(rig, callback):
    stamp = SimpleNamespace(sec=1000, nanosec=0)
    msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp))

    def transform(*_args):
        rig.clock.now += .01
        rig.node._mark_d405_reacquisition()
        return rig.pose

    rig.node._transform_vision_pose = transform
    getattr(rig.node, callback)(msg)
    assert rig.node.pose_count == rig.node.pregrasp_pose_count == 0


def test_explicit_request_cannot_bypass_pending_camera_reset(rig):
    rig.node._transform_vision_pose = lambda *_args: rig.pose
    sent = False

    def publish_stable():
        for _ in range(2):
            seconds = rig.clock.time()
            msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
                sec=int(seconds), nanosec=int((seconds-int(seconds))*1e9))))
            rig.node._vision_pose_callback(msg)
        rig.node.width_count += 1
        rig.node.height_count += 1
        rig.node.length_count += 1
        rig.node.aspect_ratio_count += 1
        rig.node.handoff_state_count += 1

    def deliver():
        nonlocal sent
        if not sent:
            sent = True
            publish_stable()  # Applied ACK deliberately absent.

    rig.clock.on_sleep = deliver
    assert rig.node._request_vision_target() is None
    assert rig.node.pose_count == 0


def test_hover_keeps_immediate_path_for_valid_tracking(rig):
    rig.sample(.1)
    samples, age = rig.node._wait_for_fresh_pregrasp_observations(0)
    assert len(samples) == 1 and age == pytest.approx(.1)
    assert rig.clock.sleeps == []
    rig.node.trigger_publisher.publish.assert_not_called()


def test_hover_waits_for_late_frame_without_restarting_yolo_or_moving(rig):
    rig.sample(.8)
    rig.clock.on_sleep = lambda: rig.sample(.01) if rig.clock.now >= 100.2 else None
    samples, age = rig.node._wait_for_fresh_pregrasp_observations(0)
    assert 100.2 <= rig.clock.now < 100.3
    assert age <= .65 and samples[-1].frame_time_s > 1000.
    assert rig.node.controller.mock_calls == []
    rig.node.trigger_publisher.publish.assert_not_called()
    rig.node.d405_pretracking_publisher.publish.assert_not_called()


@pytest.mark.parametrize('failure', ['no_frames', 'stale', 'old_capture', 'previous_attempt', 'future_capture'])
def test_hover_wait_is_bounded_and_does_not_weaken_freshness(rig, failure):
    previous = 0
    if failure == 'stale':
        rig.sample(.8)
    elif failure == 'old_capture':
        rig.sample(0., .8)
    elif failure == 'previous_attempt':
        rig.sample()
        previous = rig.node.pregrasp_pose_count
    elif failure == 'future_capture':
        rig.sample(0., -10.)
    with pytest.raises(rig.mod.RecoverableGraspError, match='no fresh D405'):
        rig.node._wait_for_fresh_pregrasp_observations(previous)
    assert rig.clock.now == pytest.approx(100.6)
    rig.node.trigger_publisher.publish.assert_not_called()


def test_stop_or_safety_interlock_interrupts_hover_wait(rig):
    def check(_stage):
        if rig.clock.now >= 100.1:
            raise RuntimeError('safety Stop')
    rig.node._require_cycle_active = check
    with pytest.raises(RuntimeError, match='safety Stop'):
        rig.node._wait_for_fresh_pregrasp_observations(0)
    assert rig.clock.now < 100.2


def test_insufficient_hover_clearance_fails_before_waiting(rig):
    rig.node._current_command_pose = lambda: rig.pose
    rig.node._wait_for_fresh_pregrasp_observations = MagicMock()
    with pytest.raises(rig.mod.RecoverableGraspError, match='safe hover clearance'):
        rig.node._revalidate_target_at_hover(rig.pose, {}, 0., 0., 0, rig.clock.time())
    rig.node._wait_for_fresh_pregrasp_observations.assert_not_called()


def test_zero_hover_wait_preserves_instant_failure_option(rig):
    rig.values['pregrasp_fresh_wait_timeout_s'] = 0.
    with pytest.raises(rig.mod.RecoverableGraspError):
        rig.node._wait_for_fresh_pregrasp_observations(0)
    assert rig.clock.sleeps == []


@pytest.mark.parametrize('bad_ack', ['acquisition_started:nan:1001', 'acquisition_started:1001:1000', 'acquisition_started:garbage'])
def test_bad_camera_ack_cannot_release_reset_barrier(rig, bad_ack):
    rig.node._mark_d405_reacquisition()
    rig.node._vision_result_callback(SimpleNamespace(data=bad_ack))
    assert not rig.node._d405_frame_after_reacquire(1001.)
    assert rig.node.vision_result_count == 0


def test_camera_ack_occurs_in_capture_loop_and_does_not_lose_later_reset(rig):
    # Execute only pure callbacks and the request gate, never import a vision
    # entry point: its top-level code connects cameras and loads CUDA models.
    script = Path(__file__).resolve().parents[2] / 'Fast-FoundationStereoPose-dul_cam/dul_cam' / (
        f'd405_cosmetic_box_leftmost_height75_panel_v{rig.version}.py')
    tree = ast.parse(script.read_text())
    names = {'trigger_callback', 'pretracking_enable_callback', 'acknowledge_d405_acquisition'}
    gate = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'acquisition_request_epoch_s > acquisition_applied_epoch_s')
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(time=rig.clock, Bool=SimpleNamespace, String=SimpleNamespace,
              logging=logging, acquisition_request_epoch_s=0., acquisition_applied_epoch_s=0.,
              vision_result_pub=MagicMock(), trigger_requested=False)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(script), 'exec'), ns)
    ns['pretracking_enable_callback'](SimpleNamespace(data=True))
    ns['vision_result_pub'].publish.assert_not_called()
    ns['acknowledge_d405_acquisition']()
    first_ack = ns['vision_result_pub'].publish.call_args.args[0].data
    rig.node._mark_d405_reacquisition()
    rig.node._vision_result_callback(SimpleNamespace(data=first_ack))
    assert not rig.node._d405_frame_after_reacquire(rig.clock.time())
    rig.clock.now += .02
    ns['trigger_callback'](SimpleNamespace(data=True))
    ns['trigger_requested'] = False  # Previous inference finishes after callback.
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(script), 'exec'), ns)
    assert ns['trigger_requested'] is True
    ns['acknowledge_d405_acquisition']()
    assert ns['acquisition_applied_epoch_s'] == pytest.approx(rig.clock.time())
    ns['pretracking_enable_callback'](SimpleNamespace(data=False))
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(script), 'exec'), ns)
    assert ns['trigger_requested'] is False


def test_explicit_detection_resumes_with_new_stable_pair_after_ack(rig):
    rig.node._transform_vision_pose = lambda *_args: rig.pose
    sent = False

    def deliver():
        nonlocal sent
        if sent:
            return
        sent = True
        rig.ack(request_epoch=1000., capture_epoch=rig.clock.time() - .003)
        for _ in range(2):
            seconds = rig.clock.time() - .001
            msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
                sec=int(seconds), nanosec=int((seconds-int(seconds))*1e9))))
            rig.node._vision_pose_callback(msg)
        rig.node.width_count += 1
        rig.node.height_count += 1
        rig.node.length_count += 1
        rig.node.aspect_ratio_count += 1
        rig.node.handoff_state_count += 1

    rig.clock.on_sleep = deliver
    assert rig.node._request_vision_target()[0] == rig.pose
    assert rig.node.pose_count == 2
    assert rig.clock.now < 100.1


def test_camera_does_not_publish_a_pre_reset_frame_as_new_stable_evidence(rig):
    script = Path(__file__).resolve().parents[2] / 'Fast-FoundationStereoPose-dul_cam/dul_cam' / (
        f'd405_cosmetic_box_leftmost_height75_panel_v{rig.version}.py')
    tree = ast.parse(script.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'is_current_acquisition_frame')
    ns = dict(acquisition_request_epoch_s=1000., acquisition_applied_epoch_s=999.,
              acquisition_capture_epoch_s=999.1)
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(script), 'exec'), ns)
    assert not ns['is_current_acquisition_frame'](1000.1)
    ns.update(acquisition_applied_epoch_s=1000., acquisition_capture_epoch_s=1000.2)
    assert not ns['is_current_acquisition_frame'](1000.1)
    assert not ns['is_current_acquisition_frame'](1000.2)
    assert ns['is_current_acquisition_frame'](1000.3)


def test_complete_hover_revalidation_uses_late_frame_and_preserves_grasp_pose(rig):
    # Load literal parameter defaults only; never construct a ROS node/controller.
    tree = ast.parse(Path(rig.mod.__file__).read_text())
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'declare_parameter' and len(call.args) == 2):
            try:
                rig.values.setdefault(ast.literal_eval(call.args[0]), ast.literal_eval(call.args[1]))
            except (ValueError, TypeError):
                pass
    rig.node._current_command_pose = lambda: rig.mod.TcpPose(.52, -.114, .24, 180., 0., 0.)
    rig.clock.on_sleep = lambda: rig.sample(.001) if rig.clock.now >= 100.2 else None
    result = rig.node._revalidate_target_at_hover(rig.pose, {}, 0., .01, 0, rig.clock.time())
    assert result == rig.pose
    assert 100.2 <= rig.clock.now < 100.3
    assert rig.node.controller.mock_calls == []


def test_frame_arriving_after_hover_deadline_does_not_authorize_descent(rig):
    def overslept():
        rig.clock.now = 100.7
        rig.sample()
    rig.clock.on_sleep = overslept
    with pytest.raises(rig.mod.RecoverableGraspError, match='no fresh D405'):
        rig.node._wait_for_fresh_pregrasp_observations(0)
    assert rig.node.controller.mock_calls == []
