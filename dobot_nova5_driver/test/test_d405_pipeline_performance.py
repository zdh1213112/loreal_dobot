"""Offline checks: diagnostics cannot block or change V4/V5 grasp evidence."""

import ast
import copy
import importlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from dobot_nova5_driver.d405_pipeline_performance import (
    DiagnosticWriter, LocalizationTiming, paint_depth_sorted_points,
    post_stop_wait_diagnostics,
)


CAMERA_DIR = Path(__file__).resolve().parents[2] / 'Fast-FoundationStereoPose-dul_cam/dul_cam'


def camera_functions(version, names, **namespace):
    """Load only pure functions; never execute camera/GPU/ROS initialization."""
    path = CAMERA_DIR / f'd405_cosmetic_box_leftmost_height75_panel_v{version}.py'
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace.update(np=np, cv2=cv2, copy=copy, time=time)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_diagnostic_writer_never_waits_for_a_busy_disk_and_drains_on_close():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def write(value):
        entered.set()
        assert release.wait(2)
        calls.append(value)

    writer = DiagnosticWriter(write, capacity=1)
    try:
        assert writer.submit(1)
        assert entered.wait(1)
        assert writer.submit(2)
        assert not writer.submit(3)
        assert writer.dropped == 1
        assert not writer.close(timeout=0.01)
        assert not writer.submit(4)
    finally:
        release.set()
        assert writer.close(timeout=2)
    assert calls == [1, 2]


def test_diagnostic_error_does_not_kill_worker(caplog):
    calls = []

    def write(value):
        if value == 1:
            raise OSError('simulated disk failure')
        calls.append(value)

    writer = DiagnosticWriter(write)
    assert writer.submit(1)
    assert writer.submit(2)
    assert writer.close(timeout=2)
    assert calls == [2]
    assert 'debug save failed' in caplog.text


@pytest.mark.parametrize('version', [4, 5])
def test_camera_queues_owned_diagnostic_snapshot_without_writing(version):
    writer = SimpleNamespace(submit=Mock())
    ns = camera_functions(
        version, {'save_top_barcode_debug_frame'},
        top_barcode_debug_writer=writer,
        top_surface_barcode_last_debug_write_at=0., TOP_BARCODE_DEBUG_WRITE_PERIOD_S=1.,
        top_surface_barcode_requested=True,
        top_surface_barcode_detector=SimpleNamespace(model_path='unchanged.pt', model_confidence=.40),
    )
    image = np.full((10, 12, 3), 37, np.uint8)
    mask = np.ones((10, 12), np.uint8)
    corners = np.array([[1, 1], [8, 1], [8, 8], [1, 8]])
    debug = {'candidate_boxes': [{'confidence': .41}]}
    capture = {'session': 'first', 'phase': 'approach'}
    ns['save_top_barcode_debug_frame'](image, mask, corners, debug, None, capture=capture)
    args, kwargs = writer.submit.call_args
    image[:] = 0
    mask[:] = 0
    corners[:] = 0
    debug['candidate_boxes'][0]['confidence'] = 0
    capture['session'] = 'next'
    assert np.all(args[0] == 37)
    assert np.all(args[1] == 1)
    assert args[2][0, 0] == 1
    assert args[3]['candidate_boxes'][0]['confidence'] == .41
    assert kwargs['capture']['session'] == 'first'
    assert kwargs['requested']
    assert kwargs['confidence_threshold'] == .40


@pytest.mark.parametrize('version', [4, 5])
@pytest.mark.parametrize('requested,session,acquisition,expected', [
    (True, 'current', 10., True), (False, 'current', 10., False),
    (True, 'previous', 10., False), (True, 'current', 9., False),
])
def test_cancelled_or_old_top_attempt_is_never_current(
    version, requested, session, acquisition, expected,
):
    ns = camera_functions(
        version, {'top_barcode_attempt_current'},
        top_surface_barcode_requested=requested,
        top_barcode_capture_session='current', acquisition_request_epoch_s=10.,
    )
    assert ns['top_barcode_attempt_current'](
        {'session': session, 'acquisition': acquisition}
    ) is expected


@pytest.mark.parametrize('count', [0, 1, 30000])
def test_vectorized_cloud_pixels_match_old_depth_order_including_ties(count):
    rng = np.random.default_rng(31)
    px = rng.integers(-3, 85, count)
    py = rng.integers(-3, 65, count)
    depths = rng.integers(0, 10, count)  # Deliberately many depth ties.
    colors = rng.integers(0, 256, (count, 3), dtype=np.uint8)
    old = np.full((60, 80, 3), 18, np.uint8)
    new = old.copy()
    for index in np.argsort(depths)[::-1]:
        if 0 <= px[index] < 80 and 0 <= py[index] < 60:
            old[py[index], px[index]] = colors[index]
    paint_depth_sorted_points(new, px, py, depths, colors)
    np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize('version', [4, 5])
def test_complete_cloud_preview_is_pixel_identical_in_both_versions(version):
    def reference(canvas, px, py, depths, bgr):
        for index in np.argsort(depths)[::-1]:
            if 0 <= px[index] < canvas.shape[1] and 0 <= py[index] < canvas.shape[0]:
                canvas[py[index], px[index]] = bgr[index]

    ns = camera_functions(version, {'render_cloud'}, CLOUD_VIEW_H=480, CLOUD_VIEW_W=560,
                          paint_depth_sorted_points=reference)
    rng = np.random.default_rng(23)
    points = rng.uniform([-.2, -.2, .3], [.3, .2, .7], (62000, 3))
    colors = rng.uniform(0, 1, points.shape)
    points[3] = np.nan
    kwargs = dict(tracked_target_points=points[:300], center=np.array([.1, 0., .5]),
                  extent=np.array([.12, .04, .05]), rotation=np.eye(3))
    old = ns['render_cloud'](points, colors, **kwargs)
    ns['paint_depth_sorted_points'] = paint_depth_sorted_points
    np.testing.assert_array_equal(ns['render_cloud'](points, colors, **kwargs), old)


def test_localization_timing_counts_only_work_before_target_publication():
    clock = [1.]
    timing = LocalizationTiming(clock=lambda: clock[0])
    timing.start()
    clock[0] = 1.1
    timing.record('object_yolo', 1.)
    timing.frames = 2
    clock[0] = 1.2
    report = timing.finish(geometry_started=1.1)
    timing.record('preview', 1.1)
    assert report == {'request_to_target_ms': 200., 'frames': 2,
                      'stage_ms': {'object_yolo': 100., 'geometry': 100.}}
    assert not timing.active
    assert 'preview' not in timing.stages


@pytest.mark.parametrize('version', [4, 5])
@pytest.mark.parametrize('new_request', [False, True])
def test_actual_stage_gates_yield_without_consuming_new_request(version, new_request):
    path = CAMERA_DIR / f'd405_cosmetic_box_leftmost_height75_panel_v{version}.py'
    tree = ast.parse(path.read_text())
    gates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
             and isinstance(node.test, ast.Call)
             and isinstance(node.test.func, ast.Name)
             and node.test.func.id == 'd405_reacquisition_pending']
    stages = {node.test.args[0].value for node in gates}
    assert {'object_yolo', 'sam_track', 'top_barcode', 'stereo_inference',
            'target_select_sam_init', 'geometry'} <= stages
    for gate in gates:
        ns = camera_functions(version, {'d405_reacquisition_pending'}, logging=Mock(),
                              acquisition_request_epoch_s=10., acquisition_applied_epoch_s=10.)
        events = []

        def gpu_returned():
            events.append('gpu_returned')
            if new_request:
                ns['acquisition_request_epoch_s'] = 11.

        ns.update(gpu_returned=gpu_returned, expensive_old_work=lambda: events.append('old_work'))
        harness = ast.parse('for _ in range(1):\n    gpu_returned()\n    expensive_old_work()')
        harness.body[0].body.insert(1, gate)
        exec(compile(ast.fix_missing_locations(harness), str(path), 'exec'), ns)
        assert events == (['gpu_returned'] if new_request else ['gpu_returned', 'old_work'])
        assert ns['acquisition_applied_epoch_s'] == 10.  # Only the next capture loop may ACK.
        assert ns['acquisition_request_epoch_s'] == (11. if new_request else 10.)


def test_timeout_diagnostic_separates_stable_and_tracking_topics_and_is_read_only():
    sample = SimpleNamespace(count=1, frame_time_s=999.9, received_at_monotonic=99.9)
    observations = [sample]
    report = post_stop_wait_diagnostics(
        observations=observations, baseline_count=0, stop_s=99., frame_barrier=999.,
        request_epoch=999., applied_epoch=999., now_monotonic=100., now_epoch=1000.,
        max_age_s=.65, required=2, handoff_fresh=True, handoff_clear=True,
        handoff_state='CLEAR', metadata={'width': .08, 'length': .12, 'height': .04, 'aspect': 3.},
        stable_messages_since_wait=2, last_pair_check=None,
    )
    assert report['stable_topic_messages_since_wait'] == 2
    assert report['fresh_tracking_samples'] == 1
    assert report['waiting_for'] == ['insufficient_fresh_tracking_samples']
    assert observations == [sample] and vars(sample) == {
        'count': 1, 'frame_time_s': 999.9, 'received_at_monotonic': 99.9}


@pytest.mark.parametrize('frame,received,reason', [
    (998., 99.9, 'captured_before_barrier'),
    (999.2, 99.9, 'capture_too_old'),
    (999.9, 99.2, 'receipt_too_old'),
    (1001., 99.9, 'future_timestamp'),
])
def test_timeout_diagnostic_explains_timestamp_rejections(frame, received, reason):
    report = post_stop_wait_diagnostics(
        observations=[SimpleNamespace(count=1, frame_time_s=frame, received_at_monotonic=received)],
        baseline_count=0, stop_s=99., frame_barrier=999., request_epoch=999., applied_epoch=999.,
        now_monotonic=100., now_epoch=1000., max_age_s=.65, required=2,
        handoff_fresh=False, handoff_clear=False, handoff_state='BLOCKED',
        metadata={'width': None}, stable_messages_since_wait=0, last_pair_check=None,
    )
    assert report['rejected_tracking_samples'] == {reason: 1}
    assert {'no_new_handoff_state', 'handoff_not_clear', 'missing_dimensions'} <= set(report['waiting_for'])


@pytest.fixture
def barcode(monkeypatch):
    monkeypatch.syspath_prepend(str(CAMERA_DIR))
    return importlib.import_module('top_surface_barcode_detector_v3')


def test_zbar_keeps_all_33_variants_in_identical_order(barcode, monkeypatch):
    gray = np.arange(240, dtype=np.uint8).reshape(12, 20)
    originals = [gray]
    for scale in (2., 3., 4.):
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        originals.extend([
            up, cv2.detailEnhance(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR), sigma_s=10, sigma_r=.15)[:, :, 0],
            cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX),
        ])
    originals.append(cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray))
    expected = [variant for original in originals for variant in (
        original, cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(original, cv2.ROTATE_90_COUNTERCLOCKWISE))]
    seen = []
    monkeypatch.setattr(barcode, 'zbar_decode', lambda image: seen.append(image.copy()) or [])
    detector = barcode.TopSurfaceBarcodeDetector()
    assert detector._decode_zbar(gray, np.ones_like(gray, bool), (0, 0)) is None
    assert len(seen) == len(expected) == 33
    for actual, old in zip(seen, expected):
        np.testing.assert_array_equal(actual, old)


def test_zbar_success_does_not_eagerly_build_unused_variants(barcode, monkeypatch):
    hit = SimpleNamespace(data=b'123456', type='EAN13', rect=None)
    decode = Mock(return_value=[hit])
    enhance = Mock(side_effect=AssertionError('unnecessary work'))
    monkeypatch.setattr(barcode, 'zbar_decode', decode)
    monkeypatch.setattr(cv2, 'detailEnhance', enhance)
    detector = barcode.TopSurfaceBarcodeDetector()
    gray = np.ones((20, 30), np.uint8)
    assert detector._decode_zbar(gray, gray.astype(bool), (0, 0)).value == '123456'
    decode.assert_called_once()
    enhance.assert_not_called()


def test_top_cancellation_interrupts_fallback_without_changing_valid_windows(barcode, monkeypatch):
    calls = []
    monkeypatch.setattr(barcode, 'zbar_decode', lambda pixels: calls.append(1) or [])
    detector = barcode.TopSurfaceBarcodeDetector()
    gray = np.ones((20, 30), np.uint8)
    assert detector._decode_zbar(
        gray, gray.astype(bool), (0, 0), cancel_check=lambda: bool(calls)
    ) is None
    assert len(calls) == 1


def test_top_hit_returning_after_cancellation_cannot_be_accepted(barcode, monkeypatch):
    cancelled = [False]
    detector = barcode.TopSurfaceBarcodeDetector()

    def yolo(*_args):
        cancelled[0] = True
        return barcode.TopSurfaceBarcode('barcode_detected', (1, 1, 5, 5))

    monkeypatch.setattr(detector, '_decode_yolo', yolo)
    assert detector.detect(np.ones((20, 30, 3), np.uint8), np.ones((20, 30), np.uint8),
                           cancel_check=lambda: cancelled[0]) is None
    assert detector.last_debug['accepted_candidate'] is None
    assert 'superseded' in detector.last_debug['message']


@pytest.mark.parametrize('version', [4, 5])
@pytest.mark.parametrize('disarm_failure', [False, True])
def test_side_confirmation_closes_top_before_stop_and_never_rearms_at_settle(version, disarm_failure):
    module = importlib.import_module(f'dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v{version}')
    node = module.CosmeticBoxSingleArmNode.__new__(module.CosmeticBoxSingleArmNode)
    clock = SimpleNamespace(now=100.)
    original_time = module.time
    module.time = SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now + 900.)
    params = dict(turntable_enabled=True, turntable_stationary_barcode_check_s=.01,
                  turntable_scan_timeout_s=1., turntable_mid_scan_restart_s=0.,
                  turntable_settle_s=.05, d405_early_top_barcode_enabled=True,
                  top_surface_barcode_enabled=True)
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node.turntable_lock = threading.RLock()
    node.turntable_scan_cancel = threading.Event()
    node.running, node.shutting_down = True, False
    node.early_top_at_startup = True
    node.turntable_state, node.turntable_barcode_value = 'STOPPED', ''
    node._wait_for_secondary_y_clearance = Mock()
    node._wait_for_turntable_camera_ready = Mock()
    node._set_turntable_barcode_window = Mock()
    node._notify_d405_post_stop_revalidate = Mock()
    node._current_top_surface_barcode = lambda: ''
    node._top_surface_barcode_is_uncertain = lambda: False
    node._recheck_turntable_barcode_candidate = lambda **_kwargs: False
    node._publish_status = Mock()
    top_states = []

    def set_top(active):
        top_states.append(active)
        if not active and disarm_failure:
            raise RuntimeError('simulated top disarm publish failure')

    node._set_top_surface_barcode_window = set_top

    def sleep(seconds):
        clock.now += seconds
        if node.turntable_state == 'RUNNING':
            node.turntable_barcode_value = 'barcode_detected'

    module.time.sleep = sleep

    def toggle(_old, new, _reason):
        if new == 'STOPPED':
            assert top_states[-1] is False
        node.turntable_state = new
        clock.now += .4
        return clock.now

    node._toggle_turntable = toggle
    try:
        if disarm_failure:
            with pytest.raises(RuntimeError, match='top disarm publish failure'):
                node._scan_turntable_for_side_barcode(require_cycle_active=False)
        else:
            assert node._scan_turntable_for_side_barcode(require_cycle_active=False) == 'barcode_detected'
    finally:
        module.time = original_time
    if disarm_failure:
        assert top_states[0] is True and len(top_states) >= 2
        assert not any(top_states[1:])  # Cleanup may retry disarm, but never rearm.
    else:
        assert top_states == [True, False]
    assert node.turntable_state == 'STOPPED'
    node._notify_d405_post_stop_revalidate.assert_called_once()
