"""D435 confirmation regressions without camera/model/controller initialization."""

import ast
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from dobot_nova5_driver.barcode_confirmation_fast import BarcodeEvidenceTracks
from dobot_nova5_driver.turntable_barcode_detector_fast import TurntableBarcodeDetector


def hit(source="yolo_detail", rect=(20, 20, 120, 80), confidence=.41):
    return SimpleNamespace(source=source, rect=rect, value="barcode_detected", confidence=confidence)


@pytest.fixture
def scanner():
    # Execute the real node methods while excluding its RealSense/ROS startup.
    p = Path(__file__).resolve().parents[1] / 'dobot_nova5_driver/d435_turntable_barcode_node_fast.py'
    tree = ast.parse(p.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'D435TurntableBarcodeNode')
    names = {'_accept_hit', '_rect_iou', '_moving_stripe_spatially_consistent',
             '_trigger_callback', '_publish_candidate_hint', '_reset_candidate_for_roi_change'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    clock = SimpleNamespace(now=100.)
    clock.monotonic = lambda: clock.now
    ns = dict(time=clock, math=math, json=__import__('json'), Bool=SimpleNamespace, String=SimpleNamespace)
    wrapper = ast.Module(body=[ast.ClassDef(name='Scanner', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(wrapper), str(p), 'exec'), ns)
    node = ns['Scanner']()
    node.state_lock = threading.Lock()
    node.scan_active, node.scan_generation, node.continuous_scan_active = True, 1, True
    node.evidence_tracks = BarcodeEvidenceTracks()
    node.candidate, node.candidate_source, node.candidate_rect = '', '', None
    node.candidate_first_rect, node.candidate_hits, node.candidate_last_at = None, 0, 0.
    node.continuous_presence_latched = False
    node.result_publisher = SimpleNamespace(publish=Mock())
    node.continuous_result_publisher = SimpleNamespace(publish=Mock())
    node.candidate_publisher = SimpleNamespace(publish=Mock())
    node._publish_status = Mock()
    node._publish_continuous_presence = Mock()
    params = dict(scanner_assist_hit_gap_s=.25, scanner_assist_stable_hits=3,
                  scanner_assist_min_iou=.20, stable_hit_gap_s=.70, stable_hits=2, yolo_min_iou=.15)
    node.get_parameter = lambda key: SimpleNamespace(value=params[key])
    return node, clock


def test_mixed_sources_cannot_replace_two_yolo_frames(scanner):
    node, clock = scanner
    assert not node._accept_hit(hit('scanner_pattern'), 1, 1)
    clock.now += .03
    assert not node._accept_hit(hit(), 1, 2)
    node.result_publisher.publish.assert_not_called()
    clock.now += .03
    assert node._accept_hit(hit(), 1, 3)
    node.result_publisher.publish.assert_called_once()


def test_scanner_only_still_confirms_without_any_yolo(scanner):
    node, clock = scanner
    for frame in (1, 2):
        assert not node._accept_hit(hit('scanner_pattern'), 1, frame)
        clock.now += .03
    assert node._accept_hit(hit('scanner_pattern'), 1, 3)


def test_yolo_views_share_only_their_own_counter(scanner):
    node, clock = scanner
    assert not node._accept_hit(hit('yolo_detail'), 1, 1)
    clock.now += .03
    assert node._accept_hit(hit('yolo_tiled'), 1, 2)


def test_same_frame_cannot_be_counted_twice(scanner):
    node, clock = scanner
    assert not node._accept_hit(hit(), 1, 1)
    assert not node._accept_hit(hit('yolo_tiled'), 1, 1)
    clock.now += .02
    assert node._accept_hit(hit(), 1, 2)


def test_different_location_and_expired_evidence_restart_counts(scanner):
    node, clock = scanner
    assert not node._accept_hit(hit(), 1, 1)
    clock.now += .03
    assert not node._accept_hit(hit(rect=(600, 20, 120, 80)), 1, 2)
    clock.now += .8
    assert not node._accept_hit(hit(rect=(600, 20, 120, 80)), 1, 3)


def test_weak_pattern_can_request_recheck_but_never_success_or_presence(scanner):
    node, clock = scanner
    for frame in range(1, 8):
        assert not node._accept_hit(hit('scanner_unverified'), 1, frame)
        clock.now += .03
    node.result_publisher.publish.assert_not_called()
    node._publish_continuous_presence.assert_not_called()
    node._publish_candidate_hint(hit('scanner_unverified'), 1, 1000.)
    assert '"pending":true' in node.candidate_publisher.publish.call_args.args[0].data


def test_scan_reset_rejects_inflight_old_frame_even_with_continuous_mode(scanner):
    node, _ = scanner
    node._accept_hit(hit(), 1, 1)
    node._trigger_callback(SimpleNamespace(data=True))
    assert not node._accept_hit(hit(), 1, 2)
    assert not node._accept_hit(hit(), 2, 3)


def test_both_routes_are_evaluated_when_scanner_already_has_a_proposal():
    detector = TurntableBarcodeDetector(scanner_assist=False, require_cuda=False)
    detector.detect_scanner_pattern = Mock(return_value=hit('scanner_pattern'))
    detector.detect_yolo = Mock(return_value=hit())
    results = detector.detect_candidates(np.zeros((100, 200, 3), dtype=np.uint8))
    assert {h.source for h in results} == {'yolo_detail', 'scanner_pattern'}
    detector.detect_yolo.assert_called_once()


@pytest.mark.parametrize('dim,blur,horizontal', [(False, False, False), (True, False, False),
                                               (False, True, False), (False, False, True)])
def test_scanner_keeps_real_bar_patterns_with_dim_blur_and_rotation(dim, blur, horizontal):
    image = np.full((120, 180, 3), 50 if dim else 180, dtype=np.uint8)
    for x in range(10, 170, 8):
        image[10:110, x:x+4] = 59 if dim else 25
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), .8)
    if horizontal:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    h, w = image.shape[:2]
    detector = TurntableBarcodeDetector(scanner_assist=False, require_cuda=False)
    detector.scanner_detector = SimpleNamespace(detect=lambda _: (True, np.array([
        [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]], dtype=np.float32)))
    assert detector.detect_scanner_pattern(image) is not None


def test_flat_and_wrinkled_scanner_proposals_remain_unconfirmed_hints():
    detector = TurntableBarcodeDetector(scanner_assist=False, require_cuda=False)
    detector.scanner_detector = SimpleNamespace(detect=lambda _: (True, np.array([
        [[0, 0], [179, 0], [179, 119], [0, 119]]], dtype=np.float32)))
    for wrinkles in (False, True):
        image = np.full((120, 180, 3), 145, dtype=np.uint8)
        if wrinkles:
            for x in (30, 80, 130):
                cv2.line(image, (x, 5), (x+18, 115), (220, 220, 220), 5)
        assert detector.detect_scanner_pattern(image) is None
        assert detector.last_unverified_scanner_hit.source == 'scanner_unverified'


def test_scanner_rectifies_a_diagonal_barcode_before_rejecting_its_structure():
    patch = np.full((80, 120, 3), 180, dtype=np.uint8)
    for x in range(0, 120, 8):
        patch[:, x:x+4] = 25
    canvas = np.full((240, 240, 3), 180, dtype=np.uint8)
    canvas[80:160, 60:180] = patch
    matrix = cv2.getRotationMatrix2D((120, 120), 45, 1)
    image = cv2.warpAffine(canvas, matrix, (240, 240), borderValue=(180, 180, 180))
    polygon = cv2.transform(np.array([[[60, 80], [179, 80], [179, 159], [60, 159]]],
                                    dtype=np.float32), matrix)
    detector = TurntableBarcodeDetector(scanner_assist=False, require_cuda=False)
    detector.scanner_detector = SimpleNamespace(detect=lambda _: (True, polygon))
    assert detector.detect_scanner_pattern(image) is not None


def test_low_confidence_yolo_threshold_remains_point_four():
    detector = TurntableBarcodeDetector(confidence=.40, scanner_assist=False, require_cuda=False)
    detector.onnx_session = SimpleNamespace(run=lambda *args: [np.array([[
        [20, 20, 140, 100, .41, 0], [200, 20, 300, 100, .39, 0]
    ]], dtype=np.float32)])
    boxes, _, confidence = detector._predict_onnx(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(boxes) == 1 and float(confidence[0]) == pytest.approx(.41)
