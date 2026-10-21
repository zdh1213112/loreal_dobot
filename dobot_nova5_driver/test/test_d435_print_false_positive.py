"""Real black-label false positive and retained pale/blurred barcode evidence."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dobot_nova5_driver.turntable_barcode_detector_fast import (
    TurntableBarcodeDetector, barcode_region_has_parallel_bar_structure,
    barcode_region_has_visual_detail,
)


DATA = Path(__file__).with_name('data') / 'd435_barcode_structure'


@pytest.mark.parametrize('enhanced', [False, True])
def test_actual_black_print_and_wrap_cannot_become_yolo_barcode_presence(enhanced):
    image = cv2.imread(str(DATA / 'black_print_false_positive_20261021.png'))
    assert image is not None
    detector = TurntableBarcodeDetector(confidence=.40, require_cuda=False,
        scanner_assist=False, detail_roi=None, wide_roi_fallback=False)
    if enhanced:
        image = detector._unsharp_for_wide_roi(image)
    height, width = image.shape[:2]
    rect = (0, 0, width, height)
    assert barcode_region_has_visual_detail(image, rect)
    assert not barcode_region_has_parallel_bar_structure(image, rect)
    detector.onnx_session = object()
    detector._predict_onnx = lambda _image: (
        np.asarray([[0, 0, width, height]], dtype=np.float32),
        np.asarray([0], dtype=np.int32), np.asarray([.493], dtype=np.float32))
    assert detector.detect_yolo(image) is None


@pytest.mark.parametrize('path,enhanced', [
    (path, enhanced) for path in sorted(DATA.glob('true_barcode_*.png'))
    for enhanced in (False, True)
    # The two far-left crops were found by the unsharpened full-frame route.
    # Their extra-sharpened versions already failed the old edge-run checks.
    if not (enhanced and path.stem in ('true_barcode_14', 'true_barcode_20'))
], ids=lambda value: value.stem if isinstance(value, Path) else str(value))
def test_recorded_pale_blurred_and_oblique_real_barcodes_remain_eligible(path, enhanced):
    image = cv2.imread(str(path))
    detector = TurntableBarcodeDetector(require_cuda=False)
    if enhanced:
        image = detector._unsharp_for_wide_roi(image)
    height, width = image.shape[:2]
    assert barcode_region_has_parallel_bar_structure(image, (0, 0, width, height))


@pytest.mark.parametrize('dim,blur,angle', [
    (False, False, 0), (True, False, 0), (False, True, 0),
    (False, False, 5), (False, False, 10), (False, False, 15),
    (True, False, 5), (True, False, 10), (False, True, 5), (False, True, 10),
])
@pytest.mark.parametrize('horizontal', [False, True])
def test_direction_check_retains_bar_patterns_under_small_rotation(dim, blur, angle, horizontal):
    light, dark = (50, 59) if dim else (180, 25)
    image = np.full((140, 220, 3), light, dtype=np.uint8)
    for x in range(10, 210, 8):
        image[10:130, x:x + 4] = dark
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), .8)
    image = cv2.warpAffine(image, cv2.getRotationMatrix2D((110, 70), angle, 1),
                           (220, 140), borderValue=(light, light, light))
    if horizontal:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    height, width = image.shape[:2]
    assert barcode_region_has_parallel_bar_structure(image, (0, 0, width, height))


def test_opencv_keeps_the_same_black_print_as_a_hint_without_confirming():
    image = cv2.imread(str(DATA / 'black_print_false_positive_20261021.png'))
    height, width = image.shape[:2]
    detector = TurntableBarcodeDetector(require_cuda=False, scanner_assist=False)
    detector.scanner_detector = SimpleNamespace(detect=lambda _image: (True, np.asarray([
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]], dtype=np.float32)))
    assert detector.detect_scanner_pattern(image) is None
    assert detector.last_unverified_scanner_hit.source == 'scanner_unverified'
