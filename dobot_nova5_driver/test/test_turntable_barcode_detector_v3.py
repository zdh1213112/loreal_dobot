"""Image-evidence checks for the V3 D435 presence-only detector."""

import cv2
import numpy as np

from dobot_nova5_driver.turntable_barcode_detector_v3 import (
    TurntableBarcodeDetector,
    barcode_region_has_parallel_bar_structure,
    barcode_region_has_visual_detail,
)


def test_flat_dark_robot_background_is_not_barcode_evidence():
    image = np.full((120, 160, 3), 10, dtype=np.uint8)
    assert not barcode_region_has_visual_detail(image, (30, 20, 90, 80))


def test_visible_barcode_at_roi_top_left_edge_remains_eligible():
    image = np.full((120, 160, 3), 180, dtype=np.uint8)
    for x in range(0, 60, 6):
        image[0:58, x : x + 3] = 25
    assert barcode_region_has_visual_detail(image, (0, 0, 60, 58))


def test_visible_barcode_at_roi_bottom_right_edge_remains_eligible():
    image = np.full((120, 160, 3), 180, dtype=np.uint8)
    for y in range(60, 120, 6):
        image[y : y + 3, 100:160] = 25
    assert barcode_region_has_visual_detail(image, (100, 60, 60, 60))


def test_dim_barcode_uses_repeated_edges_even_with_low_global_contrast():
    image = np.full((80, 80, 3), 50, dtype=np.uint8)
    for x in range(0, 80, 8):
        image[:, x : x + 4] = 59
    assert barcode_region_has_visual_detail(image, (0, 0, 80, 80))
    assert barcode_region_has_parallel_bar_structure(image, (0, 0, 80, 80))


def test_shrink_wrap_wrinkles_are_not_parallel_barcode_structure():
    image = np.full((120, 180, 3), 145, dtype=np.uint8)
    # Three broad diagonal highlights resemble the false-positive plastic
    # folds on the purple package, but they are not repeated barcode bars.
    for offset in (35, 85, 135):
        cv2.line(image, (offset - 8, 10), (offset + 8, 110), (195, 195, 195), 5)
        cv2.line(image, (offset, 10), (offset + 16, 110), (105, 105, 105), 2)
    assert barcode_region_has_visual_detail(image, (0, 0, 180, 120))
    assert not barcode_region_has_parallel_bar_structure(
        image, (0, 0, 180, 120)
    )


def test_high_confidence_shrink_wrap_yolo_proposal_is_rejected():
    image = np.full((120, 180, 3), 145, dtype=np.uint8)
    for offset in (35, 85, 135):
        cv2.line(image, (offset - 8, 10), (offset + 8, 110), (195, 195, 195), 5)
        cv2.line(image, (offset, 10), (offset + 16, 110), (105, 105, 105), 2)
    detector = TurntableBarcodeDetector(
        scanner_assist=False,
        wide_roi_fallback=False,
    )
    detector.onnx_session = object()
    detector._predict_onnx = lambda _image: (
        np.array([[0, 0, 180, 120]], dtype=np.float32),
        np.array([0], dtype=np.int32),
        np.array([0.85], dtype=np.float32),
    )

    assert detector.detect_yolo(image) is None


def test_yolo_skips_stronger_flat_background_and_keeps_edge_barcode():
    image = np.full((120, 160, 3), 10, dtype=np.uint8)
    for x in range(0, 60, 6):
        image[0:58, x : x + 3] = 180
    detector = TurntableBarcodeDetector(scanner_assist=False)
    detector.onnx_session = object()
    detector._predict_onnx = lambda _image: (
        np.array([[70, 20, 150, 100], [0, 0, 60, 58]], dtype=np.float32),
        np.array([0, 0], dtype=np.int32),
        np.array([0.90, 0.72], dtype=np.float32),
    )

    hit = detector.detect_yolo(image)

    assert hit is not None
    assert hit.rect == (0, 0, 60, 58)
    assert hit.confidence > 0.70


def test_wide_roi_uses_overlapping_detail_preserving_tiles_directly():
    image = np.full((100, 300, 3), 150, dtype=np.uint8)
    for x in range(0, 300, 8):
        image[:, x : x + 3] = 50
    detector = TurntableBarcodeDetector(
        scanner_assist=False,
        wide_roi_fallback=True,
        wide_roi_aspect_ratio=2.0,
        wide_roi_tile_fraction=0.70,
    )
    detector.onnx_session = object()
    shapes = []

    def predict(view):
        shapes.append(view.shape[:2])
        return (
            np.array([[20, 10, 80, 90]], dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0.78], dtype=np.float32),
        )

    detector._predict_onnx = predict

    hit = detector.detect_yolo(image)

    assert shapes == [(100, 210), (100, 210)]
    assert hit is not None
    assert hit.rect == (20, 10, 60, 80)
    assert hit.confidence > 0.70
    assert hit.source == "yolo_tiled"


def test_normal_aspect_roi_does_not_pay_for_tiled_fallback():
    image = np.full((100, 150, 3), 150, dtype=np.uint8)
    detector = TurntableBarcodeDetector(
        scanner_assist=False,
        wide_roi_fallback=True,
        wide_roi_aspect_ratio=2.0,
    )
    detector.onnx_session = object()
    calls = []

    def predict(view):
        calls.append(view.shape[:2])
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    detector._predict_onnx = predict

    assert detector.detect_yolo(image) is None
    assert calls == [(100, 150)]


def test_tiny_textured_yolo_box_is_rejected_by_relative_area():
    image = np.full((100, 300, 3), 150, dtype=np.uint8)
    for x in range(0, 300, 6):
        image[:, x : x + 2] = 30
    detector = TurntableBarcodeDetector(
        scanner_assist=False,
        wide_roi_fallback=False,
        yolo_min_candidate_area_ratio=0.01,
    )
    detector.onnx_session = object()
    detector._predict_onnx = lambda _image: (
        np.array([[120, 10, 135, 25]], dtype=np.float32),
        np.array([0], dtype=np.int32),
        np.array([0.92], dtype=np.float32),
    )

    assert detector.detect_yolo(image) is None


def _moving_barcode_frame(left):
    image = np.full((720, 1280, 3), 145, dtype=np.uint8)
    for x in range(left, left + 110, 6):
        image[470:530, x:x + 3] = 25
    return image


def _moving_horizontal_barcode_frame(left):
    image = np.full((720, 1280, 3), 145, dtype=np.uint8)
    for y in range(440, 550, 6):
        image[y:y + 3, left:left + 60] = 25
    return image


def test_moving_stripe_fallback_finds_barcode_but_not_stationary_label():
    detector = TurntableBarcodeDetector(scanner_assist=False)
    first = _moving_barcode_frame(740)
    second = _moving_barcode_frame(770)

    assert detector.detect_moving_stripes(first) is None
    hit = detector.detect_moving_stripes(second)
    assert hit is not None
    assert hit.source == "moving_stripes"
    assert hit.rect[0] > 750
    assert detector.detect_moving_stripes(second) is None


def test_moving_stripe_fallback_finds_horizontal_barcode():
    detector = TurntableBarcodeDetector(scanner_assist=False)
    first = _moving_horizontal_barcode_frame(740)
    second = _moving_horizontal_barcode_frame(770)

    assert detector.detect_moving_stripes(first) is None
    hit = detector.detect_moving_stripes(second)
    assert hit is not None
    assert hit.source == "moving_stripes"
    assert 760 < hit.rect[0] < 830
    assert hit.rect[3] > hit.rect[2]
    assert detector.detect_moving_stripes(second) is None


def test_moving_horizontal_barcode_respects_acceptance_roi():
    first = _moving_horizontal_barcode_frame(740)
    second = _moving_horizontal_barcode_frame(770)
    detector = TurntableBarcodeDetector(scanner_assist=False)

    assert detector.detect_moving_stripes(first, (700, 400, 850, 600)) is None
    assert detector.detect_moving_stripes(
        second, (700, 400, 850, 600)
    ) is not None

    detector = TurntableBarcodeDetector(scanner_assist=False)
    assert detector.detect_moving_stripes(first, (0, 0, 300, 300)) is None
    assert detector.detect_moving_stripes(second, (0, 0, 300, 300)) is None


def test_moving_stripe_fallback_keeps_full_frame_and_filters_by_roi():
    detector = TurntableBarcodeDetector(scanner_assist=False)
    first = _moving_barcode_frame(740)
    second = _moving_barcode_frame(770)
    selected_roi = (324, 385, 1077, 630)

    assert detector.detect_moving_stripes(first, selected_roi) is None
    hit = detector.detect_moving_stripes(second, selected_roi)
    assert hit is not None
    assert hit.source == "moving_stripes"
    assert 750 < hit.rect[0] < 850

    detector = TurntableBarcodeDetector(scanner_assist=False)
    assert detector.detect_moving_stripes(first, (0, 0, 300, 300)) is None
    assert detector.detect_moving_stripes(second, (0, 0, 300, 300)) is None


def test_moving_stripe_fallback_rejects_tiny_manual_crop():
    detector = TurntableBarcodeDetector(scanner_assist=False)
    image = _moving_barcode_frame(740)[430:565, 665:865]

    assert detector.detect_moving_stripes(image) is None
    assert detector.detect_moving_stripes(image) is None


def test_yolo_roi_only_filters_candidates_without_changing_model_view():
    image = np.full((720, 1280, 3), 140, dtype=np.uint8)
    for x in range(100, 190, 6):
        image[450:510, x:x + 3] = 20
    for x in range(700, 790, 6):
        image[450:510, x:x + 3] = 20
    detector = TurntableBarcodeDetector(
        scanner_assist=False, yolo_min_candidate_area_ratio=0.005
    )
    detector.onnx_session = object()
    shapes = []

    def predict(view):
        shapes.append(view.shape[:2])
        return (
            np.array([[100, 450, 190, 510], [700, 450, 790, 510]],
                     dtype=np.float32),
            np.array([0, 0], dtype=np.int32),
            np.array([0.90, 0.70], dtype=np.float32),
        )

    detector._predict_onnx = predict
    hit_full = detector.detect_yolo(image)
    hit_roi = detector.detect_yolo(image, (600, 400, 850, 550))

    assert shapes == [(720, 1280), (720, 1280)]
    assert hit_full is not None and hit_full.rect[0] == 100
    assert hit_roi is not None and hit_roi.rect[0] == 700


def test_scanner_roi_selects_eligible_polygon_not_largest_outside():
    image = np.full((720, 1280, 3), 140, dtype=np.uint8)
    detector = TurntableBarcodeDetector(scanner_assist=False)

    class Scanner:
        def detect(self, _image):
            return True, np.array([
                [[100, 400], [250, 400], [250, 550], [100, 550]],
                [[700, 450], [790, 450], [790, 510], [700, 510]],
            ], dtype=np.float32)

    detector.scanner_detector = Scanner()
    hit = detector.detect_scanner_pattern(image, (600, 400, 850, 550))

    assert hit is not None
    assert hit.rect[0] == 700
