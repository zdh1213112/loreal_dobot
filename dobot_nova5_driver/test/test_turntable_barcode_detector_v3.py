"""Image-evidence checks for the V3 D435 presence-only detector."""

import numpy as np

from dobot_nova5_driver.turntable_barcode_detector_v3 import (
    TurntableBarcodeDetector,
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
