"""Replay real D405 text/true-barcode proposals without models, cameras or ROS."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Fast-FoundationStereoPose-dul_cam/dul_cam/top_surface_barcode_detector_v3.py'
SPEC = importlib.util.spec_from_file_location('top_barcode_structure_replay', SOURCE)
top = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = top
SPEC.loader.exec_module(top)
DATA = Path(__file__).with_name('data') / 'd405_top_barcode_structure'


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def detector(monkeypatch):
    monkeypatch.setattr(top, 'YOLO', object())
    monkeypatch.setattr(top, 'zbar_decode', None)
    result = top.TopSurfaceBarcodeDetector()
    result._cv_detector = None
    result.load_model = lambda: True

    def predict(image, **_kwargs):
        h, w = image.shape[:2]
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=Tensor([[0, 0, w, h]]), cls=Tensor([0]), conf=Tensor([.99]),
        ))]

    result._yolo_model = SimpleNamespace(predict=predict)
    return result


@pytest.mark.parametrize('path', sorted(DATA.glob('false_text_*.png')), ids=lambda p: p.stem)
def test_real_text_proposals_cannot_confirm_even_with_high_yolo_confidence(detector, path):
    image = cv2.imread(str(path))
    assert image is not None
    mask = np.ones(image.shape[:2], np.uint8)
    assert detector.detect(image, mask) is None
    assert detector.last_debug['accepted_candidate'] is None


@pytest.mark.parametrize('path', sorted(DATA.glob('true_barcode_*.png')), ids=lambda p: p.stem)
def test_recorded_small_blurred_and_rotated_barcodes_remain_candidates(detector, path):
    image = cv2.imread(str(path))
    assert image is not None
    hit = detector.detect(image, np.ones(image.shape[:2], np.uint8))
    assert hit is not None
    assert hit.value == 'barcode_detected'


@pytest.mark.parametrize('angle', [0, 15, 30, 45, 60, 75, 90, 120, 150, 180])
@pytest.mark.parametrize('contrast', [15, 140])
def test_bar_structure_is_rotation_independent_for_pale_and_clear_bars(detector, angle, contrast):
    image = np.full((160, 160, 3), 180, np.uint8)
    for x in range(35, 125, 6):
        image[40:120, x:x+3] = 180 - contrast
    image = cv2.warpAffine(image, cv2.getRotationMatrix2D((80, 80), angle, 1),
                           (160, 160), borderValue=(180, 180, 180))
    assert detector.detect(image, np.ones(image.shape[:2], np.uint8)) is not None


def test_true_decoder_result_can_still_confirm_without_yolo_bar_presence(detector):
    image = cv2.imread(str(DATA / 'false_text_00.png'))
    detector._decode_zbar = lambda *_args, **_kwargs: top.TopSurfaceBarcode('6901234567892', (1, 2, 20, 30))
    hit = detector.detect(image, np.ones(image.shape[:2], np.uint8))
    assert hit.value == '6901234567892'
    assert detector.last_debug['decoder'] == 'ZBar'


def test_invalid_or_uniform_images_do_not_confirm(detector):
    assert detector.detect(None, np.ones((10, 10), np.uint8)) is None
    image = np.full((80, 100, 3), 180, np.uint8)
    assert detector.detect(image, np.ones(image.shape[:2], np.uint8)) is None


def test_yolo_hit_keeps_global_coordinates_after_target_roi_crop(detector):
    crop = cv2.imread(str(DATA / 'true_barcode_06.png'))
    height, width = crop.shape[:2]
    image = np.full((height+30, width+50, 3), 180, np.uint8)
    image[10:10+height, 20:20+width] = crop
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:10+height, 20:20+width] = 1
    hit = detector.detect(image, mask)
    assert hit.rect == (20, 10, width, height)


def test_cancelled_window_cannot_publish_a_candidate(detector):
    image = cv2.imread(str(DATA / 'true_barcode_06.png'))
    assert detector.detect(image, np.ones(image.shape[:2], np.uint8), cancel_check=lambda: True) is None
