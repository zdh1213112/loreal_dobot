"""D435 confidence-only overlays, without starting ROS, a camera or a model."""

import ast
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dobot_nova5_driver.d435_confirmation_review import BarcodeConfirmationReview


@pytest.fixture
def drawn_text(monkeypatch):
    labels = []
    original = cv2.putText

    def capture(image, text, origin, *args, **kwargs):
        labels.append((text, origin))
        return original(image, text, origin, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    return labels


@pytest.fixture
def preview_renderer():
    source = (Path(__file__).resolve().parents[1] / "dobot_nova5_driver"
              / "d435_turntable_barcode_node_fast.py")
    tree = ast.parse(source.read_text())
    constants = [item for item in tree.body if isinstance(item, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id.startswith("D435_BRIGHTNESS_")
                         for target in item.targets)]
    helper = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                  and item.name == "brightness_button_rects")
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef)
               and item.name == "D435TurntableBarcodeNode")
    render = next(item for item in cls.body if isinstance(item, ast.FunctionDef)
                  and item.name == "_render_preview")
    code = ast.Module(body=constants + [helper, render], type_ignores=[])
    namespace = dict(cv2=cv2, np=np)
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
    node = SimpleNamespace(
        _roi_bounds=lambda image: (0, 0, image.shape[1], image.shape[0]),
        _drawing_roi=lambda: (False, (0, 0), (0, 0)),
        detector=SimpleNamespace(detail_roi=None),
        camera_settings_text="test", detect_rate_fps=30., confirmation_review=None,
    )
    return lambda image, hit: namespace["_render_preview"](node, image, hit, True, True)


@pytest.mark.parametrize("source,expected", [
    ("yolo", "0.86"), ("yolo_detail", "0.86"), ("yolo_tiled", "0.86"),
    ("scanner_pattern", "SCANNER"), ("scanner_unverified", "SCANNER CANDIDATE"),
    ("moving_stripes", "MOVING STRIPES"),
])
def test_live_box_label_preserves_confidence_and_evidence(
        preview_renderer, drawn_text, source, expected):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    hit = SimpleNamespace(source=source, confidence=.856, rect=(100, 140, 100, 70))
    rendered = preview_renderer(image, hit)
    assert [text for text, origin in drawn_text if origin == (100, 134)] == [expected]
    assert all("yolo" not in text.lower() for text, _ in drawn_text)
    assert rendered.shape == image.shape and rendered.any() and not image.any()
    assert hit.source == source and hit.confidence == .856


@pytest.mark.parametrize("source,confidence,expected", [
    ("yolo", .856, "0.86"), ("yolo_detail", .419, "0.42"),
    ("yolo_tiled", .725, "0.72"), ("scanner_pattern", 1., "n/a"),
])
def test_confirmation_image_and_history_only_show_confidence(
        tmp_path, drawn_text, source, confidence, expected):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    metadata = dict(source=source, confidence=confidence, rect=[100, 140, 100, 70])
    review = BarcodeConfirmationReview(tmp_path, save_enabled=False)
    try:
        review.record(image, metadata)
        assert drawn_text == [(expected, (100, 132))]
        assert review.latest.metadata["source"] == source
        assert review.latest.metadata["confidence"] == confidence
        assert np.array_equal(review.latest.image, image)
        drawn_text.clear()
        review.render_on(image.copy())
        assert len(drawn_text) == 2
        assert "HISTORY" in drawn_text[0][0]
        assert drawn_text[1][0] == expected
        assert all("yolo" not in text.lower() for text, _ in drawn_text)
        assert review.worker is None
    finally:
        review.close()
