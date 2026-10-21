"""Exact-frame history and bounded diagnostic IO, without camera/ROS/model startup."""

import json
import threading
from pathlib import Path
import ast

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from dobot_nova5_driver.d435_confirmation_review import BarcodeConfirmationReview


def evidence(frame=1):
    return dict(capture_received_unix_ns=1_000_000_000 + frame, generation=4,
                detection_frame_id=frame, camera_frame_number=100 + frame,
                camera_timestamp_ms=frame * 33.3, source="yolo_detail",
                confidence=.419, rect=[30, 40, 80, 60], independent_hits=2,
                required_hits=2, value="barcode_detected", workflow_confirmation=True)


def picture():
    image = np.full((240, 400, 3), 160, dtype=np.uint8)
    for x in range(30, 110, 6):
        image[40:100, x:x + 3] = 25
    return image


def test_saved_files_use_the_exact_accepted_image_and_numeric_evidence(tmp_path):
    saved, errors = [], []
    review = BarcodeConfirmationReview(tmp_path, on_saved=saved.append, on_error=errors.append)
    original = picture()
    expected = original.copy()
    try:
        review.record(original, evidence())
        original[:] = 0  # SDK/caller reuse cannot replace the accepted frame.
    finally:
        review.close()
    assert not review.worker.is_alive() and not errors
    assert len(saved) == 1
    folder = Path(saved[0])
    assert set(p.name for p in folder.iterdir()) == {"raw.png", "annotated.jpg", "crop.png", "record.json"}
    assert np.array_equal(cv2.imread(str(folder / "raw.png")), expected)
    assert np.array_equal(cv2.imread(str(folder / "crop.png")), expected[40:100, 30:110])
    annotated = cv2.imread(str(folder / "annotated.jpg"))
    assert not np.array_equal(annotated, expected)
    data = json.loads((folder / "record.json").read_text())
    assert data["complete"] is True and data["confidence"] == .419
    assert data["camera_frame_number"] == 101
    assert data["rect"] == [30, 40, 80, 60]


def test_history_survives_live_detections_disappearing_then_expires(tmp_path):
    now = [0.]
    review = BarcodeConfirmationReview(tmp_path, clock=lambda: now[0])
    try:
        review.record(picture(), evidence())
        live = np.zeros((240, 400, 3), dtype=np.uint8)
        now[0] = 2.
        rendered = review.render_on(live.copy())
        assert rendered.any()  # A separate historical inset, no current hit needed.
        assert not live.any()
        now[0] = 5.01
        assert np.array_equal(review.render_on(live.copy()), live)
    finally:
        review.close()


def test_preview_only_keeps_exact_history_without_disk_io_or_writer(tmp_path, monkeypatch):
    now = [0.]
    directory = tmp_path / "disabled_recording"
    user_file = tmp_path / "keep.txt"
    user_file.write_text("existing record")

    def unexpected_io(*_args, **_kwargs):
        pytest.fail("preview-only confirmation must not write or prune files")

    monkeypatch.setattr(BarcodeConfirmationReview, '_save', unexpected_io)
    monkeypatch.setattr(BarcodeConfirmationReview, '_prune', unexpected_io)
    review = BarcodeConfirmationReview(directory, save_enabled=False, clock=lambda: now[0])
    original = picture()
    expected = original.copy()
    try:
        review.record(original, evidence())
        original[:] = 0
        assert np.array_equal(review.latest.image, expected)
        assert review.worker is None and review.jobs is None
        live = np.zeros((240, 400, 3), dtype=np.uint8)
        now[0] = 2.
        assert review.render_on(live.copy()).any()
        now[0] = 5.01
        assert np.array_equal(review.render_on(live.copy()), live)
    finally:
        review.close()
    review.close()  # Shutdown also works with no writer and is idempotent.
    review.record(picture(), evidence(2))
    assert review.latest.metadata['detection_frame_id'] == 1
    assert not directory.exists()
    assert user_file.read_text() == "existing record"


def test_saving_can_stay_enabled_when_history_preview_is_disabled(tmp_path):
    saved = []
    review = BarcodeConfirmationReview(tmp_path, preview_enabled=False, on_saved=saved.append)
    live = np.zeros((240, 400, 3), dtype=np.uint8)
    try:
        review.record(picture(), evidence())
        assert np.array_equal(review.render_on(live.copy()), live)
    finally:
        review.close()
    assert len(saved) == 1 and (Path(saved[0]) / 'raw.png').is_file()


@pytest.mark.parametrize('preview_enabled,save_enabled', [(True, False), (False, True),
                                                         (True, True), (False, False)])
def test_node_respects_independent_preview_and_save_settings(tmp_path, preview_enabled, save_enabled):
    # Run the real diagnostic setup while excluding camera/model/ROS initialization.
    source = Path(__file__).resolve().parents[1] / 'dobot_nova5_driver/d435_turntable_barcode_node_fast.py'
    node_tree = ast.parse(source.read_text())
    cls = next(item for item in node_tree.body if isinstance(item, ast.ClassDef)
               and item.name == 'D435TurntableBarcodeNode')
    init = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == '__init__')
    start = next(index for index, item in enumerate(init.body)
                 if isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == 'self.confirmation_review')
    end = next(index for index, item in enumerate(init.body)
               if isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == 'self.capture_thread')
    params = dict(confirmation_review_enabled=preview_enabled, confirmation_save_enabled=save_enabled,
                  confirmation_review_dir=str(tmp_path / 'records'),
                  confirmation_review_hold_s=5., confirmation_review_limit=100)
    errors = []
    logger = SimpleNamespace(info=lambda _message: None, warning=errors.append)
    node = SimpleNamespace(get_parameter=lambda key: SimpleNamespace(value=params[key]),
                           get_logger=lambda: logger)
    code = ast.Module(body=init.body[start:end], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(source), 'exec'),
         {'self': node, 'BarcodeConfirmationReview': BarcodeConfirmationReview})
    review = node.confirmation_review
    try:
        if not preview_enabled and not save_enabled:
            assert review is None
        else:
            assert review is not None and not errors
            review.record(picture(), evidence())
            rendered = review.render_on(np.zeros((240, 400, 3), dtype=np.uint8))
            assert bool(rendered.any()) is preview_enabled
    finally:
        if review is not None:
            review.close()
    assert bool(list(tmp_path.rglob('record.json'))) is save_enabled


def test_record_limit_prunes_only_owned_diagnostics_and_preserves_user_files(tmp_path):
    user_folder = tmp_path / "user_photos"
    user_folder.mkdir()
    (user_folder / "keep.png").write_bytes(b"not a diagnostic")
    protected = tmp_path / "d435_confirm_1_g1_f1"
    protected.mkdir()
    (protected / "record.json").write_text('{"kind":"d435_confirmation_review_v1"}')
    (protected / "user_note.txt").write_text("retain this user annotation")
    link = tmp_path / "d435_confirm_2_g1_f1"
    link.symlink_to(user_folder, target_is_directory=True)
    review = BarcodeConfirmationReview(tmp_path, limit=2)
    try:
        for frame in (1, 2, 3):
            review.record(picture(), evidence(frame))
    finally:
        review.close()
    generated = sorted(tmp_path.glob("d435_confirm_100000000*_g4_f*"))
    assert len(generated) == 2
    assert generated[0].name.endswith("f2")
    assert (protected / "user_note.txt").read_text().startswith("retain")
    assert (user_folder / "keep.png").is_file() and link.is_symlink()


def test_disk_failure_does_not_remove_the_visible_history_or_raise_to_detection(tmp_path):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("existing file")
    errors = []
    review = BarcodeConfirmationReview(blocked, on_error=errors.append)
    try:
        review.record(picture(), evidence())
    finally:
        review.close()
    assert errors and not review.worker.is_alive()
    assert review.render_on(np.zeros((240, 400, 3), dtype=np.uint8)).any()
    assert blocked.read_text() == "existing file"


def test_slow_disk_has_a_bounded_queue_and_never_waits_for_writing_in_record(tmp_path):
    entered, release = threading.Event(), threading.Event()
    errors, completed = [], []
    review = BarcodeConfirmationReview(tmp_path, on_error=errors.append)

    def slow_save(snapshot):
        entered.set()
        release.wait(2.)
        completed.append(snapshot.metadata["detection_frame_id"])

    review._save = slow_save
    try:
        review.record(picture(), evidence(1))
        assert entered.wait(1.)
        for frame in range(2, 7):
            review.record(picture(), evidence(frame))
        assert review.jobs.qsize() == 4
        assert len(errors) == 1 and "queue is full" in errors[0]
        assert review.latest.metadata["detection_frame_id"] == 6
    finally:
        release.set()
        review.close()
    assert completed == [1, 2, 3, 4, 5]
    review.record(picture(), evidence(7))
    assert review.latest.metadata["detection_frame_id"] == 6


def test_confirmation_frame_forces_preview_and_v5_alone_enables_review():
    root = Path(__file__).resolve().parents[1]
    node = ast.parse((root / 'dobot_nova5_driver/d435_turntable_barcode_node_fast.py').read_text())
    cls = next(item for item in node.body if isinstance(item, ast.ClassDef) and item.name == 'D435TurntableBarcodeNode')
    capture = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == '_capture_loop')
    gates = [item for item in ast.walk(capture) if isinstance(item, ast.If)
             and 'confirmation_accepted' in ast.unparse(item.test)]
    assert any('preview' in ast.unparse(item.test) and isinstance(item.test, ast.BoolOp) for item in gates)
    for version in ('v3_fast', 'v4', 'v5'):
        launch = (root / f'launch/cosmetic_box_single_arm_cycle_{version}_d435.launch.py').read_text()
        assert ('confirmation_review_enabled:=' in launch) == (version == 'v5')
        assert ('confirmation_save_enabled:=' in launch) == (version == 'v5')


def test_v5_launch_defaults_keep_history_without_automatic_saving():
    root = Path(__file__).resolve().parents[1]
    launch = ast.parse((root / 'launch/cosmetic_box_single_arm_cycle_v5_d435.launch.py').read_text())
    defaults = {item.args[0].value: next(arg.value.value for arg in item.keywords if arg.arg == 'default_value')
                for item in ast.walk(launch)
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                and item.func.id == 'DeclareLaunchArgument'}
    assert defaults['turntable_d435_confirmation_review'] == 'true'
    assert defaults['turntable_d435_confirmation_save'] == 'false'
