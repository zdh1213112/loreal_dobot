"""Bounded diagnostic snapshots of decisions already made by D435.

No camera, model, ROS or robot calls. Optional PNG/JPEG encoding and disk IO
run on a single bounded worker; preview-only mode starts no writer. This class
never accepts or rejects a barcode.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import re
import threading
import time

import cv2
import numpy as np


RECORD_KIND = "d435_confirmation_review_v1"
RECORD_NAME = re.compile(r"d435_confirm_[0-9]+_g[0-9]+_f[0-9]+")
RECORD_FILES = {"raw.png", "annotated.jpg", "crop.png", "record.json"}


@dataclass(frozen=True)
class ConfirmationSnapshot:
    image: np.ndarray
    annotated: np.ndarray
    metadata: dict
    noted_at: float


class BarcodeConfirmationReview:
    def __init__(self, directory, *, hold_s=5.0, limit=100, clock=time.monotonic,
                 preview_enabled=True, save_enabled=True,
                 on_saved=lambda _path: None, on_error=lambda _error: None):
        if not math.isfinite(hold_s) or hold_s <= 0 or int(limit) < 1:
            raise ValueError("confirmation review needs a positive duration and record limit")
        self.directory, self.hold_s, self.limit = Path(directory).expanduser(), hold_s, int(limit)
        self.clock, self.on_saved, self.on_error = clock, on_saved, on_error
        self.preview_enabled, self.save_enabled = bool(preview_enabled), bool(save_enabled)
        self.lock = threading.Lock()
        self.latest = None
        self.jobs = queue.Queue(maxsize=4) if self.save_enabled else None
        self.closed = threading.Event()
        self.worker = None
        if self.save_enabled:
            self.worker = threading.Thread(target=self._run, name="d435-review-writer", daemon=True)
            self.worker.start()

    @staticmethod
    def _notify(callback, value):
        try:
            callback(value)
        except Exception:
            pass

    def record(self, image, metadata):
        if self.closed.is_set():
            return
        raw = np.array(image, copy=True)
        data = dict(metadata, kind=RECORD_KIND)
        x, y, width, height = [int(value) for value in data["rect"]]
        if raw.ndim != 3 or raw.shape[2] != 3 or width <= 0 or height <= 0:
            raise ValueError("invalid confirmation image/rectangle")
        annotated = raw.copy()
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 0, 255), 3)
        score = f"{data['confidence']:.2f}" if data['source'].startswith('yolo') else "n/a"
        cv2.putText(annotated, f"CONFIRMED REGION {data['source']} score={score}",
                    (max(0, x), max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (255, 0, 255), 2, cv2.LINE_AA)
        snapshot = ConfirmationSnapshot(raw, annotated, data, self.clock())
        with self.lock:
            if self.closed.is_set():
                return
            self.latest = snapshot
            if self.save_enabled:
                try:
                    self.jobs.put_nowait(snapshot)
                except queue.Full:
                    self._notify(self.on_error, "confirmation snapshot queue is full; this record was not saved")

    def render_on(self, display):
        if not self.preview_enabled:
            return display
        with self.lock:
            snapshot = self.latest
        if snapshot is None:
            return display
        age = self.clock() - snapshot.noted_at
        if age < 0 or age > self.hold_s:
            return display
        height, width = display.shape[:2]
        if height < 160 or width < 240:
            return display
        thumb_width = min(440, width // 2)
        thumb_height = int(round(thumb_width * snapshot.image.shape[0] / snapshot.image.shape[1]))
        thumb_height = min(thumb_height, height // 2)
        thumb = cv2.resize(snapshot.annotated, (thumb_width, thumb_height))
        left, bottom = width - thumb_width - 12, height - 12
        top = bottom - thumb_height
        display[top:bottom, left:left + thumb_width] = thumb
        cv2.rectangle(display, (left, top), (left + thumb_width, bottom), (255, 0, 255), 2)
        cv2.rectangle(display, (left, top - 43), (left + thumb_width, top), (20, 20, 20), -1)
        title = f"LAST CONFIRM - HISTORY {age:.1f}s ago"
        title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1., 1)[0][0]
        title_scale = min(.62, (thumb_width - 10) / max(1, title_width))
        cv2.putText(display, title, (left + 5, top - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 180, 255), 1, cv2.LINE_AA)
        data = snapshot.metadata
        score = f"{data['confidence']:.2f}" if data['source'].startswith('yolo') else "n/a"
        cv2.putText(display, f"{data['source']}  confidence={score}",
                    (left + 5, top - 7), cv2.FONT_HERSHEY_SIMPLEX, min(.55, title_scale),
                    (255, 180, 255), 1, cv2.LINE_AA)
        return display

    def _save(self, snapshot):
        data = dict(snapshot.metadata, complete=False)
        name = (f"d435_confirm_{int(data['capture_received_unix_ns'])}"
                f"_g{int(data['generation'])}_f{int(data['detection_frame_id'])}")
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / name
        destination.mkdir(exist_ok=False)
        record = destination / "record.json"
        record.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        x, y, width, height = data["rect"]
        image_height, image_width = snapshot.image.shape[:2]
        crop = snapshot.image[max(0, y):min(image_height, y + height),
                              max(0, x):min(image_width, x + width)]
        try:
            for filename, image, options in (
                ("raw.png", snapshot.image, [cv2.IMWRITE_PNG_COMPRESSION, 3]),
                ("annotated.jpg", snapshot.annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]),
                ("crop.png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]),
            ):
                if not image.size or not cv2.imwrite(str(destination / filename), image, options):
                    raise OSError(f"could not save {filename}")
            data["complete"] = True
            record.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            self._notify(self.on_saved, str(destination))
        finally:
            self._prune()

    def _prune(self):
        owned = []
        for path in self.directory.iterdir():
            if path.is_symlink() or not path.is_dir() or not RECORD_NAME.fullmatch(path.name):
                continue
            try:
                data = json.loads((path / "record.json").read_text())
            except (OSError, ValueError):
                continue
            if data.get("kind") == RECORD_KIND:
                owned.append(path)
        owned.sort(key=lambda path: int(path.name.split("_")[2]))
        for path in owned[:-self.limit]:
            children = list(path.iterdir())
            # Never remove user-added files, symlinks or unrelated directories.
            if any(child.name not in RECORD_FILES or child.is_symlink() or not child.is_file()
                   for child in children):
                continue
            for child in children:
                child.unlink()
            path.rmdir()

    def _run(self):
        while not self.closed.is_set() or not self.jobs.empty():
            try:
                snapshot = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self._save(snapshot)
            except Exception as error:
                self._notify(self.on_error, str(error))
            finally:
                self.jobs.task_done()

    def close(self, timeout=2.0):
        with self.lock:
            self.closed.set()
        if self.worker is not None:
            self.worker.join(timeout=timeout)
