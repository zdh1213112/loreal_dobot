"""Hardware-free helpers that keep D405 diagnostics off the grasp critical path."""

import logging
import math
import threading
import time
from collections import defaultdict
from queue import Empty, Full, Queue

import numpy as np


def latest_synchronized_frames(pipeline, *, max_poll=32):
    """Drain available frames without mixing RGB/IR/depth or waiting again."""
    frames = pipeline.wait_for_frames()
    poll = getattr(pipeline, "poll_for_frames", None)
    if poll is None:
        return frames
    for _ in range(max(0, min(64, int(max_poll)))):
        candidate = poll()
        if not candidate:
            break
        # A duplicate synchronized set must not replace a newer color capture.
        try:
            if (candidate.get_color_frame().get_frame_number()
                    <= frames.get_color_frame().get_frame_number()):
                continue
        except AttributeError:
            pass  # Older SDK builds still preserve poll delivery order.
        frames = candidate
    return frames


def pregrasp_recovery_progress(
    *, observations, baseline_capture_s, baseline_count, started_s,
    base_deadline_s, now_monotonic, now_epoch, max_age_s, grace_s,
):
    """Permit extra WAITING only; this never makes a stale pose usable."""
    eligible = []
    for item in observations:
        receive_age = now_monotonic - item.received_at_monotonic
        capture_age = now_epoch - item.frame_time_s
        if (item.count > baseline_count
                and math.isfinite(item.frame_time_s)
                and started_s <= item.received_at_monotonic < base_deadline_s
                and item.frame_time_s > baseline_capture_s
                and 0.0 <= receive_age <= 0.20
                and max_age_s < capture_age <= max_age_s + min(1.0, grace_s)):
            eligible.append(item)
    if not eligible:
        return None
    latest = max(eligible, key=lambda item: item.frame_time_s)
    return {
        "capture_age_ms": round((now_epoch - latest.frame_time_s) * 1000.0, 1),
        "receive_age_ms": round((now_monotonic - latest.received_at_monotonic) * 1000.0, 1),
    }


class DiagnosticWriter:
    """Bounded, non-blocking diagnostic sink; it never handles robot evidence.

    Callers own snapshotting mutable inputs before submitting. A full queue
    drops only the new diagnostic job, never a detection or a camera frame.
    """

    def __init__(self, write, *, capacity=8):
        self._write = write
        self._queue = Queue(maxsize=max(1, capacity))
        self._lock = threading.Lock()
        self._closed = False
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name="d405-top-debug-writer", daemon=True
        )
        self._thread.start()

    def submit(self, *args, **kwargs):
        with self._lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait((args, kwargs))
            except Full:
                self.dropped += 1
                if self.dropped == 1 or self.dropped % 50 == 0:
                    logging.warning(
                        "[D405] Top-barcode debug queue full; skipped %d diagnostic "
                        "job(s), detection continues unchanged", self.dropped
                    )
                return False
        return True

    def _run(self):
        while True:
            try:
                args, kwargs = self._queue.get(timeout=0.1)
            except Empty:
                with self._lock:
                    if self._closed and self._queue.empty():
                        return
                continue
            try:
                self._write(*args, **kwargs)
            except Exception:
                logging.exception("[D405] Background top-barcode debug save failed")
            finally:
                self._queue.task_done()

    def close(self, *, timeout=2.0):
        """Drain accepted jobs, with a bounded wait even for an unresponsive disk."""
        with self._lock:
            self._closed = True
        self._thread.join(timeout=max(0.0, timeout))
        return not self._thread.is_alive()


def paint_depth_sorted_points(canvas, px, py, depths, bgr):
    """Paint exactly the old far-to-near result without a Python point loop.

    Pick one winner per pixel explicitly: repeated-index NumPy assignment
    alone would not guarantee which depth/color survives. Only display pixels
    are touched; the perception cloud is neither filtered nor downsampled.
    """
    height, width = canvas.shape[:2]
    order = np.argsort(depths)
    visible = order[
        (px[order] >= 0) & (px[order] < width)
        & (py[order] >= 0) & (py[order] < height)
    ]
    pixels = py[visible] * width + px[visible]
    _, first = np.unique(pixels, return_index=True)
    winners = visible[first]
    canvas[py[winners], px[winners]] = bgr[winners]


class LocalizationTiming:
    """Aggregate CPU wall times until the current fresh target is published."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.active = False
        self.frame_started = None
        self.last_slow_frame_at = float("-inf")

    def begin_frame(self):
        self.frame_started = self._clock()
        self.frame_stages = {}

    def start(self):
        self.started = self._clock()
        self.stages = defaultdict(float)
        self.frames = 0
        self.active = True

    def record(self, stage, started):
        elapsed = max(0.0, self._clock() - started)
        if self.active:
            self.stages[stage] += elapsed
        if self.frame_started is not None:
            # Geometry is recorded at initial publication and again at its end.
            # Keep the final span, rather than counting it twice in frame timing.
            self.frame_stages[stage] = elapsed

    def slow_frame_report(self, *, capture_s, returned_s, now_epoch, limit_s=0.65):
        now = self._clock()
        if (self.frame_started is None or not math.isfinite(capture_s) or capture_s <= 0
                or now_epoch - capture_s <= limit_s or now - self.last_slow_frame_at < 2.0):
            return None
        self.last_slow_frame_at = now
        return {
            "capture_age_ms": round((now_epoch - capture_s) * 1000.0, 1),
            "age_at_camera_read_ms": round((returned_s - capture_s) * 1000.0, 1),
            "loop_ms": round((now - self.frame_started) * 1000.0, 1),
            "stage_ms": {key: round(value * 1000.0, 1) for key, value in self.frame_stages.items()},
        }

    def finish(self, *, geometry_started):
        self.record("geometry", geometry_started)
        self.active = False
        return {
            "request_to_target_ms": round((self._clock() - self.started) * 1000, 1),
            "frames": self.frames,
            "stage_ms": {key: round(value * 1000, 1) for key, value in self.stages.items()},
        }


def post_stop_wait_diagnostics(
    *, observations, baseline_count, stop_s, frame_barrier, request_epoch,
    applied_epoch, now_monotonic, now_epoch, max_age_s, required,
    handoff_fresh, handoff_clear, handoff_state, metadata,
    stable_messages_since_wait, last_pair_check,
):
    """Describe rejection facts only; never authorize or modify a grasp.

    This runs once on timeout, not for every tracking frame. Stable-topic
    messages and pregrasp-topic samples are deliberately reported separately.
    """
    candidates = [item for item in observations if item.count > baseline_count]
    rejected = defaultdict(int)
    fresh_count = 0
    for item in candidates:
        receive_age = now_monotonic - item.received_at_monotonic
        capture_age = now_epoch - item.frame_time_s
        if not math.isfinite(receive_age) or not math.isfinite(capture_age):
            rejected['nonfinite_time'] += 1
        elif item.received_at_monotonic < stop_s:
            rejected['received_before_stop'] += 1
        elif item.frame_time_s <= frame_barrier:
            rejected['captured_before_barrier'] += 1
        elif receive_age < 0.0 or capture_age < 0.0:
            rejected['future_timestamp'] += 1
        elif receive_age > max_age_s:
            rejected['receipt_too_old'] += 1
        elif capture_age > max_age_s:
            rejected['capture_too_old'] += 1
        elif applied_epoch < request_epoch:
            rejected['acquisition_not_acknowledged'] += 1
        else:
            fresh_count += 1

    reasons = []
    if applied_epoch < request_epoch:
        reasons.append('waiting_for_acquisition_ack')
    if fresh_count < required:
        reasons.append('insufficient_fresh_tracking_samples')
    if not handoff_fresh:
        reasons.append('no_new_handoff_state')
    if not handoff_clear:
        reasons.append('handoff_not_clear')
    missing = [name for name, value in metadata.items() if value is None]
    if missing:
        reasons.append('missing_dimensions')

    def milliseconds(age):
        return round(age * 1000.0, 1) if math.isfinite(age) else None

    latest = max(candidates, key=lambda item: item.received_at_monotonic, default=None)
    return {
        'waiting_for': reasons,
        'request_epoch': request_epoch,
        'applied_epoch': applied_epoch,
        'frame_barrier_epoch': frame_barrier,
        'required_tracking_samples': required,
        'candidate_tracking_samples': len(candidates),
        'fresh_tracking_samples': fresh_count,
        'stable_topic_messages_since_wait': stable_messages_since_wait,
        'rejected_tracking_samples': dict(rejected),
        'max_age_ms': milliseconds(max_age_s),
        'latest_receive_age_ms': None if latest is None else milliseconds(now_monotonic - latest.received_at_monotonic),
        'latest_capture_age_ms': None if latest is None else milliseconds(now_epoch - latest.frame_time_s),
        'handoff_fresh': bool(handoff_fresh),
        'handoff_clear': bool(handoff_clear),
        'handoff_state': handoff_state,
        'missing_dimensions': missing,
        'last_pair_check': last_pair_check,
    }
