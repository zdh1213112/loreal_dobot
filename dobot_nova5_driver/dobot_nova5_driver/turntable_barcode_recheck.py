"""Bounded stopped-face reinspection shared by the V4/V5 turntable scans."""

import json
import math
import time

from .barcode_confirmation_fast import rect_iou


class TurntableBarcodeRecheckMixin:
    def _turntable_barcode_candidate_callback(self, msg):
        try:
            hint = json.loads(msg.data)
            captured_at = float(hint["captured_at"])
            pending = hint["pending"]
            if not isinstance(pending, bool) or not math.isfinite(captured_at):
                return
            if pending:
                rect = tuple(float(v) for v in hint["rect"])
                if (len(rect) != 4 or not all(math.isfinite(v) for v in rect)
                        or min(rect[:2]) < 0 or min(rect[2:]) <= 0):
                    return
                source = str(hint["source"])
                if not (source.startswith("yolo") or source in {
                    "scanner_pattern", "scanner_unverified", "moving_stripes"
                }):
                    return
        except (ValueError, TypeError, KeyError):
            return
        now, now_epoch = time.monotonic(), time.time()
        max_age = float(self.get_parameter("turntable_candidate_max_age_s").value)
        with self.turntable_lock:
            if (not self.turntable_barcode_window_active
                    or captured_at <= self.turntable_candidate_window_epoch
                    or captured_at < self.turntable_candidate_last_frame_epoch
                    or not 0 <= now_epoch - captured_at <= max_age):
                return
            self.turntable_candidate_last_frame_epoch = captured_at
            if not pending:
                # One blurred/missing frame is not enough to discard a weak
                # candidate. It expires by age; ROI/scan invalidation is immediate.
                if hint.get("invalidated") is True:
                    self.turntable_candidate = None
                return
            previous = self.turntable_candidate
            if (previous is None or now - previous["received_at"] > max_age
                    or rect_iou(previous["rect"], rect) < 0.15):
                self.turntable_candidate_since = now
            self.turntable_candidate = {
                "rect": rect, "source": source, "received_at": now,
                "captured_at": captured_at,
            }

    def _recheck_turntable_barcode_candidate(self, *, require_cycle_active, top_reader):
        """Pause only persistent unresolved proposals; success still comes from D435."""

        with self.turntable_lock:
            hint = getattr(self, "turntable_candidate", None)
            if hint is None or self.turntable_state != "RUNNING":
                return False
            now = time.monotonic()
            if (now - hint["received_at"] > float(self.get_parameter(
                    "turntable_candidate_max_age_s").value)
                    or time.time() - hint["captured_at"] > float(self.get_parameter(
                        "turntable_candidate_max_age_s").value)
                    or now - self.turntable_candidate_since < float(self.get_parameter(
                        "turntable_candidate_grace_s").value)
                    or self.turntable_candidate_rechecks >= int(self.get_parameter(
                        "turntable_candidate_max_rechecks").value)
                    or self.turntable_barcode_value):
                return False
            self.turntable_candidate_rechecks += 1

        def require_active():
            if self.turntable_scan_cancel.is_set():
                raise RuntimeError("D435 candidate recheck cancelled by a new placement/operator")
            if not self.running or self.shutting_down:
                raise RuntimeError("node stopped during D435 candidate recheck")
            with self.turntable_lock:
                if not self.turntable_barcode_window_active:
                    raise RuntimeError("D435 scan window closed during candidate recheck")
            if require_cycle_active:
                self._require_cycle_active("during D435 candidate recheck")

        def confirmed():
            with self.turntable_lock:
                side = bool(self.turntable_barcode_value)
            return side or bool(top_reader())

        require_active()
        if confirmed():
            return False
        wait_s = float(self.get_parameter("turntable_candidate_recheck_s").value)
        if not math.isfinite(wait_s) or not 0 <= wait_s <= 2:
            raise ValueError("turntable_candidate_recheck_s must be within [0, 2] seconds")
        self._publish_status(
            f"D435 unresolved {hint['source']} candidate; briefly stopping the "
            "turntable for clearer frames (candidate is not barcode confirmation)"
        )
        self._toggle_turntable("RUNNING", "STOPPED", "candidate recheck stop")
        deadline = time.monotonic() + wait_s
        while True:
            require_active()
            if confirmed():
                self._publish_status("D435 candidate recheck confirmed a barcode; turntable remains stopped")
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.005, remaining))
        require_active()
        self._wait_for_secondary_y_clearance(
            "turntable resume after candidate recheck", require_cycle_active=require_cycle_active,
        )
        require_active()
        # A confirmation received while waiting for 102 clearance also wins.
        if confirmed():
            return True
        self._publish_status(
            "D435 candidate still unconfirmed; resuming the remaining rotation "
            "search without counting the stopped interval"
        )
        self._toggle_turntable("STOPPED", "RUNNING", "candidate recheck resume")
        with self.turntable_lock:
            self.turntable_candidate_since = time.monotonic()
        return True
