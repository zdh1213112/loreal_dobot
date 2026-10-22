"""Triggered D435 side-barcode scanner for the opt-in fast V3 fork."""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from .barcode_confirmation_fast import BarcodeEvidenceTracks
from .d435_confirmation_review import BarcodeConfirmationReview
from .turntable_barcode_detector_fast import (
    DEFAULT_BARCODE_MODEL,
    TurntableBarcodeDetector,
)
from .turntable_v3 import (
    MIN_TURNTABLE_ROI_SIZE_PX,
    normalize_image_roi,
)


D435_BRIGHTNESS_EXPOSURE_STEP = 10.0
D435_BRIGHTNESS_BUTTON_WIDTH = 165
D435_BRIGHTNESS_BUTTON_HEIGHT = 46
D435_BRIGHTNESS_BUTTON_GAP = 10
D435_BRIGHTNESS_BUTTON_MARGIN = 18


def brightness_button_rects(
    image_width: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return native-image rectangles for the darker and brighter buttons."""

    right = max(2 * D435_BRIGHTNESS_BUTTON_WIDTH + D435_BRIGHTNESS_BUTTON_GAP,
                int(image_width) - D435_BRIGHTNESS_BUTTON_MARGIN)
    bright_left = right - D435_BRIGHTNESS_BUTTON_WIDTH
    dark_right = bright_left - D435_BRIGHTNESS_BUTTON_GAP
    dark_left = dark_right - D435_BRIGHTNESS_BUTTON_WIDTH
    top = D435_BRIGHTNESS_BUTTON_MARGIN
    bottom = top + D435_BRIGHTNESS_BUTTON_HEIGHT
    return (
        (dark_left, top, dark_right, bottom),
        (bright_left, top, right, bottom),
    )


def brightness_button_direction(x: int, y: int, image_width: int) -> int:
    """Return -1/1 for a brightness button hit, otherwise zero."""

    darker, brighter = brightness_button_rects(image_width)
    for direction, (left, top, right, bottom) in (
        (-1, darker),
        (1, brighter),
    ):
        if left <= int(x) < right and top <= int(y) < bottom:
            return direction
    return 0


class D435TurntableBarcodeNode(Node):
    def __init__(self) -> None:
        super().__init__("d435_turntable_barcode_node_fast")
        self.declare_parameter("serial_number", "254322071102")
        self.declare_parameter("color_width", 1280)
        self.declare_parameter("color_height", 720)
        self.declare_parameter("fps", 30)
        # The D435 RGB sensor's automatic exposure commonly chooses a shutter
        # long enough to blur barcode bars on the moving turntable.  A short
        # manual exposure keeps the bars spatially separated; gain restores
        # brightness without reintroducing motion smear.
        self.declare_parameter("auto_exposure", False)
        # Keep these ROS parameters integral because launch/CLI values such as
        # exposure:=50 are parsed as INTEGER. RealSense options are converted
        # to float only when they are applied to the sensor.
        self.declare_parameter("exposure", 70)
        # Keep the short shutter that freezes the rotating bars, and recover
        # indoor brightness with sensor gain instead of a longer exposure.
        # _set_sensor_option_safe() clamps this to the actual camera range.
        self.declare_parameter("gain", 128)
        self.declare_parameter("auto_exposure_priority", 0.0)
        self.declare_parameter("model_path", DEFAULT_BARCODE_MODEL)
        # The fixed black robot pedestal in the current cell produced a
        # repeatable YOLO false positive at 0.47.  Real package barcodes in
        # the recorded cell tests were 0.72 or higher.
        self.declare_parameter("model_confidence", 0.60)
        self.declare_parameter("model_image_size", 640)
        self.declare_parameter("inference_provider", "cuda")
        self.declare_parameter("require_cuda", True)
        self.declare_parameter("scanner_assist_enabled", True)
        self.declare_parameter("scanner_assist_stable_hits", 3)
        self.declare_parameter("scanner_assist_hit_gap_s", 0.25)
        self.declare_parameter("scanner_assist_min_iou", 0.20)
        self.declare_parameter("stable_hits", 2)
        self.declare_parameter("stable_hit_gap_s", 0.70)
        self.declare_parameter("yolo_min_iou", 0.15)
        # A wide turntable ROI loses barcode detail when letterboxed into the
        # square YOLO input. Retry two overlapping sharpened views only after
        # the established full-ROI inference misses.
        self.declare_parameter("wide_roi_fallback_enabled", True)
        self.declare_parameter("wide_roi_fallback_aspect_ratio", 2.0)
        self.declare_parameter("wide_roi_tile_fraction", 0.70)
        self.declare_parameter("wide_roi_unsharp_sigma", 1.2)
        self.declare_parameter("wide_roi_unsharp_amount", 1.0)
        self.declare_parameter("detail_roi_x", 280)
        self.declare_parameter("detail_roi_y", 220)
        self.declare_parameter("detail_roi_width", 760)
        self.declare_parameter("detail_roi_height", 440)
        self.declare_parameter("full_frame_interval", 6)
        self.declare_parameter("yolo_min_candidate_area_ratio", 0.01)
        self.declare_parameter("moving_stripes_enabled", True)
        self.declare_parameter("moving_stripes_stable_hits", 3)
        # The RealSense stream itself limits this loop to 30 FPS. Zero means
        # every acquired frame is eligible; inference time remains the natural
        # back-pressure instead of an additional timer throttle.
        self.declare_parameter("detect_interval_s", 0.0)
        self.declare_parameter("roi_x", 0)
        self.declare_parameter("roi_y", 0)
        self.declare_parameter("roi_width", 0)
        self.declare_parameter("roi_height", 0)
        self.declare_parameter("preview", True)
        # V5 opts in; earlier launch files retain their original preview/IO.
        self.declare_parameter("confirmation_review_enabled", False)
        self.declare_parameter("confirmation_save_enabled", False)
        self.declare_parameter("confirmation_review_hold_s", 5.0)
        self.declare_parameter("confirmation_review_limit", 100)
        self.declare_parameter("confirmation_review_dir", "/home/zdh/ffs_ws/d435_barcode_confirmations")
        self.declare_parameter(
            "sample_save_dir", "/home/zdh/ffs_ws/d435_barcode_samples"
        )
        self.declare_parameter(
            "preview_topic", "/vision_panel/d435_turntable/image/compressed"
        )
        self.declare_parameter("event_topic", "/vision_panel/d435_turntable/event")
        # The old 0.10 s/JPEG-80 preview was capped at 10 FPS and looked much
        # softer than the raw 1280x720 frames. Keep inference on the raw image,
        # but publish a higher-quality local preview at up to the camera's
        # configured 30 FPS. Inference load may temporarily reduce delivery.
        self.declare_parameter("preview_publish_interval_s", 0.100)
        self.declare_parameter("preview_jpeg_quality", 92)
        self.declare_parameter("trigger_topic", "/trigger_turntable_barcode")
        self.declare_parameter(
            "continuous_trigger_topic", "/trigger_d435_continuous_detection"
        )
        self.declare_parameter("result_topic", "/turntable_barcode_result")
        self.declare_parameter("candidate_topic", "/turntable_barcode_candidate")
        self.declare_parameter(
            "continuous_result_topic", "/d435_continuous_barcode_result"
        )
        self.declare_parameter(
            "continuous_presence_topic", "/d435_continuous_barcode_presence"
        )
        self.declare_parameter("continuous_rearm_absence_s", 1.0)
        self.declare_parameter("ready_topic", "/turntable_barcode_camera_ready")

        ready_qos = QoSProfile(depth=1)
        ready_qos.reliability = ReliabilityPolicy.RELIABLE
        ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.ready_publisher = self.create_publisher(
            Bool, str(self.get_parameter("ready_topic").value), ready_qos
        )
        self.result_publisher = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), 10
        )
        self.candidate_publisher = self.create_publisher(
            String, str(self.get_parameter("candidate_topic").value), 10
        )
        self.continuous_result_publisher = self.create_publisher(
            String,
            str(self.get_parameter("continuous_result_topic").value),
            10,
        )
        self.continuous_presence_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("continuous_presence_topic").value),
            ready_qos,
        )
        self.status_publisher = self.create_publisher(
            String, "/turntable_barcode_camera_status", 10
        )
        self.preview_publisher = self.create_publisher(
            CompressedImage, str(self.get_parameter("preview_topic").value), 1
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("trigger_topic").value),
            self._trigger_callback,
            10,
        )
        continuous_qos = QoSProfile(depth=1)
        continuous_qos.reliability = ReliabilityPolicy.RELIABLE
        continuous_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool,
            str(self.get_parameter("continuous_trigger_topic").value),
            self._continuous_trigger_callback,
            continuous_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("event_topic").value),
            self._preview_event_callback,
            10,
        )

        self.state_lock = threading.Lock()
        self.scan_active = False
        self.continuous_scan_active = False
        self.scan_generation = 0
        self.evidence_tracks = BarcodeEvidenceTracks()
        self.detection_frame_id = 0
        self.last_confirmation_evidence = None
        self.candidate = ""
        self.candidate_source = ""
        self.candidate_rect: tuple[int, int, int, int] | None = None
        self.candidate_first_rect: tuple[int, int, int, int] | None = None
        self.candidate_hits = 0
        self.candidate_last_at = 0.0
        self.continuous_last_report_value = ""
        self.continuous_last_report_at = 0.0
        self.continuous_last_seen_at = 0.0
        self.continuous_presence_latched = False
        self._publish_continuous_presence(False)
        self.roi_lock = threading.Lock()
        self.manual_roi: tuple[int, int, int, int] | None = None
        self.roi_generation = 0
        self.roi_drawing = False
        self.roi_start = (0, 0)
        self.roi_end = (0, 0)
        self.preview_image_size = (
            int(self.get_parameter("color_width").value),
            int(self.get_parameter("color_height").value),
        )
        self.running = True
        self.pipeline = None
        self.color_sensor = None
        self.sample_save_dir = Path(
            str(self.get_parameter("sample_save_dir").value)
        ).expanduser()
        self.sample_save_lock = threading.Lock()
        self.pending_sample_saves = 0
        self.camera_option_lock = threading.Lock()
        self.manual_exposure = float(self.get_parameter("exposure").value)
        self.manual_gain = float(self.get_parameter("gain").value)
        self.manual_auto_exposure = bool(
            self.get_parameter("auto_exposure").value
        )
        self.camera_settings_text = "camera settings pending"
        self.detect_rate_started_at = time.monotonic()
        self.detect_rate_count = 0
        self.detect_rate_fps = 0.0
        self.detector = TurntableBarcodeDetector(
            model_path=str(self.get_parameter("model_path").value),
            confidence=float(self.get_parameter("model_confidence").value),
            image_size=int(self.get_parameter("model_image_size").value),
            inference_provider=str(
                self.get_parameter("inference_provider").value
            ),
            require_cuda=bool(self.get_parameter("require_cuda").value),
            scanner_assist=bool(
                self.get_parameter("scanner_assist_enabled").value
            ),
            wide_roi_fallback=bool(
                self.get_parameter("wide_roi_fallback_enabled").value
            ),
            wide_roi_aspect_ratio=float(
                self.get_parameter("wide_roi_fallback_aspect_ratio").value
            ),
            wide_roi_tile_fraction=float(
                self.get_parameter("wide_roi_tile_fraction").value
            ),
            wide_roi_unsharp_sigma=float(
                self.get_parameter("wide_roi_unsharp_sigma").value
            ),
            wide_roi_unsharp_amount=float(
                self.get_parameter("wide_roi_unsharp_amount").value
            ),
            detail_roi=(
                int(self.get_parameter("detail_roi_x").value),
                int(self.get_parameter("detail_roi_y").value),
                int(self.get_parameter("detail_roi_width").value),
                int(self.get_parameter("detail_roi_height").value),
            ),
            full_frame_interval=int(
                self.get_parameter("full_frame_interval").value
            ),
            yolo_min_candidate_area_ratio=float(
                self.get_parameter("yolo_min_candidate_area_ratio").value
            ),
            moving_stripes_enabled=bool(
                self.get_parameter("moving_stripes_enabled").value
            ),
        )
        self.confirmation_review = None
        review_enabled = bool(self.get_parameter("confirmation_review_enabled").value)
        save_enabled = bool(self.get_parameter("confirmation_save_enabled").value)
        if review_enabled or save_enabled:
            try:
                self.confirmation_review = BarcodeConfirmationReview(
                    str(self.get_parameter("confirmation_review_dir").value),
                    hold_s=float(self.get_parameter("confirmation_review_hold_s").value),
                    limit=int(self.get_parameter("confirmation_review_limit").value),
                    preview_enabled=review_enabled,
                    save_enabled=save_enabled,
                    on_saved=lambda path: self.get_logger().info(f"D435 confirmation evidence saved: {path}"),
                    on_error=lambda error: self.get_logger().warning(f"D435 confirmation review: {error}"),
                )
            except Exception as error:
                self.get_logger().warning(f"D435 confirmation review unavailable: {error}")
        self.get_logger().info(
            f"D435 confirmation history={'on' if review_enabled else 'off'}, "
            f"automatic evidence saving={'on' if save_enabled else 'off'}"
        )
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="d435-turntable-barcode",
            daemon=True,
        )
        self.capture_thread.start()

    @staticmethod
    def _set_sensor_option_safe(sensor, option, requested: float) -> float | None:
        if sensor is None or not sensor.supports(option):
            return None
        option_range = sensor.get_option_range(option)
        value = max(float(option_range.min), min(float(option_range.max), float(requested)))
        step = float(option_range.step)
        if step > 0.0:
            value = float(option_range.min) + round(
                (value - float(option_range.min)) / step
            ) * step
            value = max(float(option_range.min), min(float(option_range.max), value))
        sensor.set_option(option, value)
        return float(sensor.get_option(option))

    def _configure_color_sensor(self, pipeline_profile) -> None:
        device = pipeline_profile.get_device()
        try:
            sensor = device.first_color_sensor()
        except (AttributeError, RuntimeError):
            sensor = next(
                (
                    candidate
                    for candidate in device.query_sensors()
                    if candidate.supports(rs.option.exposure)
                ),
                None,
            )
        if sensor is None:
            raise RuntimeError("D435 color sensor was not found")
        self.color_sensor = sensor
        with self.camera_option_lock:
            auto_exposure = self.manual_auto_exposure
            priority = self._set_sensor_option_safe(
                sensor,
                rs.option.auto_exposure_priority,
                float(self.get_parameter("auto_exposure_priority").value),
            )
            auto_value = self._set_sensor_option_safe(
                sensor,
                rs.option.enable_auto_exposure,
                1.0 if auto_exposure else 0.0,
            )
            if auto_exposure:
                self.camera_settings_text = (
                    f"AE=ON priority={priority if priority is not None else 'unsupported'}"
                )
                return
            exposure = self._set_sensor_option_safe(
                sensor,
                rs.option.exposure,
                self.manual_exposure,
            )
            gain = self._set_sensor_option_safe(
                sensor,
                rs.option.gain,
                self.manual_gain,
            )
            if exposure is not None:
                self.manual_exposure = exposure
            if gain is not None:
                self.manual_gain = gain
            self._update_camera_settings_text(auto_value)

    def _update_camera_settings_text(self, auto_value: float | None = 0.0) -> None:
        self.camera_settings_text = (
            f"AE=OFF exposure={self.manual_exposure:.1f} "
            f"gain={self.manual_gain:.1f}"
        )
        if auto_value is None:
            self.camera_settings_text += " (AE control unsupported)"

    def _adjust_manual_brightness(self, direction: int) -> None:
        """Adjust only D435 manual exposure, following the old panel logic."""

        direction = -1 if int(direction) < 0 else 1
        with self.camera_option_lock:
            sensor = self.color_sensor
            if sensor is None:
                self._publish_status("D435 brightness adjustment ignored: camera is not ready")
                return
            if self.manual_auto_exposure:
                self._publish_status(
                    "D435 brightness adjustment ignored: disable auto exposure first"
                )
                return
            before = self.manual_exposure
            exposure = self._set_sensor_option_safe(
                sensor,
                rs.option.exposure,
                before + direction * D435_BRIGHTNESS_EXPOSURE_STEP,
            )
            if exposure is None:
                self._publish_status(
                    "D435 brightness adjustment unavailable: exposure is unsupported"
                )
                return
            self.manual_exposure = exposure
            gain = self._set_sensor_option_safe(
                sensor,
                rs.option.gain,
                self.manual_gain,
            )
            if gain is not None:
                self.manual_gain = gain
            self._update_camera_settings_text()
            changed = abs(self.manual_exposure - before) > 1e-6
            text = (
                f"D435 brightness {'increased' if direction > 0 else 'decreased'}: "
                f"exposure={self.manual_exposure:.1f}, gain={self.manual_gain:.1f}"
                if changed
                else f"D435 brightness already at the {'maximum' if direction > 0 else 'minimum'} exposure"
            )
        self._publish_status(text)

    def _publish_ready(self, ready: bool) -> None:
        message = Bool()
        message.data = bool(ready)
        self.ready_publisher.publish(message)

    def _publish_continuous_presence(self, present: bool) -> None:
        message = Bool()
        message.data = bool(present)
        self.continuous_presence_publisher.publish(message)

    def _publish_status(self, text: str) -> None:
        message = String()
        message.data = text
        self.status_publisher.publish(message)
        self.get_logger().info(text)

    def _trigger_callback(self, message: Bool) -> None:
        with self.state_lock:
            self.scan_active = bool(message.data)
            self.scan_generation += 1
            self.candidate = ""
            self.candidate_source = ""
            self.candidate_rect = None
            self.candidate_first_rect = None
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
            self.evidence_tracks.clear()
            generation = self.scan_generation
        self._publish_status(
            f"D435 turntable barcode scanner {'armed' if message.data else 'disarmed'} "
            f"(generation={generation})"
        )

    def _continuous_trigger_callback(self, message: Bool) -> None:
        enabled = bool(message.data)
        with self.state_lock:
            self.continuous_scan_active = enabled
            self.candidate = ""
            self.candidate_source = ""
            self.candidate_rect = None
            self.candidate_first_rect = None
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
            self.evidence_tracks.clear()
            self.continuous_last_report_value = ""
            self.continuous_last_report_at = 0.0
            self.continuous_last_seen_at = 0.0
            self.continuous_presence_latched = False
        self._publish_continuous_presence(False)
        self._publish_status(
            f"D435 continuous detection {'enabled' if enabled else 'disabled'}; "
            "turntable motion is not controlled by this switch"
        )

    def _roi_bounds_with_generation(
        self, image: np.ndarray
    ) -> tuple[tuple[int, int, int, int], int]:
        height, width = image.shape[:2]
        with self.roi_lock:
            manual_roi = self.manual_roi
            generation = self.roi_generation
        if manual_roi is not None:
            left, top, right, bottom = manual_roi
            left = max(0, min(width - 1, int(left)))
            top = max(0, min(height - 1, int(top)))
            right = max(left + 1, min(width, int(right)))
            bottom = max(top + 1, min(height, int(bottom)))
            return (left, top, right, bottom), generation

        x = max(0, min(width - 1, int(self.get_parameter("roi_x").value)))
        y = max(0, min(height - 1, int(self.get_parameter("roi_y").value)))
        roi_width = int(self.get_parameter("roi_width").value)
        roi_height = int(self.get_parameter("roi_height").value)
        right = width if roi_width <= 0 else min(width, x + roi_width)
        bottom = height if roi_height <= 0 else min(height, y + roi_height)
        return (x, y, right, bottom), generation

    def _roi_bounds(self, image: np.ndarray) -> tuple[int, int, int, int]:
        return self._roi_bounds_with_generation(image)[0]

    def _reset_candidate_for_roi_change(self) -> None:
        """Discard evidence collected under a previous spatial selection."""

        with self.state_lock:
            self.candidate = ""
            self.candidate_source = ""
            self.candidate_rect = None
            self.candidate_first_rect = None
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
            self.evidence_tracks.clear()
            self.continuous_last_seen_at = 0.0
            self.continuous_presence_latched = False
        self._publish_continuous_presence(False)

    def _mouse_roi_callback(self, event, x, y, flags, param) -> None:
        del flags, param
        with self.roi_lock:
            width, height = self.preview_image_size
        point = (
            max(0, min(max(0, width - 1), int(x))),
            max(0, min(max(0, height - 1), int(y))),
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            brightness_direction = brightness_button_direction(
                point[0], point[1], width
            )
            if brightness_direction:
                self._adjust_manual_brightness(brightness_direction)
                return
            with self.roi_lock:
                self.roi_drawing = True
                self.roi_start = point
                self.roi_end = point
            return
        if event == cv2.EVENT_MOUSEMOVE:
            with self.roi_lock:
                if self.roi_drawing:
                    self.roi_end = point
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            self._clear_manual_roi(width, height)
            return
        if event != cv2.EVENT_LBUTTONUP:
            return

        with self.roi_lock:
            if not self.roi_drawing:
                return
            self.roi_drawing = False
            self.roi_end = point
            start = self.roi_start
            end = self.roi_end
        new_roi = normalize_image_roi(start, end, width, height)
        if new_roi is None:
            self._publish_status(
                "D435 ROI drag ignored: select at least "
                f"{MIN_TURNTABLE_ROI_SIZE_PX}x{MIN_TURNTABLE_ROI_SIZE_PX}px"
            )
            return
        with self.roi_lock:
            self.manual_roi = new_roi
            self.roi_generation += 1
            self._reset_candidate_for_roi_change()
        left, top, right, bottom = new_roi
        self._publish_status(
            "D435 barcode acceptance ROI selected (full-frame inference unchanged): "
            f"x={left}, y={top}, width={right-left}, height={bottom-top}"
        )

    def _clear_manual_roi(self, width: int, height: int) -> None:
        full_frame = (0, 0, max(1, int(width)), max(1, int(height)))
        with self.roi_lock:
            self.manual_roi = full_frame
            self.roi_drawing = False
            self.roi_generation += 1
            self._reset_candidate_for_roi_change()
        self._publish_status("D435 ROI cleared; full-frame barcode acceptance restored")

    def _drawing_roi(self) -> tuple[bool, tuple[int, int], tuple[int, int]]:
        with self.roi_lock:
            return self.roi_drawing, self.roi_start, self.roi_end

    def _preview_event_callback(self, message: String) -> None:
        """Receive mouse/key events forwarded by the combined D405 window."""

        try:
            event = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        event_type = event.get("type")
        if event_type == "mouse":
            self._mouse_roi_callback(
                int(event.get("event", -1)),
                int(event.get("x", 0)),
                int(event.get("y", 0)),
                int(event.get("flags", 0)),
                None,
            )
        elif event_type == "key":
            key = int(event.get("key", -1))
            if key in (ord("c"), ord("C")):
                with self.roi_lock:
                    width, height = self.preview_image_size
                self._clear_manual_roi(width, height)
            elif key == ord(" "):
                with self.sample_save_lock:
                    self.pending_sample_saves += 1
                    pending = self.pending_sample_saves
                self._publish_status(
                    f"D435 raw image save queued ({pending} pending)"
                )
            elif key == ord("["):
                self._adjust_manual_brightness(-1)
            elif key == ord("]"):
                self._adjust_manual_brightness(1)

    def _save_pending_sample(self, image: np.ndarray) -> None:
        """Save the next raw D435 frame requested from the combined window."""

        with self.sample_save_lock:
            if self.pending_sample_saves <= 0:
                return
            self.pending_sample_saves -= 1
            pending = self.pending_sample_saves

        filename = f"d435_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        output_path = self.sample_save_dir / filename
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(output_path),
                image,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
            ):
                raise OSError("OpenCV could not encode the PNG image")
        except Exception as exc:
            self.get_logger().error(f"D435 sample image save failed: {exc}")
            self._publish_status(f"D435 sample image save failed: {exc}")
            return

        self.get_logger().info(
            f"Saved raw D435 training image: {output_path}"
            + (f" ({pending} pending)" if pending else "")
        )
        self._publish_status(f"Saved raw D435 training image: {output_path}")

    def _render_preview(
        self,
        image: np.ndarray,
        hit,
        workflow_active: bool,
        continuous_active: bool,
    ) -> np.ndarray:
        display = image.copy()
        left, top, right, bottom = self._roi_bounds(image)
        detail_roi = self.detector.detail_roi
        if detail_roi is not None:
            detail_x, detail_y, detail_width, detail_height = detail_roi
            detail_left = max(0, min(image.shape[1] - 1, int(detail_x)))
            detail_top = max(0, min(image.shape[0] - 1, int(detail_y)))
            detail_right = max(
                detail_left + 1,
                min(image.shape[1], detail_left + int(detail_width)),
            )
            detail_bottom = max(
                detail_top + 1,
                min(image.shape[0], detail_top + int(detail_height)),
            )
            cv2.rectangle(
                display,
                (detail_left, detail_top),
                (detail_right - 1, detail_bottom - 1),
                (255, 120, 0),
                2,
            )
            cv2.putText(
                display,
                "DETAIL",
                (detail_left + 6, max(18, detail_top + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 120, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.rectangle(
            display,
            (left, top),
            (right - 1, bottom - 1),
            (0, 220, 255),
            2,
        )
        drawing, draw_start, draw_end = self._drawing_roi()
        if drawing:
            cv2.rectangle(display, draw_start, draw_end, (255, 180, 0), 2)
        if hit is not None:
            x, y, width, height = hit.rect
            evidence_color = (0, 180, 255) if hit.source == "scanner_unverified" else (0, 255, 0)
            cv2.rectangle(
                display,
                (x, y),
                (x + width, y + height),
                evidence_color,
                3,
            )
            label = (
                "SCANNER"
                if hit.source == "scanner_pattern"
                else "SCANNER CANDIDATE"
                if hit.source == "scanner_unverified"
                else "MOVING STRIPES"
                if hit.source == "moving_stripes"
                else f"{hit.confidence:.2f}"
            )
            cv2.putText(
                display,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                evidence_color,
                2,
                cv2.LINE_AA,
            )
        if workflow_active and continuous_active:
            state_text = "SCANNING + CONTINUOUS"
        elif continuous_active:
            state_text = "CONTINUOUS"
        elif workflow_active:
            state_text = "SCANNING"
        else:
            state_text = "IDLE"
        active = workflow_active or continuous_active
        cv2.putText(
            display,
            f"D435 TURNTABLE  {state_text}  ROI=({left},{top}) {right-left}x{bottom-top}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0) if active else (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        darker_rect, brighter_rect = brightness_button_rects(display.shape[1])
        for label, rect, color in (
            ("DARKER -", darker_rect, (70, 70, 70)),
            ("BRIGHTER +", brighter_rect, (40, 120, 40)),
        ):
            button_left, button_top, button_right, button_bottom = rect
            cv2.rectangle(
                display,
                (button_left, button_top),
                (button_right, button_bottom),
                color,
                -1,
            )
            cv2.rectangle(
                display,
                (button_left, button_top),
                (button_right, button_bottom),
                (240, 240, 240),
                2,
            )
            text_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
            )
            text_x = button_left + max(
                4, (button_right - button_left - text_size[0]) // 2
            )
            text_y = button_top + (
                button_bottom - button_top + text_size[1]
            ) // 2
            cv2.putText(
                display,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            display,
            "Left-drag: select ROI   Right-click/C: full frame   SPACE: save raw PNG",
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        roi_gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        sharpness = (
            float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
            if roi_gray.size
            else 0.0
        )
        cv2.putText(
            display,
            f"{self.camera_settings_text}  sharpness={sharpness:.0f}  "
            f"detect={self.detect_rate_fps:.1f}fps",
            (16, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 220, 120),
            2,
            cv2.LINE_AA,
        )
        review = getattr(self, "confirmation_review", None)
        if review is not None:
            try:
                review.render_on(display)
            except Exception as error:
                self.get_logger().debug(f"D435 confirmation history rendering failed: {error}")
        return display

    def _record_confirmation(self, image, captured_at, color_frame, acceptance_roi=None):
        """Preserve the exact accepted frame after success was already published."""
        review = self.confirmation_review
        if review is None:
            return
        try:
            metadata = dict(self.last_confirmation_evidence)
            metadata.update(capture_received_unix_ns=int(captured_at * 1_000_000_000),
                model_path=str(self.get_parameter("model_path").value),
                model_confidence_threshold=float(self.get_parameter("model_confidence").value),
                model_image_size=self.detector.image_size,
                inference_provider=self.detector.active_provider,
                detail_roi=self.detector.detail_roi,
                acceptance_roi=list(self._roi_bounds(image) if acceptance_roi is None else acceptance_roi),
                camera_settings=self.camera_settings_text,
                image_size=[int(image.shape[1]), int(image.shape[0])])
            try:
                metadata.update(camera_frame_number=int(color_frame.get_frame_number()),
                    camera_timestamp_ms=float(color_frame.get_timestamp()),
                    camera_timestamp_domain=str(color_frame.get_frame_timestamp_domain()))
            except Exception as error:
                metadata["camera_metadata_error"] = str(error)
            review.record(image, metadata)
        except Exception as error:
            self.get_logger().warning(f"D435 confirmation recording failed: {error}")

    def _publish_preview(self, image: np.ndarray) -> None:
        quality = max(
            20,
            min(100, int(self.get_parameter("preview_jpeg_quality").value)),
        )
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "d435_turntable"
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.preview_publisher.publish(message)

    @staticmethod
    def _rect_iou(
        first: tuple[int, int, int, int] | None,
        second: tuple[int, int, int, int] | None,
    ) -> float:
        if first is None or second is None:
            return 0.0
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = max(1, aw * ah + bw * bh - intersection)
        return float(intersection) / float(union)

    @staticmethod
    def _moving_stripe_spatially_consistent(
        first: tuple[int, int, int, int] | None,
        second: tuple[int, int, int, int] | None,
    ) -> bool:
        """Associate a moving barcode through perspective/rotation changes.

        IoU is appropriate for a static YOLO proposal, but a side label on a
        rotating box changes width, height and skew between frames. Requiring
        the old IoU threshold can therefore reset the three-hit confirmation
        exactly while the label is crossing the camera. Keep a conservative
        center/area gate instead: the candidate must remain local, while its
        rectangle is allowed to change shape.
        """

        if first is None or second is None:
            return False
        ax, ay, aw, ah = [float(value) for value in first]
        bx, by, bw, bh = [float(value) for value in second]
        first_area = max(1.0, aw * ah)
        second_area = max(1.0, bw * bh)
        area_ratio = second_area / first_area
        if not 0.30 <= area_ratio <= 3.30:
            return False
        first_cx, first_cy = ax + aw * 0.5, ay + ah * 0.5
        second_cx, second_cy = bx + bw * 0.5, by + bh * 0.5
        center_distance = math.hypot(
            second_cx - first_cx,
            second_cy - first_cy,
        )
        max_dimension = max(aw, ah, bw, bh)
        return center_distance <= max(90.0, 1.75 * max_dimension)

    def _accept_hit(self, hit, generation: int, frame_id: int) -> bool:
        now = time.monotonic()
        value = str(hit.value)
        source = str(hit.source)
        with self.state_lock:
            # A frame captured before a scan reset must not contribute even
            # when the separate continuous detector is still enabled.
            if generation != self.scan_generation:
                return False
            workflow_active = (
                self.scan_active and generation == self.scan_generation
            )
            continuous_active = self.continuous_scan_active
            if not workflow_active and not continuous_active:
                return False
            if continuous_active and source != "scanner_unverified":
                # Any candidate hit keeps the current visible-presence latch
                # alive, even before the spatial stability count is complete.
                self.continuous_last_seen_at = now
            scanner_pattern = source in ("scanner_pattern", "scanner_unverified")
            moving_stripes = source == "moving_stripes"
            if scanner_pattern:
                max_gap = max(
                    0.05,
                    float(
                        self.get_parameter("scanner_assist_hit_gap_s").value
                    ),
                )
                required = max(
                    2,
                    int(
                        self.get_parameter("scanner_assist_stable_hits").value
                    ),
                )
                min_iou = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.get_parameter("scanner_assist_min_iou").value
                        ),
                        ),
                    )
            else:
                max_gap = max(
                    0.05, float(self.get_parameter("stable_hit_gap_s").value)
                )
                required = (
                    max(3, int(self.get_parameter("moving_stripes_stable_hits").value))
                    if moving_stripes else
                    max(1, int(self.get_parameter("stable_hits").value))
                )
                min_iou = max(
                    0.0,
                    min(
                        1.0,
                        float(self.get_parameter("yolo_min_iou").value),
                        ),
                    )
            track = self.evidence_tracks.observe(
                hit, now=now, frame_id=frame_id, max_gap=max_gap, min_iou=min_iou,
                match=self._moving_stripe_spatially_consistent if moving_stripes else None,
            )
            self.candidate, self.candidate_source = value, source
            self.candidate_hits = track.hits
            self.candidate_rect, self.candidate_first_rect = track.rect, track.first_rect
            self.candidate_last_at = track.last_at
            if source == "scanner_unverified" or track.hits < required:
                return False
            if moving_stripes and self.candidate_first_rect is not None:
                first_x, first_y, first_w, first_h = self.candidate_first_rect
                x, y, box_w, box_h = hit.rect
                movement = abs(
                    (x + box_w * 0.5) - (first_x + first_w * 0.5)
                )
                if movement < max(10.0, box_w * 0.10):
                    return False
            if workflow_active:
                self.scan_active = False
            report_continuous = (
                continuous_active and not self.continuous_presence_latched
            )
            if report_continuous:
                self.continuous_last_report_value = value
                self.continuous_last_report_at = now
                self.continuous_presence_latched = True
            self.last_confirmation_evidence = dict(
                generation=generation, detection_frame_id=frame_id,
                source=source, value=value, confidence=float(hit.confidence),
                confidence_is_model_score=source.startswith("yolo"),
                rect=[int(value) for value in hit.rect],
                independent_hits=track.hits, required_hits=required,
                workflow_confirmation=workflow_active,
                continuous_confirmation=report_continuous,
            )
            # Start a fresh stability sequence. Continuous inference remains
            # active, but the visible-presence latch suppresses duplicates.
            self.candidate_hits = 0
            self.candidate_rect = None
            self.candidate_first_rect = None
            self.evidence_tracks.clear()
        if workflow_active:
            confidence_text = f"{float(hit.confidence):.3f}" if source.startswith("yolo") else "n/a"
            result = String()
            result.data = f"success:{value}"
            self.result_publisher.publish(result)
            self._publish_status(
                f"D435 side barcode confirmed by {source}: {value!r}; "
                f"independent_hits={track.hits}/{required}; requesting turntable stop; "
                f"confidence={confidence_text}, rect={tuple(hit.rect)}, "
                f"detection_frame={frame_id}, generation={generation}"
            )
        if report_continuous:
            result = String()
            result.data = f"success:{value}"
            self.continuous_result_publisher.publish(result)
            self._publish_continuous_presence(True)
            self._publish_status(
                f"D435 continuous detection confirmed by {source}: {value!r}"
                f"; independent_hits={track.hits}/{required}"
            )
        return workflow_active or report_continuous

    def _note_continuous_no_hit(self, now: float) -> None:
        """Re-arm continuous reporting only after the barcode has disappeared."""

        presence_cleared = False
        with self.state_lock:
            if not (
                self.continuous_scan_active
                and self.continuous_presence_latched
            ):
                return
            absence_s = max(
                0.1,
                float(
                    self.get_parameter("continuous_rearm_absence_s").value
                ),
            )
            if now - self.continuous_last_seen_at < absence_s:
                return
            self.continuous_presence_latched = False
            self.continuous_last_report_value = ""
            self.candidate = ""
            self.candidate_source = ""
            self.candidate_rect = None
            self.candidate_first_rect = None
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
            self.evidence_tracks.clear()
            presence_cleared = True
        if presence_cleared:
            self._publish_continuous_presence(False)

    def _publish_candidate_hint(self, hit, generation: int, captured_at: float, *, invalidated=False) -> None:
        """A hint can request reinspection, but never declares barcode success."""

        with self.state_lock:
            if generation != self.scan_generation:
                return
            pending = self.scan_active and hit is not None
            invalidated = invalidated or not self.scan_active
        message = String()
        message.data = json.dumps({
            "pending": pending, "captured_at": captured_at,
            "invalidated": invalidated,
            "generation": generation,
            "source": hit.source if pending else "",
            "rect": list(hit.rect) if pending else None,
        }, separators=(",", ":"))
        self.candidate_publisher.publish(message)

    def _capture_loop(self) -> None:
        try:
            self._publish_ready(False)
            self._publish_status("loading D435 turntable barcode model")
            self.detector.load_model()
            pipeline = rs.pipeline()
            config = rs.config()
            serial = str(self.get_parameter("serial_number").value)
            config.enable_device(serial)
            config.enable_stream(
                rs.stream.color,
                int(self.get_parameter("color_width").value),
                int(self.get_parameter("color_height").value),
                rs.format.bgr8,
                int(self.get_parameter("fps").value),
            )
            pipeline_profile = pipeline.start(config)
            self.pipeline = pipeline
            self._configure_color_sensor(pipeline_profile)
            # Discard startup frames before advertising readiness.
            for _ in range(15):
                if not self.running:
                    return
                pipeline.wait_for_frames(1000)
            self._publish_ready(True)
            self._publish_status(
                f"D435 turntable scanner ready: serial={serial}; "
                f"{self.camera_settings_text}; presence-only model="
                f"{self.detector.model_path}, imgsz={self.detector.image_size}, "
                f"confidence={self.detector.confidence:.2f}, provider="
                f"{self.detector.active_provider}; scanner_assist="
                f"{'on' if self.detector.scanner_detector is not None else 'off'}; "
                f"wide_roi_fallback="
                f"{'on' if self.detector.wide_roi_fallback else 'off'}; "
                f"detail_roi={self.detector.detail_roi}; "
                f"full_frame_every={self.detector.full_frame_interval}; "
                f"min_candidate_area="
                f"{self.detector.yolo_min_candidate_area_ratio:.3f}; "
                "flat-background rejection=on; parallel-bar verification=YOLO+scanner; "
                "orientation-coherence rejection=on; "
                "confirmation=independent per source; weak patterns=recheck-only; "
                f"moving_stripes={'on' if self.detector.moving_stripes_enabled else 'off'}"
            )
            if self.detector.provider_fallback_reason:
                self.get_logger().warning(
                    "D435 ONNX provider fallback: "
                    f"{self.detector.provider_fallback_reason}"
                )
            last_detect_at = 0.0
            last_preview_at = 0.0
            preview = bool(self.get_parameter("preview").value)
            if preview:
                self._publish_status(
                    "D435 integrated preview enabled in the D405 window: "
                    "left-drag limits barcode acceptance ROI without cropping "
                    "inference; right-click or C restores full frame"
                )
            while self.running and rclpy.ok():
                frames = pipeline.wait_for_frames(1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                captured_at = time.time()
                image = np.asanyarray(color_frame.get_data())
                self._save_pending_sample(image)
                with self.roi_lock:
                    self.preview_image_size = (image.shape[1], image.shape[0])
                with self.state_lock:
                    workflow_active = self.scan_active
                    continuous_active = self.continuous_scan_active
                    generation = self.scan_generation
                active = workflow_active or continuous_active
                hit = None
                confirmation_accepted = False
                now = time.monotonic()
                if active and now - last_detect_at >= max(
                    0.0, float(self.get_parameter("detect_interval_s").value)
                ):
                    last_detect_at = now
                    allowed_roi, roi_generation = self._roi_bounds_with_generation(image)
                    provider_before = self.detector.active_provider
                    hits = self.detector.detect_candidates(image, allowed_roi)
                    self.detection_frame_id += 1
                    hit = hits[0] if hits else None
                    self.detect_rate_count += 1
                    detect_finished_at = time.monotonic()
                    detect_rate_elapsed = (
                        detect_finished_at - self.detect_rate_started_at
                    )
                    if detect_rate_elapsed >= 1.0:
                        self.detect_rate_fps = (
                            self.detect_rate_count / detect_rate_elapsed
                        )
                        self.detect_rate_count = 0
                        self.detect_rate_started_at = detect_finished_at
                    if self.detector.active_provider != provider_before:
                        self.get_logger().warning(
                            "D435 ONNX provider changed during inference: "
                            f"{provider_before} -> {self.detector.active_provider}; "
                            f"{self.detector.provider_fallback_reason}"
                        )
                    with self.roi_lock:
                        roi_changed = roi_generation != self.roi_generation
                        if roi_changed:
                            # An ROI change during inference invalidates the
                            # old frame's spatial evidence.
                            hit = None
                        else:
                            for candidate in hits:
                                if self._accept_hit(candidate, generation, self.detection_frame_id):
                                    hit = candidate
                                    confirmation_accepted = True
                                    break
                            if not any(c.source != "scanner_unverified" for c in hits):
                                self._note_continuous_no_hit(now)
                        if workflow_active:
                            self._publish_candidate_hint(
                                hit, generation, captured_at, invalidated=roi_changed,
                            )
                    if confirmation_accepted:
                        self._record_confirmation(image, captured_at, color_frame, allowed_roi)
                preview_interval = max(
                    0.02,
                    float(self.get_parameter("preview_publish_interval_s").value),
                )
                if preview and (now - last_preview_at >= preview_interval or confirmation_accepted):
                    last_preview_at = now
                    self._publish_preview(
                        self._render_preview(
                            image,
                            hit,
                            workflow_active,
                            continuous_active,
                        )
                    )
        except Exception as exc:
            self._publish_ready(False)
            self.get_logger().fatal(f"D435 turntable barcode node failed: {exc}")
            self._publish_status(f"FAULT: D435 turntable barcode node failed: {exc}")
        finally:
            self._publish_ready(False)
            self._publish_continuous_presence(False)
            pipeline = self.pipeline
            self.pipeline = None
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def destroy_node(self):
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.confirmation_review is not None:
            self.confirmation_review.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = D435TurntableBarcodeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
