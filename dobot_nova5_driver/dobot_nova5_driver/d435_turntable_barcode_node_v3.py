"""Triggered D435 side-barcode scanner for the V3 turntable workflow."""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

from .turntable_barcode_detector_v3 import (
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
        super().__init__("d435_turntable_barcode_node_v3")
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
        self.declare_parameter("scanner_assist_enabled", True)
        self.declare_parameter("scanner_assist_stable_hits", 3)
        self.declare_parameter("scanner_assist_hit_gap_s", 0.25)
        self.declare_parameter("scanner_assist_min_iou", 0.20)
        self.declare_parameter("stable_hits", 2)
        self.declare_parameter("stable_hit_gap_s", 0.70)
        self.declare_parameter("yolo_min_iou", 0.15)
        self.declare_parameter("detect_interval_s", 0.03)
        self.declare_parameter("roi_x", 0)
        self.declare_parameter("roi_y", 0)
        self.declare_parameter("roi_width", 0)
        self.declare_parameter("roi_height", 0)
        self.declare_parameter("preview", True)
        self.declare_parameter(
            "preview_topic", "/vision_panel/d435_turntable/image/compressed"
        )
        self.declare_parameter("event_topic", "/vision_panel/d435_turntable/event")
        self.declare_parameter("preview_publish_interval_s", 0.10)
        self.declare_parameter("preview_jpeg_quality", 80)
        self.declare_parameter("trigger_topic", "/trigger_turntable_barcode")
        self.declare_parameter(
            "continuous_trigger_topic", "/trigger_d435_continuous_detection"
        )
        self.declare_parameter("result_topic", "/turntable_barcode_result")
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
        self.candidate = ""
        self.candidate_source = ""
        self.candidate_rect: tuple[int, int, int, int] | None = None
        self.candidate_hits = 0
        self.candidate_last_at = 0.0
        self.continuous_last_report_value = ""
        self.continuous_last_report_at = 0.0
        self.continuous_last_seen_at = 0.0
        self.continuous_presence_latched = False
        self._publish_continuous_presence(False)
        self.roi_lock = threading.Lock()
        self.manual_roi: tuple[int, int, int, int] | None = None
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
        self.camera_option_lock = threading.Lock()
        self.manual_exposure = float(self.get_parameter("exposure").value)
        self.manual_gain = float(self.get_parameter("gain").value)
        self.manual_auto_exposure = bool(
            self.get_parameter("auto_exposure").value
        )
        self.camera_settings_text = "camera settings pending"
        self.detector = TurntableBarcodeDetector(
            model_path=str(self.get_parameter("model_path").value),
            confidence=float(self.get_parameter("model_confidence").value),
            image_size=int(self.get_parameter("model_image_size").value),
            inference_provider=str(
                self.get_parameter("inference_provider").value
            ),
            scanner_assist=bool(
                self.get_parameter("scanner_assist_enabled").value
            ),
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
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
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
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
            self.continuous_last_report_value = ""
            self.continuous_last_report_at = 0.0
            self.continuous_last_seen_at = 0.0
            self.continuous_presence_latched = False
        self._publish_continuous_presence(False)
        self._publish_status(
            f"D435 continuous detection {'enabled' if enabled else 'disabled'}; "
            "turntable motion is not controlled by this switch"
        )

    def _roi_bounds(self, image: np.ndarray) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        with self.roi_lock:
            manual_roi = self.manual_roi
        if manual_roi is not None:
            left, top, right, bottom = manual_roi
            left = max(0, min(width - 1, int(left)))
            top = max(0, min(height - 1, int(top)))
            right = max(left + 1, min(width, int(right)))
            bottom = max(top + 1, min(height, int(bottom)))
            return left, top, right, bottom

        x = max(0, min(width - 1, int(self.get_parameter("roi_x").value)))
        y = max(0, min(height - 1, int(self.get_parameter("roi_y").value)))
        roi_width = int(self.get_parameter("roi_width").value)
        roi_height = int(self.get_parameter("roi_height").value)
        right = width if roi_width <= 0 else min(width, x + roi_width)
        bottom = height if roi_height <= 0 else min(height, y + roi_height)
        return x, y, right, bottom

    def _configured_roi(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        left, top, right, bottom = self._roi_bounds(image)
        return image[top:bottom, left:right], (left, top)

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
        left, top, right, bottom = new_roi
        self._publish_status(
            "D435 ROI selected by mouse: "
            f"x={left}, y={top}, width={right-left}, height={bottom-top}"
        )

    def _clear_manual_roi(self, width: int, height: int) -> None:
        full_frame = (0, 0, max(1, int(width)), max(1, int(height)))
        with self.roi_lock:
            self.manual_roi = full_frame
            self.roi_drawing = False
        self._publish_status("D435 ROI cleared; full-frame detection restored")

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
            elif key == ord("["):
                self._adjust_manual_brightness(-1)
            elif key == ord("]"):
                self._adjust_manual_brightness(1)

    def _render_preview(
        self,
        image: np.ndarray,
        hit,
        workflow_active: bool,
        continuous_active: bool,
    ) -> np.ndarray:
        display = image.copy()
        left, top, right, bottom = self._roi_bounds(image)
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
            cv2.rectangle(
                display,
                (left + x, top + y),
                (left + x + width, top + y + height),
                (0, 255, 0),
                3,
            )
            label = (
                "SCANNER"
                if hit.source == "scanner_pattern"
                else f"YOLO {hit.confidence:.2f}"
            )
            cv2.putText(
                display,
                label,
                (left + x, max(18, top + y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
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
            "Left-drag: select ROI   Right-click/C: full frame   Click brightness buttons",
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
            f"{self.camera_settings_text}  sharpness={sharpness:.0f}",
            (16, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 220, 120),
            2,
            cv2.LINE_AA,
        )
        return display

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

    def _accept_hit(self, hit, generation: int) -> bool:
        now = time.monotonic()
        value = str(hit.value)
        source = str(hit.source)
        with self.state_lock:
            workflow_active = (
                self.scan_active and generation == self.scan_generation
            )
            continuous_active = self.continuous_scan_active
            if not workflow_active and not continuous_active:
                return False
            if continuous_active:
                # Any candidate hit keeps the current visible-presence latch
                # alive, even before the spatial stability count is complete.
                self.continuous_last_seen_at = now
            scanner_pattern = source == "scanner_pattern"
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
                spatially_consistent = self._rect_iou(
                    self.candidate_rect, hit.rect
                ) >= max(
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
                required = max(1, int(self.get_parameter("stable_hits").value))
                spatially_consistent = self._rect_iou(
                    self.candidate_rect, hit.rect
                ) >= max(
                    0.0,
                    min(
                        1.0,
                        float(self.get_parameter("yolo_min_iou").value),
                    ),
                )
            same_candidate = (
                value == self.candidate
                and source == self.candidate_source
                and now - self.candidate_last_at <= max_gap
                and spatially_consistent
            )
            if same_candidate:
                self.candidate_hits += 1
            else:
                self.candidate = value
                self.candidate_source = source
                self.candidate_hits = 1
            self.candidate_rect = tuple(hit.rect)
            self.candidate_last_at = now
            if self.candidate_hits < required:
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
            # Start a fresh stability sequence. Continuous inference remains
            # active, but the visible-presence latch suppresses duplicates.
            self.candidate_hits = 0
            self.candidate_rect = None
        if workflow_active:
            result = String()
            result.data = f"success:{value}"
            self.result_publisher.publish(result)
            self._publish_status(
                f"D435 side barcode confirmed by {source}: {value!r}; "
                "requesting turntable stop"
            )
        if report_continuous:
            result = String()
            result.data = f"success:{value}"
            self.continuous_result_publisher.publish(result)
            self._publish_continuous_presence(True)
            self._publish_status(
                f"D435 continuous detection confirmed by {source}: {value!r}"
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
            self.candidate_hits = 0
            self.candidate_last_at = 0.0
            presence_cleared = True
        if presence_cleared:
            self._publish_continuous_presence(False)

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
                "flat-background rejection=on"
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
                    "left-drag selects ROI; right-click or C restores full frame"
                )
            while self.running and rclpy.ok():
                frames = pipeline.wait_for_frames(1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                with self.roi_lock:
                    self.preview_image_size = (image.shape[1], image.shape[0])
                with self.state_lock:
                    workflow_active = self.scan_active
                    continuous_active = self.continuous_scan_active
                    generation = self.scan_generation
                active = workflow_active or continuous_active
                hit = None
                now = time.monotonic()
                if active and now - last_detect_at >= max(
                    0.0, float(self.get_parameter("detect_interval_s").value)
                ):
                    last_detect_at = now
                    crop, origin = self._configured_roi(image)
                    provider_before = self.detector.active_provider
                    hit = self.detector.detect(crop)
                    if self.detector.active_provider != provider_before:
                        self.get_logger().warning(
                            "D435 ONNX provider changed during inference: "
                            f"{provider_before} -> {self.detector.active_provider}; "
                            f"{self.detector.provider_fallback_reason}"
                        )
                    if hit is not None:
                        self._accept_hit(hit, generation)
                    else:
                        self._note_continuous_no_hit(now)
                preview_interval = max(
                    0.02,
                    float(self.get_parameter("preview_publish_interval_s").value),
                )
                if preview and now - last_preview_at >= preview_interval:
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
