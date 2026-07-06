#!/usr/bin/env python3
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO


DEFAULT_MODEL_PATH = "/home/zdh/yolo_one/yolo_train_xense_load_image/outputs/train/obb_demo111/weights/best.pt"


@dataclass
class Detection:
    box: Tuple[int, int, int, int]
    score: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def nms_indices(boxes: Sequence[Tuple[int, int, int, int]], scores: Sequence[float], score_threshold: float, nms_threshold: float) -> List[int]:
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(list(boxes), list(scores), score_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    return np.array(indices).reshape(-1).astype(int).tolist()


def draw_modern_ui(img: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], label: str, is_locked: bool) -> None:
    x, y, w, h = box
    thickness = 3 if is_locked else 2
    corner_len = max(10, min(w, h) // 5)

    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

    if not is_locked:
        return

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, dst=img)

    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_y = max(0, y - label_h - 10)
    cv2.rectangle(img, (x, label_y), (x + label_w + 10, label_y + label_h + 10), color, cv2.FILLED)
    cv2.putText(img, label, (x + 5, label_y + label_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def stripe_pattern_score(tile_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(tile_bgr.shape[1], x + w)
    y1 = min(tile_bgr.shape[0], y + h)
    if x1 - x0 < 18 or y1 - y0 < 10:
        return 0.0

    gray = cv2.cvtColor(tile_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)

    grad_x = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    mean_x = float(cv2.mean(np.abs(grad_x))[0])
    mean_y = float(cv2.mean(np.abs(grad_y))[0])
    dominant = max(mean_x, mean_y)
    orthogonal = min(mean_x, mean_y)
    if dominant <= 1e-3:
        return 0.0

    vertical_bars = mean_x >= mean_y
    orientation_ratio = dominant / (orthogonal + 1e-3)
    orientation_score = clamp01((orientation_ratio - 1.05) / 1.4)

    axis = 0 if vertical_bars else 1
    reduced = cv2.reduce(eq, axis, cv2.REDUCE_AVG, dtype=cv2.CV_32F)
    signal = reduced.reshape(-1)
    signal_len = int(signal.size)
    if signal_len < 16:
        return 0.0

    signal_mean = float(np.mean(signal))
    bits = (signal >= signal_mean).astype(np.uint8)
    transitions = int(np.count_nonzero(bits[1:] != bits[:-1]))

    run_lengths = []
    cur_run = 1
    for i in range(1, signal_len):
        if bits[i] == bits[i - 1]:
            cur_run += 1
        else:
            run_lengths.append(cur_run)
            cur_run = 1
    run_lengths.append(cur_run)
    if len(run_lengths) < 6:
        return 0.0

    run_array = np.array(run_lengths, dtype=np.float32)
    run_mean = float(np.mean(run_array))
    if run_mean <= 1e-3:
        return 0.0
    run_cv = float(np.std(run_array) / run_mean)

    transition_score = clamp01((transitions - 5.0) / 11.0)
    run_cv_score = clamp01((run_cv - 0.10) / 0.30)
    return 0.45 * orientation_score + 0.30 * transition_score + 0.25 * run_cv_score


class D405YoloPtBase(Node):
    color_width = 640
    color_height = 480
    slice_rows = 1
    slice_cols = 2
    slice_overlap = 0.20
    conf_threshold = 0.45
    nms_threshold = 0.45
    score_keep = 0.30
    input_width = 640
    input_height = 640
    window_name = "YOLO PT Detection"
    enable_shape_filter = False
    enable_mouse_roi = False
    min_roi_size = 20
    panel_name = "barcode"
    panel_image_topic = "/vision_panel/barcode/image/compressed"
    panel_event_topic = "/vision_panel/barcode/event"

    min_aspect_ratio = 1.60
    suspicious_aspect_ratio = 2.40
    suspicious_area_ratio = 0.035
    min_pattern_score = 0.38

    def __init__(self, node_name: str, startup_label: str):
        super().__init__(node_name)
        self.publisher = self.create_publisher(String, "detected_barcodes", 10)

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("device", "", ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("camera_sn", "409122274792")
        self.declare_parameter("color_width", self.color_width)
        self.declare_parameter("color_height", self.color_height)
        self.declare_parameter("input_size", 640)
        self.declare_parameter("conf_threshold", float(self.conf_threshold))
        self.declare_parameter("nms_threshold", float(self.nms_threshold))
        self.declare_parameter("show", True)
        self.declare_parameter("drain_frames", 6)
        self.declare_parameter("batch_size", self.slice_rows * self.slice_cols)
        self.declare_parameter("process_period", 0.03)
        self.declare_parameter("infer_period", 0.15)
        self.declare_parameter("use_slicing", False)
        self.declare_parameter("display_scale", 0.5)
        self.declare_parameter("torch_threads", 1)
        self.declare_parameter("auto_exposure", False)
        # exposure 是相机曝光时间，数值越大画面越亮、运动拖影/过曝风险越高；白色塑封盒反光强时应调低。
        self.declare_parameter("exposure", 11000.0)
        # gain 是传感器增益，数值越大暗部越亮、噪声也越多；条码纹理变糊或噪点多时应调低。
        self.declare_parameter("gain", 8.0)
        self.declare_parameter("auto_white_balance", True)
        self.declare_parameter("standalone_window", False)
        self.declare_parameter("panel_image_topic", self.panel_image_topic)
        self.declare_parameter("panel_event_topic", self.panel_event_topic)
        self.declare_parameter("panel_jpeg_quality", 80)

        self.model_path = str(self.get_parameter("model_path").value)
        self.device = str(self.get_parameter("device").value)
        self.camera_sn = str(self.get_parameter("camera_sn").value).strip()
        self.color_width = int(self.get_parameter("color_width").value)
        self.color_height = int(self.get_parameter("color_height").value)
        self.input_width = int(self.get_parameter("input_size").value)
        self.input_height = self.input_width
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.nms_threshold = float(self.get_parameter("nms_threshold").value)
        self.show = bool(self.get_parameter("show").value)
        self.drain_frames = max(1, int(self.get_parameter("drain_frames").value))
        self.batch_size = max(1, int(self.get_parameter("batch_size").value))
        self.process_period = max(0.01, float(self.get_parameter("process_period").value))
        self.infer_period = max(0.03, float(self.get_parameter("infer_period").value))
        self.use_slicing = bool(self.get_parameter("use_slicing").value)
        self.display_scale = max(0.1, min(1.0, float(self.get_parameter("display_scale").value)))
        self.torch_threads = max(1, int(self.get_parameter("torch_threads").value))
        self.auto_exposure = bool(self.get_parameter("auto_exposure").value)
        self.exposure = float(self.get_parameter("exposure").value)
        self.gain = float(self.get_parameter("gain").value)
        self.auto_white_balance = bool(self.get_parameter("auto_white_balance").value)
        self.standalone_window = bool(self.get_parameter("standalone_window").value)
        self.panel_image_topic = str(self.get_parameter("panel_image_topic").value)
        self.panel_event_topic = str(self.get_parameter("panel_event_topic").value)
        self.panel_jpeg_quality = max(1, min(100, int(self.get_parameter("panel_jpeg_quality").value)))

        self.drawing_roi = False
        self.roi_start = (-1, -1)
        self.roi_end = (-1, -1)
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.latest_frame_size = (self.color_width, self.color_height)

        cv2.setNumThreads(1)
        torch.set_num_threads(self.torch_threads)
        torch.set_num_interop_threads(1)

        self.get_logger().info(f"正在加载 YOLO .pt 模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO .pt 模型加载成功: {self.model_path}")

        self.pipeline = rs.pipeline()
        self.pipeline_started = False
        self.color_sensor = None
        config = rs.config()
        if self.camera_sn:
            config.enable_device(self.camera_sn)
            self.get_logger().info(f"指定 D405 序列号: {self.camera_sn}")
        else:
            self.get_logger().warn("未指定 D405 序列号，将使用系统默认 RealSense 设备")
        config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, 30)

        try:
            profile = self.pipeline.start(config)
            self.pipeline_started = True
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    self.color_sensor = sensor
                    self.apply_camera_exposure_settings(sensor)
            self.get_logger().info(startup_label)
        except Exception as exc:
            self.get_logger().error(f"相机启动失败: {exc}")
            raise

        self.panel_image_pub = self.create_publisher(CompressedImage, self.panel_image_topic, 1)
        self.panel_event_sub = self.create_subscription(String, self.panel_event_topic, self.panel_event_callback, 10)

        if self.show and self.standalone_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.window_name, 30, 50)
            if self.enable_mouse_roi:
                cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.total_frames = 0
        self.last_time = time.monotonic()
        self.last_infer_time = self.last_time - self.infer_period
        self.current_fps = 0.0
        self.last_detections: List[Detection] = []
        self.last_infer_ms = 0.0
        self._processing_frame = False
        self.timer = self.create_timer(self.process_period, self.process_frame)
        self.get_logger().info(
            f"低占用参数: device={self.device or 'auto'} infer_period={self.infer_period:.2f}s "
            f"use_slicing={self.use_slicing} torch_threads={self.torch_threads} "
            f"color={self.color_width}x{self.color_height} panel_topic={self.panel_image_topic} "
            f"standalone_window={self.standalone_window}"
        )

    def set_sensor_option_safe(self, sensor, option, value: float) -> bool:
        try:
            option_range = sensor.get_option_range(option)
            clamped = max(option_range.min, min(option_range.max, value))
            sensor.set_option(option, clamped)
            return True
        except Exception as exc:
            self.get_logger().warn(f"设置相机参数失败: {option}={value}, {exc}")
            return False

    def apply_camera_exposure_settings(self, sensor) -> None:
        self.set_sensor_option_safe(sensor, rs.option.enable_auto_exposure, 1.0 if self.auto_exposure else 0.0)
        if self.auto_exposure:
            self.get_logger().info("相机曝光: auto_exposure=True，使用 RealSense 自动曝光")
            return

        if sensor.supports(rs.option.exposure):
            self.set_sensor_option_safe(sensor, rs.option.exposure, self.exposure)
        if sensor.supports(rs.option.gain):
            self.set_sensor_option_safe(sensor, rs.option.gain, self.gain)
        self.get_logger().info(f"相机曝光: auto_exposure=False exposure={self.exposure:.1f} gain={self.gain:.1f}")

    def adjust_manual_exposure(self, exposure_delta: float = 0.0, gain_delta: float = 0.0) -> None:
        if self.color_sensor is None or self.auto_exposure:
            return

        if exposure_delta and self.color_sensor.supports(rs.option.exposure):
            option_range = self.color_sensor.get_option_range(rs.option.exposure)
            self.exposure = max(option_range.min, min(option_range.max, self.exposure + exposure_delta))
            self.color_sensor.set_option(rs.option.exposure, self.exposure)
        if gain_delta and self.color_sensor.supports(rs.option.gain):
            option_range = self.color_sensor.get_option_range(rs.option.gain)
            self.gain = max(option_range.min, min(option_range.max, self.gain + gain_delta))
            self.color_sensor.set_option(rs.option.gain, self.gain)
        self.get_logger().info(f"当前手动曝光: exposure={self.exposure:.1f} gain={self.gain:.1f}")

    def destroy_node(self) -> bool:
        if self.pipeline_started:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline_started = False
        if self.show and self.standalone_window:
            cv2.destroyAllWindows()
        return super().destroy_node()

    def publish_panel_frame(self, image: np.ndarray) -> None:
        if not self.show:
            return
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.panel_jpeg_quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.panel_name
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.panel_image_pub.publish(msg)

    def panel_event_callback(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        event_type = event.get("type")
        if event_type == "mouse":
            self.mouse_callback(
                int(event.get("event", -1)),
                int(event.get("x", 0)),
                int(event.get("y", 0)),
                int(event.get("flags", 0)),
                None,
            )
        elif event_type == "key":
            self.handle_key(int(event.get("key", -1)))

    def handle_key(self, key: int) -> None:
        if key == ord("["):
            self.adjust_manual_exposure(exposure_delta=-50.0)
        elif key == ord("]"):
            self.adjust_manual_exposure(exposure_delta=50.0)
        elif key == ord("-"):
            self.adjust_manual_exposure(gain_delta=-1.0)
        elif key in (ord("="), ord("+")):
            self.adjust_manual_exposure(gain_delta=1.0)
        elif key in (ord("r"), ord("R")):
            self.clear_roi()

    def clamp_point_to_frame(self, x: int, y: int) -> Tuple[int, int]:
        frame_w, frame_h = self.latest_frame_size
        clamped_x = max(0, min(frame_w - 1, int(x)))
        clamped_y = max(0, min(frame_h - 1, int(y)))
        return clamped_x, clamped_y

    def normalize_roi(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        frame_w, frame_h = self.latest_frame_size
        x0, y0 = self.clamp_point_to_frame(*start)
        x1, y1 = self.clamp_point_to_frame(*end)
        left = max(0, min(x0, x1))
        top = max(0, min(y0, y1))
        right = min(frame_w, max(x0, x1) + 1)
        bottom = min(frame_h, max(y0, y1) + 1)
        if right - left <= self.min_roi_size or bottom - top <= self.min_roi_size:
            return None
        return left, top, right, bottom

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if not self.enable_mouse_roi:
            return

        point = self.clamp_point_to_frame(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_roi = True
            self.roi_start = point
            self.roi_end = point
            self.roi = None
            self.last_detections = []
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_roi:
            self.roi_end = point
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_roi = False
            self.roi_end = point
            self.roi = self.normalize_roi(self.roi_start, self.roi_end)
            self.last_detections = []
            if self.roi is not None:
                self.get_logger().info(f"ROI Locked: {self.roi}")
            else:
                self.get_logger().info("ROI too small, cleared. Drag a larger area to enable ROI detection.")

    def active_roi(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        image_h, image_w = image.shape[:2]
        self.latest_frame_size = (image_w, image_h)
        if not self.enable_mouse_roi or self.roi is None:
            return None

        left, top, right, bottom = self.roi
        left = max(0, min(image_w - 1, left))
        top = max(0, min(image_h - 1, top))
        right = max(left + 1, min(image_w, right))
        bottom = max(top + 1, min(image_h, bottom))
        if right - left <= self.min_roi_size or bottom - top <= self.min_roi_size:
            return None
        return left, top, right, bottom

    def draw_roi_overlay(self, image: np.ndarray) -> None:
        if not self.enable_mouse_roi:
            return

        roi_color = (0, 200, 255)
        if self.drawing_roi:
            roi = self.normalize_roi(self.roi_start, self.roi_end)
            if roi is not None:
                left, top, right, bottom = roi
                cv2.rectangle(image, (left, top), (right, bottom), roi_color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(image, self.roi_start, self.roi_end, roi_color, 1, cv2.LINE_AA)
        elif self.roi is not None:
            roi = self.active_roi(image)
            if roi is None:
                return
            left, top, right, bottom = roi
            cv2.rectangle(image, (left, top), (right, bottom), roi_color, 2, cv2.LINE_AA)
            label_y = max(15, top - 8)
            cv2.putText(image, "ROI LOCKED", (left, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2, cv2.LINE_AA)

    def run_infer_with_roi(self, image: np.ndarray) -> List[Detection]:
        roi = self.active_roi(image)
        if roi is None:
            return self.run_sliced_infer(image) if self.use_slicing else self.run_yolo_infer(image)

        left, top, right, bottom = roi
        roi_image = image[top:bottom, left:right]
        detections = self.run_sliced_infer(roi_image) if self.use_slicing else self.run_yolo_infer(roi_image)
        shifted_detections: List[Detection] = []
        for det in detections:
            x, y, w, h = det.box
            shifted_detections.append(Detection(box=(x + left, y + top, w, h), score=det.score))
        return shifted_detections

    def clear_roi(self) -> None:
        if not self.enable_mouse_roi or self.roi is None:
            return
        self.roi = None
        self.drawing_roi = False
        self.last_detections = []
        self.get_logger().info("ROI Cleared by user.")

    def passes_detection_filter(self, tile_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        if not self.enable_shape_filter:
            return True

        x, y, w, h = box
        src_h, src_w = tile_bgr.shape[:2]
        if w < 10 or h < 7:
            return False
        if w > src_w * 0.92 or h > src_h * 0.92:
            return False

        aspect = max(w, h) / max(1, min(w, h))
        if aspect < self.min_aspect_ratio or aspect > 12.0:
            return False

        area_ratio = (w * h) / max(1, src_w * src_h)
        if aspect < self.suspicious_aspect_ratio or area_ratio > self.suspicious_area_ratio:
            return stripe_pattern_score(tile_bgr, box) >= self.min_pattern_score
        return True

    def get_latest_color_frame(self):
        if not self.pipeline_started:
            return None

        frames = None
        for _ in range(self.drain_frames):
            try:
                polled = self.pipeline.poll_for_frames()
            except RuntimeError as exc:
                self.get_logger().warn(f"RealSense 取帧失败，pipeline 未就绪: {exc}")
                self.pipeline_started = False
                return None
            if not polled:
                break
            frames = polled

        if frames is None:
            try:
                frames = self.pipeline.wait_for_frames(10)
            except Exception:
                return None

            for _ in range(self.drain_frames - 1):
                try:
                    polled = self.pipeline.poll_for_frames()
                except RuntimeError as exc:
                    self.get_logger().warn(f"RealSense 取帧失败，pipeline 未就绪: {exc}")
                    self.pipeline_started = False
                    return None
                if not polled:
                    break
                frames = polled

        return frames.get_color_frame() if frames else None

    def result_to_detections(self, result, tile_bgr: np.ndarray) -> List[Detection]:
        src_h, src_w = tile_bgr.shape[:2]
        detections: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        obb = getattr(result, "obb", None)

        if obb is not None and getattr(obb, "xyxy", None) is not None:
            xyxy_array = obb.xyxy.cpu().numpy()
            conf_array = obb.conf.cpu().numpy() if obb.conf is not None else np.ones((len(xyxy_array),), dtype=np.float32)
        elif boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy_array = boxes.xyxy.cpu().numpy()
            conf_array = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy_array),), dtype=np.float32)
        else:
            return detections

        for xyxy, score_value in zip(xyxy_array, conf_array):
            score = float(score_value)
            if score < self.conf_threshold:
                continue

            x1, y1, x2, y2 = xyxy[:4]
            left = max(0, int(round(float(x1))))
            top = max(0, int(round(float(y1))))
            right = min(src_w - 1, int(round(float(x2))))
            bottom = min(src_h - 1, int(round(float(y2))))
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                continue

            box = (left, top, width, height)
            if self.passes_detection_filter(tile_bgr, box):
                detections.append(Detection(box=box, score=score))

        return detections

    def run_yolo_infer(self, tile_bgr: np.ndarray) -> List[Detection]:
        results = self.predict_tiles([tile_bgr])
        if not results:
            return []
        return self.result_to_detections(results[0], tile_bgr)

    def predict_tiles(self, tiles: Sequence[np.ndarray]):
        if not tiles:
            return []

        predict_kwargs = {
            "source": list(tiles),
            "imgsz": self.input_width,
            "conf": self.conf_threshold,
            "iou": self.nms_threshold,
            "verbose": False,
            "stream": False,
            "batch": self.batch_size,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        return self.model.predict(**predict_kwargs)

    def run_sliced_infer(self, image: np.ndarray) -> List[Detection]:
        image_h, image_w = image.shape[:2]
        tile_w = image_w // self.slice_cols
        tile_h = image_h // self.slice_rows
        overlap_w = int(tile_w * self.slice_overlap)
        overlap_h = int(tile_h * self.slice_overlap)

        all_boxes: List[Tuple[int, int, int, int]] = []
        all_scores: List[float] = []
        tiles: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []

        for row in range(self.slice_rows):
            for col in range(self.slice_cols):
                x0 = max(0, col * tile_w - overlap_w)
                y0 = max(0, row * tile_h - overlap_h)
                x1 = min(image_w, (col + 1) * tile_w + overlap_w)
                y1 = min(image_h, (row + 1) * tile_h + overlap_h)
                tiles.append(image[y0:y1, x0:x1])
                offsets.append((x0, y0))

        results = self.predict_tiles(tiles)
        for tile, result, (x0, y0) in zip(tiles, results, offsets):
            for det in self.result_to_detections(result, tile):
                x, y, w, h = det.box
                all_boxes.append((x + x0, y + y0, w, h))
                all_scores.append(det.score)

        keep = nms_indices(all_boxes, all_scores, self.score_keep, self.nms_threshold)
        return [Detection(box=all_boxes[index], score=all_scores[index]) for index in keep]

    def process_frame(self) -> None:
        if self._processing_frame:
            return

        self._processing_frame = True
        color_frame = self.get_latest_color_frame()
        try:
            if not color_frame:
                if self.show and self.standalone_window:
                    cv2.waitKey(1)
                return

            now = time.monotonic()
            dt = now - self.last_time
            self.last_time = now
            if 0.0 < dt < 1.0:
                instant_fps = 1.0 / dt
                self.current_fps = instant_fps if self.current_fps == 0.0 else self.current_fps * 0.9 + instant_fps * 0.1

            image = np.asanyarray(color_frame.get_data()).copy()
            self.latest_frame_size = (image.shape[1], image.shape[0])
            self.total_frames += 1

            inferred_this_frame = False
            if now - self.last_infer_time >= self.infer_period:
                infer_t0 = time.monotonic()
                self.last_detections = self.run_infer_with_roi(image)
                self.last_infer_ms = (time.monotonic() - infer_t0) * 1000.0
                self.last_infer_time = now
                inferred_this_frame = True

            for det in self.last_detections:
                draw_modern_ui(image, det.box, (0, 255, 0), f"{det.score:.2f}", True)

            if inferred_this_frame and self.last_detections:
                msg = String()
                msg.data = f"detected {len(self.last_detections)} barcodes"
                self.publisher.publish(msg)

            exposure_text = "AE" if self.auto_exposure else f"Exp:{self.exposure:.0f} Gain:{self.gain:.0f}"
            roi_text = "ROI:Drag/R" if self.enable_mouse_roi and self.roi is None else "ROI:ON/R" if self.enable_mouse_roi else "ROI:OFF"
            hud_text = f"FPS: {self.current_fps:.1f} | Infer: {self.last_infer_ms:.1f}ms | Dets: {len(self.last_detections)} | {exposure_text} | Input: {self.input_width}x{self.input_height} | {roi_text}"
            cv2.putText(image, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            self.draw_roi_overlay(image)
            self.publish_panel_frame(image)

            if self.show and self.standalone_window:
                display = image
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                self.handle_key(key)
        finally:
            self._processing_frame = False


def spin_node(node_cls) -> None:
    rclpy.init()
    node = node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
