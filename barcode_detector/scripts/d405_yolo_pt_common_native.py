#!/usr/bin/env python3
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
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

        self.model_path = str(self.get_parameter("model_path").value)
        self.device = str(self.get_parameter("device").value)
        self.camera_sn = str(self.get_parameter("camera_sn").value).strip()
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

        cv2.setNumThreads(1)
        torch.set_num_threads(self.torch_threads)
        torch.set_num_interop_threads(1)

        self.get_logger().info(f"正在加载 YOLO .pt 模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO .pt 模型加载成功: {self.model_path}")

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.camera_sn:
            config.enable_device(self.camera_sn)
            self.get_logger().info(f"指定 D405 序列号: {self.camera_sn}")
        else:
            self.get_logger().warn("未指定 D405 序列号，将使用系统默认 RealSense 设备")
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        try:
            profile = self.pipeline.start(config)
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1.0)
                if sensor.supports(rs.option.enable_auto_white_balance):
                    sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
            self.get_logger().info("已使用 RealSense 原生自动曝光/自动白平衡，未手动设置 exposure/gain")
            self.get_logger().info(startup_label)
        except Exception as exc:
            self.get_logger().error(f"相机启动失败: {exc}")
            raise

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
            f"use_slicing={self.use_slicing} torch_threads={self.torch_threads}"
        )

    def destroy_node(self) -> bool:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self.show:
            cv2.destroyAllWindows()
        return super().destroy_node()

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
        frames = None
        for _ in range(self.drain_frames):
            polled = self.pipeline.poll_for_frames()
            if not polled:
                break
            frames = polled

        if frames is None:
            try:
                frames = self.pipeline.wait_for_frames(10)
            except Exception:
                return None

            for _ in range(self.drain_frames - 1):
                polled = self.pipeline.poll_for_frames()
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
                if self.show:
                    cv2.waitKey(1)
                return

            now = time.monotonic()
            dt = now - self.last_time
            self.last_time = now
            if 0.0 < dt < 1.0:
                instant_fps = 1.0 / dt
                self.current_fps = instant_fps if self.current_fps == 0.0 else self.current_fps * 0.9 + instant_fps * 0.1

            image = np.asanyarray(color_frame.get_data()).copy()
            self.total_frames += 1

            inferred_this_frame = False
            if now - self.last_infer_time >= self.infer_period:
                infer_t0 = time.monotonic()
                self.last_detections = self.run_sliced_infer(image) if self.use_slicing else self.run_yolo_infer(image)
                self.last_infer_ms = (time.monotonic() - infer_t0) * 1000.0
                self.last_infer_time = now
                inferred_this_frame = True

            for det in self.last_detections:
                draw_modern_ui(image, det.box, (0, 255, 0), f"{det.score:.2f}", True)

            if inferred_this_frame and self.last_detections:
                msg = String()
                msg.data = f"detected {len(self.last_detections)} barcodes"
                self.publisher.publish(msg)

            hud_text = f"FPS: {self.current_fps:.1f} | Infer: {self.last_infer_ms:.1f}ms | Dets: {len(self.last_detections)} | Input: {self.input_width}x{self.input_height}"
            cv2.putText(image, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

            if self.show:
                display = cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale)
                cv2.imshow(self.window_name, display)
                cv2.waitKey(1)
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
