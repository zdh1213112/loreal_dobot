#!/usr/bin/env python3
"""
Standalone D435 eye-to-hand RRT grasp planner for Dobot Nova5.

This script does not modify any existing vision or robot-control files.
It uses only the top D435 camera for:
1. YOLO target localization in camera coordinates
2. D435 depth point-cloud obstacle extraction
3. 3D RRT planning with obstacle clearance
4. Qt button-driven Nova5 execution
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciPyRot
from ultralytics import YOLO

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from PySide6 import QtCore
    from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
    from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PyQt5 import QtCore
        from PyQt5.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal as Signal
        from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PyQt5"
    except ImportError:
        from PySide2 import QtCore
        from PySide2.QtCore import QPoint, QRect, QSize, Qt, Signal
        from PySide2.QtGui import QImage, QPainter, QPen, QPixmap
        from PySide2.QtWidgets import (
            QApplication,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PySide2"


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d435_nova5_rrt_config.yaml")


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def euler_deg_to_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    return SciPyRot.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


def rotation_matrix_to_euler_deg(rot: np.ndarray) -> tuple[float, float, float]:
    rx_deg, ry_deg, rz_deg = SciPyRot.from_matrix(rot).as_euler("xyz", degrees=True)
    return float(rx_deg), float(ry_deg), float(rz_deg)


def make_transform(rotation: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return transform


def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) == 0:
        return points_xyz.copy()
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return (rot @ points_xyz.T).T + trans


def project_point_to_pixel(point_cam: np.ndarray, intrinsics: rs.intrinsics) -> Optional[tuple[int, int]]:
    z = float(point_cam[2])
    if z <= 1e-6:
        return None
    u = int(round(point_cam[0] * intrinsics.fx / z + intrinsics.ppx))
    v = int(round(point_cam[1] * intrinsics.fy / z + intrinsics.ppy))
    if 0 <= u < intrinsics.width and 0 <= v < intrinsics.height:
        return u, v
    return None


def deep_update(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config_dict() -> dict:
    return {
        "camera": {
            "serial": "254622078230",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "min_depth_m": 0.15,
            "max_depth_m": 1.60,
            "depth_stride": 2,
            "yolo_weights": "/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-4/weights/best.pt",
            "yolo_conf": 0.50,
            "target_selection": "leftmost_x",
            "depth_window_px": 4,
        },
        "planner": {
            "obstacle_clearance_m": 0.10,
            "approach_offset_m": 0.12,
            "step_size_m": 0.04,
            "edge_resolution_m": 0.02,
            "goal_tolerance_m": 0.03,
            "max_iterations": 4000,
            "goal_sample_rate": 0.20,
            "workspace_margin_m": 0.08,
            "start_exclusion_radius_m": 0.12,
            "target_exclusion_radius_m": 0.05,
            "pregrasp_exclusion_radius_m": 0.05,
            "smoothing_passes": 200,
            "random_seed": 7,
            "remove_support_plane": True,
            "support_plane_distance_threshold_m": 0.012,
            "support_plane_normal_cos_threshold": 0.80,
            "support_plane_min_points": 1500,
        },
        "robot": {
            "driver_root": "/home/zdh/dobot_nova5/_staging_dobot_nova5_driver",
            "ip": "192.168.142.102",
            "dashboard_port": 29999,
            "feedback_port": 30004,
            "startup_joint": [178.0, -7.60, 90.0, 2.40, -90.0, -1.50],
            "startup_speed": 35,
            "auto_enable": False,
            "go_to_start": False,
            "linear_speed": 20,
            "linear_acc": 40,
            "joint_speed": 25,
            "joint_acc": 40,
            "use_robot_feedback_for_start": True,
            "manual_start_cam_xyz": [0.0, 0.0, 0.45],
            "handeye_base_to_d435": [
                0.9938108, 0.10750777, -0.02796736, -0.02212362452,
                0.10480691, -0.99087883, -0.08470334, 0.63295122513,
                -0.03681853, 0.08124792, -0.99601364, 1.3125790433,
                0.0, 0.0, 0.0, 1.0,
            ],
            "grasp_x_flip_deg": 180.0,
            "grasp_pitch_bias_deg": 0.0,
            "grasp_yaw_bias_deg": -90.0,
            "grasp_roll_extra_deg": 0.0,
            "tcp_tip_offset_m": [0.0, 0.0, 0.03],
            "auto_close_gripper_after_grasp": False,
            # "dh_enable": False,
            "dh_enable": True,
            "dh_tool_identify": 1,
            "dh_slave_id": 1,
            "dh_force": 30,
            "dh_grasp_force": 30,
            "dh_max_opening_m": 0.095,
            "dh_wait_timeout_s": 10.0,
        },
        "ui": {
            "preview_width": 960,
            "preview_height": 540,
        },
    }


def load_config(config_path: str) -> dict:
    config = default_config_dict()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = deep_update(config, loaded)
    return config


@dataclass
class CameraConfig:
    serial: str
    width: int
    height: int
    fps: int
    min_depth_m: float
    max_depth_m: float
    depth_stride: int
    yolo_weights: str
    yolo_conf: float
    target_selection: str
    depth_window_px: int

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        return cls(
            serial=str(data["serial"]),
            width=int(data["width"]),
            height=int(data["height"]),
            fps=int(data["fps"]),
            min_depth_m=float(data["min_depth_m"]),
            max_depth_m=float(data["max_depth_m"]),
            depth_stride=int(data["depth_stride"]),
            yolo_weights=str(data["yolo_weights"]),
            yolo_conf=float(data["yolo_conf"]),
            target_selection=str(data["target_selection"]),
            depth_window_px=int(data["depth_window_px"]),
        )


@dataclass
class PlannerConfig:
    obstacle_clearance_m: float
    approach_offset_m: float
    step_size_m: float
    edge_resolution_m: float
    goal_tolerance_m: float
    max_iterations: int
    goal_sample_rate: float
    workspace_margin_m: float
    start_exclusion_radius_m: float
    target_exclusion_radius_m: float
    pregrasp_exclusion_radius_m: float
    smoothing_passes: int
    random_seed: int
    remove_support_plane: bool
    support_plane_distance_threshold_m: float
    support_plane_normal_cos_threshold: float
    support_plane_min_points: int

    @classmethod
    def from_dict(cls, data: dict) -> "PlannerConfig":
        return cls(
            obstacle_clearance_m=float(data["obstacle_clearance_m"]),
            approach_offset_m=float(data["approach_offset_m"]),
            step_size_m=float(data["step_size_m"]),
            edge_resolution_m=float(data["edge_resolution_m"]),
            goal_tolerance_m=float(data["goal_tolerance_m"]),
            max_iterations=int(data["max_iterations"]),
            goal_sample_rate=float(data["goal_sample_rate"]),
            workspace_margin_m=float(data["workspace_margin_m"]),
            start_exclusion_radius_m=float(data["start_exclusion_radius_m"]),
            target_exclusion_radius_m=float(data["target_exclusion_radius_m"]),
            pregrasp_exclusion_radius_m=float(data["pregrasp_exclusion_radius_m"]),
            smoothing_passes=int(data["smoothing_passes"]),
            random_seed=int(data["random_seed"]),
            remove_support_plane=bool(data["remove_support_plane"]),
            support_plane_distance_threshold_m=float(data["support_plane_distance_threshold_m"]),
            support_plane_normal_cos_threshold=float(data["support_plane_normal_cos_threshold"]),
            support_plane_min_points=int(data["support_plane_min_points"]),
        )


@dataclass
class RobotConfig:
    driver_root: str
    ip: str
    dashboard_port: int
    feedback_port: int
    startup_joint: list[float]
    startup_speed: int
    auto_enable: bool
    go_to_start: bool
    linear_speed: int
    linear_acc: int
    joint_speed: int
    joint_acc: int
    use_robot_feedback_for_start: bool
    manual_start_cam_xyz: list[float]
    handeye_base_to_d435: list[float]
    grasp_x_flip_deg: float
    grasp_pitch_bias_deg: float
    grasp_yaw_bias_deg: float
    grasp_roll_extra_deg: float
    tcp_tip_offset_m: list[float]
    auto_close_gripper_after_grasp: bool
    dh_enable: bool
    dh_tool_identify: int
    dh_slave_id: int
    dh_force: int
    dh_grasp_force: int
    dh_max_opening_m: float
    dh_wait_timeout_s: float

    @classmethod
    def from_dict(cls, data: dict) -> "RobotConfig":
        return cls(
            driver_root=str(data["driver_root"]),
            ip=str(data["ip"]),
            dashboard_port=int(data["dashboard_port"]),
            feedback_port=int(data["feedback_port"]),
            startup_joint=[float(v) for v in data["startup_joint"]],
            startup_speed=int(data["startup_speed"]),
            auto_enable=bool(data["auto_enable"]),
            go_to_start=bool(data["go_to_start"]),
            linear_speed=int(data["linear_speed"]),
            linear_acc=int(data["linear_acc"]),
            joint_speed=int(data["joint_speed"]),
            joint_acc=int(data["joint_acc"]),
            use_robot_feedback_for_start=bool(data["use_robot_feedback_for_start"]),
            manual_start_cam_xyz=[float(v) for v in data["manual_start_cam_xyz"]],
            handeye_base_to_d435=[float(v) for v in data["handeye_base_to_d435"]],
            grasp_x_flip_deg=float(data["grasp_x_flip_deg"]),
            grasp_pitch_bias_deg=float(data["grasp_pitch_bias_deg"]),
            grasp_yaw_bias_deg=float(data["grasp_yaw_bias_deg"]),
            grasp_roll_extra_deg=float(data["grasp_roll_extra_deg"]),
            tcp_tip_offset_m=[float(v) for v in data["tcp_tip_offset_m"]],
            auto_close_gripper_after_grasp=bool(data["auto_close_gripper_after_grasp"]),
            dh_enable=bool(data["dh_enable"]),
            dh_tool_identify=int(data["dh_tool_identify"]),
            dh_slave_id=int(data["dh_slave_id"]),
            dh_force=int(data["dh_force"]),
            dh_grasp_force=int(data["dh_grasp_force"]),
            dh_max_opening_m=float(data["dh_max_opening_m"]),
            dh_wait_timeout_s=float(data["dh_wait_timeout_s"]),
        )


@dataclass
class UIConfig:
    preview_width: int
    preview_height: int

    @classmethod
    def from_dict(cls, data: dict) -> "UIConfig":
        return cls(
            preview_width=int(data["preview_width"]),
            preview_height=int(data["preview_height"]),
        )


@dataclass
class AppConfig:
    camera: CameraConfig
    planner: PlannerConfig
    robot: RobotConfig
    ui: UIConfig

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            camera=CameraConfig.from_dict(data["camera"]),
            planner=PlannerConfig.from_dict(data["planner"]),
            robot=RobotConfig.from_dict(data["robot"]),
            ui=UIConfig.from_dict(data["ui"]),
        )


@dataclass
class DetectionResult:
    center_uv: tuple[int, int]
    center_xyz_cam: np.ndarray
    yaw_rad: float
    confidence: float
    corners_uv: np.ndarray


class PreviewLabel(QLabel):
    roi_changed = Signal(object)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setMouseTracking(True)
        self._base_pixmap: Optional[QPixmap] = None
        self._display_pixmap: Optional[QPixmap] = None
        self._image_size: Optional[tuple[int, int]] = None
        self._pixmap_rect = QRect()
        self._roi_rect_image: Optional[QRect] = None
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None

    def set_preview_pixmap(self, pixmap: QPixmap, image_size: tuple[int, int]) -> None:
        self._base_pixmap = pixmap
        self._image_size = image_size
        self._render()

    def clear_roi(self) -> None:
        self._roi_rect_image = None
        self._drag_start = None
        self._drag_current = None
        self._render()
        self.roi_changed.emit(None)

    def roi_rect_image(self) -> Optional[QRect]:
        if self._roi_rect_image is None:
            return None
        return QRect(self._roi_rect_image)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event) -> None:
        point = self._event_point(event)
        if event.button() != Qt.LeftButton or self._base_pixmap is None or self._image_size is None:
            return super().mousePressEvent(event)
        if not self._pixmap_rect.contains(point):
            return super().mousePressEvent(event)
        self._drag_start = point
        self._drag_current = point
        self._render()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            return super().mouseMoveEvent(event)
        self._drag_current = self._event_point(event)
        self._render()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._drag_start is None or self._image_size is None:
            return super().mouseReleaseEvent(event)
        self._drag_current = self._event_point(event)
        rect_widget = QRect(self._drag_start, self._drag_current).normalized().intersected(self._pixmap_rect)
        self._drag_start = None
        self._drag_current = None
        rect_image = self._widget_rect_to_image_rect(rect_widget)
        if rect_image is None:
            self._render()
            return
        self._roi_rect_image = rect_image
        self._render()
        self.roi_changed.emit(QRect(self._roi_rect_image))

    def _render(self) -> None:
        if self._base_pixmap is None:
            self._display_pixmap = None
            return
        scaled = self._base_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(self.size())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        x = max(0, (self.width() - scaled.width()) // 2)
        y = max(0, (self.height() - scaled.height()) // 2)
        self._pixmap_rect = QRect(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(self._pixmap_rect.topLeft(), scaled)

        if self._roi_rect_image is not None:
            roi_widget = self._image_rect_to_widget_rect(self._roi_rect_image)
            if roi_widget is not None:
                pen = QPen(Qt.green, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(roi_widget)

        if self._drag_start is not None and self._drag_current is not None:
            drag_rect = QRect(self._drag_start, self._drag_current).normalized().intersected(self._pixmap_rect)
            pen = QPen(Qt.yellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(drag_rect)

        painter.end()
        self._display_pixmap = canvas
        self.setPixmap(canvas)

    def _widget_rect_to_image_rect(self, rect_widget: QRect) -> Optional[QRect]:
        if self._image_size is None or self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
            return None
        rect = rect_widget.normalized().intersected(self._pixmap_rect)
        if rect.width() < 5 or rect.height() < 5:
            return None
        image_w, image_h = self._image_size
        x0 = int(round((rect.left() - self._pixmap_rect.left()) * image_w / self._pixmap_rect.width()))
        y0 = int(round((rect.top() - self._pixmap_rect.top()) * image_h / self._pixmap_rect.height()))
        x1 = int(round((rect.right() - self._pixmap_rect.left()) * image_w / self._pixmap_rect.width()))
        y1 = int(round((rect.bottom() - self._pixmap_rect.top()) * image_h / self._pixmap_rect.height()))
        x0 = max(0, min(image_w - 1, x0))
        y0 = max(0, min(image_h - 1, y0))
        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return QRect(QPoint(x0, y0), QPoint(x1, y1))

    def _image_rect_to_widget_rect(self, rect_image: QRect) -> Optional[QRect]:
        if self._image_size is None or self._pixmap_rect.width() <= 0 or self._pixmap_rect.height() <= 0:
            return None
        image_w, image_h = self._image_size
        x0 = self._pixmap_rect.left() + int(round(rect_image.left() * self._pixmap_rect.width() / image_w))
        y0 = self._pixmap_rect.top() + int(round(rect_image.top() * self._pixmap_rect.height() / image_h))
        x1 = self._pixmap_rect.left() + int(round(rect_image.right() * self._pixmap_rect.width() / image_w))
        y1 = self._pixmap_rect.top() + int(round(rect_image.bottom() * self._pixmap_rect.height() / image_h))
        return QRect(QPoint(x0, y0), QPoint(x1, y1)).normalized()

    def _event_point(self, event) -> QPoint:
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()


@dataclass
class SceneCapture:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: rs.intrinsics
    point_cloud_cam: np.ndarray
    detection: DetectionResult
    target_transform_cam: np.ndarray
    start_cam_xyz: np.ndarray
    pregrasp_cam_xyz: np.ndarray
    annotated_bgr: np.ndarray
    plane_removed: bool
    plane_inlier_count: int
    roi_xyxy: Optional[tuple[int, int, int, int]]


@dataclass
class LiveFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    point_cloud_cam: np.ndarray
    point_cloud_colors_rgb: np.ndarray


@dataclass
class PlanResult:
    capture: SceneCapture
    filtered_obstacle_points_cam: np.ndarray
    rrt_path_cam: np.ndarray
    full_path_cam: np.ndarray
    flange_path_cam: np.ndarray
    full_path_base: np.ndarray
    full_path_base_poses: list[object]
    grasp_pose_base: object
    grasp_tip_base: np.ndarray
    pregrasp_pose_base: object


class RRTPlanningError(RuntimeError):
    pass


class LivePointCloudViewer:
    def __init__(self) -> None:
        self._vis = None
        self._pcd = None
        self._coord = None
        self._markers = None
        self._path = None
        self._flange_path = None
        self._initialized = False
        self._view_initialized = False

    def ensure_open(self) -> None:
        if o3d is None or self._initialized:
            return
        window_width = 960
        window_height = 720
        window_left = 900
        window_top = 50
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            window_width = min(window_width, max(640, int(available.width() * 0.42)))
            window_height = min(window_height, max(480, int(available.height() * 0.60)))
            window_left = max(available.left() + 16, available.right() - window_width - 16)
            window_top = max(available.top() + 40, 40)
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(
            "D435 Live Point Cloud",
            width=window_width,
            height=window_height,
            left=window_left,
            top=window_top,
        )
        self._vis.get_render_option().point_size = 2.0
        self._vis.get_render_option().background_color = np.array([0.08, 0.08, 0.08])
        self._pcd = o3d.geometry.PointCloud()
        self._coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        self._markers = o3d.geometry.PointCloud()
        self._path = o3d.geometry.LineSet()
        self._flange_path = o3d.geometry.LineSet()
        self._vis.add_geometry(self._pcd)
        self._vis.add_geometry(self._coord)
        self._vis.add_geometry(self._markers)
        self._vis.add_geometry(self._path)
        self._vis.add_geometry(self._flange_path)
        self._initialized = True
        self._view_initialized = False

    def update(
        self,
        points_xyz: np.ndarray,
        colors_rgb: np.ndarray,
        capture: Optional[SceneCapture] = None,
        plan: Optional[PlanResult] = None,
    ) -> None:
        if o3d is None:
            return
        self.ensure_open()
        assert self._vis is not None and self._pcd is not None and self._markers is not None and self._path is not None and self._flange_path is not None
        self._pcd.points = o3d.utility.Vector3dVector(np.asarray(points_xyz, dtype=np.float64))
        self._pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors_rgb, dtype=np.float64))

        marker_points = np.zeros((0, 3), dtype=np.float64)
        marker_colors = np.zeros((0, 3), dtype=np.float64)
        if capture is not None:
            marker_points = np.vstack(
                [
                    np.asarray(capture.start_cam_xyz, dtype=np.float64).reshape(1, 3),
                    np.asarray(capture.pregrasp_cam_xyz, dtype=np.float64).reshape(1, 3),
                    np.asarray(capture.detection.center_xyz_cam, dtype=np.float64).reshape(1, 3),
                ]
            )
            marker_colors = np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.5, 0.0],
                ],
                dtype=np.float64,
            )
        self._markers.points = o3d.utility.Vector3dVector(marker_points)
        self._markers.colors = o3d.utility.Vector3dVector(marker_colors)

        if plan is not None:
            path_points = np.asarray(plan.full_path_cam, dtype=np.float64)
        elif capture is not None:
            path_points = np.vstack(
                [
                    np.asarray(capture.start_cam_xyz, dtype=np.float64).reshape(1, 3),
                    np.asarray(capture.pregrasp_cam_xyz, dtype=np.float64).reshape(1, 3),
                    np.asarray(capture.detection.center_xyz_cam, dtype=np.float64).reshape(1, 3),
                ]
            )
        else:
            path_points = np.zeros((0, 3), dtype=np.float64)

        if len(path_points) >= 2:
            lines = np.asarray([[i, i + 1] for i in range(len(path_points) - 1)], dtype=np.int32)
            line_colors = np.tile(np.asarray([[0.0, 1.0, 1.0]], dtype=np.float64), (len(lines), 1))
            self._path.points = o3d.utility.Vector3dVector(path_points)
            self._path.lines = o3d.utility.Vector2iVector(lines)
            self._path.colors = o3d.utility.Vector3dVector(line_colors)
        else:
            self._path.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self._path.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            self._path.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        flange_path_points = np.asarray(plan.flange_path_cam, dtype=np.float64) if plan is not None else np.zeros((0, 3), dtype=np.float64)
        if len(flange_path_points) >= 2:
            flange_lines = np.asarray([[i, i + 1] for i in range(len(flange_path_points) - 1)], dtype=np.int32)
            flange_line_colors = np.tile(np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64), (len(flange_lines), 1))
            self._flange_path.points = o3d.utility.Vector3dVector(flange_path_points)
            self._flange_path.lines = o3d.utility.Vector2iVector(flange_lines)
            self._flange_path.colors = o3d.utility.Vector3dVector(flange_line_colors)
        else:
            self._flange_path.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            self._flange_path.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            self._flange_path.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        self._vis.update_geometry(self._pcd)
        self._vis.update_geometry(self._markers)
        self._vis.update_geometry(self._path)
        self._vis.update_geometry(self._flange_path)
        if len(points_xyz) > 0 and not self._view_initialized:
            try:
                self._vis.reset_view_point(True)
                ctr = self._vis.get_view_control()
                center = np.mean(np.asarray(points_xyz, dtype=np.float64), axis=0)
                ctr.set_lookat(center.tolist())
                ctr.set_front([0.0, 0.0, -1.0])
                ctr.set_up([0.0, -1.0, 0.0])
                ctr.set_zoom(0.45)
            except Exception:
                pass
            self._view_initialized = True
        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
        self._vis = None
        self._pcd = None
        self._coord = None
        self._markers = None
        self._path = None
        self._initialized = False
        self._view_initialized = False


class D435TopSceneManager:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.pipeline: Optional[rs.pipeline] = None
        self.profile = None
        self.depth_scale = 0.001
        self.align = rs.align(rs.stream.color)
        self.intrinsics: Optional[rs.intrinsics] = None
        self.yolo = YOLO(self.config.yolo_weights)

    def start(self) -> None:
        if self.pipeline is not None:
            return
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self.config.serial)
        rs_config.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps)
        rs_config.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps)
        profile = pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = color_intr
        self.pipeline = pipeline
        self.profile = profile

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.profile = None
            self.intrinsics = None

    def get_live_frame(self) -> LiveFrame:
        if self.pipeline is None or self.intrinsics is None:
            raise RuntimeError("D435 is not started")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned D435 color/depth frames")

        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        points_xyz, colors_rgb = self._depth_to_colored_points(depth_m, color_bgr, self.intrinsics)
        return LiveFrame(
            color_bgr=color_bgr,
            depth_m=depth_m,
            point_cloud_cam=points_xyz,
            point_cloud_colors_rgb=colors_rgb,
        )

    def capture(
        self,
        start_cam_xyz: np.ndarray,
        planner_cfg: PlannerConfig,
        roi_xyxy: Optional[tuple[int, int, int, int]] = None,
    ) -> SceneCapture:
        if self.pipeline is None or self.intrinsics is None:
            raise RuntimeError("D435 is not started")

        live_frame = None
        for _ in range(3):
            live_frame = self.get_live_frame()
        assert live_frame is not None
        color_bgr = live_frame.color_bgr
        depth_m = live_frame.depth_m

        detection = self._detect_target(color_bgr, depth_m, roi_xyxy=roi_xyxy)
        target_transform_cam = make_transform(
            SciPyRot.from_euler("z", detection.yaw_rad).as_matrix(),
            detection.center_xyz_cam,
        )
        pregrasp_cam_xyz = self._compute_pregrasp_point(detection.center_xyz_cam, planner_cfg.approach_offset_m)
        point_cloud_cam = live_frame.point_cloud_cam

        filtered_points, plane_removed, plane_inlier_count = self._remove_support_plane_if_needed(
            point_cloud_cam,
            planner_cfg,
        )
        annotated = self._build_annotated_image(
            color_bgr,
            detection,
            start_cam_xyz,
            pregrasp_cam_xyz,
        )

        return SceneCapture(
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            point_cloud_cam=filtered_points,
            detection=detection,
            target_transform_cam=target_transform_cam,
            start_cam_xyz=np.asarray(start_cam_xyz, dtype=np.float64),
            pregrasp_cam_xyz=pregrasp_cam_xyz,
            annotated_bgr=annotated,
            plane_removed=plane_removed,
            plane_inlier_count=plane_inlier_count,
            roi_xyxy=roi_xyxy,
        )

    def _detect_target(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        roi_xyxy: Optional[tuple[int, int, int, int]] = None,
    ) -> DetectionResult:
        roi = self._normalize_roi_xyxy(roi_xyxy, color_bgr.shape[1], color_bgr.shape[0])
        if roi is None:
            color_input = color_bgr
            x_offset = 0
            y_offset = 0
        else:
            x0, y0, x1, y1 = roi
            color_input = color_bgr[y0:y1, x0:x1]
            x_offset = x0
            y_offset = y0

        results = self.yolo(color_input, conf=self.config.yolo_conf, verbose=False)
        if not results or results[0].obb is None or len(results[0].obb) == 0:
            if roi is None:
                raise RuntimeError("YOLO did not find any OBB target in the current D435 frame")
            raise RuntimeError("YOLO did not find any OBB target inside the selected ROI")

        obbs = results[0].obb
        centers_x = obbs.xyxyxyxy[:, :, 0].mean(dim=1)
        centers_y = obbs.xyxyxyxy[:, :, 1].mean(dim=1)
        confidences = obbs.conf

        strategy = self.config.target_selection.lower()
        if strategy == "leftmost_x":
            best_idx = int(torch_argmin(centers_x))
        elif strategy == "highest_confidence":
            best_idx = int(torch_argmax(confidences))
        else:
            raise ValueError(f"Unsupported target_selection strategy: {self.config.target_selection}")

        u_local = int(round(float(centers_x[best_idx])))
        v_local = int(round(float(centers_y[best_idx])))
        u = u_local + x_offset
        v = v_local + y_offset
        z_m = self._robust_depth_at(depth_m, u, v)
        if not (self.config.min_depth_m < z_m < self.config.max_depth_m):
            raise RuntimeError(f"Target depth is invalid: z={z_m:.3f}m")

        intr = self.intrinsics
        assert intr is not None
        x_m = (u - intr.ppx) * z_m / intr.fx
        y_m = (v - intr.ppy) * z_m / intr.fy
        yaw_rad = float(obbs.xywhr[best_idx, 4].item())
        confidence = float(confidences[best_idx].item())
        corners_uv = obbs.xyxyxyxy[best_idx].cpu().numpy().astype(np.int32)
        corners_uv[:, 0] += x_offset
        corners_uv[:, 1] += y_offset

        return DetectionResult(
            center_uv=(u, v),
            center_xyz_cam=np.array([x_m, y_m, z_m], dtype=np.float64),
            yaw_rad=yaw_rad,
            confidence=confidence,
            corners_uv=corners_uv,
        )

    def _normalize_roi_xyxy(
        self,
        roi_xyxy: Optional[tuple[int, int, int, int]],
        width: int,
        height: int,
    ) -> Optional[tuple[int, int, int, int]]:
        if roi_xyxy is None:
            return None
        x0, y0, x1, y1 = [int(v) for v in roi_xyxy]
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        return x0, y0, x1, y1

    def _robust_depth_at(self, depth_m: np.ndarray, u: int, v: int) -> float:
        half = max(1, self.config.depth_window_px)
        u0 = max(0, u - half)
        u1 = min(depth_m.shape[1], u + half + 1)
        v0 = max(0, v - half)
        v1 = min(depth_m.shape[0], v + half + 1)
        patch = depth_m[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.config.min_depth_m) & (valid < self.config.max_depth_m)]
        if len(valid) == 0:
            raise RuntimeError(f"No valid depth around target pixel ({u}, {v})")
        return float(np.median(valid))

    def _depth_to_points(self, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> np.ndarray:
        stride = max(1, self.config.depth_stride)
        us, vs = np.meshgrid(
            np.arange(0, depth_m.shape[1], stride, dtype=np.float32),
            np.arange(0, depth_m.shape[0], stride, dtype=np.float32),
        )
        z = depth_m[::stride, ::stride].reshape(-1)
        valid = np.isfinite(z) & (z > self.config.min_depth_m) & (z < self.config.max_depth_m)
        us = us.reshape(-1)[valid]
        vs = vs.reshape(-1)[valid]
        z = z[valid].astype(np.float64)
        x = (us - intrinsics.ppx) * z / intrinsics.fx
        y = (vs - intrinsics.ppy) * z / intrinsics.fy
        return np.stack([x, y, z], axis=1)

    def _depth_to_colored_points(
        self,
        depth_m: np.ndarray,
        color_bgr: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> tuple[np.ndarray, np.ndarray]:
        stride = max(1, self.config.depth_stride)
        us, vs = np.meshgrid(
            np.arange(0, depth_m.shape[1], stride, dtype=np.int32),
            np.arange(0, depth_m.shape[0], stride, dtype=np.int32),
        )
        z = depth_m[::stride, ::stride].reshape(-1)
        us_flat = us.reshape(-1)
        vs_flat = vs.reshape(-1)
        valid = np.isfinite(z) & (z > self.config.min_depth_m) & (z < self.config.max_depth_m)
        z = z[valid].astype(np.float64)
        us_valid = us_flat[valid].astype(np.float64)
        vs_valid = vs_flat[valid].astype(np.float64)
        x = (us_valid - intrinsics.ppx) * z / intrinsics.fx
        y = (vs_valid - intrinsics.ppy) * z / intrinsics.fy
        points = np.stack([x, y, z], axis=1)
        colors_rgb = color_bgr[vs_flat[valid], us_flat[valid], ::-1].astype(np.float64) / 255.0
        return points, colors_rgb

    def _compute_pregrasp_point(self, target_cam_xyz: np.ndarray, approach_offset_m: float) -> np.ndarray:
        direction = np.asarray(target_cam_xyz, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return target_cam_xyz.copy()
        return target_cam_xyz - direction / norm * approach_offset_m

    def _remove_support_plane_if_needed(
        self,
        points_xyz: np.ndarray,
        planner_cfg: PlannerConfig,
    ) -> tuple[np.ndarray, bool, int]:
        if not planner_cfg.remove_support_plane or o3d is None or len(points_xyz) < planner_cfg.support_plane_min_points:
            return points_xyz, False, 0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        try:
            plane_model, inliers = cloud.segment_plane(
                distance_threshold=planner_cfg.support_plane_distance_threshold_m,
                ransac_n=3,
                num_iterations=200,
            )
        except Exception:
            return points_xyz, False, 0

        if len(inliers) < planner_cfg.support_plane_min_points:
            return points_xyz, False, 0

        plane_normal = np.asarray(plane_model[:3], dtype=np.float64)
        plane_normal_norm = float(np.linalg.norm(plane_normal))
        if plane_normal_norm < 1e-6:
            return points_xyz, False, 0
        plane_normal = plane_normal / plane_normal_norm
        if abs(float(plane_normal[2])) < planner_cfg.support_plane_normal_cos_threshold:
            return points_xyz, False, 0

        mask = np.ones(len(points_xyz), dtype=bool)
        mask[np.asarray(inliers, dtype=np.int64)] = False
        filtered = points_xyz[mask]
        if len(filtered) == 0:
            return points_xyz, False, 0
        return filtered, True, len(inliers)

    def _build_annotated_image(
        self,
        color_bgr: np.ndarray,
        detection: DetectionResult,
        start_cam_xyz: np.ndarray,
        pregrasp_cam_xyz: np.ndarray,
    ) -> np.ndarray:
        annotated = color_bgr.copy()
        cv2.polylines(annotated, [detection.corners_uv], True, (0, 0, 255), 3)
        cv2.circle(annotated, detection.center_uv, 5, (0, 255, 255), -1)
        cv2.putText(
            annotated,
            f"target z={detection.center_xyz_cam[2]:.3f}m conf={detection.confidence:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        for point, color, label in (
            (start_cam_xyz, (255, 0, 0), "START"),
            (pregrasp_cam_xyz, (0, 255, 255), "PRE"),
            (detection.center_xyz_cam, (0, 165, 255), "GOAL"),
        ):
            uv = project_point_to_pixel(point, self.intrinsics)
            if uv is None:
                continue
            cv2.circle(annotated, uv, 7, color, -1)
            cv2.putText(
                annotated,
                label,
                (uv[0] + 8, uv[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return annotated


class RRT3DPlanner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(self.config.random_seed)

    def plan(self, capture: SceneCapture) -> tuple[np.ndarray, np.ndarray]:
        start = capture.start_cam_xyz
        goal = capture.pregrasp_cam_xyz
        obstacles = self._prepare_obstacles(capture)

        if len(obstacles) == 0:
            path = np.stack([start, goal], axis=0)
            return obstacles, path

        tree = cKDTree(obstacles)
        self._assert_free_point(start, tree, "start")
        self._assert_free_point(goal, tree, "goal/pregrasp")

        if self._edge_is_free(start, goal, tree):
            path = np.stack([start, goal], axis=0)
            return obstacles, path

        bounds_min, bounds_max = self._build_workspace_bounds(start, goal, obstacles)
        nodes = [start.copy()]
        parents = [-1]

        for _ in range(self.config.max_iterations):
            sample = self._sample(bounds_min, bounds_max, goal)
            nearest_idx = self._nearest_index(nodes, sample)
            new_point = self._steer(nodes[nearest_idx], sample)

            if not self._point_is_free(new_point, tree):
                continue
            if not self._edge_is_free(nodes[nearest_idx], new_point, tree):
                continue

            nodes.append(new_point)
            parents.append(nearest_idx)
            new_idx = len(nodes) - 1

            if np.linalg.norm(new_point - goal) <= max(self.config.goal_tolerance_m, self.config.step_size_m):
                if self._edge_is_free(new_point, goal, tree):
                    nodes.append(goal.copy())
                    parents.append(new_idx)
                    raw_path = self._backtrack(nodes, parents, len(nodes) - 1)
                    smooth_path = self._smooth_path(raw_path, tree)
                    return obstacles, smooth_path

        raise RRTPlanningError(
            f"RRT failed after {self.config.max_iterations} iterations. "
            "Try increasing workspace margin, step size, or reducing obstacle clearance."
        )

    def _prepare_obstacles(self, capture: SceneCapture) -> np.ndarray:
        points = np.asarray(capture.point_cloud_cam, dtype=np.float64)
        if len(points) == 0:
            return points

        min_corner = np.minimum.reduce(
            [capture.start_cam_xyz, capture.pregrasp_cam_xyz, capture.detection.center_xyz_cam]
        ) - self.config.workspace_margin_m
        max_corner = np.maximum.reduce(
            [capture.start_cam_xyz, capture.pregrasp_cam_xyz, capture.detection.center_xyz_cam]
        ) + self.config.workspace_margin_m
        roi_mask = np.all((points >= min_corner) & (points <= max_corner), axis=1)
        points = points[roi_mask]

        if len(points) == 0:
            return points

        points = self._exclude_sphere(points, capture.start_cam_xyz, self.config.start_exclusion_radius_m)
        points = self._exclude_sphere(points, capture.pregrasp_cam_xyz, self.config.pregrasp_exclusion_radius_m)
        points = self._exclude_sphere(points, capture.detection.center_xyz_cam, self.config.target_exclusion_radius_m)
        return points

    def _exclude_sphere(self, points: np.ndarray, center: np.ndarray, radius_m: float) -> np.ndarray:
        if len(points) == 0 or radius_m <= 0.0:
            return points
        distances = np.linalg.norm(points - center.reshape(1, 3), axis=1)
        return points[distances > radius_m]

    def _assert_free_point(self, point: np.ndarray, tree: cKDTree, label: str) -> None:
        if not self._point_is_free(point, tree):
            nearest = float(tree.query(point, k=1)[0])
            raise RRTPlanningError(
                f"{label} is already inside inflated obstacles: nearest distance={nearest:.4f}m, "
                f"required clearance={self.config.obstacle_clearance_m:.4f}m"
            )

    def _build_workspace_bounds(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_points = np.vstack([obstacles, start.reshape(1, 3), goal.reshape(1, 3)])
        bounds_min = np.min(all_points, axis=0) - self.config.workspace_margin_m
        bounds_max = np.max(all_points, axis=0) + self.config.workspace_margin_m
        return bounds_min, bounds_max

    def _sample(self, bounds_min: np.ndarray, bounds_max: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if float(self.rng.random()) < self.config.goal_sample_rate:
            return goal.copy()
        return self.rng.uniform(bounds_min, bounds_max)

    def _nearest_index(self, nodes: list[np.ndarray], sample: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node - sample)) for node in nodes]
        return int(np.argmin(distances))

    def _steer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = target - source
        distance = float(np.linalg.norm(direction))
        if distance <= self.config.step_size_m:
            return target.copy()
        return source + direction / distance * self.config.step_size_m

    def _point_is_free(self, point: np.ndarray, tree: cKDTree) -> bool:
        nearest = float(tree.query(point, k=1)[0])
        return nearest >= self.config.obstacle_clearance_m

    def _edge_is_free(self, p0: np.ndarray, p1: np.ndarray, tree: cKDTree) -> bool:
        distance = float(np.linalg.norm(p1 - p0))
        if distance < 1e-9:
            return self._point_is_free(p0, tree)
        steps = max(2, int(math.ceil(distance / self.config.edge_resolution_m)) + 1)
        ts = np.linspace(0.0, 1.0, steps)
        samples = p0.reshape(1, 3) + (p1 - p0).reshape(1, 3) * ts.reshape(-1, 1)
        nearest = tree.query(samples, k=1)[0]
        return bool(np.all(nearest >= self.config.obstacle_clearance_m))

    def _backtrack(self, nodes: list[np.ndarray], parents: list[int], index: int) -> np.ndarray:
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        path.reverse()
        return np.asarray(path, dtype=np.float64)

    def _smooth_path(self, path: np.ndarray, tree: cKDTree) -> np.ndarray:
        if len(path) <= 2:
            return path

        smoothed = path.copy()
        for _ in range(max(0, self.config.smoothing_passes)):
            if len(smoothed) <= 2:
                break
            i = int(self.rng.integers(0, len(smoothed) - 1))
            j = int(self.rng.integers(i + 1, len(smoothed)))
            if j <= i + 1:
                continue
            if self._edge_is_free(smoothed[i], smoothed[j], tree):
                smoothed = np.vstack([smoothed[: i + 1], smoothed[j:]])

        greedy = [smoothed[0]]
        index = 0
        while index < len(smoothed) - 1:
            next_index = len(smoothed) - 1
            while next_index > index + 1:
                if self._edge_is_free(smoothed[index], smoothed[next_index], tree):
                    break
                next_index -= 1
            greedy.append(smoothed[next_index])
            index = next_index
        return np.asarray(greedy, dtype=np.float64)


class Nova5RobotBridge:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.base_to_d435 = np.array(self.config.handeye_base_to_d435, dtype=np.float64).reshape(4, 4)
        self.d435_to_base = np.linalg.inv(self.base_to_d435)
        self.tcp_tip_offset_flange = np.array(self.config.tcp_tip_offset_m, dtype=np.float64).reshape(3)
        self._controller = None
        self._dh_gripper = None
        self._dashboard_shared = None
        self._motion_lock = threading.Lock()
        self._abort_motion = threading.Event()

        if self.config.driver_root not in sys.path:
            sys.path.insert(0, self.config.driver_root)
        driver_package_root = os.path.join(self.config.driver_root, "dobot_nova5_driver")
        if os.path.isdir(driver_package_root) and driver_package_root not in sys.path:
            sys.path.insert(0, driver_package_root)

        from dobot_nova5_driver.controller import DobotNova5Controller, TcpPose

        self.DobotNova5Controller = DobotNova5Controller
        self.TcpPose = TcpPose
        self._dh_imports_ready = False
        self._drag_enabled = False
        self._coord_type = "user"

    def connect(self) -> None:
        if self._controller is not None:
            return
        self._controller = self.DobotNova5Controller(
            robot_ip=self.config.ip,
            dashboard_port=self.config.dashboard_port,
            feedback_port=self.config.feedback_port,
            startup_joint=self.config.startup_joint,
            startup_speed=self.config.startup_speed,
        )
        self._controller.connect(
            go_to_start=self.config.go_to_start,
            auto_enable=self.config.auto_enable,
        )
        self._controller.set_linear_profile(speed=self.config.linear_speed, accel=self.config.linear_acc)
        self._controller.set_joint_profile(speed=self.config.joint_speed, accel=self.config.joint_acc)
        if self.config.dh_enable:
            self._init_dh_gripper()

    def disconnect(self) -> None:
        self._disconnect_dh_gripper()
        if self._controller is not None:
            self._controller.disconnect()
            self._controller = None

    def is_connected(self) -> bool:
        return self._controller is not None

    def robot_mode_text(self) -> str:
        if self._controller is None:
            return "DISCONNECTED"
        return self._controller.robot_mode_text()

    def current_tcp_pose_base(self):
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        return self._controller.current_tcp_pose()

    def current_tcp_point_cam(self) -> np.ndarray:
        if self.config.use_robot_feedback_for_start and self._controller is not None:
            flange_pose = self._controller.current_tcp_pose()
            flange_transform = self._pose_to_transform(flange_pose)
            tip_transform = self._flange_transform_to_tip_transform(flange_transform)
            point_cam = (self.d435_to_base @ tip_transform)[:3, 3]
            return np.asarray(point_cam, dtype=np.float64)
        return np.asarray(self.config.manual_start_cam_xyz, dtype=np.float64)

    def target_pose_base_from_camera(self, target_transform_cam: np.ndarray):
        target_transform_base = self.base_to_d435 @ target_transform_cam
        _, _, target_yaw_deg = rotation_matrix_to_euler_deg(target_transform_base[:3, :3])
        rx_deg = self.config.grasp_x_flip_deg + self.config.grasp_roll_extra_deg
        ry_deg = self.config.grasp_pitch_bias_deg
        rz_deg = normalize_angle_deg(target_yaw_deg + self.config.grasp_yaw_bias_deg)
        tip_pose = self.TcpPose(
            x=float(target_transform_base[0, 3]),
            y=float(target_transform_base[1, 3]),
            z=float(target_transform_base[2, 3]),
            rx=float(rx_deg),
            ry=float(ry_deg),
            rz=float(rz_deg),
        )
        return self._tip_pose_to_flange_pose(tip_pose)

    def pose_base_from_camera_point(self, point_cam_xyz: np.ndarray, template_pose):
        point_base = transform_points(self.base_to_d435, point_cam_xyz.reshape(1, 3))[0]
        tip_pose = self.TcpPose(
            x=float(point_base[0]),
            y=float(point_base[1]),
            z=float(point_base[2]),
            rx=float(template_pose.rx),
            ry=float(template_pose.ry),
            rz=float(template_pose.rz),
        )
        return self._tip_pose_to_flange_pose(tip_pose)

    def _pose_to_transform(self, pose) -> np.ndarray:
        return make_transform(
            euler_deg_to_rotation_matrix(pose.rx, pose.ry, pose.rz),
            np.array([pose.x, pose.y, pose.z], dtype=np.float64),
        )

    def _transform_to_pose(self, transform: np.ndarray):
        rx_deg, ry_deg, rz_deg = rotation_matrix_to_euler_deg(transform[:3, :3])
        return self.TcpPose(
            x=float(transform[0, 3]),
            y=float(transform[1, 3]),
            z=float(transform[2, 3]),
            rx=float(rx_deg),
            ry=float(ry_deg),
            rz=float(rz_deg),
        )

    def _flange_transform_to_tip_transform(self, flange_transform: np.ndarray) -> np.ndarray:
        tip_transform = np.array(flange_transform, dtype=np.float64, copy=True)
        tip_transform[:3, 3] = flange_transform[:3, 3] + flange_transform[:3, :3] @ self.tcp_tip_offset_flange
        return tip_transform

    def _tip_pose_to_flange_pose(self, tip_pose):
        tip_transform = self._pose_to_transform(tip_pose)
        flange_transform = np.array(tip_transform, dtype=np.float64, copy=True)
        flange_transform[:3, 3] = tip_transform[:3, 3] - tip_transform[:3, :3] @ self.tcp_tip_offset_flange
        return self._transform_to_pose(flange_transform)

    def flange_pose_to_tip_point_base(self, flange_pose) -> np.ndarray:
        flange_transform = self._pose_to_transform(flange_pose)
        tip_transform = self._flange_transform_to_tip_transform(flange_transform)
        return np.asarray(tip_transform[:3, 3], dtype=np.float64)

    def execute_path(self, path_base: np.ndarray, grasp_pose_base) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")

        self._abort_motion.clear()
        with self._motion_lock:
            path_points = [np.asarray(point, dtype=np.float64) for point in path_base]
            executable_points = path_points[1:] if len(path_points) >= 2 else []
            current_pose = self._controller.current_tcp_pose()
            travel_rx = float(current_pose.rx)
            travel_ry = float(current_pose.ry)
            travel_rz = float(current_pose.rz)

            if executable_points:
                travel_points = executable_points[:-1]
                pregrasp_point = executable_points[-1]

                if travel_points:
                    self._raise_if_motion_aborted()
                    first_point = travel_points[0]
                    first_pose = self.TcpPose(
                        x=float(first_point[0]),
                        y=float(first_point[1]),
                        z=float(first_point[2]),
                        rx=travel_rx,
                        ry=travel_ry,
                        rz=travel_rz,
                    )
                    print(
                        "[ExecutePath] MovJ travel[0] flange="
                        f"({first_pose.x:.3f}, {first_pose.y:.3f}, {first_pose.z:.3f}, "
                        f"{first_pose.rx:.1f}, {first_pose.ry:.1f}, {first_pose.rz:.1f})"
                    )
                    self._controller.move_joint_tcp(
                        first_pose,
                        speed=self.config.joint_speed,
                        accel=self.config.joint_acc,
                    )
                    for index, point in enumerate(travel_points[1:], start=1):
                        self._raise_if_motion_aborted()
                        pose = self.TcpPose(
                            x=float(point[0]),
                            y=float(point[1]),
                            z=float(point[2]),
                            rx=travel_rx,
                            ry=travel_ry,
                            rz=travel_rz,
                        )
                        print(
                            f"[ExecutePath] MovL travel[{index}] flange="
                            f"({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}, "
                            f"{pose.rx:.1f}, {pose.ry:.1f}, {pose.rz:.1f})"
                        )
                        self._controller.move_linear_tcp(
                            pose,
                            speed=self.config.linear_speed,
                            accel=self.config.linear_acc,
                        )

                self._raise_if_motion_aborted()
                pregrasp_pose = self.TcpPose(
                    x=float(pregrasp_point[0]),
                    y=float(pregrasp_point[1]),
                    z=float(pregrasp_point[2]),
                    rx=float(grasp_pose_base.rx),
                    ry=float(grasp_pose_base.ry),
                    rz=float(grasp_pose_base.rz),
                )
                print(
                    "[ExecutePath] MovJ pregrasp flange="
                    f"({pregrasp_pose.x:.3f}, {pregrasp_pose.y:.3f}, {pregrasp_pose.z:.3f}, "
                    f"{pregrasp_pose.rx:.1f}, {pregrasp_pose.ry:.1f}, {pregrasp_pose.rz:.1f})"
                )
                self._controller.move_joint_tcp(
                    pregrasp_pose,
                    speed=self.config.joint_speed,
                    accel=self.config.joint_acc,
                )
            self._raise_if_motion_aborted()
            print(
                "[ExecutePath] MovL grasp flange="
                f"({grasp_pose_base.x:.3f}, {grasp_pose_base.y:.3f}, {grasp_pose_base.z:.3f}, "
                f"{grasp_pose_base.rx:.1f}, {grasp_pose_base.ry:.1f}, {grasp_pose_base.rz:.1f})"
            )
            self._controller.move_linear_tcp(
                grasp_pose_base,
                speed=self.config.linear_speed,
                accel=self.config.linear_acc,
            )

        if self.config.auto_close_gripper_after_grasp and self._dh_gripper is not None:
            self.close_gripper()

    def execute_retreat(self, path_base: np.ndarray, grasp_pose_base) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._abort_motion.clear()
        with self._motion_lock:
            retreat_points = [np.asarray(point, dtype=np.float64) for point in reversed(path_base)]
            for index, point in enumerate(retreat_points):
                self._raise_if_motion_aborted()
                pose = self.TcpPose(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(point[2]),
                    rx=float(grasp_pose_base.rx),
                    ry=float(grasp_pose_base.ry),
                    rz=float(grasp_pose_base.rz),
                )
                if index == 0:
                    self._controller.move_joint_tcp(
                        pose,
                        speed=self.config.joint_speed,
                        accel=self.config.joint_acc,
                    )
                else:
                    self._controller.move_linear_tcp(
                        pose,
                        speed=self.config.linear_speed,
                        accel=self.config.linear_acc,
                    )

    def move_home(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._abort_motion.set()
        try:
            self._controller.stop_motion()
        except Exception:
            pass
        try:
            self._controller.continue_motion()
        except Exception:
            pass
        with self._motion_lock:
            self._controller.move_to_startup()

    def power_on(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.power_on()

    def enable_robot(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.enable_robot()
        if self.config.dh_enable and self._dh_gripper is None:
            self._init_dh_gripper()

    def disable_robot(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.disable_robot()

    def clear_error(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.clear_error()

    def reset_robot(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.reset_robot()

    def stop_motion(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._abort_motion.set()
        self._controller.stop_motion()

    def pause_motion(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._abort_motion.set()
        self._controller.pause_motion()

    def continue_motion(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.continue_motion()

    def set_speed_factor(self, speed: int) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.set_speed_factor(int(speed))

    def set_coord_type(self, coord_type: str) -> None:
        coord_type = str(coord_type).strip().lower()
        if coord_type not in ("user", "tool"):
            raise ValueError(f"Unsupported coord_type: {coord_type}")
        self._coord_type = coord_type

    def toggle_drag(self) -> bool:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        if self._drag_enabled:
            self._controller.stop_drag()
            self._drag_enabled = False
        else:
            self._controller.start_drag()
            self._drag_enabled = True
        return self._drag_enabled

    def start_jog(self, axis_id: str, user: int = 0, tool: int = 0) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        coord_type = 1 if self._coord_type == "user" else 2
        self._controller.move_jog(axis_id=axis_id, coord_type=coord_type, user=user, tool=tool)

    def stop_jog(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._controller.move_jog("")

    def read_current_pose(self):
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        return self._controller.read_pose()

    def current_joint(self) -> list[float]:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        return self._controller.current_joint()

    def manual_move_joint(self, pose) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._abort_motion.clear()
        with self._motion_lock:
            self._controller.move_joint_tcp(
                pose,
                speed=self.config.joint_speed,
                accel=self.config.joint_acc,
            )

    def manual_move_linear(self, pose) -> None:
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        self._abort_motion.clear()
        with self._motion_lock:
            self._controller.move_linear_tcp(
                pose,
                speed=self.config.linear_speed,
                accel=self.config.linear_acc,
            )

    def _raise_if_motion_aborted(self) -> None:
        if self._abort_motion.is_set():
            raise RuntimeError("Motion interrupted by user")

    def open_gripper(self) -> None:
        self._ensure_dh_gripper_ready()
        self._dh_gripper.set_position(
            1.0,
            wait=True,
            timeout_s=self.config.dh_wait_timeout_s,
        )

    def close_gripper(self) -> None:
        self._ensure_dh_gripper_ready()
        self._dh_gripper.set_force(self.config.dh_grasp_force)
        self._dh_gripper.set_position(
            0.0,
            wait=True,
            timeout_s=self.config.dh_wait_timeout_s,
        )

    def _ensure_dh_gripper_ready(self) -> None:
        if not self.config.dh_enable:
            raise RuntimeError(
                "DH gripper is disabled in config. Set robot.dh_enable: true in d435_nova5_rrt_config.yaml"
            )
        if self._controller is None:
            raise RuntimeError("Robot is not connected")
        if self._dh_gripper is None:
            self._init_dh_gripper()
        if self._dh_gripper is None:
            raise RuntimeError("DH gripper initialization failed")

    def _init_dh_gripper(self) -> None:
        if self._controller is None:
            raise RuntimeError("Robot must be connected before DH gripper init")
        if self._dh_gripper is not None:
            return
        if self._dh_imports_ready is False:
            from dobot_nova5_driver.dobot_dh_api import DobotDHConfig, DHGripper, raise_if_error

            self.DobotDHConfig = DobotDHConfig
            self.DHGripper = DHGripper
            self.raise_if_error = raise_if_error
            self._dh_imports_ready = True

        dashboard = self._controller.dashboard
        if dashboard is None:
            raise RuntimeError("Robot dashboard is not connected")

        cfg = self.DobotDHConfig(
            robot_ip=self.config.ip,
            dashboard_port=self.config.dashboard_port,
            tool_identify=self.config.dh_tool_identify,
            slave_id=self.config.dh_slave_id,
            force=self.config.dh_force,
            enable_robot=False,
        )

        self._configure_dh_tool_rs485(dashboard, cfg)
        self._cleanup_stale_dh_modbus(dashboard)

        gripper = self.DHGripper(dashboard, cfg)
        gripper.initialize(timeout_s=self.config.dh_wait_timeout_s, init_open=True)
        self._dh_gripper = gripper
        self._dashboard_shared = dashboard

    def _configure_dh_tool_rs485(self, dashboard, cfg) -> None:
        tool_mode_response = dashboard.SetToolMode(1, 1, cfg.tool_identify)
        if "Control Mode Is Not Tcp" in str(tool_mode_response):
            return
        self.raise_if_error(tool_mode_response, "SetToolMode")

        tool_485_response = dashboard.SetTool485(cfg.baudrate, cfg.parity, cfg.stop_bit, cfg.tool_identify)
        if "Control Mode Is Not Tcp" in str(tool_485_response):
            return
        self.raise_if_error(tool_485_response, "SetTool485")

    def _cleanup_stale_dh_modbus(self, dashboard) -> None:
        for master_index in range(5):
            try:
                dashboard.ModbusClose(master_index)
            except Exception:
                pass

    def _disconnect_dh_gripper(self) -> None:
        if self._dh_gripper is not None:
            try:
                self._dh_gripper.disconnect()
            finally:
                self._dh_gripper = None
        self._dashboard_shared = None


class PlannerMainWindow(QMainWindow):
    status_signal = Signal(str)
    image_signal = Signal(object)
    plan_signal = Signal(object)
    error_signal = Signal(str)
    mode_signal = Signal(str)
    scene_info_signal = Signal(object)
    robot_feedback_signal = Signal(object)
    drag_state_signal = Signal(bool)
    live_frame_signal = Signal(object)
    live_timer_signal = Signal(bool)

    def __init__(self, config: AppConfig, config_path: str) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.scene_manager = D435TopSceneManager(self.config.camera)
        self.rrt_planner = RRT3DPlanner(self.config.planner)
        self.robot_bridge = Nova5RobotBridge(self.config.robot)
        self.latest_capture: Optional[SceneCapture] = None
        self.latest_plan: Optional[PlanResult] = None
        self.latest_preview_bgr: Optional[np.ndarray] = None
        self.latest_live_frame: Optional[LiveFrame] = None
        self._busy = False
        self._busy_label = ""
        self._roi_rect_image: Optional[QRect] = None
        self._live_update_timer = QtCore.QTimer(self)
        self._live_update_timer.setInterval(120)
        self._live_update_timer.timeout.connect(self._poll_live_view)
        self._live_viewer = LivePointCloudViewer()
        self._live_polling = False

        self.status_signal.connect(self._append_log)
        self.image_signal.connect(self._set_preview_image)
        self.plan_signal.connect(self._on_plan_ready)
        self.error_signal.connect(self._show_error)
        self.robot_feedback_signal.connect(self._apply_robot_feedback)
        self.drag_state_signal.connect(self._apply_drag_state)
        self.live_frame_signal.connect(self._apply_live_frame)
        self.live_timer_signal.connect(self._set_live_timer_running)

        self._preview_display_size = self._compute_preview_display_size()
        self._build_ui()
        self.mode_signal.connect(self.mode_label.setText)
        self.scene_info_signal.connect(self._apply_scene_info)
        self.setWindowTitle(f"D435 RRT Nova5 Grasp Planner [{QT_BINDING}]")
        QtCore.QTimer.singleShot(0, self._fit_window_to_screen)
        self._append_log(f"Loaded config: {self.config_path}")

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col.setSpacing(8)

        layout.addLayout(left_col, 0)
        layout.addLayout(right_col, 1)

        self.robot_group = QGroupBox("Robot")
        robot_layout = QGridLayout(self.robot_group)
        self.robot_ip_edit = QLineEdit(self.config.robot.ip)
        self.connect_button = QPushButton("Connect Robot")
        self.power_button = QPushButton("Power On")
        self.enable_button = QPushButton("Enable")
        self.disable_button = QPushButton("Disable")
        self.clear_error_button = QPushButton("Clear Error")
        self.reset_button = QPushButton("Reset")
        self.pause_button = QPushButton("Pause")
        self.continue_button = QPushButton("Continue")
        self.drag_button = QPushButton("Enter Drag")
        self.home_button = QPushButton("Home")
        self.stop_button = QPushButton("Stop")
        self.open_gripper_button = QPushButton("Open Gripper")
        self.close_gripper_button = QPushButton("Close Gripper")
        self.mode_label = QLabel("DISCONNECTED")
        self.speed_factor_spin = QSpinBox()
        self.speed_factor_spin.setRange(1, 100)
        self.speed_factor_spin.setValue(100)
        self.apply_speed_button = QPushButton("Apply Speed")
        robot_layout.addWidget(QLabel("IP"), 0, 0)
        robot_layout.addWidget(self.robot_ip_edit, 0, 1, 1, 3)
        robot_layout.addWidget(self.connect_button, 1, 0)
        robot_layout.addWidget(self.power_button, 1, 1)
        robot_layout.addWidget(self.enable_button, 1, 2)
        robot_layout.addWidget(self.disable_button, 1, 3)
        robot_layout.addWidget(self.clear_error_button, 2, 0)
        robot_layout.addWidget(self.reset_button, 2, 1)
        robot_layout.addWidget(self.pause_button, 2, 2)
        robot_layout.addWidget(self.continue_button, 2, 3)
        robot_layout.addWidget(self.drag_button, 3, 0)
        robot_layout.addWidget(self.home_button, 3, 1)
        robot_layout.addWidget(self.stop_button, 3, 2)
        robot_layout.addWidget(self.open_gripper_button, 4, 0)
        robot_layout.addWidget(self.close_gripper_button, 4, 1)
        robot_layout.addWidget(QLabel("Speed %"), 4, 2)
        robot_layout.addWidget(self.speed_factor_spin, 4, 3)
        robot_layout.addWidget(self.apply_speed_button, 5, 2, 1, 2)
        robot_layout.addWidget(QLabel("Mode"), 5, 0)
        robot_layout.addWidget(self.mode_label, 5, 1, 1, 2)
        left_col.addWidget(self.robot_group)

        self.manual_group = QGroupBox("Manual Motion")
        manual_layout = QGridLayout(self.manual_group)
        self.user_index_spin = QSpinBox()
        self.user_index_spin.setRange(0, 9)
        self.tool_index_spin = QSpinBox()
        self.tool_index_spin.setRange(0, 9)
        self.user_coord_button = QPushButton("Jog Coord: User")
        self.tool_coord_button = QPushButton("Jog Coord: Tool")
        self.read_pose_button = QPushButton("Read Pose")
        self.write_pose_button = QPushButton("Read -> Target")
        self.movj_button = QPushButton("MovJ Target")
        self.movl_button = QPushButton("MovL Target")
        self.pose_display_edits: dict[str, QLineEdit] = {}
        self.joint_display_edits: dict[str, QLineEdit] = {}
        self.target_spinboxes: dict[str, QDoubleSpinBox] = {}

        manual_layout.addWidget(QLabel("User"), 0, 0)
        manual_layout.addWidget(self.user_index_spin, 0, 1)
        manual_layout.addWidget(QLabel("Tool"), 0, 2)
        manual_layout.addWidget(self.tool_index_spin, 0, 3)
        manual_layout.addWidget(self.user_coord_button, 1, 0, 1, 2)
        manual_layout.addWidget(self.tool_coord_button, 1, 2, 1, 2)
        manual_layout.addWidget(self.read_pose_button, 2, 0, 1, 2)
        manual_layout.addWidget(self.write_pose_button, 2, 2, 1, 2)
        self._build_pose_spinbox_columns(manual_layout, start_row=3, column_offset=0, store=self.target_spinboxes)
        manual_layout.addWidget(self.movj_button, 9, 0, 1, 2)
        manual_layout.addWidget(self.movl_button, 9, 2, 1, 2)
        self._build_pose_display_group(manual_layout, title="Current TCP", start_row=10, edits=self.pose_display_edits)
        self._build_joint_display_group(manual_layout, title="Current Joint", start_row=10, column=2, edits=self.joint_display_edits)
        left_col.addWidget(self.manual_group)

        self.jog_group = QGroupBox("Jog")
        jog_layout = QGridLayout(self.jog_group)
        self._build_jog_buttons(jog_layout)
        left_col.addWidget(self.jog_group)

        self.vision_group = QGroupBox("Vision / Planner")
        vision_layout = QGridLayout(self.vision_group)
        self.serial_edit = QLineEdit(self.config.camera.serial)
        self.clearance_spin = self._make_double_spin(self.config.planner.obstacle_clearance_m, 0.01, 0.30, 0.005)
        self.approach_spin = self._make_double_spin(self.config.planner.approach_offset_m, 0.02, 0.30, 0.01)
        self.step_spin = self._make_double_spin(self.config.planner.step_size_m, 0.01, 0.20, 0.005)
        self.start_camera_button = QPushButton("Start D435")
        self.capture_button = QPushButton("Capture Scene")
        self.plan_button = QPushButton("Plan RRT")
        self.preview_3d_button = QPushButton("Preview 3D")
        self.execute_button = QPushButton("Execute")
        self.retreat_button = QPushButton("Retreat")
        self.use_roi_button = QPushButton("Use ROI For Detect")
        self.clear_roi_button = QPushButton("Clear ROI")
        self.roi_label = QLabel("ROI: full image")
        vision_layout.addWidget(QLabel("D435 Serial"), 0, 0)
        vision_layout.addWidget(self.serial_edit, 0, 1, 1, 2)
        vision_layout.addWidget(QLabel("Clearance (m)"), 1, 0)
        vision_layout.addWidget(self.clearance_spin, 1, 1, 1, 2)
        vision_layout.addWidget(QLabel("Approach (m)"), 2, 0)
        vision_layout.addWidget(self.approach_spin, 2, 1, 1, 2)
        vision_layout.addWidget(QLabel("Step (m)"), 3, 0)
        vision_layout.addWidget(self.step_spin, 3, 1, 1, 2)
        vision_layout.addWidget(self.start_camera_button, 4, 0)
        vision_layout.addWidget(self.capture_button, 4, 1)
        vision_layout.addWidget(self.plan_button, 4, 2)
        vision_layout.addWidget(self.preview_3d_button, 5, 0)
        vision_layout.addWidget(self.execute_button, 5, 1)
        vision_layout.addWidget(self.retreat_button, 5, 2)
        vision_layout.addWidget(self.use_roi_button, 6, 0)
        vision_layout.addWidget(self.clear_roi_button, 6, 1)
        vision_layout.addWidget(self.roi_label, 6, 2)
        left_col.addWidget(self.vision_group)

        self.info_group = QGroupBox("Scene Info")
        info_layout = QFormLayout(self.info_group)
        self.start_label = QLabel("-")
        self.target_label = QLabel("-")
        self.pregrasp_label = QLabel("-")
        self.path_label = QLabel("-")
        self.plane_label = QLabel("-")
        self.scene_roi_label = QLabel("ROI: full image")
        info_layout.addRow("Start Cam XYZ", self.start_label)
        info_layout.addRow("Target Cam XYZ", self.target_label)
        info_layout.addRow("Pregrasp Cam XYZ", self.pregrasp_label)
        info_layout.addRow("Path Points", self.path_label)
        info_layout.addRow("Plane Removal", self.plane_label)
        info_layout.addRow("Detection ROI", self.scene_roi_label)
        left_col.addWidget(self.info_group)
        left_col.addStretch(1)

        self.preview_label = PreviewLabel("No D435 frame yet")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(self._preview_display_size)
        self.preview_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.preview_label.setStyleSheet("border: 1px solid #555; background: #111; color: #ddd;")
        right_col.addWidget(self.preview_label, 0, Qt.AlignTop)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(180)
        right_col.addWidget(self.log_edit, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)

        self.connect_button.clicked.connect(lambda: self._run_background("connect robot", self._connect_robot))
        self.power_button.clicked.connect(lambda: self._run_background("power on", self._power_on_robot))
        self.enable_button.clicked.connect(lambda: self._run_background("enable robot", self._enable_robot))
        self.disable_button.clicked.connect(lambda: self._run_background("disable robot", self._disable_robot))
        self.clear_error_button.clicked.connect(lambda: self._run_background("clear error", self._clear_error))
        self.reset_button.clicked.connect(lambda: self._run_background("reset robot", self._reset_robot))
        self.pause_button.clicked.connect(lambda: self._run_background("pause robot", self._pause_robot, allow_while_busy=True))
        self.continue_button.clicked.connect(lambda: self._run_background("continue robot", self._continue_robot, allow_while_busy=True))
        self.drag_button.clicked.connect(lambda: self._run_background("toggle drag", self._toggle_drag))
        self.home_button.clicked.connect(lambda: self._run_background("home robot", self._move_home, allow_while_busy=True))
        self.stop_button.clicked.connect(lambda: self._run_background("stop robot", self._stop_robot, allow_while_busy=True))
        self.open_gripper_button.clicked.connect(lambda: self._run_background("open gripper", self.robot_bridge.open_gripper))
        self.close_gripper_button.clicked.connect(lambda: self._run_background("close gripper", self.robot_bridge.close_gripper))
        self.apply_speed_button.clicked.connect(lambda: self._run_background("apply speed", self._apply_speed_factor))
        self.user_coord_button.clicked.connect(lambda: self._set_coord_type("user"))
        self.tool_coord_button.clicked.connect(lambda: self._set_coord_type("tool"))
        self.read_pose_button.clicked.connect(self._read_pose_feedback)
        self.write_pose_button.clicked.connect(self._write_feedback_to_target)
        self.movj_button.clicked.connect(lambda: self._run_background("manual movj", self._manual_move_joint))
        self.movl_button.clicked.connect(lambda: self._run_background("manual movl", self._manual_move_linear))
        self.start_camera_button.clicked.connect(lambda: self._run_background("start camera", self._start_camera))
        self.capture_button.clicked.connect(lambda: self._run_background("capture scene", self._capture_scene))
        self.plan_button.clicked.connect(lambda: self._run_background("plan path", self._plan_path))
        self.use_roi_button.clicked.connect(self._announce_current_roi)
        self.clear_roi_button.clicked.connect(self.preview_label.clear_roi)
        self.preview_3d_button.clicked.connect(self._toggle_live_point_cloud_view)
        self.execute_button.clicked.connect(lambda: self._run_background("execute path", self._execute_plan))
        self.retreat_button.clicked.connect(lambda: self._run_background("retreat path", self._retreat_plan))
        self.preview_label.roi_changed.connect(self._on_roi_changed)

    def _compute_preview_display_size(self) -> QSize:
        base_width = max(320, int(self.config.ui.preview_width))
        base_height = max(240, int(self.config.ui.preview_height))
        screen = QApplication.primaryScreen()
        if screen is None:
            return QSize(base_width, base_height)
        available = screen.availableGeometry()
        max_width = max(320, int(available.width() * 0.40))
        max_height = max(240, int(available.height() * 0.46))
        scale = min(max_width / base_width, max_height / base_height, 1.0)
        return QSize(max(320, int(round(base_width * scale))), max(240, int(round(base_height * scale))))

    def _fit_window_to_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 820)
            return
        available = screen.availableGeometry()
        max_width = max(960, int(available.width() * 0.98))
        max_height = max(720, int(available.height() * 0.96))
        hint = self.sizeHint()
        width = min(max(960, hint.width() + 24), max_width)
        height = min(max(720, hint.height() + 24), max_height)
        self.resize(width, height)

    def _make_double_spin(self, value: float, minimum: float, maximum: float, single_step: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setDecimals(4)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(single_step)
        widget.setValue(value)
        return widget

    def _build_pose_spinbox_columns(
        self,
        layout: QGridLayout,
        start_row: int,
        column_offset: int,
        store: dict[str, QDoubleSpinBox],
    ) -> None:
        specs = (
            ("X", -2.0, 2.0, 0.005),
            ("Y", -2.0, 2.0, 0.005),
            ("Z", -2.0, 2.0, 0.005),
            ("Rx", -180.0, 180.0, 1.0),
            ("Ry", -180.0, 180.0, 1.0),
            ("Rz", -180.0, 180.0, 1.0),
        )
        for index, (axis, minimum, maximum, step) in enumerate(specs):
            row = start_row + index
            spin = self._make_double_spin(0.0, minimum, maximum, step)
            layout.addWidget(QLabel(axis), row, column_offset)
            layout.addWidget(spin, row, column_offset + 1)
            store[axis] = spin

    def _build_pose_display_group(
        self,
        layout: QGridLayout,
        title: str,
        start_row: int,
        edits: dict[str, QLineEdit],
    ) -> None:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for axis in ("X", "Y", "Z", "Rx", "Ry", "Rz"):
            edit = QLineEdit("-")
            edit.setReadOnly(True)
            edits[axis] = edit
            form.addRow(axis, edit)
        layout.addWidget(box, start_row, 0, 1, 2)

    def _build_joint_display_group(
        self,
        layout: QGridLayout,
        title: str,
        start_row: int,
        column: int,
        edits: dict[str, QLineEdit],
    ) -> None:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for axis in ("J1", "J2", "J3", "J4", "J5", "J6"):
            edit = QLineEdit("-")
            edit.setReadOnly(True)
            edits[axis] = edit
            form.addRow(axis, edit)
        layout.addWidget(box, start_row, column, 1, 2)

    def _build_jog_buttons(self, layout: QGridLayout) -> None:
        axes = ("X", "Y", "Z", "Rx", "Ry", "Rz")
        for row, axis in enumerate(axes):
            minus_btn = QPushButton(f"{axis}-")
            plus_btn = QPushButton(f"{axis}+")
            minus_btn.pressed.connect(lambda a=f"{axis}-": self._start_jog(a))
            minus_btn.released.connect(self._stop_jog)
            plus_btn.pressed.connect(lambda a=f"{axis}+": self._start_jog(a))
            plus_btn.released.connect(self._stop_jog)
            layout.addWidget(QLabel(axis), row, 0)
            layout.addWidget(minus_btn, row, 1)
            layout.addWidget(plus_btn, row, 2)

    def _sync_ui_to_config(self) -> None:
        self.config.robot.ip = self.robot_ip_edit.text().strip()
        self.config.camera.serial = self.serial_edit.text().strip()
        self.config.planner.obstacle_clearance_m = float(self.clearance_spin.value())
        self.config.planner.approach_offset_m = float(self.approach_spin.value())
        self.config.planner.step_size_m = float(self.step_spin.value())
        self.scene_manager.config = self.config.camera
        self.rrt_planner.config = self.config.planner
        self.robot_bridge.config.ip = self.config.robot.ip
        self.robot_bridge.config = self.config.robot

    def _run_background(self, label: str, fn, allow_while_busy: bool = False) -> None:
        if self._busy and not allow_while_busy:
            self._append_log(f"Busy: previous action still running, skip {label}")
            return
        self._sync_ui_to_config()

        def worker() -> None:
            previous_busy = self._busy
            previous_label = self._busy_label
            self._busy = True
            self._busy_label = label
            try:
                self.status_signal.emit(f"Start: {label}")
                fn()
                self.status_signal.emit(f"Done: {label}")
            except Exception as exc:
                text = f"{label} failed: {exc}\n{traceback.format_exc()}"
                self.error_signal.emit(text)
            finally:
                self._busy = previous_busy
                self._busy_label = previous_label

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _connect_robot(self) -> None:
        self.robot_bridge.connect()
        start_cam = self.robot_bridge.current_tcp_point_cam()
        self._emit_robot_feedback()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())
        self.scene_info_signal.emit({"start": self._fmt_xyz(start_cam)})
        self.status_signal.emit(
            f"Robot connected. mode={self.robot_bridge.robot_mode_text()} start_cam={self._fmt_xyz(start_cam)}"
        )

    def _power_on_robot(self) -> None:
        self.robot_bridge.power_on()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _enable_robot(self) -> None:
        self.robot_bridge.enable_robot()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())
        self._emit_robot_feedback()

    def _disable_robot(self) -> None:
        self.robot_bridge.disable_robot()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _clear_error(self) -> None:
        self.robot_bridge.clear_error()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _reset_robot(self) -> None:
        self.robot_bridge.reset_robot()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _move_home(self) -> None:
        self.robot_bridge.move_home()
        self._emit_robot_feedback()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _stop_robot(self) -> None:
        self.robot_bridge.stop_motion()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _pause_robot(self) -> None:
        self.robot_bridge.pause_motion()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _continue_robot(self) -> None:
        self.robot_bridge.continue_motion()
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _toggle_drag(self) -> None:
        enabled = self.robot_bridge.toggle_drag()
        self.drag_state_signal.emit(enabled)
        self.mode_signal.emit(self.robot_bridge.robot_mode_text())

    def _apply_speed_factor(self) -> None:
        self.robot_bridge.set_speed_factor(int(self.speed_factor_spin.value()))

    def _set_coord_type(self, coord_type: str) -> None:
        self.robot_bridge.set_coord_type(coord_type)
        if coord_type == "user":
            self.user_coord_button.setEnabled(False)
            self.tool_coord_button.setEnabled(True)
        else:
            self.user_coord_button.setEnabled(True)
            self.tool_coord_button.setEnabled(False)

    def _read_pose_feedback(self) -> None:
        try:
            self._emit_robot_feedback()
        except Exception as exc:
            self._show_error(f"read pose failed: {exc}")

    def _write_feedback_to_target(self) -> None:
        try:
            pose = self.robot_bridge.read_current_pose()
        except Exception as exc:
            self._show_error(f"read pose failed: {exc}")
            return
        self._set_pose_spinboxes(self.target_spinboxes, pose)
        self._append_log("Current pose written into manual target")

    def _manual_target_pose(self):
        return self.robot_bridge.TcpPose(
            x=float(self.target_spinboxes["X"].value()),
            y=float(self.target_spinboxes["Y"].value()),
            z=float(self.target_spinboxes["Z"].value()),
            rx=float(self.target_spinboxes["Rx"].value()),
            ry=float(self.target_spinboxes["Ry"].value()),
            rz=float(self.target_spinboxes["Rz"].value()),
        )

    def _manual_move_joint(self) -> None:
        self.robot_bridge.manual_move_joint(self._manual_target_pose())
        self._emit_robot_feedback()

    def _manual_move_linear(self) -> None:
        self.robot_bridge.manual_move_linear(self._manual_target_pose())
        self._emit_robot_feedback()

    def _start_jog(self, axis_id: str) -> None:
        user = int(self.user_index_spin.value())
        tool = int(self.tool_index_spin.value())
        try:
            self.robot_bridge.start_jog(axis_id=axis_id, user=user, tool=tool)
        except Exception as exc:
            self._show_error(f"start jog failed: {exc}")

    def _stop_jog(self) -> None:
        try:
            self.robot_bridge.stop_jog()
        except Exception as exc:
            self._show_error(f"stop jog failed: {exc}")

    def _emit_robot_feedback(self) -> None:
        pose = self.robot_bridge.current_tcp_pose_base()
        joints = self.robot_bridge.current_joint()
        self.robot_feedback_signal.emit(
            {
                "pose": (pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz),
                "joints": tuple(float(v) for v in joints),
            }
        )

    def _apply_robot_feedback(self, payload: dict) -> None:
        pose_values = payload["pose"]
        joints = payload["joints"]
        for axis, value in zip(("X", "Y", "Z", "Rx", "Ry", "Rz"), pose_values):
            self.pose_display_edits[axis].setText(f"{value:.3f}")
        for axis, value in zip(("J1", "J2", "J3", "J4", "J5", "J6"), joints):
            self.joint_display_edits[axis].setText(f"{value:.3f}")

    def _apply_drag_state(self, enabled: bool) -> None:
        self.drag_button.setText("Exit Drag" if enabled else "Enter Drag")

    def _set_pose_spinboxes(self, spinboxes: dict[str, QDoubleSpinBox], pose) -> None:
        spinboxes["X"].setValue(float(pose.x))
        spinboxes["Y"].setValue(float(pose.y))
        spinboxes["Z"].setValue(float(pose.z))
        spinboxes["Rx"].setValue(float(pose.rx))
        spinboxes["Ry"].setValue(float(pose.ry))
        spinboxes["Rz"].setValue(float(pose.rz))

    def _start_camera(self) -> None:
        self.scene_manager.stop()
        self.scene_manager = D435TopSceneManager(self.config.camera)
        self.scene_manager.start()
        self.live_timer_signal.emit(True)
        self.status_signal.emit(
            f"D435 started. serial={self.config.camera.serial} "
            f"size={self.config.camera.width}x{self.config.camera.height}"
        )

    def _poll_live_view(self) -> None:
        if self._live_polling or self.scene_manager.pipeline is None:
            return
        self._live_polling = True
        try:
            live_frame = self.scene_manager.get_live_frame()
            self.live_frame_signal.emit(live_frame)
        except Exception:
            pass
        finally:
            self._live_polling = False

    def _apply_live_frame(self, live_frame: LiveFrame) -> None:
        self.latest_live_frame = live_frame
        overlay = self._compose_live_overlay(live_frame)
        self._set_preview_image(overlay)
        if o3d is not None:
            self._live_viewer.update(
                live_frame.point_cloud_cam,
                live_frame.point_cloud_colors_rgb,
                capture=self.latest_capture,
                plan=self.latest_plan,
            )

    def _set_live_timer_running(self, running: bool) -> None:
        if running:
            if not self._live_update_timer.isActive():
                self._live_update_timer.start()
            return
        if self._live_update_timer.isActive():
            self._live_update_timer.stop()

    def _resolve_start_cam_xyz(self) -> np.ndarray:
        if self.config.robot.use_robot_feedback_for_start and self.robot_bridge.is_connected():
            start_cam = self.robot_bridge.current_tcp_point_cam()
        else:
            start_cam = np.asarray(self.config.robot.manual_start_cam_xyz, dtype=np.float64)
        return np.asarray(start_cam, dtype=np.float64)

    def _capture_scene(self) -> None:
        if self.scene_manager.pipeline is None:
            self.scene_manager.start()

        start_cam_xyz = self._resolve_start_cam_xyz()
        capture = self.scene_manager.capture(
            start_cam_xyz=start_cam_xyz,
            planner_cfg=self.config.planner,
            roi_xyxy=self._current_roi_xyxy(),
        )
        self.latest_capture = capture
        self.latest_plan = None
        plane_text = "removed" if capture.plane_removed else "kept"
        self.scene_info_signal.emit(
            {
                "start": self._fmt_xyz(capture.start_cam_xyz),
                "target": self._fmt_xyz(capture.detection.center_xyz_cam),
                "pregrasp": self._fmt_xyz(capture.pregrasp_cam_xyz),
                "path": "-",
                "plane": f"{plane_text} ({capture.plane_inlier_count})",
                "roi": self._fmt_roi(capture.roi_xyxy),
            }
        )
        self.image_signal.emit(capture.annotated_bgr)
        self.status_signal.emit(
            f"Captured scene. target={self._fmt_xyz(capture.detection.center_xyz_cam)} "
            f"pregrasp={self._fmt_xyz(capture.pregrasp_cam_xyz)} "
            f"plane_removed={capture.plane_removed} inliers={capture.plane_inlier_count} "
            f"roi={self._fmt_roi(capture.roi_xyxy)}"
        )
        self.status_signal.emit("Previous plan cleared after new capture. Run Plan RRT again before Execute.")

    def _plan_path(self) -> None:
        if self.latest_capture is None:
            raise RuntimeError("Capture scene first")

        self.rrt_planner = RRT3DPlanner(self.config.planner)
        capture_for_plan = self.latest_capture
        try:
            filtered_obstacles, path_cam = self.rrt_planner.plan(capture_for_plan)
        except RRTPlanningError as exc:
            if "goal/pregrasp is already inside inflated obstacles" not in str(exc):
                raise
            capture_for_plan = self._adjust_pregrasp_for_clearance(capture_for_plan)
            filtered_obstacles, path_cam = self.rrt_planner.plan(capture_for_plan)

        grasp_pose_base = self.robot_bridge.target_pose_base_from_camera(capture_for_plan.target_transform_cam)
        pregrasp_pose_base = self.robot_bridge.pose_base_from_camera_point(capture_for_plan.pregrasp_cam_xyz, grasp_pose_base)
        full_path_cam = np.vstack([path_cam, capture_for_plan.detection.center_xyz_cam.reshape(1, 3)])
        full_path_base_poses = [
            self.robot_bridge.pose_base_from_camera_point(point_cam, grasp_pose_base)
            for point_cam in full_path_cam
        ]
        full_path_base = np.asarray(
            [[pose.x, pose.y, pose.z] for pose in full_path_base_poses],
            dtype=np.float64,
        )
        flange_path_cam = transform_points(self.robot_bridge.d435_to_base, full_path_base)

        overlay = self._overlay_path_on_image(capture_for_plan.annotated_bgr, full_path_cam, capture_for_plan.intrinsics)
        self.image_signal.emit(overlay)

        result = PlanResult(
            capture=capture_for_plan,
            filtered_obstacle_points_cam=filtered_obstacles,
            rrt_path_cam=path_cam,
            full_path_cam=full_path_cam,
            flange_path_cam=flange_path_cam,
            full_path_base=full_path_base,
            full_path_base_poses=full_path_base_poses,
            grasp_pose_base=grasp_pose_base,
            grasp_tip_base=np.asarray(
                self.robot_bridge.flange_pose_to_tip_point_base(grasp_pose_base),
                dtype=np.float64,
            ),
            pregrasp_pose_base=pregrasp_pose_base,
        )
        self.plan_signal.emit(result)

    def _adjust_pregrasp_for_clearance(self, capture: SceneCapture) -> SceneCapture:
        planner = self.config.planner
        obstacles = self.rrt_planner._prepare_obstacles(capture)
        if len(obstacles) == 0:
            return capture

        tree = cKDTree(obstacles)
        target = np.asarray(capture.detection.center_xyz_cam, dtype=np.float64)
        pregrasp = np.asarray(capture.pregrasp_cam_xyz, dtype=np.float64)
        direction = pregrasp - target
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            raise RRTPlanningError("Cannot auto-adjust pregrasp: target and pregrasp are identical")
        direction = direction / norm

        nearest = float(tree.query(pregrasp, k=1)[0])
        safety_margin = 0.01
        required_offset = max(
            planner.approach_offset_m,
            norm + max(0.0, planner.obstacle_clearance_m - nearest) + safety_margin,
        )

        max_offset = 0.30
        step = max(0.005, planner.edge_resolution_m)
        adjusted_offset = required_offset
        adjusted_pregrasp = None
        while adjusted_offset <= max_offset + 1e-9:
            candidate = target + direction * adjusted_offset
            if self.rrt_planner._point_is_free(candidate, tree):
                adjusted_pregrasp = candidate
                break
            adjusted_offset += step

        if adjusted_pregrasp is None:
            raise RRTPlanningError(
                "goal/pregrasp is too close to obstacles and automatic retreat could not find a collision-free pregrasp"
            )

        adjusted_capture = copy.copy(capture)
        adjusted_capture.pregrasp_cam_xyz = np.asarray(adjusted_pregrasp, dtype=np.float64)
        adjusted_capture.annotated_bgr = self.scene_manager._build_annotated_image(
            capture.color_bgr,
            capture.detection,
            capture.start_cam_xyz,
            adjusted_capture.pregrasp_cam_xyz,
        )
        self.latest_capture = adjusted_capture
        self.scene_info_signal.emit({"pregrasp": self._fmt_xyz(adjusted_capture.pregrasp_cam_xyz)})
        self.status_signal.emit(
            "Adjusted pregrasp outward for obstacle clearance: "
            f"{self._fmt_xyz(capture.pregrasp_cam_xyz)} -> {self._fmt_xyz(adjusted_capture.pregrasp_cam_xyz)}"
        )
        return adjusted_capture

    def _on_plan_ready(self, result: PlanResult) -> None:
        self.latest_plan = result
        self.scene_info_signal.emit(
            {
                "pregrasp": self._fmt_xyz(result.capture.pregrasp_cam_xyz),
                "path": str(len(result.full_path_cam)),
            }
        )
        self._append_log(
            f"RRT planned {len(result.rrt_path_cam)} pregrasp points, total execute points={len(result.full_path_cam)}. "
            f"grasp_base=({result.grasp_pose_base.x:.3f}, {result.grasp_pose_base.y:.3f}, {result.grasp_pose_base.z:.3f}, "
            f"{result.grasp_pose_base.rx:.1f}, {result.grasp_pose_base.ry:.1f}, {result.grasp_pose_base.rz:.1f})"
        )

    def _execute_plan(self) -> None:
        if self.latest_plan is None:
            raise RuntimeError("No active plan. Capture Scene clears the previous plan; run Plan RRT again before Execute.")
        if not self.robot_bridge.is_connected():
            raise RuntimeError("Robot is not connected")

        self.robot_bridge.execute_path(
            path_base=self.latest_plan.full_path_base[:-1],
            grasp_pose_base=self.latest_plan.grasp_pose_base,
        )

    def _retreat_plan(self) -> None:
        if self.latest_plan is None:
            raise RuntimeError("No active plan. Capture Scene clears the previous plan; run Plan RRT again before Retreat.")
        if not self.robot_bridge.is_connected():
            raise RuntimeError("Robot is not connected")

        self.robot_bridge.execute_retreat(
            path_base=self.latest_plan.full_path_base[:-1],
            grasp_pose_base=self.latest_plan.grasp_pose_base,
        )

    def _overlay_path_on_image(
        self,
        image_bgr: np.ndarray,
        path_cam: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> np.ndarray:
        canvas = image_bgr.copy()
        projected = []
        for point in path_cam:
            uv = project_point_to_pixel(point, intrinsics)
            if uv is not None:
                projected.append(uv)
        if len(projected) >= 2:
            for idx in range(len(projected) - 1):
                cv2.line(canvas, projected[idx], projected[idx + 1], (255, 255, 0), 2)
        for uv in projected:
            cv2.circle(canvas, uv, 4, (255, 255, 0), -1)
        return canvas

    def _compose_live_overlay(self, live_frame: LiveFrame) -> np.ndarray:
        canvas = live_frame.color_bgr.copy()
        capture = self.latest_capture
        if capture is not None:
            detection = capture.detection
            cv2.polylines(canvas, [np.asarray(detection.corners_uv, dtype=np.int32)], True, (0, 0, 255), 3)
            cv2.circle(canvas, tuple(int(v) for v in detection.center_uv), 5, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                f"target z={detection.center_xyz_cam[2]:.3f}m conf={detection.confidence:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            for point, color, label in (
                (capture.start_cam_xyz, (255, 0, 0), "START"),
                (capture.pregrasp_cam_xyz, (0, 255, 255), "PRE"),
                (capture.detection.center_xyz_cam, (0, 165, 255), "GOAL"),
            ):
                uv = project_point_to_pixel(point, capture.intrinsics)
                if uv is None:
                    continue
                cv2.circle(canvas, uv, 7, color, -1)
                cv2.putText(
                    canvas,
                    label,
                    (uv[0] + 8, uv[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
        if self.latest_plan is not None:
            canvas = self._overlay_path_on_image(canvas, self.latest_plan.full_path_cam, self.latest_plan.capture.intrinsics)
        return canvas

    def _toggle_live_point_cloud_view(self) -> None:
        if o3d is None:
            QMessageBox.warning(self, "Open3D Missing", "open3d is not installed in this environment.")
            return
        self._live_viewer.ensure_open()
        if self.latest_live_frame is not None:
            self._live_viewer.update(self.latest_live_frame.point_cloud_cam, self.latest_live_frame.point_cloud_colors_rgb)

    def _make_sphere(self, center_xyz: np.ndarray, color_rgb: list[float], radius: float):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color(color_rgb)
        sphere.translate(center_xyz)
        return sphere

    def _set_preview_image(self, image_bgr: np.ndarray) -> None:
        self.latest_preview_bgr = image_bgr.copy()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        q_image = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self._preview_display_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.set_preview_pixmap(pixmap, (width, height))

    def _append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)
        self.mode_label.setText(self.robot_bridge.robot_mode_text())

    def _apply_scene_info(self, info: dict) -> None:
        if "start" in info:
            self.start_label.setText(str(info["start"]))
        if "target" in info:
            self.target_label.setText(str(info["target"]))
        if "pregrasp" in info:
            self.pregrasp_label.setText(str(info["pregrasp"]))
        if "path" in info:
            self.path_label.setText(str(info["path"]))
        if "plane" in info:
            self.plane_label.setText(str(info["plane"]))
        if "roi" in info:
            self.roi_label.setText(str(info["roi"]))
            self.scene_roi_label.setText(str(info["roi"]))

    def _show_error(self, text: str) -> None:
        self._append_log(text)
        QMessageBox.critical(self, "Error", text)

    def _fmt_xyz(self, xyz: np.ndarray) -> str:
        return f"{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}"

    def _current_roi_xyxy(self) -> Optional[tuple[int, int, int, int]]:
        if self._roi_rect_image is None:
            return None
        left = int(self._roi_rect_image.left())
        top = int(self._roi_rect_image.top())
        right = int(self._roi_rect_image.right()) + 1
        bottom = int(self._roi_rect_image.bottom()) + 1
        return left, top, right, bottom

    def _fmt_roi(self, roi_xyxy: Optional[tuple[int, int, int, int]]) -> str:
        if roi_xyxy is None:
            return "ROI: full image"
        x0, y0, x1, y1 = roi_xyxy
        return f"ROI: ({x0}, {y0}) - ({x1}, {y1})"

    def _on_roi_changed(self, rect: Optional[QRect]) -> None:
        self._roi_rect_image = None if rect is None else QRect(rect)
        self.scene_info_signal.emit({"roi": self._fmt_roi(self._current_roi_xyxy())})

    def _announce_current_roi(self) -> None:
        roi = self._current_roi_xyxy()
        if roi is None:
            self._append_log("ROI detection is using the full image. Drag on the preview to select a ROI.")
        else:
            self._append_log(f"ROI detection enabled: {self._fmt_roi(roi)}")

    def closeEvent(self, event) -> None:
        try:
            self._live_update_timer.stop()
            self._live_viewer.close()
            self.scene_manager.stop()
        finally:
            self.robot_bridge.disconnect()
        super().closeEvent(event)


def torch_argmin(tensor) -> int:
    return int(tensor.argmin().item())


def torch_argmax(tensor) -> int:
    return int(tensor.argmax().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D435 eye-to-hand RRT grasp planner for Nova5")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_dict(load_config(args.config))
    app = QApplication(sys.argv)
    window = PlannerMainWindow(config=config, config_path=os.path.abspath(args.config))
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
