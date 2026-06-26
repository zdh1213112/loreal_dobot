"""
功能说明：Nova5 粗精定位抓取控制端（对应 D405 传送带边缘拨散增强版）。

主要流程：
1. 连接 Dobot Nova5 机械臂 TCP/IP 控制接口，并按配置初始化 DH 夹爪。
2. 订阅 `/target_pose_cam_coarse`，接收 D435 粗定位结果，并转换到机器人 base 坐标执行粗定位移动。
3. 发布 `/coarse_target_obj_for_d405` 给 D405，并通过 `/trigger_d405_vision` 触发 D405 精定位。
4. 订阅 `/target_pose_cam_fine` 和 `/gripper_target_width`，接收 D405 精抓位姿和夹爪开口宽度。
5. 订阅传送带边缘拨散辅助话题：
   - `/target_x_extent_cam`：目标在 D405 相机系下的 X 向尺寸。
   - `/target_singulation_start_cam`：视觉建议的拨散起点。
   - `/target_singulation_needed`：视觉判断是否需要拨散。
   - `/target_singulation_hint`：保留的拨散方向兼容接口。
6. 根据视觉净空判断和 `singulation_auto_decision_enabled` 决定是否执行抓前拨散。
7. 执行固定 X 方向的传送带边缘拨散动作：从目标外沿进入，沿配置方向扫过目标，再抬起回退。
8. 拨散后重新触发 D405 精定位并采样，使用更新后的目标位姿执行精抓。
9. 抓取完成后执行用户配置的中转点、旋转检查和放置点流程。
10. 支持对拨散进入高度、接触高度、目标 X 尺寸、起止边距、速度/加速度和夹爪力进行配置。

匹配视觉端：`d405_local_ransac_coarse_lock_wait_lock_belt_edge_singulation.py`。
"""

import math
import os
import sys
import threading
import time
from typing import Optional
from scipy.spatial.transform import Rotation as SciPyRot

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Bool, Float32
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

if __package__ in (None, ""):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    from dobot_nova5_driver.controller import DobotNova5Controller, ROBOT_MODE_TEXT, TcpPose
    from dobot_nova5_driver.dobot_dh_api import DobotDHConfig, DHGripper, raise_if_error
    from dobot_nova5_driver.TCP_IP_Python_V4.dobot_api import DobotApiDashboard
else:
    from .controller import DobotNova5Controller, ROBOT_MODE_TEXT, TcpPose
    from .dobot_dh_api import DobotDHConfig, DHGripper, raise_if_error
    from .TCP_IP_Python_V4.dobot_api import DobotApiDashboard

try:
    from PySide6 import QtCore
    from PySide6.QtCore import QTimer, Signal
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
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PyQt5 import QtCore
        from PyQt5.QtCore import QTimer, pyqtSignal as Signal
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
            QPushButton,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PyQt5"
    except ImportError:
        from PySide2 import QtCore
        from PySide2.QtCore import QTimer, Signal
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
            QPushButton,
            QScrollArea,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        QT_BINDING = "PySide2"


POSE_AXES = ("X", "Y", "Z", "Rx", "Ry", "Rz")
JOINT_AXES = ("J1", "J2", "J3", "J4", "J5", "J6")


def quaternion_to_euler_deg(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_deg_to_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_mat @ ry_mat @ rx_mat


def rotation_matrix_to_euler_deg(rot: np.ndarray) -> tuple[float, float, float]:
    sy = -float(rot[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rot[0, 1]), float(rot[1, 1]))

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(values_deg: list[float]) -> float:
    if not values_deg:
        raise ValueError("values_deg must not be empty")
    sin_sum = sum(math.sin(math.radians(v)) for v in values_deg)
    cos_sum = sum(math.cos(math.radians(v)) for v in values_deg)
    return normalize_angle_deg(math.degrees(math.atan2(sin_sum, cos_sum)))


def circular_distance_deg(a_deg: float, b_deg: float) -> float:
    return abs(normalize_angle_deg(a_deg - b_deg))


def make_transform(rotation: np.ndarray, translation_xyz: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array(translation_xyz, dtype=np.float64)
    return transform


def tcp_pose_to_transform(pose: TcpPose) -> np.ndarray:
    rotation = euler_deg_to_rotation_matrix(pose.rx, pose.ry, pose.rz)
    return make_transform(rotation, (pose.x, pose.y, pose.z))


def pose_stamped_to_transform(msg: PoseStamped) -> np.ndarray:
    quat = msg.pose.orientation
    rotation = quaternion_to_rotation_matrix(quat.x, quat.y, quat.z, quat.w)
    pos = msg.pose.position
    return make_transform(rotation, (pos.x, pos.y, pos.z))


def transform_to_tcp_pose(transform: np.ndarray) -> TcpPose:
    rotation = transform[:3, :3]
    x, y, z = transform[:3, 3]
    rx_deg, ry_deg, rz_deg = rotation_matrix_to_euler_deg(rotation)
    return TcpPose(x=float(x), y=float(y), z=float(z), rx=rx_deg, ry=ry_deg, rz=rz_deg)


def flat_matrix_to_transform(values: list[float]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError(f"Expected 16 values for transform matrix, got {len(values)}")
    return np.array(values, dtype=np.float64).reshape(4, 4)


class Nova5DriverNode(Node):
    def __init__(self) -> None:
        super().__init__("nova5_driver_node")

        self.declare_parameter("robot_ip", "192.168.111.102")
        self.declare_parameter("dashboard_port", 29999)
        self.declare_parameter("feedback_port", 30004)
        self.declare_parameter("go_to_start", False)
        self.declare_parameter("auto_enable", False)
        self.declare_parameter("startup_joint", [178.0, -7.60, 90.0, 2.40, -90.0, -1.50]) #机械臂初始位姿
        self.declare_parameter("startup_speed", 35)
        self.declare_parameter("motion_topic", "/target_pose_cam_fine")
        self.declare_parameter("coarse_motion_topic", "/target_pose_cam_coarse")
        self.declare_parameter("coarse_target_lock_topic", "/coarse_target_obj_for_d405")
        self.declare_parameter("tcp_pose_topic", "/nova5/current_tcp_pose")
        self.declare_parameter("linear_speed", 10)
        self.declare_parameter("linear_acc", 20)
        self.declare_parameter("joint_speed", 20)
        self.declare_parameter("joint_acc", 20)
        self.declare_parameter("motion_user_index", 0)
        self.declare_parameter("flange_tool_index", 0)
        # self.declare_parameter("command_tool_index", 0) #0就是法兰
        self.declare_parameter("command_tool_index", 1) #1就是tcp工具
        self.declare_parameter("default_post_grasp_pose_tool_index", 0)
        self.declare_parameter("fine_tool_to_flange_z_m", 0.2285) # 末端D405抓取时，夹爪尖端到法兰中心Z差值
        # self.declare_parameter("fine_tool_to_flange_z_m", 0.10) # 末端D405抓取时，夹爪尖端到法兰中心Z差值
        self.declare_parameter("camera_frame_id", "camera_d405_link")
        self.declare_parameter("coarse_camera_frame_id", "camera_d435_link")
        self.declare_parameter("coarse_hover_z_m", 0.05)  # 粗定位时，夹爪尖端停在目标上方27cm
        self.declare_parameter("coarse_target_x_offset_m", 0.06)  # 粗定位最终base执行值在x上减10cm
        self.declare_parameter("coarse_grasp_yaw_bias_deg", -90.0)
        self.declare_parameter("coarse_grasp_pitch_bias_deg", -0.0)
        self.declare_parameter("coarse_grasp_x_flip_deg", 180.0)
        self.declare_parameter("trigger_d405_topic", "/trigger_d405_vision")
        self.declare_parameter("gripper_width_topic", "/gripper_target_width")
        self.declare_parameter("singulation_target_x_extent_topic", "/target_x_extent_cam")  # 视觉发布的目标在相机/机械臂 X 方向上的实际尺寸
        self.declare_parameter("singulation_start_pose_topic", "/target_singulation_start_cam")  # 视觉发布的目标 +X 外沿起点（仍在相机坐标系）
        self.declare_parameter("singulation_needed_topic", "/target_singulation_needed")  # 视觉发布的“当前目标是否真的需要扒拉”
        self.declare_parameter("singulation_hint_topic", "/target_singulation_hint")  # 保留旧接口；当前固定 X 方向扒拉逻辑不再依赖它
        # self.declare_parameter("dh_enable", False)
        self.declare_parameter("dh_enable", True) #dh夹爪上使能
        self.declare_parameter("dh_max_opening_m", 0.095)
        self.declare_parameter("dh_force", 30)
        self.declare_parameter("dh_grasp_force", 30)
        self.declare_parameter("dh_slave_id", 1)
        self.declare_parameter("dh_tool_identify", 1)
        self.declare_parameter("dh_wait_timeout_s", 10.0)
        self.declare_parameter("d405_trigger_delay_s", 0.7)
        self.declare_parameter("fine_refine_delay_s", 0.1)
        self.declare_parameter("fine_refine_cycles", 1)
        self.declare_parameter("fine_target_timeout_s", 2.0)
        self.declare_parameter("fine_position_stability_threshold_m", 0.032)
        self.declare_parameter("fine_angle_stability_threshold_deg", 10.0)
        self.declare_parameter("fine_max_retries", 3)
        self.declare_parameter("pre_grasp_singulation_enabled", True)  # 是否启用抓取前扒拉
        self.declare_parameter("singulation_auto_decision_enabled", True)  # 是否启用“只有净空不足时才扒拉”的自适应决策
        self.declare_parameter("singulation_hover_above_target_m", 0.008)  # 扒拉起点上方悬停高度；越小，上下动作越少
        self.declare_parameter("singulation_contact_z_offset_m", 0.006)  # 扒拉接触高度相对目标 z 的偏移；正值略高于目标，负值略低于目标
        self.declare_parameter("singulation_retreat_height_m", 0.128)  # 扒拉完成后抬起高度
        self.declare_parameter("singulation_entry_lift_m", 0.010)  # 从当前位姿先向上抬多少，作为进入扒拉起点前的安全过渡高度
        self.declare_parameter("singulation_entry_clearance_m", 0.015)  # 到达扒拉起点上方后，距离真正悬停点再额外保留的安全高度
        self.declare_parameter("singulation_nominal_target_x_extent_m", 0.050)  # 视觉 X 尺寸暂时不可用时的默认目标 X 尺寸
        self.declare_parameter("singulation_start_x_margin_m", 0.0200)  # 从目标 +X 边缘再向 +X 多走一点，作为扒拉起点
        self.declare_parameter("singulation_end_x_margin_m", -0.020)  # 扒拉到目标 -X 边缘后，再向 -X 多走一点
        self.declare_parameter("singulation_reacquire_delay_s", 0.20)  # 扒拉抬起后等待视觉重新稳定的延时
        self.declare_parameter("singulation_reacquire_samples", 2)  # 扒拉后重新平均的视觉帧数
        self.declare_parameter("singulation_joint_speed", 12)  # 扒拉阶段关节运动速度
        self.declare_parameter("singulation_joint_acc", 12)  # 扒拉阶段关节运动加速度
        self.declare_parameter("singulation_linear_speed", 8)  # 扒拉阶段直线运动速度
        self.declare_parameter("singulation_linear_acc", 10)  # 扒拉阶段直线运动加速度
        self.declare_parameter("singulation_gripper_force", 20)  # 扒拉时夹爪闭合力，DH 范围 [20, 100]
        self.declare_parameter("singulation_fixed_start_axis", "plus_x")  # 固定从目标 +X 边缘外开始
        self.declare_parameter("singulation_fixed_sweep_axis", "minus_x")  # 固定朝目标 -X 方向扒拉
        self.declare_parameter("post_grasp_rotate_step_deg", -90.0)  #中转点绕z轴逆时针旋转90度
        self.declare_parameter("post_grasp_rotate_count", 3) #旋转3次
        self.declare_parameter("post_grasp_rotate_interval_s", 0.5)
        self.declare_parameter(
            "default_transfer_pose",
            [0.415, 0.121, 0.604, -176.222, -0.160, -90.432], #中转点检查位置
        )
        self.declare_parameter(
            "default_place_pose",
            [-0.153, 0.432, 0.446, 177.157, -1.201, -2.342], #放置位置
        )
        self.declare_parameter(
            "handeye_flange_to_cam", #d405手在眼上标定矩阵 (基于相机光学坐标系)
            [
                0.99999289, 0.00303007, -0.00224455, -0.01007269571,
                -0.00207268, 0.93892954, 0.34410322, -0.09923380417,
                0.00315013, -0.34409612, 0.93892914, 0.04701274037,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        self.declare_parameter(
            "handeye_base_to_d435",
            [
                0.9938108, 0.10750777, -0.02796736, -0.02212362452,
                0.10480691, -0.99087883, -0.08470334, 0.63295122513,
                -0.03681853, 0.08124792, -0.99601364, 1.3125790433,
                0.0, 0.0, 0.0, 1.0,
            ],
        )

        self._controller = DobotNova5Controller(
            robot_ip=self.get_parameter("robot_ip").value,
            dashboard_port=int(self.get_parameter("dashboard_port").value),
            feedback_port=int(self.get_parameter("feedback_port").value),
            startup_joint=[float(v) for v in self.get_parameter("startup_joint").value],
            startup_speed=int(self.get_parameter("startup_speed").value),
        )
        self._controller.connect(
            go_to_start=bool(self.get_parameter("go_to_start").value),
            auto_enable=bool(self.get_parameter("auto_enable").value),
        )
        self._camera_frame_id = str(self.get_parameter("camera_frame_id").value)
        self._coarse_camera_frame_id = str(self.get_parameter("coarse_camera_frame_id").value)
        self._handeye_flange_to_cam = flat_matrix_to_transform(
            [float(v) for v in self.get_parameter("handeye_flange_to_cam").value]
        )
        self._handeye_base_to_d435 = flat_matrix_to_transform(
            [float(v) for v in self.get_parameter("handeye_base_to_d435").value]
        )

        self._latest_target_pose: Optional[TcpPose] = None
        self._latest_target_source: str = "none"
        self._latest_raw_target_pose: Optional[TcpPose] = None
        self._latest_raw_frame_id: str = "none"
        self._target_msg_count = 0
        self._latest_coarse_target_pose: Optional[TcpPose] = None
        self._latest_coarse_target_source: str = "none"
        self._latest_coarse_object_base_pose: Optional[TcpPose] = None
        self._latest_raw_coarse_target_pose: Optional[TcpPose] = None
        self._latest_raw_coarse_frame_id: str = "none"
        self._coarse_msg_count = 0
        self._coarse_last_error = "none"
        self._drag_enabled = False
        self._dh_dashboard: Optional[DobotApiDashboard] = None
        self._dh_dashboard_is_shared = False
        self._dh_gripper: Optional[DHGripper] = None
        self._latest_gripper_target_width_m: Optional[float] = None
        self._latest_target_x_extent_m: Optional[float] = None
        self._latest_singulation_start_pose: Optional[TcpPose] = None
        self._latest_singulation_needed: bool = True
        self._latest_singulation_direction_base: Optional[np.ndarray] = None
        self._latest_singulation_hint_frame: str = "none"
        self._gripper_width_tracking_enabled = True
        self._coord_type = "user"
        self._motion_lock = threading.Lock()
        self._latest_pose_lock = threading.Lock()

        self._tcp_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("tcp_pose_topic").value,
            10,
        )
        self._pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter("motion_topic").value,
            self._target_pose_callback,
            10,
        )
        self._coarse_pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter("coarse_motion_topic").value,
            self._coarse_target_pose_callback,
            10,
        )
        self._trigger_d405_pub = self.create_publisher(
            Bool,
            self.get_parameter("trigger_d405_topic").value,
            10,
        )
        self._coarse_target_lock_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter("coarse_target_lock_topic").value,
            10,
        )
        self._gripper_width_sub = self.create_subscription(
            Float32,
            self.get_parameter("gripper_width_topic").value,
            self._gripper_target_width_callback,
            10,
        )
        self._target_x_extent_sub = self.create_subscription(
            Float32,
            self.get_parameter("singulation_target_x_extent_topic").value,
            self._target_x_extent_callback,
            10,
        )
        self._singulation_start_pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter("singulation_start_pose_topic").value,
            self._singulation_start_pose_callback,
            10,
        )
        self._singulation_needed_sub = self.create_subscription(
            Bool,
            self.get_parameter("singulation_needed_topic").value,
            self._singulation_needed_callback,
            10,
        )
        self._singulation_hint_sub = self.create_subscription(
            Vector3Stamped,
            self.get_parameter("singulation_hint_topic").value,
            self._singulation_hint_callback,
            10,
        )
        self._timer = self.create_timer(0.1, self._publish_tcp_pose)
        self._init_dh_gripper_if_enabled()
        self._apply_motion_profiles()

        self.get_logger().info(
            "Nova5 driver connected and ready. "
            f"motion_user_index={self._motion_user_index()} "
            f"flange_tool_index={self._flange_tool_index()} "
            f"command_tool_index={self._command_tool_index()}"
        )

    def destroy_node(self):
        try:
            self._disconnect_dh_gripper()
            self._controller.disconnect()
        finally:
            super().destroy_node()

    def latest_target_pose(self) -> Optional[TcpPose]:
        with self._latest_pose_lock:
            if self._latest_target_pose is None:
                return None
            pose = self._latest_target_pose
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def set_latest_target_pose(self, pose: TcpPose, source: str) -> None:
        with self._latest_pose_lock:
            self._latest_target_pose = TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)
            self._latest_target_source = source

    def latest_raw_target_pose(self) -> Optional[TcpPose]:
        with self._latest_pose_lock:
            if self._latest_raw_target_pose is None:
                return None
            pose = self._latest_raw_target_pose
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def latest_raw_frame_id(self) -> str:
        with self._latest_pose_lock:
            return self._latest_raw_frame_id

    def target_msg_count(self) -> int:
        with self._latest_pose_lock:
            return self._target_msg_count

    def latest_target_source(self) -> str:
        with self._latest_pose_lock:
            return self._latest_target_source

    def latest_coarse_target_pose(self) -> Optional[TcpPose]:
        with self._latest_pose_lock:
            if self._latest_coarse_target_pose is None:
                return None
            pose = self._latest_coarse_target_pose
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def latest_coarse_target_source(self) -> str:
        with self._latest_pose_lock:
            return self._latest_coarse_target_source

    def latest_coarse_object_base_pose(self) -> Optional[TcpPose]:
        with self._latest_pose_lock:
            if self._latest_coarse_object_base_pose is None:
                return None
            pose = self._latest_coarse_object_base_pose
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def latest_raw_coarse_target_pose(self) -> Optional[TcpPose]:
        with self._latest_pose_lock:
            if self._latest_raw_coarse_target_pose is None:
                return None
            pose = self._latest_raw_coarse_target_pose
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def latest_raw_coarse_frame_id(self) -> str:
        with self._latest_pose_lock:
            return self._latest_raw_coarse_frame_id

    def coarse_msg_count(self) -> int:
        with self._latest_pose_lock:
            return self._coarse_msg_count

    def coarse_last_error(self) -> str:
        with self._latest_pose_lock:
            return self._coarse_last_error

    def _store_raw_target_pose(self, pose: TcpPose, frame_id: str) -> None:
        with self._latest_pose_lock:
            self._latest_raw_target_pose = TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)
            self._latest_raw_frame_id = frame_id
            self._target_msg_count += 1

    def _store_raw_coarse_target_pose(self, pose: TcpPose, frame_id: str) -> None:
        with self._latest_pose_lock:
            self._latest_raw_coarse_target_pose = TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)
            self._latest_raw_coarse_frame_id = frame_id
            self._coarse_msg_count += 1

    def _set_coarse_last_error(self, text: str) -> None:
        with self._latest_pose_lock:
            self._coarse_last_error = text

    def set_latest_coarse_target_pose(self, pose: TcpPose, source: str) -> None:
        with self._latest_pose_lock:
            self._latest_coarse_target_pose = TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)
            self._latest_coarse_target_source = source

    def set_latest_coarse_object_base_pose(self, pose: TcpPose) -> None:
        with self._latest_pose_lock:
            self._latest_coarse_object_base_pose = TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def _publish_locked_coarse_target_for_d405(self, base_target_pose: TcpPose) -> None:
        current_flange = self.current_flange_pose()
        t_base_to_flange = tcp_pose_to_transform(current_flange)
        t_base_to_cam = t_base_to_flange @ self._handeye_flange_to_cam
        t_cam_to_base = np.linalg.inv(t_base_to_cam)

        quat = SciPyRot.from_euler(
            'xyz',
            [base_target_pose.rx, base_target_pose.ry, base_target_pose.rz],
            degrees=True,
        ).as_quat()
        base_pose_msg = PoseStamped()
        base_pose_msg.header.frame_id = "base"
        base_pose_msg.pose.position.x = base_target_pose.x
        base_pose_msg.pose.position.y = base_target_pose.y
        base_pose_msg.pose.position.z = base_target_pose.z
        base_pose_msg.pose.orientation.x = float(quat[0])
        base_pose_msg.pose.orientation.y = float(quat[1])
        base_pose_msg.pose.orientation.z = float(quat[2])
        base_pose_msg.pose.orientation.w = float(quat[3])

        t_base_to_target = pose_stamped_to_transform(base_pose_msg)
        t_cam_to_target = t_cam_to_base @ t_base_to_target
        cam_target_msg = PoseStamped()
        cam_target_msg.header.stamp = self.get_clock().now().to_msg()
        cam_target_msg.header.frame_id = self._camera_frame_id
        cam_target_msg.pose.position.x = float(t_cam_to_target[0, 3])
        cam_target_msg.pose.position.y = float(t_cam_to_target[1, 3])
        cam_target_msg.pose.position.z = float(t_cam_to_target[2, 3])
        cam_quat = SciPyRot.from_matrix(t_cam_to_target[:3, :3]).as_quat()
        cam_target_msg.pose.orientation.x = float(cam_quat[0])
        cam_target_msg.pose.orientation.y = float(cam_quat[1])
        cam_target_msg.pose.orientation.z = float(cam_quat[2])
        cam_target_msg.pose.orientation.w = float(cam_quat[3])
        self._coarse_target_lock_pub.publish(cam_target_msg)

    def _init_dh_gripper_if_enabled(self) -> None: #初始化 DH 夹爪连接
        if not bool(self.get_parameter("dh_enable").value):
            return
        config = DobotDHConfig(
            robot_ip=str(self.get_parameter("robot_ip").value),
            dashboard_port=int(self.get_parameter("dashboard_port").value),
            tool_identify=int(self.get_parameter("dh_tool_identify").value),
            slave_id=int(self.get_parameter("dh_slave_id").value),
            force=int(self.get_parameter("dh_force").value),
            enable_robot=False,
        )
        if self._controller.dashboard is None:
            raise RuntimeError("Robot dashboard is not connected, cannot initialize DH gripper")

        self._dh_dashboard = self._controller.dashboard
        self._dh_dashboard_is_shared = True
        self._configure_dh_tool_rs485(config)
        self._cleanup_stale_dh_modbus()
        self._dh_gripper = DHGripper(self._dh_dashboard, config)
        self._dh_gripper.initialize(
            timeout_s=float(self.get_parameter("dh_wait_timeout_s").value),
            init_open=True,
        )
        self.get_logger().info("DH gripper connected and ready.")

    def _configure_dh_tool_rs485(self, config: DobotDHConfig) -> None:
        if self._dh_dashboard is None:
            raise RuntimeError("DH dashboard is not connected")
        set_tool_mode_response = self._dh_dashboard.SetToolMode(1, 1, config.tool_identify)
        if "Control Mode Is Not Tcp" in str(set_tool_mode_response):
            self.get_logger().warn(
                "SetToolMode skipped because controller reported non-TCP control mode; "
                "continuing with existing tool interface mode."
            )
            return
        raise_if_error(set_tool_mode_response, "SetToolMode")

        set_tool_485_response = self._dh_dashboard.SetTool485(
            config.baudrate,
            config.parity,
            config.stop_bit,
            config.tool_identify,
        )
        if "Control Mode Is Not Tcp" in str(set_tool_485_response):
            self.get_logger().warn(
                "SetTool485 skipped because controller reported non-TCP control mode; "
                "continuing with existing tool RS485 settings."
            )
            return
        raise_if_error(set_tool_485_response, "SetTool485")

    def _cleanup_stale_dh_modbus(self) -> None:
        if self._dh_dashboard is None:
            raise RuntimeError("DH dashboard is not connected")
        for master_index in range(5):
            try:
                self._dh_dashboard.ModbusClose(master_index)
            except Exception:
                pass

    def _disconnect_dh_gripper(self) -> None:
        if self._dh_gripper is not None:
            try:
                self._dh_gripper.disconnect()
            finally:
                self._dh_gripper = None
        if self._dh_dashboard is not None and not self._dh_dashboard_is_shared:
            try:
                self._dh_dashboard.close()
            finally:
                self._dh_dashboard = None
        self._dh_dashboard = None
        self._dh_dashboard_is_shared = False

    def _gripper_width_to_normalized(self, width_m: float) -> float:
        max_opening_m = float(self.get_parameter("dh_max_opening_m").value)
        if max_opening_m <= 0.0:
            raise ValueError(f"dh_max_opening_m must be > 0, got {max_opening_m}")
        return max(0.0, min(1.0, width_m / max_opening_m))

    def _set_gripper_width_tracking_enabled(self, enabled: bool, reason: str) -> None:
        if self._gripper_width_tracking_enabled == enabled:
            return
        self._gripper_width_tracking_enabled = enabled
        state_text = "enabled" if enabled else "disabled"
        self.get_logger().info(f"Gripper width tracking {state_text}: {reason}")

    def _apply_gripper_target_width(self, width_m: float, wait: bool = True) -> None:
        if self._dh_gripper is None:
            raise RuntimeError("DH gripper is not enabled or connected")
        normalized = self._gripper_width_to_normalized(width_m)
        self._dh_gripper.set_position( #调用dh_api下发
            normalized,
            wait=wait,
            timeout_s=float(self.get_parameter("dh_wait_timeout_s").value),
        )
        self.get_logger().info(
            f"Applied gripper width target {width_m:.4f} m -> normalized={normalized:.3f} raw={int(round(normalized * 1000.0))}"
        )

    def _close_gripper_for_grasp(self, wait: bool = True) -> None:
        if self._dh_gripper is None:
            raise RuntimeError("DH gripper is not enabled or connected")
        grasp_force = int(self.get_parameter("dh_grasp_force").value)
        self._dh_gripper.set_force(grasp_force)
        self._dh_gripper.set_position(
            0.0,
            wait=wait,
            timeout_s=float(self.get_parameter("dh_wait_timeout_s").value),
        )
        self.get_logger().info(f"Closed DH gripper for grasp with force={grasp_force}.")

    def _open_gripper_fully(self, wait: bool = True) -> None:
        if self._dh_gripper is None:
            raise RuntimeError("DH gripper is not enabled or connected")
        self._dh_gripper.set_position(
            1.0,
            wait=wait,
            timeout_s=float(self.get_parameter("dh_wait_timeout_s").value),
        )
        self.get_logger().info("Opened DH gripper to full-open position.")

    def test_gripper_open_close(self) -> None:
        if self._dh_gripper is None:
            raise RuntimeError("DH gripper is not enabled or connected")
        self.get_logger().info("Starting DH gripper open/close test.")
        self._close_gripper_for_grasp(wait=True)
        time.sleep(0.5)
        self._open_gripper_fully(wait=True)
        self.get_logger().info("Completed DH gripper open/close test.")

    def _gripper_target_width_callback(self, msg: Float32) -> None:
        width_m = float(msg.data)
        with self._latest_pose_lock:
            self._latest_gripper_target_width_m = width_m
        if self._dh_gripper is None or not self._gripper_width_tracking_enabled:
            return
        try:
            self._apply_gripper_target_width(width_m, wait=False)
        except Exception as exc:
            self.get_logger().error(f"Failed to apply gripper width target {width_m:.4f} m: {exc}")

    def latest_gripper_target_width_m(self) -> Optional[float]:
        with self._latest_pose_lock:
            return self._latest_gripper_target_width_m

    def _target_x_extent_callback(self, msg: Float32) -> None:
        extent_m = float(msg.data)
        if extent_m <= 0.0:
            return
        with self._latest_pose_lock:
            self._latest_target_x_extent_m = extent_m

    def latest_target_x_extent_m(self) -> Optional[float]:
        with self._latest_pose_lock:
            return self._latest_target_x_extent_m

    def _singulation_start_pose_callback(self, msg: PoseStamped) -> None:
        frame_id = msg.header.frame_id.strip() or "base"
        try:
            if frame_id == "base":
                start_pose = TcpPose(
                    x=float(msg.pose.position.x),
                    y=float(msg.pose.position.y),
                    z=float(msg.pose.position.z),
                    rx=0.0,
                    ry=0.0,
                    rz=0.0,
                )
            elif frame_id == self._camera_frame_id:
                current_flange = self.current_flange_pose()
                t_base_to_flange = tcp_pose_to_transform(current_flange)
                t_cam_to_start = pose_stamped_to_transform(msg)
                t_base_to_start = t_base_to_flange @ self._handeye_flange_to_cam @ t_cam_to_start
                start_pose = transform_to_tcp_pose(t_base_to_start)
            else:
                return
        except Exception:
            return

        with self._latest_pose_lock:
            self._latest_singulation_start_pose = start_pose

    def latest_singulation_start_pose(self) -> Optional[TcpPose]:
        with self._latest_pose_lock:
            if self._latest_singulation_start_pose is None:
                return None
            pose = self._latest_singulation_start_pose
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

    def _singulation_needed_callback(self, msg: Bool) -> None:
        with self._latest_pose_lock:
            self._latest_singulation_needed = bool(msg.data)

    def latest_singulation_needed(self) -> bool:
        with self._latest_pose_lock:
            return self._latest_singulation_needed

    def _restore_gripper_width_from_latest_target(self, wait: bool = True) -> None:
        width_m = self.latest_gripper_target_width_m()
        if width_m is None:
            self.get_logger().warning(
                "Skipped gripper width restore after singulation because no width target is available."
            )
            return
        self._apply_gripper_target_width(width_m, wait=wait)
        self.get_logger().info(
            f"Restored gripper width from latest target before final grasp: width={width_m:.4f} m"
        )

    def _singulation_hint_callback(self, msg: Vector3Stamped) -> None:
        frame_id = msg.header.frame_id.strip() or "none"
        raw_vec = np.array([float(msg.vector.x), float(msg.vector.y), float(msg.vector.z)], dtype=np.float64)
        norm = float(np.linalg.norm(raw_vec))
        if norm < 1e-6:
            return
        raw_vec /= norm

        try:
            if frame_id == self._camera_frame_id:
                current_flange = self.current_flange_pose()
                t_base_to_flange = tcp_pose_to_transform(current_flange)
                t_base_to_cam = t_base_to_flange @ self._handeye_flange_to_cam
                rot_base_to_cam = t_base_to_cam[:3, :3]
                base_vec = rot_base_to_cam @ raw_vec
            elif frame_id == "base":
                base_vec = raw_vec
            else:
                self.get_logger().warning(
                    f"Ignored singulation hint in unsupported frame_id '{frame_id}'."
                )
                return
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert singulation hint from {frame_id}: {exc}")
            return

        base_vec[2] = 0.0
        xy_norm = float(np.linalg.norm(base_vec[:2]))
        if xy_norm < 1e-6:
            return
        base_vec[:2] /= xy_norm
        base_vec[2] = 0.0

        with self._latest_pose_lock:
            self._latest_singulation_direction_base = base_vec.copy()
            self._latest_singulation_hint_frame = frame_id

    def _latest_singulation_direction_xy(self) -> Optional[np.ndarray]:
        with self._latest_pose_lock:
            if self._latest_singulation_direction_base is None:
                return None
            return self._latest_singulation_direction_base.copy()

    def _prepare_gripper_for_singulation(self, wait: bool = True) -> None:
        if self._dh_gripper is None:
            raise RuntimeError("DH gripper is not enabled or connected")
        singulation_force = int(self.get_parameter("singulation_gripper_force").value)
        if singulation_force < 20 or singulation_force > 100:
            raise ValueError(f"singulation_gripper_force must be in [20, 100], got {singulation_force}")
        self._dh_gripper.set_force(singulation_force)
        self._dh_gripper.set_position(
            0.0,
            wait=wait,
            timeout_s=float(self.get_parameter("dh_wait_timeout_s").value),
        )
        self.get_logger().info(f"Closed DH gripper for singulation with force={singulation_force}.")

    def _estimate_target_x_extent_m(self) -> float:
        tracked_extent_m = self.latest_target_x_extent_m()
        if tracked_extent_m is not None:
            return max(0.015, tracked_extent_m)
        nominal_extent_m = float(self.get_parameter("singulation_nominal_target_x_extent_m").value)
        return max(0.015, nominal_extent_m)

    def _pose_with_xyz(self, pose: TcpPose, x: float, y: float, z: float) -> TcpPose:
        return TcpPose(x=x, y=y, z=z, rx=pose.rx, ry=pose.ry, rz=pose.rz)

    def _collect_fresh_fine_target_from_tracking(
        self,
        sample_count: int,
        sample_delay_s: float,
        timeout_s: float,
    ) -> TcpPose:
        samples: list[TcpPose] = []
        previous_count = self.target_msg_count()
        for sample_index in range(sample_count):
            if sample_index > 0 and sample_delay_s > 0.0:
                time.sleep(sample_delay_s)
            sample = self._wait_for_next_fine_target(previous_count, timeout_s)
            samples.append(sample)
            previous_count = self.target_msg_count()
        self._validate_fine_target_stability(samples)
        averaged_target = self._average_tcp_poses(samples)
        self.set_latest_target_pose(
            averaged_target,
            f"{self.get_parameter('motion_topic').value} [post-singulation averaged {len(samples)} samples]",
        )
        self.get_logger().info(
            f"Reacquired fine target from {len(samples)} tracking samples: "
            f"x={averaged_target.x:.3f} y={averaged_target.y:.3f} z={averaged_target.z:.3f} "
            f"rx={averaged_target.rx:.2f} ry={averaged_target.ry:.2f} rz={averaged_target.rz:.2f}"
        )
        return averaged_target

    def _execute_pre_grasp_singulation(self, target: TcpPose) -> TcpPose:
        if self._dh_gripper is None:
            raise RuntimeError("Pre-grasp singulation requires DH gripper to be enabled")

        hover_offset_m = float(self.get_parameter("singulation_hover_above_target_m").value)
        contact_z_offset_m = float(self.get_parameter("singulation_contact_z_offset_m").value)
        retreat_height_m = float(self.get_parameter("singulation_retreat_height_m").value)
        reacquire_delay_s = float(self.get_parameter("singulation_reacquire_delay_s").value)
        reacquire_samples = max(1, int(self.get_parameter("singulation_reacquire_samples").value))
        fine_target_timeout_s = float(self.get_parameter("fine_target_timeout_s").value)
        fine_refine_delay_s = float(self.get_parameter("fine_refine_delay_s").value)

        target_x_extent_m = self._estimate_target_x_extent_m()
        half_x_extent_m = 0.5 * target_x_extent_m
        start_x_margin_m = float(self.get_parameter("singulation_start_x_margin_m").value)
        end_x_margin_m = float(self.get_parameter("singulation_end_x_margin_m").value)
        start_pose_source = "fallback_center_plus_x"
        visual_start_pose = self.latest_singulation_start_pose()

        # 固定规则：
        # 1. 优先使用视觉给出的目标 +X 外沿点
        # 2. 再从这个外沿点继续向 +X 多走一点，作为真实扒拉起点
        # 3. 如果视觉起点不可用，才退回“目标中心 + 半个 X 尺寸 + 边距”的估算方式
        if visual_start_pose is not None:
            start_x = visual_start_pose.x + start_x_margin_m
            start_y = visual_start_pose.y
            start_pose_source = "vision_positive_x_edge"
        else:
            start_x = target.x + half_x_extent_m + start_x_margin_m
            start_y = target.y
        end_x = target.x - half_x_extent_m - end_x_margin_m
        end_y = start_y
        hover_z = target.z + hover_offset_m
        contact_z = target.z + contact_z_offset_m
        retreat_z = target.z + retreat_height_m
        entry_lift_m = float(self.get_parameter("singulation_entry_lift_m").value)
        entry_clearance_m = float(self.get_parameter("singulation_entry_clearance_m").value)
        current_pose = self.read_current_pose()
        entry_z = max(current_pose.z + entry_lift_m, hover_z + entry_clearance_m)

        entry_lift_pose = self._pose_with_xyz(current_pose, current_pose.x, current_pose.y, entry_z)
        entry_over_start_pose = self._pose_with_xyz(target, start_x, start_y, entry_z)
        hover_start_pose = self._pose_with_xyz(target, start_x, start_y, hover_z)
        contact_start_pose = self._pose_with_xyz(target, start_x, start_y, contact_z)
        sweep_end_pose = self._pose_with_xyz(target, end_x, end_y, contact_z)
        retreat_pose = self._pose_with_xyz(target, end_x, end_y, retreat_z)
        visual_start_text = (
            f"visual_start_x={visual_start_pose.x:.3f} "
            if visual_start_pose is not None
            else "visual_start_x=none "
        )

        self.get_logger().info(
            "Pre-grasp singulation plan: "
            f"target_x_extent={target_x_extent_m:.3f}m "
            f"start_x_margin={start_x_margin_m:.3f}m "
            f"end_x_margin={end_x_margin_m:.3f}m "
            f"entry_lift={entry_lift_m:.3f}m "
            f"entry_clearance={entry_clearance_m:.3f}m "
            f"hover_above={hover_offset_m:.3f}m "
            f"contact_z_offset={contact_z_offset_m:.3f}m "
            f"start_source={start_pose_source} "
            f"{visual_start_text}"
            f"entry_pose=({entry_over_start_pose.x:.3f}, {entry_over_start_pose.y:.3f}, {entry_over_start_pose.z:.3f}) "
            f"start_pose=({start_x:.3f}, {start_y:.3f}, {hover_z:.3f}) "
            f"end_pose=({end_x:.3f}, {end_y:.3f}, {contact_z:.3f})"
        )

        self._set_gripper_width_tracking_enabled(
            False,
            "locking gripper width during pre-grasp singulation sweep",
        )
        try:
            self._prepare_gripper_for_singulation(wait=True)
            self._move_joint_pose_with_profile_and_log(
                entry_lift_pose,
                "Reached singulation entry-lift pose",
                speed=int(self.get_parameter("singulation_joint_speed").value),
                accel=int(self.get_parameter("singulation_joint_acc").value),
            )
            self._move_joint_pose_with_profile_and_log(
                entry_over_start_pose,
                "Reached singulation entry-over-start pose",
                speed=int(self.get_parameter("singulation_joint_speed").value),
                accel=int(self.get_parameter("singulation_joint_acc").value),
            )
            self._move_joint_pose_with_profile_and_log(
                hover_start_pose,
                "Reached singulation hover-start pose",
                speed=int(self.get_parameter("singulation_joint_speed").value),
                accel=int(self.get_parameter("singulation_joint_acc").value),
            )
            self._move_linear_with_profile_and_log(
                contact_start_pose,
                "Reached singulation contact-start pose",
                speed=int(self.get_parameter("singulation_linear_speed").value),
                accel=int(self.get_parameter("singulation_linear_acc").value),
            )
            self._move_linear_with_profile_and_log(
                sweep_end_pose,
                "Executed singulation sweep",
                speed=int(self.get_parameter("singulation_linear_speed").value),
                accel=int(self.get_parameter("singulation_linear_acc").value),
            )
            self._move_linear_with_profile_and_log(
                retreat_pose,
                "Retreated after singulation sweep",
                speed=int(self.get_parameter("singulation_linear_speed").value),
                accel=int(self.get_parameter("singulation_linear_acc").value),
            )

            if reacquire_delay_s > 0.0:
                time.sleep(reacquire_delay_s)
            refreshed_target = self._collect_fresh_fine_target_from_tracking(
                sample_count=reacquire_samples,
                sample_delay_s=fine_refine_delay_s,
                timeout_s=fine_target_timeout_s,
            )
        finally:
            self._set_gripper_width_tracking_enabled(
                True,
                "restoring gripper width tracking after singulation sweep",
            )
        self._restore_gripper_width_from_latest_target(wait=True)
        return refreshed_target

    def execute_latest_target(self) -> None:
        target = self.latest_target_pose()
        if target is None:
            raise RuntimeError("No latest target pose has been received yet")
        self._move_joint_pose_with_log(target, "Executed latest target")

    def execute_latest_target_staged(self) -> None:
        target = self.latest_target_pose()
        if target is None:
            raise RuntimeError("No latest target pose has been received yet")

        current_tcp = self.read_current_pose()
        planar_target = TcpPose(
            x=target.x,
            y=target.y,
            z=current_tcp.z,
            rx=target.rx,
            ry=target.ry,
            rz=target.rz,
        )

        self._move_joint_pose_with_log(planar_target, "Executed staged fine planar move")

        with self._motion_lock:
            self._controller.move_linear_tcp(
                target,
                speed=int(self.get_parameter("linear_speed").value),
                accel=int(self.get_parameter("linear_acc").value),
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )
        self.get_logger().info(
            f"Executed staged fine descend x={target.x:.3f} y={target.y:.3f} z={target.z:.3f} "
            f"rx={target.rx:.2f} ry={target.ry:.2f} rz={target.rz:.2f}"
        )

    def _wait_for_next_fine_target(self, previous_count: int, timeout_s: float) -> TcpPose:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.target_msg_count() > previous_count:
                target = self.latest_target_pose()
                if target is not None:
                    return target
            time.sleep(0.02)
        raise TimeoutError(f"Timed out waiting for next fine target update after count={previous_count}")

    def _average_tcp_poses(self, poses: list[TcpPose]) -> TcpPose:
        count = float(len(poses))
        return TcpPose(
            x=sum(p.x for p in poses) / count,
            y=sum(p.y for p in poses) / count,
            z=sum(p.z for p in poses) / count,
            rx=circular_mean_deg([p.rx for p in poses]),
            ry=circular_mean_deg([p.ry for p in poses]),
            rz=circular_mean_deg([p.rz for p in poses]),
        )

    def _validate_fine_target_stability(self, poses: list[TcpPose]) -> None:
        if len(poses) <= 1:
            return

        pos_threshold_m = float(self.get_parameter("fine_position_stability_threshold_m").value)
        angle_threshold_deg = float(self.get_parameter("fine_angle_stability_threshold_deg").value)

        xs = [p.x for p in poses]
        ys = [p.y for p in poses]
        zs = [p.z for p in poses]
        rxs = [p.rx for p in poses]
        rys = [p.ry for p in poses]
        rzs = [p.rz for p in poses]

        pos_span_m = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
        )
        rx_center = circular_mean_deg(rxs)
        ry_center = circular_mean_deg(rys)
        rz_center = circular_mean_deg(rzs)
        angle_span_deg = max(
            max(circular_distance_deg(v, rx_center) for v in rxs) * 2.0,
            max(circular_distance_deg(v, ry_center) for v in rys) * 2.0,
            max(circular_distance_deg(v, rz_center) for v in rzs) * 2.0,
        )

        if pos_span_m > pos_threshold_m or angle_span_deg > angle_threshold_deg:
            raise RuntimeError(
                "Fine target unstable: "
                f"pos_span={pos_span_m:.4f}m threshold={pos_threshold_m:.4f}m, "
                f"angle_span={angle_span_deg:.2f}deg threshold={angle_threshold_deg:.2f}deg"
            )

    def _apply_motion_profiles(self) -> None:
        linear_supported = self._controller.set_linear_profile(
            speed=int(self.get_parameter("linear_speed").value),
            accel=int(self.get_parameter("linear_acc").value),
        )
        joint_supported = self._controller.set_joint_profile(
            speed=int(self.get_parameter("joint_speed").value),
            accel=int(self.get_parameter("joint_acc").value),
        )
        if not linear_supported:
            self.get_logger().warn(
                "Controller does not support global linear profile commands VelL/AccL; "
                "continuing with per-motion MovL a/v parameters."
            )
        if not joint_supported:
            self.get_logger().warn(
                "Controller does not support global joint profile commands VelJ/AccJ; "
                "continuing with per-motion MovJ a/v parameters."
            )

    def _move_linear_with_log(self, pose: TcpPose, label: str) -> None:
        with self._motion_lock:
            self._controller.move_linear_tcp(
                pose,
                speed=int(self.get_parameter("linear_speed").value),
                accel=int(self.get_parameter("linear_acc").value),
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )
        self.get_logger().info(
            f"{label}: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} "
            f"rx={pose.rx:.2f} ry={pose.ry:.2f} rz={pose.rz:.2f}"
        )

    def _move_joint_pose_with_log(self, pose: TcpPose, label: str) -> None:
        with self._motion_lock:
            self._controller.move_joint_tcp(
                pose,
                speed=int(self.get_parameter("joint_speed").value),
                accel=int(self.get_parameter("joint_acc").value),
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )
        self.get_logger().info(
            f"{label}: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} "
            f"rx={pose.rx:.2f} ry={pose.ry:.2f} rz={pose.rz:.2f}"
        )

    def _move_linear_with_profile_and_log(self, pose: TcpPose, label: str, speed: int, accel: int) -> None:
        with self._motion_lock:
            self._controller.move_linear_tcp(
                pose,
                speed=speed,
                accel=accel,
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )
        self.get_logger().info(
            f"{label}: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} "
            f"rx={pose.rx:.2f} ry={pose.ry:.2f} rz={pose.rz:.2f}"
        )

    def _move_joint_pose_with_profile_and_log(self, pose: TcpPose, label: str, speed: int, accel: int) -> None:
        with self._motion_lock:
            self._controller.move_joint_tcp(
                pose,
                speed=speed,
                accel=accel,
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )
        self.get_logger().info(
            f"{label}: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} "
            f"rx={pose.rx:.2f} ry={pose.ry:.2f} rz={pose.rz:.2f}"
        )

    def _rotate_tool_z_with_log(self, delta_rz_deg: float, label: str) -> None:
        with self._motion_lock:
            self._controller.rel_move_tool_joint(
                TcpPose(0.0, 0.0, 0.0, 0.0, 0.0, delta_rz_deg),
                speed=int(self.get_parameter("joint_speed").value),
                accel=int(self.get_parameter("joint_acc").value),
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )
        current_tcp = self.current_tcp_pose()
        self.get_logger().info(
            f"{label}: x={current_tcp.x:.3f} y={current_tcp.y:.3f} z={current_tcp.z:.3f} "
            f"rx={current_tcp.rx:.2f} ry={current_tcp.ry:.2f} rz={current_tcp.rz:.2f}"
        )

    def execute_post_grasp_place_sequence(self, transfer_pose: TcpPose, place_pose: TcpPose) -> None:
        rotate_step_deg = float(self.get_parameter("post_grasp_rotate_step_deg").value)
        rotate_count = max(0, int(self.get_parameter("post_grasp_rotate_count").value))
        rotate_interval_s = max(0.0, float(self.get_parameter("post_grasp_rotate_interval_s").value))

        self.get_logger().info("========== [阶段 4] 抓取后搬运流程 ==========")
        self.get_logger().info(
            f"Post-grasp transfer target (tool={self._command_tool_index()}): "
            f"x={transfer_pose.x:.3f} y={transfer_pose.y:.3f} z={transfer_pose.z:.3f} "
            f"rx={transfer_pose.rx:.2f} ry={transfer_pose.ry:.2f} rz={transfer_pose.rz:.2f}"
        )
        self.get_logger().info(
            f"Post-grasp place target (tool={self._command_tool_index()}): "
            f"x={place_pose.x:.3f} y={place_pose.y:.3f} z={place_pose.z:.3f} "
            f"rx={place_pose.rx:.2f} ry={place_pose.ry:.2f} rz={place_pose.rz:.2f}"
        )
        self._move_joint_pose_with_log(transfer_pose, "Moved to post-grasp transfer pose")

        for rotate_index in range(rotate_count):
            self._rotate_tool_z_with_log(
                rotate_step_deg,
                f"Post-grasp Z rotation {rotate_index + 1}/{rotate_count}",
            )
            if rotate_index < rotate_count - 1 and rotate_interval_s > 0.0:
                time.sleep(rotate_interval_s)

        self._move_joint_pose_with_log(place_pose, "Moved to post-grasp place pose")
        if self._dh_gripper is not None:
            self._open_gripper_fully(wait=True)
        self.get_logger().info("Post-grasp place sequence completed.")

    def execute_coarse_then_fine_sequence(
        self,
        transfer_pose: Optional[TcpPose] = None,
        place_pose: Optional[TcpPose] = None,
    ) -> None: #粗定位--》精定位--》夹爪开合  执行流程
        coarse_target = self.latest_coarse_target_pose()
        if coarse_target is None:
            raise RuntimeError("No coarse target pose has been received yet")
        coarse_object_base_pose = self.latest_coarse_object_base_pose()
        if coarse_object_base_pose is None:
            raise RuntimeError("No coarse object base pose has been received yet")
        self._set_gripper_width_tracking_enabled(True, "starting coarse-to-fine vision sequence")
        self._move_joint_pose_with_log(coarse_target, "Executed coarse target")
        self._publish_locked_coarse_target_for_d405(coarse_object_base_pose)
        self.get_logger().info(
            f"Published coarse-locked target for D405 x={coarse_object_base_pose.x:.3f} "
            f"y={coarse_object_base_pose.y:.3f} z={coarse_object_base_pose.z:.3f}"
        )

        refine_cycles = max(0, int(self.get_parameter("fine_refine_cycles").value))
        fine_target_timeout_s = float(self.get_parameter("fine_target_timeout_s").value)
        fine_refine_delay_s = float(self.get_parameter("fine_refine_delay_s").value)
        max_retries = max(0, int(self.get_parameter("fine_max_retries").value))

        success = False
        self.get_logger().info("========== [阶段 2] 启动精定位视觉采样 ==========")

        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.get_logger().warning(
                    f"--> [重试机制] 正在进行第 {attempt} 次重新采样..."
                )

            fine_samples: list[TcpPose] = []
            trigger_msg = Bool()
            trigger_msg.data = True
            previous_count = self.target_msg_count()
            self._publish_locked_coarse_target_for_d405(coarse_object_base_pose)
            self._trigger_d405_pub.publish(trigger_msg)
            self.get_logger().info("Published D405 trigger true.")

            time.sleep(float(self.get_parameter("d405_trigger_delay_s").value))
            try:
                fine_samples.append(self._wait_for_next_fine_target(previous_count, fine_target_timeout_s))
            except TimeoutError:
                self.get_logger().error("--> 获取初始精定位数据超时！")
                if attempt < max_retries:
                    time.sleep(0.5)
                continue

            refine_success = True
            for cycle_index in range(refine_cycles):
                previous_count = self.target_msg_count()
                self.get_logger().info(
                    f"Sampling fine refine cycle {cycle_index + 1}/{refine_cycles} from current SAM2 track."
                )
                time.sleep(fine_refine_delay_s)
                try:
                    fine_samples.append(self._wait_for_next_fine_target(previous_count, fine_target_timeout_s))
                except TimeoutError:
                    self.get_logger().error(f"--> 第 {cycle_index + 1} 次 Refine 获取数据超时！")
                    refine_success = False
                    break

            if not refine_success:
                if attempt < max_retries:
                    time.sleep(0.5)
                continue

            try:
                self._validate_fine_target_stability(fine_samples)
                averaged_target = self._average_tcp_poses(fine_samples)
                self.set_latest_target_pose(
                    averaged_target,
                    f"{self.get_parameter('motion_topic').value} [fine averaged {len(fine_samples)} samples]",
                )
                self.get_logger().info(
                    f"--> ✅ 校验通过！共平均了 {len(fine_samples)} 帧有效位姿。"
                )
                self.get_logger().info(
                    f"Averaged fine target from {len(fine_samples)} samples: "
                    f"x={averaged_target.x:.3f} y={averaged_target.y:.3f} z={averaged_target.z:.3f} "
                    f"rx={averaged_target.rx:.2f} ry={averaged_target.ry:.2f} rz={averaged_target.rz:.2f}"
                )
                if bool(self.get_parameter("pre_grasp_singulation_enabled").value):
                    singulation_should_run = True
                    if bool(self.get_parameter("singulation_auto_decision_enabled").value):
                        singulation_should_run = self.latest_singulation_needed()
                    if singulation_should_run:
                        self.get_logger().info("========== [阶段 2.5] 精定位前拨料/推挤 ==========")
                        averaged_target = self._execute_pre_grasp_singulation(averaged_target)
                        self.get_logger().info(
                            f"Post-singulation target ready: "
                            f"x={averaged_target.x:.3f} y={averaged_target.y:.3f} z={averaged_target.z:.3f} "
                            f"rx={averaged_target.rx:.2f} ry={averaged_target.ry:.2f} rz={averaged_target.rz:.2f}"
                        )
                    else:
                        self.get_logger().info(
                            "========== [阶段 2.5] 跳过拨料 =========="
                        )
                        self.get_logger().info(
                            "Skipped pre-grasp singulation because point-cloud side clearance is sufficient."
                        )
                success = True
                break
            except RuntimeError as exc:
                self.get_logger().error(f"--> ❌ 误差超标，目标不稳定: {exc}")
                if attempt < max_retries:
                    self.get_logger().info("--> 丢弃脏数据，准备重新触发视觉...")
                    time.sleep(0.5)

        if success:
            self.get_logger().info("========== [阶段 3] 自动执行精定位下探 ==========")
            self._set_gripper_width_tracking_enabled(False, "locking gripper width before final fine approach and grasp")
            self.execute_latest_target_staged()
            if self._dh_gripper is not None:
                self._close_gripper_for_grasp(wait=True)
            if transfer_pose is not None and place_pose is not None:
                self.execute_post_grasp_place_sequence(transfer_pose, place_pose)
        else:
            self.get_logger().fatal(
                f"🛑 精定位连续 {max_retries + 1} 次采样失败 (误差超标或超时)，已终止下探保护系统！"
            )

    def move_to_startup(self) -> None:
        self._set_gripper_width_tracking_enabled(False, "moving robot back to startup pose")
        self._apply_motion_profiles()
        with self._motion_lock:
            self._controller.move_to_startup()
        if self._dh_gripper is not None:
            self._open_gripper_fully(wait=True)

    def manual_move_linear(self, pose: TcpPose) -> None:
        self.set_latest_target_pose(pose, "manual_target")
        with self._motion_lock:
            self._controller.move_linear_tcp(
                pose,
                speed=int(self.get_parameter("linear_speed").value),
                accel=int(self.get_parameter("linear_acc").value),
                user_index=self._motion_user_index(),
                tool_index=self._command_tool_index(),
            )

    def power_on(self) -> None:
        self._controller.power_on()

    def enable_robot(self) -> None:
        self._controller.enable_robot()

    def disable_robot(self) -> None:
        self._controller.disable_robot()

    def clear_error(self) -> None:
        self._controller.clear_error()

    def reset_robot(self) -> None:
        self._controller.reset_robot()

    def stop_motion(self) -> None:
        self._controller.stop_motion()

    def pause_motion(self) -> None:
        self._controller.pause_motion()

    def continue_motion(self) -> None:
        self._controller.continue_motion()

    def apply_speed_factor(self, speed: int) -> None:
        self._controller.set_speed_factor(speed)

    def apply_motion_profile(self) -> None:
        self._apply_motion_profiles()

    def set_user_index(self, index: int) -> None:
        self._controller.set_user_index(index)

    def set_tool_index(self, index: int) -> None:
        self._controller.set_tool_index(index)

    def set_coord_type(self, coord_type: str) -> None:
        if coord_type not in ("user", "tool"):
            raise ValueError(f"Unsupported coord_type: {coord_type}")
        self._coord_type = coord_type

    def toggle_drag(self) -> bool:
        if self._drag_enabled:
            self._controller.stop_drag()
            self._drag_enabled = False
        else:
            self._controller.start_drag()
            self._drag_enabled = True
        return self._drag_enabled

    def start_jog(self, axis_id: str, user: int = 0, tool: int = 0) -> None:
        coord_type = 1 if self._coord_type == "user" else 2
        self._controller.move_jog(axis_id=axis_id, coord_type=coord_type, user=user, tool=tool)

    def stop_jog(self) -> None:
        self._controller.move_jog("")

    def read_current_pose(self) -> TcpPose:
        return self._controller.read_pose(
            user_index=self._motion_user_index(),
            tool_index=self._command_tool_index(),
        )

    def current_tcp_pose(self) -> TcpPose:
        return self._controller.current_tcp_pose(
            user_index=self._motion_user_index(),
            tool_index=self._command_tool_index(),
        )

    def current_flange_pose(self) -> TcpPose:
        if self._command_uses_flange_frame():
            return self._controller.current_tcp_pose(
                user_index=self._motion_user_index(),
                tool_index=self._flange_tool_index(),
            )

        # 当执行工具坐标系不是法兰时，不再强依赖 GetPose(user=..., tool=0)。
        # 直接用当前工具 TCP 位姿减去已知的 flange->tool 尖端偏移，反推出法兰位姿。
        active_tool_pose = self.current_tcp_pose()
        t_base_to_tool = tcp_pose_to_transform(active_tool_pose)
        z_lift_m = float(self.get_parameter("fine_tool_to_flange_z_m").value)
        t_tool_to_flange = make_transform(np.eye(3), (0.0, 0.0, -z_lift_m))
        t_base_to_flange = t_base_to_tool @ t_tool_to_flange
        return transform_to_tcp_pose(t_base_to_flange)

    def current_joint(self) -> list[float]:
        return self._controller.current_joint()

    def robot_mode_text(self) -> str:
        return ROBOT_MODE_TEXT.get(self._controller.robot_mode, str(self._controller.robot_mode))

    def current_command_id(self) -> int:
        return self._controller.current_command_id()

    def _motion_user_index(self) -> int:
        return int(self.get_parameter("motion_user_index").value)

    def _flange_tool_index(self) -> int:
        return int(self.get_parameter("flange_tool_index").value)

    def _command_tool_index(self) -> int:
        return int(self.get_parameter("command_tool_index").value)

    def _command_uses_flange_frame(self) -> bool:
        return self._command_tool_index() == self._flange_tool_index()

    def _relative_target_from_tool_frame(self, msg: PoseStamped, tool_index: int) -> TcpPose:
        current_tool = self._controller.current_tcp_pose(
            user_index=self._motion_user_index(),
            tool_index=tool_index,
        )
        base_tool = tcp_pose_to_transform(current_tool)
        tool_target = pose_stamped_to_transform(msg)
        base_target = base_tool @ tool_target
        return transform_to_tcp_pose(base_target)

    def _convert_absolute_pose_between_tools(self, pose: TcpPose, src_tool_index: int, dst_tool_index: int) -> TcpPose:
        if src_tool_index == dst_tool_index:
            return TcpPose(pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz)

        t_base_to_src = tcp_pose_to_transform(pose)
        z_lift_m = float(self.get_parameter("fine_tool_to_flange_z_m").value)
        t_flange_to_tip = make_transform(np.eye(3), (0.0, 0.0, z_lift_m))
        t_tip_to_flange = make_transform(np.eye(3), (0.0, 0.0, -z_lift_m))

        if src_tool_index == self._flange_tool_index() and dst_tool_index == self._command_tool_index():
            return transform_to_tcp_pose(t_base_to_src @ t_flange_to_tip)
        if src_tool_index == self._command_tool_index() and dst_tool_index == self._flange_tool_index():
            return transform_to_tcp_pose(t_base_to_src @ t_tip_to_flange)

        raise ValueError(
            f"Unsupported tool conversion src={src_tool_index} dst={dst_tool_index}; "
            f"supported indices are flange={self._flange_tool_index()} and command={self._command_tool_index()}"
        )

    def _coarse_target_pose_callback(self, msg: PoseStamped) -> None:
        raw_rx_deg, raw_ry_deg, raw_rz_deg = quaternion_to_euler_deg(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        raw_target = TcpPose(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            z=float(msg.pose.position.z),
            rx=raw_rx_deg,
            ry=raw_ry_deg,
            rz=raw_rz_deg,
        )
        frame_id = msg.header.frame_id.strip() or "base"
        self._store_raw_coarse_target_pose(raw_target, frame_id)
        self._set_coarse_last_error("none")
        # self.get_logger().info(
        #     f"[coarse raw #{self.coarse_msg_count()}] frame={frame_id} "
        #     f"x={raw_target.x:.3f} y={raw_target.y:.3f} z={raw_target.z:.3f} "
        #     f"rx={raw_target.rx:.2f} ry={raw_target.ry:.2f} rz={raw_target.rz:.2f}"
        # )

        try:
            if frame_id == "base":
                coarse_target = raw_target
                coarse_object_base_pose = raw_target
                source = f"{self.get_parameter('coarse_motion_topic').value} [base]"
            elif frame_id == self._coarse_camera_frame_id:
                t_cam_to_target = pose_stamped_to_transform(msg)
                t_base_to_target_obj = self._handeye_base_to_d435 @ t_cam_to_target
                coarse_object_base_pose = transform_to_tcp_pose(t_base_to_target_obj)

                yaw_bias_deg = float(self.get_parameter("coarse_grasp_yaw_bias_deg").value)
                pitch_bias_deg = float(self.get_parameter("coarse_grasp_pitch_bias_deg").value)
                x_flip_deg = float(self.get_parameter("coarse_grasp_x_flip_deg").value)
                _, _, target_yaw_deg = rotation_matrix_to_euler_deg(t_base_to_target_obj[:3, :3])
                grasp_yaw_deg = target_yaw_deg + yaw_bias_deg

                coarse_hover_z_m = float(self.get_parameter("coarse_hover_z_m").value)
                fine_tool_to_flange_z_m = float(self.get_parameter("fine_tool_to_flange_z_m").value)
                r_base_to_tool_tip = euler_deg_to_rotation_matrix(
                    x_flip_deg, pitch_bias_deg, grasp_yaw_deg
                )
                hover_offset_z_m = coarse_hover_z_m
                if self._command_uses_flange_frame():
                    hover_offset_z_m += fine_tool_to_flange_z_m
                tool_frame_offset = np.array(
                    [0.0, 0.0, -hover_offset_z_m],
                    dtype=np.float64,
                )
                command_translation = (
                    np.array(t_base_to_target_obj[:3, 3], dtype=np.float64)
                    + r_base_to_tool_tip @ tool_frame_offset
                )
                coarse_target_x_offset_m = float(self.get_parameter("coarse_target_x_offset_m").value)

                # 粗定位阶段直接下发固定的 Rx/Ry/Rz，避免矩阵反解成另一组等效欧拉角。
                coarse_target = TcpPose(
                    x=float(command_translation[0]) - coarse_target_x_offset_m,
                    y=float(command_translation[1]),
                    z=float(command_translation[2]),
                    rx=x_flip_deg + 6,
                    # ry=pitch_bias_deg,
                    # rz=grasp_yaw_deg,
                    ry=0.27,
                    rz=-90,
                )
                # self.get_logger().info(
                #     "[coarse debug] "
                #     f"base_target_xyz=({t_base_to_target_obj[0, 3]:.3f}, {t_base_to_target_obj[1, 3]:.3f}, {t_base_to_target_obj[2, 3]:.3f}) "
                #     f"target_yaw={target_yaw_deg:.2f} grasp_yaw={grasp_yaw_deg:.2f} "
                #     f"cmd_rxyz=({coarse_target.rx:.2f}, {coarse_target.ry:.2f}, {coarse_target.rz:.2f}) "
                #     f"tool_to_flange_offset_local=({tool_tip_to_flange_offset[0]:.3f}, {tool_tip_to_flange_offset[1]:.3f}, {tool_tip_to_flange_offset[2]:.3f}) "
                #     f"flange_xyz_before_x_offset=({flange_translation[0]:.3f}, {flange_translation[1]:.3f}, {flange_translation[2]:.3f}) "
                #     f"x_offset={coarse_target_x_offset_m:.3f} "
                #     f"final_coarse_xyz=({coarse_target.x:.3f}, {coarse_target.y:.3f}, {coarse_target.z:.3f})"
                # )
                source = f"{self.get_parameter('coarse_motion_topic').value} [d435->base coarse]"
            else:
                self.get_logger().error(
                    f"Unsupported coarse frame_id '{frame_id}'. Only 'base' and '{self._coarse_camera_frame_id}' are supported."
                )
                self._set_coarse_last_error(f"unsupported frame_id: {frame_id}")
                return
        except Exception as exc:
            self._set_coarse_last_error(str(exc))
            self.get_logger().error(f"Failed to convert coarse target pose from {frame_id} to base: {exc}")
            return

        self.set_latest_coarse_object_base_pose(coarse_object_base_pose)
        self.set_latest_coarse_target_pose(coarse_target, source)
        # self.get_logger().info(
        #     f"Received coarse target frame={frame_id}, cached coarse base target: "
        #     f"x={coarse_target.x:.3f} y={coarse_target.y:.3f} z={coarse_target.z:.3f} "
        #     f"rx={coarse_target.rx:.2f} ry={coarse_target.ry:.2f} rz={coarse_target.rz:.2f}"
        # )

    def _target_pose_callback(self, msg: PoseStamped) -> None:
        raw_rx_deg, raw_ry_deg, raw_rz_deg = quaternion_to_euler_deg(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        raw_target = TcpPose(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            z=float(msg.pose.position.z),
            rx=raw_rx_deg,
            ry=raw_ry_deg,
            rz=raw_rz_deg,
        )
        frame_id = msg.header.frame_id.strip() or "base"
        self._store_raw_target_pose(raw_target, frame_id)

        try:
            if frame_id == "base":
                target = raw_target
                source = f"{self.get_parameter('motion_topic').value} [base]"
            elif frame_id.startswith("tool") and frame_id[4:].isdigit():
                tool_index = int(frame_id[4:])
                target = self._relative_target_from_tool_frame(msg, tool_index)
                source = f"{self.get_parameter('motion_topic').value} [tool{tool_index}->base]"
            # elif frame_id == self._camera_frame_id:
            #     # 1. 获取拍照瞬间的基座(Base)到法兰盘(Flange/Tool0)的 4x4 齐次变换矩阵
            #     current_tcp = self.current_tcp_pose()
            #     T_base_to_flange = tcp_pose_to_transform(current_tcp)
                
            #     # 2. 将视觉节点发来的 PoseStamped (基于相机光学坐标系) 转换为 4x4 变换矩阵
            #     # 此时该矩阵代表了目标物体相对于相机坐标系的位置和姿态
            #     T_cam_to_target = pose_stamped_to_transform(msg)
                
            #     # 3. 核心空间几何推导：通过矩阵连乘，将目标物体映射到 Base 坐标系下
            #     # 公式: T_base_to_target_obj = T_base_to_flange * T_flange_to_cam * T_cam_to_target
            #     # 此时得到的矩阵描述了物体的真实空间位姿。由于视觉侧规定物体的 Z 轴指向相机，
            #     # 所以 T_base_to_target_obj 中的 Z 轴目前是朝上的。
            #     T_base_to_target_obj = T_base_to_flange @ self._handeye_flange_to_cam @ T_cam_to_target
                
            #     # ==========================================
            #     # 核心修正：引入抓取姿态偏置 (Grasp Offset)
            #     # ==========================================
            #     # 目的: 将“物体所在的空间位姿”转换为“机械臂去抓取它时末端夹爪应有的位姿”
            #     #
            #     # 物理对齐逻辑：
            #     # (1) 物体 Z 轴朝上，但我们需要机械臂 TCP 的 Z 轴朝下对准物体进行抓取，
            #     #     所以我们让局部坐标系绕 X 轴旋转 180 度，实现夹爪的翻转动作。
            #     # (2) 视觉根据物体包围盒长边算出的 Rz 导致抓取方向存在约 103 度的偏差（根据测试数据得来），
            #     #     为了使得夹爪垂直或平行于物体抓取，我们需要补偿一个偏航角（Yaw）。
            #     #     这里暂定绕 Z 轴旋转 -90 度进行补偿对齐。
            #     #
            #     # 提示: 如果实机运行发现抓取角度与期望的长/宽边存在正交偏差，可将 -90.0 改为 90.0 或 0.0。
                
            #     # 创建偏置旋转矩阵: 绕 X 轴转 180 度, 绕 Z 轴转 -90 度
            #     R_grasp_offset = euler_deg_to_rotation_matrix(180.0, 0.0, -90.0) 
            #     T_target_to_grasp = make_transform(R_grasp_offset, (0.0, 0.0, 0.0))
                
            #     # 4. 最终连乘：Base -> 物体 -> 适合抓取的姿态
            #     T_base_to_target = T_base_to_target_obj @ T_target_to_grasp
                
            #     # 将 4x4 矩阵转换回机械臂控制器可接收的 [x, y, z, rx, ry, rz] 格式
            #     target = transform_to_tcp_pose(T_base_to_target)
            #     source = f"{self.get_parameter('motion_topic').value} [camera_optical->base]"


            elif frame_id == self._camera_frame_id:
                # 1. 获取拍照瞬间的基座(Base)到法兰盘(Flange/Tool0)的 4x4 齐次变换矩阵
                current_flange = self.current_flange_pose()
                T_base_to_flange = tcp_pose_to_transform(current_flange)
                
                # 2. 将视觉节点发来的 PoseStamped (基于相机光学坐标系) 转换为 4x4 变换矩阵
                T_cam_to_target = pose_stamped_to_transform(msg)
                
                # 3. 算出物体的真实空间位姿
                T_base_to_target_obj = T_base_to_flange @ self._handeye_flange_to_cam @ T_cam_to_target
                
                # 4. 引入抓取姿态偏置 (Grasp Offset) - 让夹爪 Z 轴朝下，并调整偏航角对齐物体长边
                R_grasp_offset = euler_deg_to_rotation_matrix(180.0, 0.0, -90.0) 
                T_target_to_grasp = make_transform(R_grasp_offset, (0.0, 0.0, 0.0))
                
                if self._command_uses_flange_frame():
                    z_lift_m = float(self.get_parameter("fine_tool_to_flange_z_m").value)
                    T_tool_to_flange = make_transform(np.eye(3), (0.0, 0.0, -z_lift_m))
                    T_base_to_command_target = T_base_to_target_obj @ T_target_to_grasp @ T_tool_to_flange
                else:
                    T_base_to_command_target = T_base_to_target_obj @ T_target_to_grasp

                target = transform_to_tcp_pose(T_base_to_command_target)
                source = (
                    f"{self.get_parameter('motion_topic').value} "
                    f"[camera_optical->base tool={self._command_tool_index()}]"
                )
            else:
                self.get_logger().error(
                    f"Unsupported frame_id '{frame_id}'. Only 'base', 'toolN' and '{self._camera_frame_id}' are supported."
                )
                return
        except Exception as exc:
            self.get_logger().error(f"Failed to convert target pose from {frame_id} to base: {exc}")
            return

        self.set_latest_target_pose(target, source)
        # self.get_logger().info(
        #     f"Received target frame={frame_id}, cached base target: "
        #     f"x={target.x:.3f} y={target.y:.3f} z={target.z:.3f} "
        #     f"rx={target.rx:.2f} ry={target.ry:.2f} rz={target.rz:.2f}"
        # )

    # def _publish_tcp_pose(self) -> None:
    #     try:
    #         tcp = self.current_tcp_pose()
    #     except Exception:
    #         return

    #     msg = PoseStamped()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = "base"
    #     msg.pose.position.x = tcp.x
    #     msg.pose.position.y = tcp.y
    #     msg.pose.position.z = tcp.z
    #     msg.pose.orientation.x = 0.0
    #     msg.pose.orientation.y = 0.0
    #     msg.pose.orientation.z = 0.0
    #     msg.pose.orientation.w = 1.0
    #     self._tcp_pub.publish(msg)
    def _publish_tcp_pose(self) -> None:
        try:
            tcp = self.current_tcp_pose()
        except Exception:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base"
        msg.pose.position.x = tcp.x
        msg.pose.position.y = tcp.y
        msg.pose.position.z = tcp.z
        
        # 将机械臂发来的欧拉角 (rx, ry, rz 角度) 转换为四元数
        # 'xyz' 代表 extrinsic x-y-z 旋转顺序，这与你代码里的 Rz * Ry * Rx 矩阵乘法数学上是一致的
        quat = SciPyRot.from_euler('xyz', [tcp.rx, tcp.ry, tcp.rz], degrees=True).as_quat()

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        self._tcp_pub.publish(msg)

class WorkerThread(QtCore.QThread):
    finished = Signal(bool, str, object)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            self.finished.emit(False, str(exc), None)
            return
        self.finished.emit(True, "OK", result)


PRIORITY_ACTIONS = {"Stop", "Pause", "Continue"}


class Nova5ControlWindow(QMainWindow):
    def __init__(self, node: Nova5DriverNode):
        super().__init__()
        self.node = node
        self._worker: Optional[WorkerThread] = None
        self._running_workers: set[WorkerThread] = set()

        self.pose_display_edits: dict[str, QLineEdit] = {}
        self.joint_display_edits: dict[str, QLineEdit] = {}
        self.target_spinboxes: dict[str, QDoubleSpinBox] = {}
        self.transfer_target_spinboxes: dict[str, QDoubleSpinBox] = {}
        self.place_target_spinboxes: dict[str, QDoubleSpinBox] = {}
        self._transfer_target_configured = False
        self._place_target_configured = False
        self.latest_target_edits: dict[str, QLineEdit] = {}
        self.latest_raw_target_edits: dict[str, QLineEdit] = {}
        self.latest_coarse_target_edits: dict[str, QLineEdit] = {}
        self.latest_raw_coarse_target_edits: dict[str, QLineEdit] = {}

        self.setWindowTitle("Nova5 Driver Control - User Defined Post Grasp Targets")
        self.resize(1080, 760)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self._build_status_group(left_layout)
        self._build_feedback_group(left_layout)
        self._build_latest_raw_coarse_target_group(left_layout)
        self._build_latest_coarse_target_group(left_layout)
        self._build_latest_raw_target_group(left_layout)
        self._build_latest_target_group(left_layout)
        self._build_target_group(left_layout)
        self._build_post_grasp_group(left_layout)
        self._build_command_group(left_layout)
        self._build_test_group(left_layout)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        self._build_jog_group(right_layout)
        self._build_log_group(right_layout)
        right_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_panel)

        root_layout.addWidget(left_scroll, 3)
        root_layout.addWidget(right_scroll, 2)

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(200)
        self._ui_timer.timeout.connect(self.refresh_ui)
        self._ui_timer.start()
        self._load_default_post_grasp_targets()
        self.refresh_ui()

    def closeEvent(self, event) -> None:
        if self._ui_timer.isActive():
            self._ui_timer.stop()
        for worker in list(self._running_workers):
            worker.wait(1000)
        super().closeEvent(event)

    def _build_status_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("Nova5 状态")
        layout = QFormLayout(box)

        self.ip_label = QLabel(self.node.get_parameter("robot_ip").value)
        self.mode_label = QLabel("-")
        self.command_id_label = QLabel("-")
        self.topic_label = QLabel(self.node.get_parameter("motion_topic").value)
        self.coord_label = QLabel("user")
        self.qt_binding_label = QLabel(QT_BINDING)

        layout.addRow("机械臂 IP", self.ip_label)
        layout.addRow("RobotMode", self.mode_label)
        layout.addRow("当前命令 ID", self.command_id_label)
        layout.addRow("订阅话题", self.topic_label)
        layout.addRow("点动坐标系", self.coord_label)
        layout.addRow("Qt 绑定", self.qt_binding_label)
        parent_layout.addWidget(box)

    def _build_command_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("基础控制")
        layout = QGridLayout(box)

        self.speed_factor_spin = QSpinBox()
        self.speed_factor_spin.setRange(1, 100)
        self.speed_factor_spin.setValue(20)
        self.linear_speed_spin = QSpinBox()
        self.linear_speed_spin.setRange(1, 100)
        self.linear_speed_spin.setValue(int(self.node.get_parameter("linear_speed").value))
        self.linear_acc_spin = QSpinBox()
        self.linear_acc_spin.setRange(1, 100)
        self.linear_acc_spin.setValue(int(self.node.get_parameter("linear_acc").value))
        self.joint_speed_spin = QSpinBox()
        self.joint_speed_spin.setRange(1, 100)
        self.joint_speed_spin.setValue(int(self.node.get_parameter("joint_speed").value))
        self.joint_acc_spin = QSpinBox()
        self.joint_acc_spin.setRange(1, 100)
        self.joint_acc_spin.setValue(int(self.node.get_parameter("joint_acc").value))
        self.user_index_spin = QSpinBox()
        self.user_index_spin.setRange(0, 9)
        self.tool_index_spin = QSpinBox()
        self.tool_index_spin.setRange(0, 9)

        self.power_button = QPushButton("上电")
        self.enable_button = QPushButton("使能")
        self.disable_button = QPushButton("下使能")
        self.clear_error_button = QPushButton("清错")
        self.reset_button = QPushButton("复位")
        self.stop_button = QPushButton("停止")
        self.pause_button = QPushButton("暂停")
        self.continue_button = QPushButton("继续")
        self.drag_button = QPushButton("进入拖拽")
        self.read_pose_button = QPushButton("读取姿态")
        self.sync_target_button = QPushButton("当前值写入目标")
        self.apply_speed_button = QPushButton("下发速度")
        self.apply_motion_profile_button = QPushButton("下发运动参数")
        self.user_coord_button = QPushButton("点动坐标: User")
        self.tool_coord_button = QPushButton("点动坐标: Tool")
        self.go_home_button = QPushButton("回初始位姿")

        layout.addWidget(self.power_button, 0, 0)
        layout.addWidget(self.enable_button, 0, 1)
        layout.addWidget(self.disable_button, 0, 2)
        layout.addWidget(self.clear_error_button, 1, 0)
        layout.addWidget(self.reset_button, 1, 1)
        layout.addWidget(self.stop_button, 1, 2)
        layout.addWidget(self.pause_button, 2, 0)
        layout.addWidget(self.continue_button, 2, 1)
        layout.addWidget(self.drag_button, 2, 2)

        layout.addWidget(QLabel("速度 %"), 3, 0)
        layout.addWidget(self.speed_factor_spin, 3, 1)
        layout.addWidget(self.apply_speed_button, 3, 2)
        layout.addWidget(QLabel("L-Speed"), 4, 0)
        layout.addWidget(self.linear_speed_spin, 4, 1)
        layout.addWidget(QLabel("L-Acc"), 5, 0)
        layout.addWidget(self.linear_acc_spin, 5, 1)
        layout.addWidget(QLabel("J-Speed"), 4, 2)
        layout.addWidget(self.joint_speed_spin, 4, 3)
        layout.addWidget(QLabel("J-Acc"), 5, 2)
        layout.addWidget(self.joint_acc_spin, 5, 3)
        layout.addWidget(self.apply_motion_profile_button, 6, 0, 1, 4)
        layout.addWidget(QLabel("User"), 7, 0)
        layout.addWidget(self.user_index_spin, 7, 1)
        layout.addWidget(QLabel("Tool"), 8, 0)
        layout.addWidget(self.tool_index_spin, 8, 1)
        layout.addWidget(self.user_coord_button, 7, 2)
        layout.addWidget(self.tool_coord_button, 8, 2)
        layout.addWidget(self.read_pose_button, 9, 0)
        layout.addWidget(self.sync_target_button, 9, 1)
        layout.addWidget(self.go_home_button, 9, 2)

        self.power_button.clicked.connect(lambda: self.run_action("PowerOn", self.node.power_on))
        self.enable_button.clicked.connect(lambda: self.run_action("EnableRobot", self.node.enable_robot))
        self.disable_button.clicked.connect(lambda: self.run_action("DisableRobot", self.node.disable_robot))
        self.clear_error_button.clicked.connect(lambda: self.run_action("ClearError", self.node.clear_error))
        self.reset_button.clicked.connect(lambda: self.run_action("ResetRobot", self.node.reset_robot))
        self.stop_button.clicked.connect(lambda: self.run_action("Stop", self.node.stop_motion))
        self.pause_button.clicked.connect(lambda: self.run_action("Pause", self.node.pause_motion))
        self.continue_button.clicked.connect(lambda: self.run_action("Continue", self.node.continue_motion))
        self.drag_button.clicked.connect(self.toggle_drag)
        self.read_pose_button.clicked.connect(self.sync_feedback_to_target)
        self.sync_target_button.clicked.connect(self.sync_feedback_to_target)
        self.apply_speed_button.clicked.connect(
            lambda: self.run_action(
                "SpeedFactor",
                self.node.apply_speed_factor,
                int(self.speed_factor_spin.value()),
            )
        )
        self.apply_motion_profile_button.clicked.connect(self.apply_motion_profile)
        self.user_coord_button.clicked.connect(lambda: self.set_coord_type("user"))
        self.tool_coord_button.clicked.connect(lambda: self.set_coord_type("tool"))
        self.go_home_button.clicked.connect(lambda: self.run_action("MoveToStartup", self.node.move_to_startup))

        parent_layout.addWidget(box)

    def _build_feedback_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("当前位姿")
        layout = QHBoxLayout(box)

        tcp_box = QGroupBox("TCP")
        tcp_layout = QFormLayout(tcp_box)
        for axis in POSE_AXES:
            edit = QLineEdit("0.000")
            edit.setReadOnly(True)
            self.pose_display_edits[axis] = edit
            tcp_layout.addRow(axis, edit)

        joint_box = QGroupBox("关节")
        joint_layout = QFormLayout(joint_box)
        for axis in JOINT_AXES:
            edit = QLineEdit("0.000")
            edit.setReadOnly(True)
            self.joint_display_edits[axis] = edit
            joint_layout.addRow(axis, edit)

        layout.addWidget(tcp_box)
        layout.addWidget(joint_box)
        parent_layout.addWidget(box)

    def _build_latest_target_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("最新订阅目标(Base执行值/精定位)")
        layout = QFormLayout(box)

        self.latest_target_source_label = QLabel("none")
        layout.addRow("来源", self.latest_target_source_label)

        for axis in POSE_AXES:
            edit = QLineEdit("0.000")
            edit.setReadOnly(True)
            self.latest_target_edits[axis] = edit
            layout.addRow(axis, edit)

        parent_layout.addWidget(box)

    def _build_latest_coarse_target_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("最新粗定位目标(Base执行值)")
        layout = QFormLayout(box)

        self.latest_coarse_target_source_label = QLabel("none")
        layout.addRow("来源", self.latest_coarse_target_source_label)

        for axis in POSE_AXES:
            edit = QLineEdit("0.000")
            edit.setReadOnly(True)
            self.latest_coarse_target_edits[axis] = edit
            layout.addRow(axis, edit)

        parent_layout.addWidget(box)

    def _build_latest_raw_coarse_target_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("最新粗定位原始值")
        layout = QFormLayout(box)

        self.latest_raw_coarse_frame_label = QLabel("none")
        layout.addRow("frame_id", self.latest_raw_coarse_frame_label)

        for axis in POSE_AXES:
            edit = QLineEdit("0.000")
            edit.setReadOnly(True)
            self.latest_raw_coarse_target_edits[axis] = edit
            layout.addRow(axis, edit)

        parent_layout.addWidget(box)

    def _build_latest_raw_target_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("最新订阅原始值")
        layout = QFormLayout(box)

        self.latest_raw_frame_label = QLabel("none")
        layout.addRow("frame_id", self.latest_raw_frame_label)

        for axis in POSE_AXES:
            edit = QLineEdit("0.000")
            edit.setReadOnly(True)
            self.latest_raw_target_edits[axis] = edit
            layout.addRow(axis, edit)

        parent_layout.addWidget(box)

    def _build_target_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("手动目标位姿")
        layout = QGridLayout(box)

        for row, axis in enumerate(POSE_AXES):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            if row < 3:
                spin.setRange(-2.0, 2.0)
                spin.setSingleStep(0.005)
            else:
                spin.setRange(-360.0, 360.0)
                spin.setSingleStep(1.0)
            self.target_spinboxes[axis] = spin
            layout.addWidget(QLabel(axis), row, 0)
            layout.addWidget(spin, row, 1)

        self.movl_button = QPushButton("MovL 到手动目标")
        self.write_latest_button = QPushButton("最新订阅值写入手动目标")
        self.movl_button.clicked.connect(self.execute_manual_target)
        self.write_latest_button.clicked.connect(self.copy_latest_to_manual_target)

        layout.addWidget(self.write_latest_button, 6, 0)
        layout.addWidget(self.movl_button, 6, 1)
        parent_layout.addWidget(box)

    def _build_post_grasp_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("抓取后搬运位姿")
        layout = QGridLayout(box)

        layout.addWidget(QLabel("中转点"), 0, 0, 1, 2)
        self._build_pose_spinbox_columns(layout, start_row=1, column_offset=0, store=self.transfer_target_spinboxes)

        layout.addWidget(QLabel("放置点"), 0, 2, 1, 2)
        self._build_pose_spinbox_columns(layout, start_row=1, column_offset=2, store=self.place_target_spinboxes)

        self.write_transfer_button = QPushButton("当前位姿写入中转点")
        self.write_place_button = QPushButton("当前位姿写入放置点")
        self.post_grasp_only_button = QPushButton("仅执行抓取后搬运")

        self.write_transfer_button.clicked.connect(self.sync_feedback_to_transfer_target)
        self.write_place_button.clicked.connect(self.sync_feedback_to_place_target)
        self.post_grasp_only_button.clicked.connect(self.execute_post_grasp_only)

        layout.addWidget(self.write_transfer_button, 7, 0, 1, 2)
        layout.addWidget(self.write_place_button, 7, 2, 1, 2)
        layout.addWidget(self.post_grasp_only_button, 8, 0, 1, 4)
        parent_layout.addWidget(box)

    def _load_default_post_grasp_targets(self) -> None:
        transfer_values = [float(v) for v in self.node.get_parameter("default_transfer_pose").value]
        place_values = [float(v) for v in self.node.get_parameter("default_place_pose").value]
        if len(transfer_values) != 6:
            raise ValueError(f"default_transfer_pose must contain 6 values, got {len(transfer_values)}")
        if len(place_values) != 6:
            raise ValueError(f"default_place_pose must contain 6 values, got {len(place_values)}")

        src_tool_index = int(self.node.get_parameter("default_post_grasp_pose_tool_index").value)
        transfer_pose = self.node._convert_absolute_pose_between_tools(
            TcpPose(*transfer_values),
            src_tool_index,
            self.node._command_tool_index(),
        )
        place_pose = self.node._convert_absolute_pose_between_tools(
            TcpPose(*place_values),
            src_tool_index,
            self.node._command_tool_index(),
        )

        self._set_pose_spinboxes(self.transfer_target_spinboxes, transfer_pose)
        self._set_pose_spinboxes(self.place_target_spinboxes, place_pose)
        self._transfer_target_configured = True
        self._place_target_configured = True

    def _build_test_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("测试执行")
        layout = QVBoxLayout(box)

        self.test_execute_button = QPushButton("测试执行最新订阅目标")
        self.coarse_then_fine_button = QPushButton("粗定位 -> 精定位 -> 抓取 -> 搬运放置")
        self.test_gripper_button = QPushButton("测试夹爪开合")
        self.test_execute_button.clicked.connect(
            lambda: self.run_action("ExecuteLatestTarget", self.node.execute_latest_target)
        )
        self.coarse_then_fine_button.clicked.connect(self.execute_coarse_then_fine_with_post_grasp)
        self.test_gripper_button.clicked.connect(
            lambda: self.run_action("TestGripperOpenClose", self.node.test_gripper_open_close)
        )

        tip = QLabel("只有点击这个按钮，机械臂才会执行最近一次订阅到的目标位姿。")
        tip.setWordWrap(True)

        layout.addWidget(self.test_execute_button)
        layout.addWidget(self.coarse_then_fine_button)
        layout.addWidget(self.test_gripper_button)
        layout.addWidget(tip)
        parent_layout.addWidget(box)

    def _build_jog_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("XYZRxRyRz 点动")
        layout = QGridLayout(box)
        layout.addWidget(QLabel("按下开始点动，松开停止。"), 0, 0, 1, 3)

        for row, axis in enumerate(POSE_AXES, start=1):
            minus_btn = QPushButton(f"{axis}-")
            plus_btn = QPushButton(f"{axis}+")
            minus_btn.pressed.connect(lambda a=f"{axis}-": self.start_jog(a))
            minus_btn.released.connect(self.stop_jog)
            plus_btn.pressed.connect(lambda a=f"{axis}+": self.start_jog(a))
            plus_btn.released.connect(self.stop_jog)
            layout.addWidget(QLabel(axis), row, 0)
            layout.addWidget(minus_btn, row, 1)
            layout.addWidget(plus_btn, row, 2)

        parent_layout.addWidget(box)

    def _build_log_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("运行信息")
        layout = QVBoxLayout(box)
        self.message_label = QLabel("ready")
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        parent_layout.addWidget(box)

    def run_action(self, name: str, fn, *args) -> None:
        if self._worker is not None and self._worker.isRunning() and name not in PRIORITY_ACTIONS:
            self.set_message(f"{name} ignored: previous action still running")
            return
        self.set_message(f"{name} running...")
        worker = WorkerThread(fn, *args)
        self._running_workers.add(worker)
        if name not in PRIORITY_ACTIONS:
            self._worker = worker
        worker.finished.connect(
            lambda ok, msg, result, n=name, w=worker: self._on_action_finished(n, ok, msg, result, w)
        )
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_action_finished(self, name: str, ok: bool, msg: str, result: object, worker: WorkerThread) -> None:
        self._running_workers.discard(worker)
        if self._worker is worker:
            self._worker = None
        if ok and name == "ToggleDrag":
            enabled = bool(result)
            self.drag_button.setText("退出拖拽" if enabled else "进入拖拽")
        if ok:
            self.set_message(f"{name} finished")
        else:
            if name == "MoveToStartup":
                msg = (
                    f"{msg} | current startup_joint may be unreachable, in collision/singularity, "
                    "or rejected by controller planning."
                )
            self.set_message(f"{name} failed: {msg}")
        self.refresh_ui()

    def set_message(self, text: str) -> None:
        self.message_label.setText(text)

    def toggle_drag(self) -> None:
        self.run_action("ToggleDrag", self.node.toggle_drag)

    def set_coord_type(self, coord_type: str) -> None:
        self.node.set_coord_type(coord_type)
        self.coord_label.setText(coord_type)
        self.set_message(f"Jog coord type set to {coord_type}")

    def sync_feedback_to_target(self) -> None:
        try:
            pose = self.node.read_current_pose()
        except Exception as exc:
            self.set_message(f"Read current pose failed: {exc}")
            return
        self._set_manual_target(pose)
        self.set_message("Current pose written into manual target")

    def copy_latest_to_manual_target(self) -> None:
        pose = self.node.latest_target_pose()
        if pose is None:
            self.set_message("No latest target received")
            return
        self._set_manual_target(pose)
        self.set_message("Latest subscribed target written into manual target")

    def execute_manual_target(self) -> None:
        pose = self._manual_target_pose()
        self.run_action("ManualMovL", self.node.manual_move_linear, pose)

    def apply_motion_profile(self) -> None:
        self.node.set_parameters([
            rclpy.parameter.Parameter("linear_speed", value=int(self.linear_speed_spin.value())),
            rclpy.parameter.Parameter("linear_acc", value=int(self.linear_acc_spin.value())),
            rclpy.parameter.Parameter("joint_speed", value=int(self.joint_speed_spin.value())),
            rclpy.parameter.Parameter("joint_acc", value=int(self.joint_acc_spin.value())),
        ])
        self.run_action("ApplyMotionProfile", self.node.apply_motion_profile)

    def sync_feedback_to_transfer_target(self) -> None:
        try:
            pose = self.node.read_current_pose()
        except Exception as exc:
            self.set_message(f"Read current pose failed: {exc}")
            return
        self._set_pose_spinboxes(self.transfer_target_spinboxes, pose)
        self.set_message("Current pose written into transfer target")

    def sync_feedback_to_place_target(self) -> None:
        try:
            pose = self.node.read_current_pose()
        except Exception as exc:
            self.set_message(f"Read current pose failed: {exc}")
            return
        self._set_pose_spinboxes(self.place_target_spinboxes, pose)
        self.set_message("Current pose written into place target")

    def execute_post_grasp_only(self) -> None:
        if not self._transfer_target_configured or not self._place_target_configured:
            self.set_message("Transfer/place target not configured")
            return
        transfer_pose = self._pose_from_spinboxes(self.transfer_target_spinboxes)
        place_pose = self._pose_from_spinboxes(self.place_target_spinboxes)
        self.run_action(
            "ExecutePostGraspPlace",
            self.node.execute_post_grasp_place_sequence,
            transfer_pose,
            place_pose,
        )

    def execute_coarse_then_fine_with_post_grasp(self) -> None:
        if not self._transfer_target_configured or not self._place_target_configured:
            self.set_message("Transfer/place target not configured")
            return
        transfer_pose = self._pose_from_spinboxes(self.transfer_target_spinboxes)
        place_pose = self._pose_from_spinboxes(self.place_target_spinboxes)
        self.run_action(
            "ExecuteCoarseThenFine",
            self.node.execute_coarse_then_fine_sequence,
            transfer_pose,
            place_pose,
        )

    def start_jog(self, axis_id: str) -> None:
        user = int(self.user_index_spin.value())
        tool = int(self.tool_index_spin.value())
        try:
            self.node.start_jog(axis_id, user=user, tool=tool)
        except Exception as exc:
            self.set_message(f"Start jog failed: {exc}")

    def stop_jog(self) -> None:
        try:
            self.node.stop_jog()
        except Exception as exc:
            self.set_message(f"Stop jog failed: {exc}")

    def _manual_target_pose(self) -> TcpPose:
        return self._pose_from_spinboxes(self.target_spinboxes)

    def _set_manual_target(self, pose: TcpPose) -> None:
        self._set_pose_spinboxes(self.target_spinboxes, pose)

    def _build_pose_spinbox_columns(
        self,
        layout: QGridLayout,
        start_row: int,
        column_offset: int,
        store: dict[str, QDoubleSpinBox],
    ) -> None:
        for row, axis in enumerate(POSE_AXES, start=start_row):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            if axis in ("X", "Y", "Z"):
                spin.setRange(-2.0, 2.0)
                spin.setSingleStep(0.005)
            else:
                spin.setRange(-360.0, 360.0)
                spin.setSingleStep(1.0)
            store[axis] = spin
            if store is self.transfer_target_spinboxes:
                spin.valueChanged.connect(self._mark_transfer_target_configured)
            elif store is self.place_target_spinboxes:
                spin.valueChanged.connect(self._mark_place_target_configured)
            layout.addWidget(QLabel(axis), row, column_offset)
            layout.addWidget(spin, row, column_offset + 1)

    def _pose_from_spinboxes(self, spinboxes: dict[str, QDoubleSpinBox]) -> TcpPose:
        return TcpPose(
            x=float(spinboxes["X"].value()),
            y=float(spinboxes["Y"].value()),
            z=float(spinboxes["Z"].value()),
            rx=float(spinboxes["Rx"].value()),
            ry=float(spinboxes["Ry"].value()),
            rz=float(spinboxes["Rz"].value()),
        )

    def _set_pose_spinboxes(self, spinboxes: dict[str, QDoubleSpinBox], pose: TcpPose) -> None:
        spinboxes["X"].setValue(pose.x)
        spinboxes["Y"].setValue(pose.y)
        spinboxes["Z"].setValue(pose.z)
        spinboxes["Rx"].setValue(pose.rx)
        spinboxes["Ry"].setValue(pose.ry)
        spinboxes["Rz"].setValue(pose.rz)

    def _mark_transfer_target_configured(self) -> None:
        self._transfer_target_configured = True

    def _mark_place_target_configured(self) -> None:
        self._place_target_configured = True

    def refresh_ui(self) -> None:
        self.mode_label.setText(self.node.robot_mode_text())
        self.command_id_label.setText(str(self.node.current_command_id()))
        self.coord_label.setText(self.node._coord_type)

        try:
            tcp = self.node.current_tcp_pose()
            tcp_values = (tcp.x, tcp.y, tcp.z, tcp.rx, tcp.ry, tcp.rz)
            for axis, value in zip(POSE_AXES, tcp_values):
                self.pose_display_edits[axis].setText(f"{value:.3f}")
        except Exception:
            pass

        try:
            joints = self.node.current_joint()
            for axis, value in zip(JOINT_AXES, joints):
                self.joint_display_edits[axis].setText(f"{value:.3f}")
        except Exception:
            pass

        latest = self.node.latest_target_pose()
        if latest is not None:
            latest_values = (latest.x, latest.y, latest.z, latest.rx, latest.ry, latest.rz)
            for axis, value in zip(POSE_AXES, latest_values):
                self.latest_target_edits[axis].setText(f"{value:.3f}")
        self.latest_target_source_label.setText(self.node.latest_target_source())

        latest_coarse = self.node.latest_coarse_target_pose()
        if latest_coarse is not None:
            latest_coarse_values = (
                latest_coarse.x,
                latest_coarse.y,
                latest_coarse.z,
                latest_coarse.rx,
                latest_coarse.ry,
                latest_coarse.rz,
            )
            for axis, value in zip(POSE_AXES, latest_coarse_values):
                self.latest_coarse_target_edits[axis].setText(f"{value:.3f}")
        self.latest_coarse_target_source_label.setText(self.node.latest_coarse_target_source())

        raw_latest = self.node.latest_raw_target_pose()
        if raw_latest is not None:
            raw_values = (raw_latest.x, raw_latest.y, raw_latest.z, raw_latest.rx, raw_latest.ry, raw_latest.rz)
            for axis, value in zip(POSE_AXES, raw_values):
                self.latest_raw_target_edits[axis].setText(f"{value:.3f}")
        self.latest_raw_frame_label.setText(self.node.latest_raw_frame_id())

        raw_latest_coarse = self.node.latest_raw_coarse_target_pose()
        if raw_latest_coarse is not None:
            raw_coarse_values = (
                raw_latest_coarse.x,
                raw_latest_coarse.y,
                raw_latest_coarse.z,
                raw_latest_coarse.rx,
                raw_latest_coarse.ry,
                raw_latest_coarse.rz,
            )
            for axis, value in zip(POSE_AXES, raw_coarse_values):
                self.latest_raw_coarse_target_edits[axis].setText(f"{value:.3f}")
        self.latest_raw_coarse_frame_label.setText(self.node.latest_raw_coarse_frame_id())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Nova5DriverNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = QApplication.instance() or QApplication(sys.argv)
    window = Nova5ControlWindow(node)
    window.show()

    try:
        exec_fn = getattr(app, "exec", None)
        if exec_fn is None:
            exec_fn = app.exec_
        exec_fn()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
