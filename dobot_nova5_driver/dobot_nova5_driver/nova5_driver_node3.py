import math
import os
import sys
import threading
import time
from typing import Optional
from scipy.spatial.transform import Rotation as SciPyRot

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

if __package__ in (None, ""):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_ROOT = os.path.dirname(CURRENT_DIR)
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    from dobot_nova5_driver.controller import DobotNova5Controller, ROBOT_MODE_TEXT, TcpPose
else:
    from .controller import DobotNova5Controller, ROBOT_MODE_TEXT, TcpPose

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

        self.declare_parameter("robot_ip", "192.168.5.102")
        self.declare_parameter("dashboard_port", 29999)
        self.declare_parameter("feedback_port", 30004)
        self.declare_parameter("go_to_start", False)
        self.declare_parameter("auto_enable", False)
        self.declare_parameter("startup_joint", [178.0, -7.60, 90.0, 2.40, -90.0, -1.50]) #机械臂初始位姿
        self.declare_parameter("startup_speed", 35)
        self.declare_parameter("motion_topic", "/target_pose_cam_fine")
        self.declare_parameter("coarse_motion_topic", "/target_pose_cam_coarse")
        self.declare_parameter("tcp_pose_topic", "/nova5/current_tcp_pose")
        self.declare_parameter("linear_speed", 25)
        self.declare_parameter("fine_tool_to_flange_z_m", 0.23) # 末端D405抓取时，夹爪尖端到法兰中心Z差值
        self.declare_parameter("camera_frame_id", "camera_d405_link")
        self.declare_parameter("coarse_camera_frame_id", "camera_d435_link")
        self.declare_parameter("coarse_hover_z_m", 0.10)  # 粗定位时，夹爪尖端停在目标上方27cm
        self.declare_parameter("coarse_target_x_offset_m", 0.10)  # 粗定位最终base执行值在x上减10cm
        self.declare_parameter("coarse_grasp_yaw_bias_deg", -90.0)
        self.declare_parameter("coarse_grasp_pitch_bias_deg", -0.0)
        self.declare_parameter("coarse_grasp_x_flip_deg", 180.0)
        self.declare_parameter("trigger_d405_topic", "/trigger_d405_vision")
        self.declare_parameter("d405_trigger_delay_s", 0.7)
        self.declare_parameter("fine_refine_delay_s", 0.1)
        self.declare_parameter("fine_refine_cycles", 1)
        self.declare_parameter("fine_target_timeout_s", 2.0)
        self.declare_parameter("fine_position_stability_threshold_m", 0.032)
        self.declare_parameter("fine_angle_stability_threshold_deg", 10.0)
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
        self._latest_raw_coarse_target_pose: Optional[TcpPose] = None
        self._latest_raw_coarse_frame_id: str = "none"
        self._coarse_msg_count = 0
        self._coarse_last_error = "none"
        self._drag_enabled = False
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
        self._timer = self.create_timer(0.1, self._publish_tcp_pose)

        self.get_logger().info("Nova5 driver connected and ready.")

    def destroy_node(self):
        try:
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

    def execute_latest_target(self) -> None:
        target = self.latest_target_pose()
        if target is None:
            raise RuntimeError("No latest target pose has been received yet")
        with self._motion_lock:
            self._controller.move_linear_tcp(
                target,
                speed=int(self.get_parameter("linear_speed").value),
            )
        self.get_logger().info(
            f"Executed latest target x={target.x:.3f} y={target.y:.3f} z={target.z:.3f} "
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

    def execute_coarse_then_fine_sequence(self) -> None:
        coarse_target = self.latest_coarse_target_pose()
        if coarse_target is None:
            raise RuntimeError("No coarse target pose has been received yet")

        with self._motion_lock:
            self._controller.move_linear_tcp(
                coarse_target,
                speed=int(self.get_parameter("linear_speed").value),
            )

        self.get_logger().info(
            f"Executed coarse target x={coarse_target.x:.3f} y={coarse_target.y:.3f} z={coarse_target.z:.3f} "
            f"rx={coarse_target.rx:.2f} ry={coarse_target.ry:.2f} rz={coarse_target.rz:.2f}"
        )

        refine_cycles = max(0, int(self.get_parameter("fine_refine_cycles").value))
        fine_target_timeout_s = float(self.get_parameter("fine_target_timeout_s").value)
        fine_samples: list[TcpPose] = []
        trigger_msg = Bool()
        trigger_msg.data = True #到达粗定位点位，设置为 true
        previous_count = self.target_msg_count()
        self._trigger_d405_pub.publish(trigger_msg)
        self.get_logger().info("Published D405 trigger true.")

        time.sleep(float(self.get_parameter("d405_trigger_delay_s").value)) #sleep一下
        fine_samples.append(self._wait_for_next_fine_target(previous_count, fine_target_timeout_s))

        fine_refine_delay_s = float(self.get_parameter("fine_refine_delay_s").value)
        for cycle_index in range(refine_cycles):
            trigger_msg = Bool()
            trigger_msg.data = True
            previous_count = self.target_msg_count()
            self._trigger_d405_pub.publish(trigger_msg)
            self.get_logger().info(
                f"Published D405 trigger true for fine refine cycle {cycle_index + 1}/{refine_cycles}."
            )
            time.sleep(fine_refine_delay_s)
            fine_samples.append(self._wait_for_next_fine_target(previous_count, fine_target_timeout_s))

        self._validate_fine_target_stability(fine_samples)
        averaged_target = self._average_tcp_poses(fine_samples)
        self.set_latest_target_pose(
            averaged_target,
            f"{self.get_parameter('motion_topic').value} [fine averaged {len(fine_samples)} samples]",
        )
        self.get_logger().info(
            f"Averaged fine target from {len(fine_samples)} samples: "
            f"x={averaged_target.x:.3f} y={averaged_target.y:.3f} z={averaged_target.z:.3f} "
            f"rx={averaged_target.rx:.2f} ry={averaged_target.ry:.2f} rz={averaged_target.rz:.2f}"
        )

        self.execute_latest_target()

    def move_to_startup(self) -> None:
        with self._motion_lock:
            self._controller.move_to_startup()

    def manual_move_linear(self, pose: TcpPose) -> None:
        self.set_latest_target_pose(pose, "manual_target")
        with self._motion_lock:
            self._controller.move_linear_tcp(
                pose,
                speed=int(self.get_parameter("linear_speed").value),
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
        return self._controller.read_pose()

    def current_tcp_pose(self) -> TcpPose:
        return self._controller.current_tcp_pose()

    def current_joint(self) -> list[float]:
        return self._controller.current_joint()

    def robot_mode_text(self) -> str:
        return ROBOT_MODE_TEXT.get(self._controller.robot_mode, str(self._controller.robot_mode))

    def current_command_id(self) -> int:
        return self._controller.current_command_id()

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
                source = f"{self.get_parameter('coarse_motion_topic').value} [base]"
            elif frame_id == self._coarse_camera_frame_id:
                t_cam_to_target = pose_stamped_to_transform(msg)
                t_base_to_target_obj = self._handeye_base_to_d435 @ t_cam_to_target

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
                tool_tip_to_flange_offset = np.array(
                    [0.0, 0.0, -(coarse_hover_z_m + fine_tool_to_flange_z_m)],
                    dtype=np.float64,
                )
                flange_translation = (
                    np.array(t_base_to_target_obj[:3, 3], dtype=np.float64)
                    + r_base_to_tool_tip @ tool_tip_to_flange_offset
                )
                coarse_target_x_offset_m = float(self.get_parameter("coarse_target_x_offset_m").value)

                # 粗定位阶段直接下发固定的 Rx/Ry/Rz，避免矩阵反解成另一组等效欧拉角。
                coarse_target = TcpPose(
                    x=float(flange_translation[0]) - coarse_target_x_offset_m,
                    y=float(flange_translation[1]),
                    z=float(flange_translation[2]),
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
            elif frame_id == "tool0":
                current_tcp = self.current_tcp_pose()
                base_tool = tcp_pose_to_transform(current_tcp)
                tool_target = pose_stamped_to_transform(msg)
                base_target = base_tool @ tool_target
                target = transform_to_tcp_pose(base_target)
                source = f"{self.get_parameter('motion_topic').value} [tool0->base]"
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
                current_tcp = self.current_tcp_pose()
                T_base_to_flange = tcp_pose_to_transform(current_tcp)
                
                # 2. 将视觉节点发来的 PoseStamped (基于相机光学坐标系) 转换为 4x4 变换矩阵
                T_cam_to_target = pose_stamped_to_transform(msg)
                
                # 3. 算出物体的真实空间位姿
                T_base_to_target_obj = T_base_to_flange @ self._handeye_flange_to_cam @ T_cam_to_target
                
                # 4. 引入抓取姿态偏置 (Grasp Offset) - 让夹爪 Z 轴朝下，并调整偏航角对齐物体长边
                R_grasp_offset = euler_deg_to_rotation_matrix(180.0, 0.0, -90.0) 
                T_target_to_grasp = make_transform(R_grasp_offset, (0.0, 0.0, 0.0))
                
                # ==========================================
                # 核心修正：引入末端夹爪负载 (Tool Offset)
                # ==========================================
                # 上面算出的 `T_base_to_target_obj @ T_target_to_grasp` 是“夹爪尖端 (Tool Tip)”应该在的空间位姿。
                # 但机械臂是以“法兰盘 (Flange)”为控制中心的。法兰到夹爪尖端在 Z 轴上有 23cm 的正向突出。
                # 为了让夹爪尖端刚好碰到物体，法兰盘必须停在沿局部 Z 轴“倒退 23cm”的位置。
                # 所以我们构造一个沿 Z 轴平移 -0.23m 的逆矩阵 (T_tool_to_flange)
                z_lift_m = float(self.get_parameter("fine_tool_to_flange_z_m").value)
                T_tool_to_flange = make_transform(np.eye(3), (0.0, 0.0, -z_lift_m))
                
                # 5. 最终完美矩阵连乘：Base -> 物体 -> 抓取姿态 -> 法兰盘实际应该停靠的位姿
                T_base_to_flange_target = T_base_to_target_obj @ T_target_to_grasp @ T_tool_to_flange
                
                # 将 4x4 矩阵转换回机械臂控制器可接收的 [x, y, z, rx, ry, rz] 格式
                target = transform_to_tcp_pose(T_base_to_flange_target)
                source = f"{self.get_parameter('motion_topic').value} [camera_optical->base]"    
            else:
                self.get_logger().error(
                    f"Unsupported frame_id '{frame_id}'. Only 'base', 'tool0' and '{self._camera_frame_id}' are supported."
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
        self.latest_target_edits: dict[str, QLineEdit] = {}
        self.latest_raw_target_edits: dict[str, QLineEdit] = {}
        self.latest_coarse_target_edits: dict[str, QLineEdit] = {}
        self.latest_raw_coarse_target_edits: dict[str, QLineEdit] = {}

        self.setWindowTitle("Nova5 Driver Control")
        self.resize(1080, 760)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self._build_status_group(left_layout)
        self._build_command_group(left_layout)
        self._build_feedback_group(left_layout)
        self._build_latest_raw_coarse_target_group(left_layout)
        self._build_latest_coarse_target_group(left_layout)
        self._build_latest_raw_target_group(left_layout)
        self._build_latest_target_group(left_layout)
        self._build_target_group(left_layout)
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
        layout.addWidget(QLabel("User"), 4, 0)
        layout.addWidget(self.user_index_spin, 4, 1)
        layout.addWidget(QLabel("Tool"), 5, 0)
        layout.addWidget(self.tool_index_spin, 5, 1)
        layout.addWidget(self.user_coord_button, 4, 2)
        layout.addWidget(self.tool_coord_button, 5, 2)
        layout.addWidget(self.read_pose_button, 6, 0)
        layout.addWidget(self.sync_target_button, 6, 1)
        layout.addWidget(self.go_home_button, 6, 2)

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

    def _build_test_group(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("测试执行")
        layout = QVBoxLayout(box)

        self.test_execute_button = QPushButton("测试执行最新订阅目标")
        self.coarse_then_fine_button = QPushButton("粗定位 -> 触发D405 -> 精定位")
        self.test_execute_button.clicked.connect(
            lambda: self.run_action("ExecuteLatestTarget", self.node.execute_latest_target)
        )
        self.coarse_then_fine_button.clicked.connect(
            lambda: self.run_action("ExecuteCoarseThenFine", self.node.execute_coarse_then_fine_sequence)
        )

        tip = QLabel("只有点击这个按钮，机械臂才会执行最近一次订阅到的目标位姿。")
        tip.setWordWrap(True)

        layout.addWidget(self.test_execute_button)
        layout.addWidget(self.coarse_then_fine_button)
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
        return TcpPose(
            x=float(self.target_spinboxes["X"].value()),
            y=float(self.target_spinboxes["Y"].value()),
            z=float(self.target_spinboxes["Z"].value()),
            rx=float(self.target_spinboxes["Rx"].value()),
            ry=float(self.target_spinboxes["Ry"].value()),
            rz=float(self.target_spinboxes["Rz"].value()),
        )

    def _set_manual_target(self, pose: TcpPose) -> None:
        self.target_spinboxes["X"].setValue(pose.x)
        self.target_spinboxes["Y"].setValue(pose.y)
        self.target_spinboxes["Z"].setValue(pose.z)
        self.target_spinboxes["Rx"].setValue(pose.rx)
        self.target_spinboxes["Ry"].setValue(pose.ry)
        self.target_spinboxes["Rz"].setValue(pose.rz)

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
