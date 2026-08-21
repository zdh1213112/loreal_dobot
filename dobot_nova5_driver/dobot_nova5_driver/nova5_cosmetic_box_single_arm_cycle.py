"""Single Nova5 (192.168.111.101) cosmetic-box pick/scan/place cycle.

The state machine intentionally contains only the requested path:

startup -> trigger D405 -> grasp 75%-depth target -> transfer joint ->
consume any barcode seen during transfer, otherwise rotate wrist J6 -90 degrees
per barcode face with the legacy safety logic until one value is stable ->
user-frame XYZ PTP to scan exit -> hold TCP XYZ and MoveJog about user Ry- ->
place -> startup.

The D405 pose represents the TCP-tip point 75% down from the measured top
surface. A small operator-visible Z correction compensates residual hand-eye
height bias, and an absolute TCP-Z floor prevents contact with the tabletop.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation as SciPyRot
from std_msgs.msg import Bool, Float32, String

try:
    from PySide6 import QtCore
    from PySide6.QtCore import QTimer, Signal
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout,
        QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
        QSpinBox, QVBoxLayout, QWidget,
    )
except ImportError:
    from PyQt5 import QtCore
    from PyQt5.QtCore import QTimer, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication, QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout,
        QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
        QSpinBox, QVBoxLayout, QWidget,
    )

from .controller import DobotNova5Controller, TcpPose
from .dobot_dh_api import (
    GRIP_DROPPED,
    GRIP_GRIPPED,
    GRIP_IN_MOTION,
    GRIP_REACHED,
    DobotDHConfig,
    DHGripper,
    raise_if_error,
)


def pose_to_transform(pose: TcpPose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = SciPyRot.from_euler("xyz", [pose.rx, pose.ry, pose.rz], degrees=True).as_matrix()
    transform[:3, 3] = [pose.x, pose.y, pose.z]
    return transform


def message_to_transform(msg: PoseStamped) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    quat = msg.pose.orientation
    transform[:3, :3] = SciPyRot.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    pos = msg.pose.position
    transform[:3, 3] = [pos.x, pos.y, pos.z]
    return transform


def transform_to_pose(transform: np.ndarray) -> TcpPose:
    rx, ry, rz = SciPyRot.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    x, y, z = transform[:3, 3]
    return TcpPose(float(x), float(y), float(z), float(rx), float(ry), float(rz))


def circular_mean(values: list[float]) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


class RecoverableGraspError(RuntimeError):
    """A confirmed empty/lost grasp for which returning to startup is safe."""

    def __init__(self, stage: str, message: str, needs_vertical_retreat: bool = False):
        super().__init__(message)
        self.stage = stage
        self.needs_vertical_retreat = needs_vertical_retreat


class CosmeticBoxSingleArmNode(Node):
    def __init__(self) -> None:
        super().__init__("nova5_cosmetic_box_single_arm_cycle")

        self.declare_parameter("robot_ip", "192.168.111.101")
        self.declare_parameter("dashboard_port", 29999)
        self.declare_parameter("feedback_port", 30004)
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("startup_joint", [14.0, 14.0, -115.0, 25.0, 83.0, 10.0])
        self.declare_parameter("transfer_joint", [14.0, -29.0, -99.0, 39.0, 88.0, 10.0])
        self.declare_parameter("place_pose", [0.531, 0.328, 0.085, -179.0, -1.39, -85.18])
        self.declare_parameter("scan_exit_user_xyz", [0.503, 0.121, 0.471])
        self.declare_parameter("user_index", 0)
        self.declare_parameter("flange_tool_index", 0)
        self.declare_parameter("command_tool_index", 1)
        self.declare_parameter("joint_speed", 25)
        self.declare_parameter("joint_acc", 25)
        self.declare_parameter("linear_speed", 12)
        self.declare_parameter("linear_acc", 20)
        self.declare_parameter("jog_speed_factor", 8)
        self.declare_parameter("jog_tolerance_m", 0.002)
        self.declare_parameter("jog_axis_timeout_s", 20.0)
        self.declare_parameter("grasp_lift_m", 0.060)
        # Current real-cell trials show an approximately 10 mm vertical bias
        # between the transformed vision pose and the physical gripper tip.
        # Positive is shallower/safer and remains editable in the GUI.
        self.declare_parameter("grasp_z_offset_m", 0.010)
        self.declare_parameter("grasp_z_offset_limit_m", 0.020)
        self.declare_parameter("minimum_safe_tcp_z_m", 0.010)

        self.declare_parameter("camera_frame_id", "camera_d405_link")
        self.declare_parameter("vision_pose_topic", "/target_pose_cam_fine")
        self.declare_parameter("vision_width_topic", "/gripper_target_width")
        self.declare_parameter("vision_height_topic", "/cosmetic_box_height")
        self.declare_parameter("vision_trigger_topic", "/trigger_d405_vision")
        self.declare_parameter("vision_samples", 3)
        self.declare_parameter("vision_timeout_s", 8.0)
        self.declare_parameter("vision_retry_delay_s", 1.0)
        self.declare_parameter("vision_position_stability_m", 0.010)
        self.declare_parameter("vision_angle_stability_deg", 10.0)
        self.declare_parameter("min_box_height_m", 0.005)
        self.declare_parameter("max_box_height_m", 0.150)
        self.declare_parameter(
            "handeye_flange_to_cam",
            [
                0.99999289, 0.00303007, -0.00224455, -0.01007269571,
                -0.00207268, 0.93892954, 0.34410322, -0.09923380417,
                0.00315013, -0.34409612, 0.93892914, 0.04701274037,
                0.0, 0.0, 0.0, 1.0,
            ],
        )
        self.declare_parameter("grasp_offset_rxyz_deg", [180.0, 0.0, -90.0])

        self.declare_parameter("barcode_topic", "/detected_barcodes")
        # A HID scanner emits one complete decoded string per successful scan;
        # unlike frame-by-frame vision detections it need not be seen 3 times.
        self.declare_parameter("barcode_stable_hits", 1)
        self.declare_parameter("barcode_hit_gap_s", 0.7)
        self.declare_parameter("barcode_face_wait_s", 1.5)
        self.declare_parameter("barcode_max_face_rotations", 4)
        # Match the previous production node exactly: rotate wrist J6 by -90
        # degrees for each next-face search, with a joint-limit guard.
        self.declare_parameter("barcode_flip_step_deg", -90.0)
        self.declare_parameter("barcode_flip_safe_joint_limit_deg", 355.0)
        self.declare_parameter("barcode_flip_watch_joint_index", 5)
        self.declare_parameter("face_up_user_ry_deg", -90.0)
        self.declare_parameter("face_up_jog_tolerance_deg", 2.0)
        self.declare_parameter("face_up_jog_timeout_s", 60.0)
        self.declare_parameter("face_up_fixed_xyz_tolerance_m", 0.003)

        self.declare_parameter("dh_max_opening_m", 0.095)
        self.declare_parameter("dh_force", 30)
        self.declare_parameter("dh_grasp_force", 30)
        self.declare_parameter("dh_slave_id", 1)
        self.declare_parameter("dh_tool_identify", 1)
        self.declare_parameter("dh_timeout_s", 10.0)
        self.declare_parameter("grasp_close_settle_s", 0.8)
        self.declare_parameter("grasp_confirm_samples", 3)
        self.declare_parameter("grasp_confirm_interval_s", 0.15)
        self.declare_parameter("grasp_feedback_wait_s", 1.5)
        self.declare_parameter("single_cycle_grasp_retry_limit", 3)
        self.declare_parameter("grasp_success_min_opening_m", 0.003)
        self.declare_parameter("grasp_feedback_required", True)

        handeye_values = [float(value) for value in self.get_parameter("handeye_flange_to_cam").value]
        if len(handeye_values) != 16:
            raise ValueError("handeye_flange_to_cam must contain 16 values")
        self.handeye_flange_to_cam = np.asarray(handeye_values, dtype=np.float64).reshape(4, 4)

        self.controller = DobotNova5Controller(
            robot_ip=str(self.get_parameter("robot_ip").value),
            dashboard_port=int(self.get_parameter("dashboard_port").value),
            feedback_port=int(self.get_parameter("feedback_port").value),
            startup_joint=self._six_values("startup_joint"),
            startup_speed=int(self.get_parameter("joint_speed").value),
        )
        self.controller.connect(go_to_start=False, auto_enable=bool(self.get_parameter("auto_enable").value))
        self.controller.set_joint_profile(
            speed=int(self.get_parameter("joint_speed").value),
            accel=int(self.get_parameter("joint_acc").value),
        )
        self.controller.set_linear_profile(
            speed=int(self.get_parameter("linear_speed").value),
            accel=int(self.get_parameter("linear_acc").value),
        )
        self.gripper = self._initialize_gripper()

        self.data_lock = threading.Lock()
        self.pose_samples: deque[tuple[int, TcpPose]] = deque(maxlen=20)
        self.pose_count = 0
        self.width_count = 0
        self.height_count = 0
        self.latest_width_m: Optional[float] = None
        self.latest_height_m: Optional[float] = None

        self.barcode_lock = threading.Lock()
        self.barcode_window_active = False
        self.barcode_value = ""
        self.barcode_hits = 0
        self.barcode_last_time = 0.0

        self.running = True
        self.cycle_enabled = False
        self.worker: Optional[threading.Thread] = None
        # Serialize complete robot sequences. Emergency Stop intentionally does
        # not take this lock, so it can interrupt a blocking sequence; recovery
        # to startup waits for that interrupted sequence to unwind before it
        # sends a new movement command.
        self.action_lock = threading.RLock()
        self.last_status = "ready - waiting for operator"
        self.last_accepted_target: Optional[TcpPose] = None
        self.last_accepted_width_m: Optional[float] = None
        self.last_accepted_height_m: Optional[float] = None

        self.create_subscription(PoseStamped, str(self.get_parameter("vision_pose_topic").value), self._vision_pose_callback, 10)
        self.create_subscription(Float32, str(self.get_parameter("vision_width_topic").value), self._vision_width_callback, 10)
        self.create_subscription(Float32, str(self.get_parameter("vision_height_topic").value), self._vision_height_callback, 10)
        self.create_subscription(String, str(self.get_parameter("barcode_topic").value), self._barcode_callback, 20)
        self.create_subscription(Bool, "/cosmetic_pick_cycle_enable", self._cycle_enable_callback, 10)
        self.trigger_publisher = self.create_publisher(Bool, str(self.get_parameter("vision_trigger_topic").value), 10)
        self.status_publisher = self.create_publisher(String, "/cosmetic_pick_cycle_status", 10)

        self.start_timer = self.create_timer(1.0, self._start_automatically_once)
        self.get_logger().warning(
            "Using handeye_flange_to_cam parameter for robot 192.168.111.101 / D405 409122274792; "
            "verify this calibration on the real cell before enabling motion."
        )
        self.get_logger().info("Single-arm cosmetic-box controller connected; no 192.168.111.102 connection is created.")

    def _six_values(self, name: str) -> list[float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != 6:
            raise ValueError(f"{name} must contain 6 values")
        return values

    def _initialize_gripper(self) -> DHGripper:
        if self.controller.dashboard is None:
            raise RuntimeError("Robot dashboard not connected")
        config = DobotDHConfig(
            robot_ip=str(self.get_parameter("robot_ip").value),
            dashboard_port=int(self.get_parameter("dashboard_port").value),
            tool_identify=int(self.get_parameter("dh_tool_identify").value),
            slave_id=int(self.get_parameter("dh_slave_id").value),
            force=int(self.get_parameter("dh_force").value),
            enable_robot=False,
        )
        dashboard = self.controller.dashboard
        mode_response = dashboard.SetToolMode(1, 1, config.tool_identify)
        if "Control Mode Is Not Tcp" not in str(mode_response):
            raise_if_error(mode_response, "SetToolMode")
        rs485_response = dashboard.SetTool485(config.baudrate, config.parity, config.stop_bit, config.tool_identify)
        if "Control Mode Is Not Tcp" not in str(rs485_response):
            raise_if_error(rs485_response, "SetTool485")
        for master_index in range(5):
            try:
                dashboard.ModbusClose(master_index)
            except Exception:
                pass
        gripper = DHGripper(dashboard, config)
        gripper.initialize(timeout_s=float(self.get_parameter("dh_timeout_s").value), init_open=True)
        return gripper

    def _vision_pose_callback(self, msg: PoseStamped) -> None:
        if msg.header.frame_id.strip() != str(self.get_parameter("camera_frame_id").value):
            self.get_logger().error(f"Ignoring vision frame {msg.header.frame_id!r}")
            return
        try:
            flange_pose = self.controller.current_tcp_pose(
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("flange_tool_index").value),
            )
            offset_angles = self._six_values_from_rotation("grasp_offset_rxyz_deg")
            target_to_grasp = np.eye(4, dtype=np.float64)
            target_to_grasp[:3, :3] = SciPyRot.from_euler("xyz", offset_angles, degrees=True).as_matrix()
            base_to_target = pose_to_transform(flange_pose) @ self.handeye_flange_to_cam @ message_to_transform(msg)
            command_pose = transform_to_pose(base_to_target @ target_to_grasp)
        except Exception as exc:
            self.get_logger().error(f"Vision pose transform failed: {exc}")
            return
        with self.data_lock:
            self.pose_count += 1
            self.pose_samples.append((self.pose_count, command_pose))

    def _six_values_from_rotation(self, name: str) -> list[float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != 3:
            raise ValueError(f"{name} must contain 3 values")
        return values

    def _vision_width_callback(self, msg: Float32) -> None:
        with self.data_lock:
            self.latest_width_m = float(msg.data)
            self.width_count += 1

    def _vision_height_callback(self, msg: Float32) -> None:
        with self.data_lock:
            self.latest_height_m = float(msg.data)
            self.height_count += 1

    def _barcode_callback(self, msg: String) -> None:
        value = msg.data.strip()
        if not value:
            return
        now = time.monotonic()
        with self.barcode_lock:
            if not self.barcode_window_active:
                return
            max_gap = float(self.get_parameter("barcode_hit_gap_s").value)
            if value == self.barcode_value and now - self.barcode_last_time <= max_gap:
                self.barcode_hits += 1
            else:
                self.barcode_value = value
                self.barcode_hits = 1
            self.barcode_last_time = now
            hits = self.barcode_hits
        self.get_logger().info(f"Barcode stability: value={value!r}, hits={hits}")

    def _cycle_enable_callback(self, msg: Bool) -> None:
        self.cycle_enabled = bool(msg.data)
        self._publish_status("cycle enabled" if self.cycle_enabled else "cycle will stop after current blocking motion")
        if self.cycle_enabled:
            self._ensure_worker()

    def _start_automatically_once(self) -> None:
        self.start_timer.cancel()
        if bool(self.get_parameter("auto_start").value):
            self.cycle_enabled = True
            self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.worker = threading.Thread(target=self._cycle_worker, daemon=True)
        self.worker.start()

    def _publish_status(self, text: str) -> None:
        self.last_status = text
        msg = String()
        msg.data = text
        self.status_publisher.publish(msg)
        self.get_logger().info(f"[cycle] {text}")

    def _cycle_worker(self) -> None:
        try:
            with self.action_lock:
                self._move_startup_and_open(require_cycle_active=True)
                cycle_index = 0
                while self.running and self.cycle_enabled:
                    cycle_index += 1
                    self._publish_status(f"cycle {cycle_index}: detecting minimum-camera-X box")
                    target_bundle = self._request_vision_target()
                    if target_bundle is None:
                        delay = max(0.1, float(self.get_parameter("vision_retry_delay_s").value))
                        self._publish_status(f"cycle {cycle_index}: no valid box; retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    target_pose, width_m, height_m = target_bundle
                    try:
                        self._execute_one_cycle(target_pose, width_m, height_m)
                    except RecoverableGraspError as exc:
                        self._recover_failed_grasp_for_retry(exc)
                        self._publish_status(
                            f"cycle {cycle_index}: grasp retry recovery completed; "
                            "requesting a fresh D405 target"
                        )
                        continue
                    self._move_startup_and_open(require_cycle_active=True)
            self._publish_status("cycle stopped")
        except Exception as exc:
            self.cycle_enabled = False
            self._publish_status(f"FAULT: {exc}")
            self.get_logger().fatal(f"Automatic cycle stopped safely: {exc}")

    def _move_startup_and_open(self, require_cycle_active: bool = False) -> None:
        self._publish_status("moving to startup joint")
        self.controller.move_joint(self._six_values("startup_joint"), speed=int(self.get_parameter("joint_speed").value))
        if require_cycle_active:
            self._require_cycle_active("at startup joint")
        self.gripper.set_force(int(self.get_parameter("dh_force").value))
        self.gripper.open(wait=True)

    def move_startup(self) -> None:
        """Always recover control, return to startup, and open the gripper."""
        self.cycle_enabled = False
        self._publish_status("return-to-startup request accepted; cancelling the previous action")

        # Stop an active/paused command before waiting for the sequence lock.
        # This is safe to call from the priority GUI worker and lets the older
        # worker leave its blocking command wait promptly.
        mode = self.controller.robot_mode
        if mode in (7, 8, 10):
            try:
                self.controller.stop_motion()
            except Exception as exc:
                self.get_logger().warning(f"Stop before startup recovery reported: {exc}")

        self._publish_status("waiting for the previous action to release robot control")
        with self.action_lock:
            self._prepare_robot_for_startup_recovery()
            self._move_startup_and_open(require_cycle_active=False)
        self._publish_status("startup reached and gripper opened")

    def _prepare_robot_for_startup_recovery(self) -> None:
        mode = self.controller.robot_mode
        if mode in (9, 11):
            self._publish_status(
                f"clearing robot {self.controller.robot_mode_text()} state before startup recovery"
            )
            self.controller.clear_error()
            deadline = time.monotonic() + 10.0
            while self.controller.robot_mode in (9, 11) and time.monotonic() < deadline:
                time.sleep(0.05)
            mode = self.controller.robot_mode

        if mode == 10:
            self.controller.stop_motion()
            mode = self.controller.robot_mode

        if mode in (7, 8):
            self._publish_status("waiting for the stopped robot to become idle")
            try:
                self.controller.wait_until_idle(timeout_s=5.0)
            except TimeoutError:
                self.controller.stop_motion()
                self.controller.wait_until_idle(timeout_s=10.0)
            mode = self.controller.robot_mode

        if mode != 5:
            self._publish_status(
                f"enabling robot from mode {self.controller.robot_mode_text()} for startup recovery"
            )
            self.controller.enable_robot()
        self.controller.wait_until_idle(timeout_s=10.0)

    def open_gripper(self) -> None:
        self.gripper.open(wait=True)
        self._publish_status("gripper opened")

    def close_gripper(self) -> None:
        self.gripper.set_force(int(self.get_parameter("dh_grasp_force").value))
        self.gripper.close(wait=True)
        position_m = self.gripper.read_position() * float(self.get_parameter("dh_max_opening_m").value)
        state = self.gripper.read_grip_state()
        self._publish_status(f"gripper closed: state={state}, opening={position_m*1000:.1f}mm")

    def enable_robot(self) -> None:
        self.controller.enable_robot()
        self._publish_status("robot enabled")

    def clear_robot_error(self) -> None:
        self.controller.clear_error()
        self._publish_status("robot error cleared")

    def stop_robot(self) -> None:
        self.cycle_enabled = False
        self.controller.stop_motion()
        self._publish_status("robot stopped; continuous cycle disabled")

    def start_continuous_cycle(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("cycle worker is already running")
        self.cycle_enabled = True
        self._ensure_worker()
        self._publish_status("continuous cycle started")

    def stop_continuous_cycle(self) -> None:
        self.cycle_enabled = False
        self._publish_status("continuous cycle will stop after the current blocking motion")

    def sample_vision_only(self) -> tuple[TcpPose, float, float]:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("continuous cycle is running")
        result = self._request_vision_target(require_cycle_enabled=False)
        if result is None:
            raise RuntimeError("no stable D405 target received")
        target, width_m, height_m = result
        self._publish_status(
            f"vision sampled: xyz=({target.x:.3f},{target.y:.3f},{target.z:.3f})m "
            f"height={height_m*1000:.1f}mm width={width_m*1000:.1f}mm"
        )
        return result

    def execute_single_cycle(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("continuous cycle is running")
        self.cycle_enabled = True
        try:
            with self.action_lock:
                self._move_startup_and_open(require_cycle_active=True)
                retry_limit = max(
                    0, int(self.get_parameter("single_cycle_grasp_retry_limit").value)
                )
                grasp_failures = 0
                while True:
                    result = self._request_vision_target()
                    if result is None:
                        raise RuntimeError("no stable D405 target received")
                    try:
                        self._execute_one_cycle(*result)
                        break
                    except RecoverableGraspError as exc:
                        grasp_failures += 1
                        self._recover_failed_grasp_for_retry(exc)
                        if grasp_failures > retry_limit:
                            raise RuntimeError(
                                f"Grasp failed after {grasp_failures} attempts; "
                                "robot recovered to startup"
                            ) from exc
                        self._publish_status(
                            f"single cycle grasp retry {grasp_failures}/{retry_limit}: "
                            "requesting a fresh D405 target"
                        )
                self._move_startup_and_open(require_cycle_active=True)
                self._publish_status("single cycle completed")
        finally:
            self.cycle_enabled = False

    def _request_vision_target(self, require_cycle_enabled: bool = True) -> Optional[tuple[TcpPose, float, float]]:
        with self.data_lock:
            previous_pose = self.pose_count
            previous_width = self.width_count
            previous_height = self.height_count
        trigger = Bool()
        trigger.data = True
        self.trigger_publisher.publish(trigger)
        required_samples = max(1, int(self.get_parameter("vision_samples").value))
        deadline = time.monotonic() + max(0.1, float(self.get_parameter("vision_timeout_s").value))
        while (
            self.running
            and (self.cycle_enabled or not require_cycle_enabled)
            and time.monotonic() < deadline
        ):
            with self.data_lock:
                ready = (
                    self.pose_count >= previous_pose + required_samples
                    and self.width_count > previous_width
                    and self.height_count > previous_height
                )
                if ready:
                    samples = [pose for count, pose in self.pose_samples if count > previous_pose][-required_samples:]
                    width_m = self.latest_width_m
                    height_m = self.latest_height_m
                    break
            time.sleep(0.02)
        else:
            return None
        if len(samples) != required_samples or width_m is None or height_m is None:
            return None
        min_height = float(self.get_parameter("min_box_height_m").value)
        max_height = float(self.get_parameter("max_box_height_m").value)
        if not min_height <= height_m <= max_height:
            self.get_logger().error(f"Vision height {height_m:.4f}m outside [{min_height:.4f}, {max_height:.4f}]")
            return None
        averaged = TcpPose(
            x=float(np.mean([pose.x for pose in samples])),
            y=float(np.mean([pose.y for pose in samples])),
            z=float(np.mean([pose.z for pose in samples])),
            rx=circular_mean([pose.rx for pose in samples]),
            ry=circular_mean([pose.ry for pose in samples]),
            rz=circular_mean([pose.rz for pose in samples]),
        )
        mean_position = np.array([averaged.x, averaged.y, averaged.z], dtype=np.float64)
        position_spread = max(
            float(np.linalg.norm(np.array([pose.x, pose.y, pose.z]) - mean_position))
            for pose in samples
        )
        angle_spread = max(
            abs((value - mean_value + 180.0) % 360.0 - 180.0)
            for pose in samples
            for value, mean_value in (
                (pose.rx, averaged.rx),
                (pose.ry, averaged.ry),
                (pose.rz, averaged.rz),
            )
        )
        if position_spread > float(self.get_parameter("vision_position_stability_m").value):
            self.get_logger().warning(f"Vision sample position spread {position_spread:.4f}m is unstable")
            return None
        if angle_spread > float(self.get_parameter("vision_angle_stability_deg").value):
            self.get_logger().warning(f"Vision sample angle spread {angle_spread:.2f}deg is unstable")
            return None
        self.get_logger().info(
            f"Accepted 75%-depth target: x={averaged.x:.3f} y={averaged.y:.3f} z={averaged.z:.3f}, "
            f"height={height_m*1000:.1f}mm width_command={width_m*1000:.1f}mm"
        )
        self.last_accepted_target = averaged
        self.last_accepted_width_m = width_m
        self.last_accepted_height_m = height_m
        return averaged, width_m, height_m

    def _execute_one_cycle(self, target: TcpPose, width_m: float, height_m: float) -> None:
        z_offset_m = float(self.get_parameter("grasp_z_offset_m").value)
        z_offset_limit_m = abs(float(self.get_parameter("grasp_z_offset_limit_m").value))
        if abs(z_offset_m) > z_offset_limit_m:
            raise RuntimeError(
                f"grasp_z_offset_m={z_offset_m:.4f} exceeds limit {z_offset_limit_m:.4f}m"
            )
        vision_target_z = target.z
        corrected_z = target.z + z_offset_m
        minimum_safe_z = float(self.get_parameter("minimum_safe_tcp_z_m").value)
        if corrected_z < minimum_safe_z:
            self.get_logger().warning(
                f"Clamping grasp TCP Z from {corrected_z:.4f}m to safe floor {minimum_safe_z:.4f}m"
            )
            corrected_z = minimum_safe_z
        target = TcpPose(target.x, target.y, corrected_z, target.rx, target.ry, target.rz)
        self._publish_status(
            f"grasp plan: vision_Z={vision_target_z*1000:.1f}mm, "
            f"Z_correction={z_offset_m*1000:+.1f}mm, command_Z={target.z*1000:.1f}mm, "
            f"box_height={height_m*1000:.1f}mm"
        )
        self._publish_status("opening gripper to measured box width")
        max_opening = float(self.get_parameter("dh_max_opening_m").value)
        width_m = max(0.0, min(max_opening, width_m))
        self.gripper.set_position(width_m / max_opening, wait=True)
        self._require_cycle_active("after pre-opening gripper")

        self.controller.inverse_kinematics(
            target,
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
            joint_near=self.controller.current_joint(),
        )
        current = self._current_command_pose()
        planar = TcpPose(target.x, target.y, current.z, target.rx, target.ry, target.rz)
        self._publish_status("moving above selected box")
        self.controller.move_joint_tcp(
            planar,
            speed=int(self.get_parameter("joint_speed").value),
            accel=int(self.get_parameter("joint_acc").value),
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )
        self._require_cycle_active("after move-above")
        self._publish_status("descending TCP tip to 75% box height")
        self.controller.move_linear_tcp(
            target,
            speed=int(self.get_parameter("linear_speed").value),
            accel=int(self.get_parameter("linear_acc").value),
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )
        self._require_cycle_active("at grasp depth")
        self._publish_status("at grasp depth; closing gripper now")
        self.gripper.set_force(int(self.get_parameter("dh_grasp_force").value))
        self.gripper.close(wait=True)
        self._require_cycle_active("after gripper close")
        self._confirm_grasp_before_lift(max_opening)

        self._relative_user_move(z=float(self.get_parameter("grasp_lift_m").value), label="lifting grasp")
        self._require_cycle_active("after grasp lift")
        self._validate_grasp_feedback("after lift", max_opening)

        # Start listening before moving to the transfer joint because the HID
        # scanner can decode the label as soon as it enters the field of view.
        self._reset_barcode_window()
        self._publish_status("moving to barcode transfer joint; barcode window armed")
        self.controller.move_joint(self._six_values("transfer_joint"), speed=int(self.get_parameter("joint_speed").value))
        self._require_cycle_active("at barcode transfer joint")
        barcode = self._rotate_until_stable_barcode()
        self._publish_status(f"stable barcode acquired: {barcode}")

        self._move_to_user_xyz([float(value) for value in self.get_parameter("scan_exit_user_xyz").value])
        self._require_cycle_active("after user-frame XYZ move")
        face_up_delta = float(self.get_parameter("face_up_user_ry_deg").value)
        self._rotate_face_up_about_user_y_jog(face_up_delta)
        self._require_cycle_active("after upward flip")
        self._validate_grasp_feedback(
            "after barcode face-up rotation",
            float(self.get_parameter("dh_max_opening_m").value),
        )

        place = self._six_values("place_pose")
        place_pose = TcpPose(*place)
        self._publish_status("moving to final place pose")
        self.controller.move_joint_tcp(
            place_pose,
            speed=int(self.get_parameter("joint_speed").value),
            accel=int(self.get_parameter("joint_acc").value),
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )
        self._require_cycle_active("at final place pose")
        self.gripper.open(wait=True)
        self._publish_status("placed box; returning to startup")

    def _require_cycle_active(self, stage: str) -> None:
        if not self.running or not self.cycle_enabled:
            raise RuntimeError(f"Cycle cancelled by operator at {stage}; no subsequent motion was issued")

    @staticmethod
    def _grip_state_text(state: int) -> str:
        return {
            GRIP_IN_MOTION: "moving",
            GRIP_REACHED: "target reached without object",
            GRIP_GRIPPED: "object gripped",
            GRIP_DROPPED: "object dropped/lost",
        }.get(state, "unknown")

    def _recover_failed_grasp_for_retry(self, exc: RecoverableGraspError) -> None:
        self._require_cycle_active("before recoverable grasp retry")
        self._publish_status(
            f"recoverable grasp failure at {exc.stage}: {exc}; returning to startup"
        )
        if exc.needs_vertical_retreat:
            self._relative_user_move(
                z=float(self.get_parameter("grasp_lift_m").value),
                label="empty grasp at lowered TCP; retreating vertically before startup",
            )
            self._require_cycle_active("after empty-grasp vertical retreat")
        self._move_startup_and_open(require_cycle_active=True)

    def _validate_grasp_feedback(self, stage: str, max_opening: float) -> None:
        opening_m = self.gripper.read_position() * max_opening
        grip_state = self.gripper.read_grip_state()
        if grip_state == GRIP_IN_MOTION:
            wait_s = max(0.1, float(self.get_parameter("grasp_feedback_wait_s").value))
            self._publish_status(
                f"gripper still moving {stage}; waiting up to {wait_s:.1f}s for terminal feedback"
            )
            try:
                self.gripper.wait_until_stopped(
                    timeout_s=wait_s,
                    target_position=0.0,
                    initial_position=opening_m / max(1e-9, max_opening),
                )
            except TimeoutError:
                pass
            opening_m = self.gripper.read_position() * max_opening
            grip_state = self.gripper.read_grip_state()
        state_text = self._grip_state_text(grip_state)
        self._publish_status(
            f"gripper feedback {stage}: state={grip_state} ({state_text}), "
            f"opening={opening_m*1000:.1f}mm"
        )
        feedback_failed = (
            grip_state != GRIP_GRIPPED
            or opening_m <= float(self.get_parameter("grasp_success_min_opening_m").value)
        )
        if feedback_failed and bool(self.get_parameter("grasp_feedback_required").value):
            raise RecoverableGraspError(
                stage=stage,
                message=(
                    f"state={grip_state} ({state_text}), opening={opening_m:.4f}m"
                ),
                needs_vertical_retreat="before lift" in stage,
            )
        if feedback_failed:
            self.get_logger().warning(
                f"Gripper feedback bypassed by operator setting {stage}: "
                f"state={grip_state} ({state_text}), opening={opening_m:.4f}m"
            )

    def _confirm_grasp_before_lift(self, max_opening: float) -> None:
        """Keep the TCP stationary until the DH gripper is stably holding an object."""
        settle_s = max(0.0, float(self.get_parameter("grasp_close_settle_s").value))
        required = max(1, int(self.get_parameter("grasp_confirm_samples").value))
        interval_s = max(0.05, float(self.get_parameter("grasp_confirm_interval_s").value))
        if settle_s > 0.0:
            self._publish_status(
                f"gripper close command completed; holding grasp pose for {settle_s:.2f}s"
            )
            deadline = time.monotonic() + settle_s
            while time.monotonic() < deadline:
                self._require_cycle_active("while gripper settles before lift")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        minimum_opening = float(self.get_parameter("grasp_success_min_opening_m").value)
        confirmed = 0
        last_state = GRIP_IN_MOTION
        last_opening_m = 0.0
        for sample_index in range(required):
            self._require_cycle_active("while confirming grasp before lift")
            last_opening_m = self.gripper.read_position() * max_opening
            last_state = self.gripper.read_grip_state()
            state_text = self._grip_state_text(last_state)
            self._publish_status(
                f"grasp confirmation {sample_index + 1}/{required}: "
                f"state={last_state} ({state_text}), opening={last_opening_m*1000:.1f}mm"
            )
            if last_state == GRIP_GRIPPED and last_opening_m > minimum_opening:
                confirmed += 1
            else:
                confirmed = 0
            if sample_index + 1 < required:
                time.sleep(interval_s)

        if confirmed == required:
            self._publish_status("grasp stable at lowered TCP; lifting is now permitted")
            return

        detail = (
            f"Grasp not stable before lift: confirmed={confirmed}/{required}, "
            f"state={last_state} ({self._grip_state_text(last_state)}), "
            f"opening={last_opening_m:.4f}m; TCP remains at grasp depth"
        )
        if bool(self.get_parameter("grasp_feedback_required").value):
            raise RecoverableGraspError(
                stage="before lift",
                message=detail,
                needs_vertical_retreat=True,
            )
        self.get_logger().warning(f"{detail}; bypassed by operator setting")

    def _current_command_pose(self) -> TcpPose:
        return self.controller.current_tcp_pose(
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )

    def _relative_user_move(self, x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0, label="relative user move") -> None:
        self._publish_status(label)
        self.controller.rel_move_user_joint(
            TcpPose(float(x), float(y), float(z), float(rx), float(ry), float(rz)),
            speed=int(self.get_parameter("joint_speed").value),
            accel=int(self.get_parameter("joint_acc").value),
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )

    def _reset_barcode_window(self) -> None:
        with self.barcode_lock:
            self.barcode_window_active = True
            self.barcode_value = ""
            self.barcode_hits = 0
            self.barcode_last_time = 0.0

    def _is_barcode_flip_joint_safe(self, joints_deg: Optional[list[float]], delta_deg: float) -> bool:
        if joints_deg is None or len(joints_deg) != 6:
            return True
        safe_limit = abs(float(self.get_parameter("barcode_flip_safe_joint_limit_deg").value))
        watch_index = max(0, min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)))
        predicted = float(joints_deg[watch_index]) + float(delta_deg)
        if abs(predicted) > safe_limit:
            self.get_logger().warning(
                f"Barcode flip {delta_deg:+.1f}deg rejected: J{watch_index + 1} "
                f"current={joints_deg[watch_index]:.1f}, predicted={predicted:.1f}, "
                f"safe_limit=+/-{safe_limit:.1f}deg"
            )
            return False
        return True

    def _select_safe_barcode_flip_delta(self, preferred_delta_deg: float) -> Optional[float]:
        joints = self.controller.current_joint()
        winding_sign = math.copysign(1.0, preferred_delta_deg)
        candidates = [
            preferred_delta_deg,
            preferred_delta_deg - winding_sign * 360.0,
            preferred_delta_deg + winding_sign * 360.0,
            preferred_delta_deg - winding_sign * 720.0,
            preferred_delta_deg + winding_sign * 720.0,
        ]
        for delta in candidates:
            if abs(delta) > 1e-6 and self._is_barcode_flip_joint_safe(joints, delta):
                if abs(delta - preferred_delta_deg) > 1e-6:
                    self.get_logger().warning(
                        f"Using equivalent J6 barcode rotation {delta:+.1f}deg instead of "
                        f"{preferred_delta_deg:+.1f}deg to stay inside the joint limit"
                    )
                return float(delta)
        return None

    def _rotate_barcode_flip_joint(self, delta_deg: float, face_index: int) -> None:
        current_joints = self.controller.current_joint()
        tcp_before = self._current_command_pose()
        if len(current_joints) != 6:
            raise RuntimeError(f"Current joint feedback must contain 6 values, got {len(current_joints)}")
        watch_index = max(0, min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)))
        if not self._is_barcode_flip_joint_safe(current_joints, delta_deg):
            raise RuntimeError(
                f"Unsafe barcode J{watch_index + 1} rotation: "
                f"current={current_joints[watch_index]:.1f}, delta={delta_deg:+.1f}deg"
            )
        target_joints = [float(value) for value in current_joints]
        target_joints[watch_index] += float(delta_deg)
        self._publish_status(
            f"barcode face {face_index}: old-logic J{watch_index + 1} "
            f"{current_joints[watch_index]:.1f}->{target_joints[watch_index]:.1f}deg"
        )
        self.controller.move_joint(target_joints, speed=int(self.get_parameter("joint_speed").value))
        self._require_cycle_active(f"after barcode J{watch_index + 1} rotation")
        tcp_after = self._current_command_pose()
        xyz_shift_mm = 1000.0 * math.sqrt(
            (tcp_after.x - tcp_before.x) ** 2
            + (tcp_after.y - tcp_before.y) ** 2
            + (tcp_after.z - tcp_before.z) ** 2
        )
        self.get_logger().info(
            f"Legacy barcode rotation completed: only J{watch_index + 1} commanded, "
            f"TCP XYZ shift={xyz_shift_mm:.2f}mm"
        )
        self._validate_grasp_feedback(
            f"after barcode face {face_index}",
            float(self.get_parameter("dh_max_opening_m").value),
        )

    def _wait_for_current_barcode(self, required_hits: int, timeout_s: float, stage: str) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._require_cycle_active(stage)
            with self.barcode_lock:
                if self.barcode_hits >= required_hits:
                    return self.barcode_value
            time.sleep(0.02)
        return ""

    def _rotate_until_stable_barcode(self) -> str:
        required_hits = max(1, int(self.get_parameter("barcode_stable_hits").value))
        # Preserve the previous production sequence exactly: inspect the face
        # already presented at the transfer joint, then make at most three J6
        # quarter turns.  Keeping this fixed at four faces also prevents an
        # accidental mouse-wheel change in the GUI from shortening the search.
        max_faces = 4
        wait_s = max(0.1, float(self.get_parameter("barcode_face_wait_s").value))
        preferred_delta = float(self.get_parameter("barcode_flip_step_deg").value)
        try:
            with self.barcode_lock:
                window_active = self.barcode_window_active
            if not window_active:
                self._reset_barcode_window()

            # Check the current face first, including a code read while moving
            # into the transfer joint, exactly as the previous search loop did.
            value = self._wait_for_current_barcode(required_hits, wait_s, "checking barcode at transfer joint")
            if value:
                self._publish_status(f"barcode already acquired at transfer joint: {value}")
                return value

            # max_faces includes the initial unrotated face, hence at most
            # max_faces - 1 wrist rotations, matching the old max_steps=3 loop.
            for flip_index in range(1, max_faces):
                self._require_cycle_active(f"before barcode face rotation {flip_index}")
                self._reset_barcode_window()
                selected_delta = self._select_safe_barcode_flip_delta(preferred_delta)
                if selected_delta is None:
                    raise RuntimeError("No safe equivalent J6 barcode rotation is available")
                self._rotate_barcode_flip_joint(selected_delta, flip_index + 1)
                value = self._wait_for_current_barcode(
                    required_hits,
                    wait_s,
                    f"checking barcode after J6 rotation {flip_index}",
                )
                if value:
                    return value
            raise RuntimeError(f"Barcode not stable after checking {max_faces} faces; object retained at transfer point")
        finally:
            with self.barcode_lock:
                self.barcode_window_active = False

    def _move_to_user_xyz(self, target_xyz: list[float]) -> None:
        if len(target_xyz) != 3:
            raise ValueError("scan_exit_user_xyz must contain 3 values")
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        current = self._current_command_pose()
        target = TcpPose(
            float(target_xyz[0]),
            float(target_xyz[1]),
            float(target_xyz[2]),
            current.rx,
            current.ry,
            current.rz,
        )
        self._publish_status(
            f"user-frame PTP XYZ to {target.x*1000:.0f}, {target.y*1000:.0f}, {target.z*1000:.0f} mm; "
            f"holding orientation ({target.rx:.1f},{target.ry:.1f},{target.rz:.1f})deg"
        )

        # Validate the complete destination before moving. Sequential MoveJog
        # reached X first but the controller then rejected Y+ (-1), because that
        # intermediate Cartesian state had no accepted continuation. A single
        # User/Tool-aware PTP command avoids that artificial intermediate state.
        self.controller.inverse_kinematics(
            target,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=self.controller.current_joint(),
        )
        self._require_cycle_active("before user-frame XYZ PTP")
        self.controller.move_joint_tcp(
            target,
            speed=int(self.get_parameter("jog_speed_factor").value),
            accel=int(self.get_parameter("joint_acc").value),
            user_index=user_index,
            tool_index=tool_index,
        )
        self._require_cycle_active("at user-frame XYZ target")

        tolerance = max(0.0005, float(self.get_parameter("jog_tolerance_m").value))
        final = self._current_command_pose()
        errors = [abs(final.x - target_xyz[0]), abs(final.y - target_xyz[1]), abs(final.z - target_xyz[2])]
        if max(errors) > tolerance * 1.5:
            raise RuntimeError(f"User-frame PTP final XYZ error too large: {[round(e, 4) for e in errors]}m")
        self._publish_status(
            f"user-frame XYZ reached: {final.x*1000:.1f}, {final.y*1000:.1f}, {final.z*1000:.1f} mm"
        )

    def _rotate_face_up_about_user_y_jog(self, total_delta_deg: float) -> None:
        if abs(total_delta_deg) <= 1e-6:
            return
        # Restore the previous production mechanism: continuously jog the
        # rotational axis in the selected user frame. With User 0 this is Ry-
        # about the base frame and the controller itself keeps TCP XYZ fixed.
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        before_pose = self.controller.current_tcp_pose(user_index=user_index, tool_index=tool_index)
        fixed_xyz = np.array([before_pose.x, before_pose.y, before_pose.z], dtype=np.float64)
        start_rotation = SciPyRot.from_euler(
            "xyz", [before_pose.rx, before_pose.ry, before_pose.rz], degrees=True
        ).as_matrix()
        target_deg = abs(float(total_delta_deg))
        direction = 1.0 if total_delta_deg > 0.0 else -1.0
        axis_name = "Ry+" if direction > 0.0 else "Ry-"
        angle_tolerance_deg = max(
            0.2,
            abs(float(self.get_parameter("face_up_jog_tolerance_deg").value)),
        )
        timeout_s = max(1.0, float(self.get_parameter("face_up_jog_timeout_s").value))
        tolerance_m = max(
            0.0005,
            float(self.get_parameter("face_up_fixed_xyz_tolerance_m").value),
        )

        self._publish_status(
            f"user-frame {axis_name} MoveJog to {total_delta_deg:+.1f}deg with fixed TCP XYZ; "
            f"User={user_index}, Tool={tool_index}, speed_factor="
            f"{int(self.get_parameter('joint_speed').value)}%, XYZ=({fixed_xyz[0]*1000:.1f},"
            f"{fixed_xyz[1]*1000:.1f},{fixed_xyz[2]*1000:.1f})mm"
        )
        jog_started = False
        directed_progress_deg = 0.0
        start_time = time.monotonic()
        last_log_progress = -10.0
        try:
            # The previous working implementation inherited the joint-motion
            # speed factor at this point.  Do not use jog_speed_factor here:
            # that GUI value belongs to the slow XYZ PTP move and made a 90
            # degree rotation exceed the old 25 second timeout.
            self.controller.set_speed_factor(int(self.get_parameter("joint_speed").value))
            self.controller.move_jog(axis_name, coord_type=1, user=user_index, tool=tool_index)
            jog_started = True
            while True:
                self._require_cycle_active("during user-frame Ry face-up jog")
                if time.monotonic() - start_time > timeout_s:
                    raise RuntimeError(
                        f"User-frame {axis_name} MoveJog timed out: "
                        f"progress={directed_progress_deg:.1f}/{target_deg:.1f}deg"
                    )
                current = self.controller.current_tcp_pose(user_index=user_index, tool_index=tool_index)
                current_rotation = SciPyRot.from_euler(
                    "xyz", [current.rx, current.ry, current.rz], degrees=True
                ).as_matrix()
                user_delta_rotation = current_rotation @ start_rotation.T
                signed_y_deg = math.degrees(
                    math.atan2(user_delta_rotation[0, 2], user_delta_rotation[0, 0])
                )
                directed_progress_deg = direction * signed_y_deg
                if directed_progress_deg - last_log_progress >= 10.0:
                    self.get_logger().info(
                        f"User-frame {axis_name} MoveJog: "
                        f"progress={directed_progress_deg:.1f}/{target_deg:.1f}deg, "
                        f"rpy=({current.rx:.1f},{current.ry:.1f},{current.rz:.1f})deg"
                    )
                    last_log_progress = directed_progress_deg
                if directed_progress_deg >= target_deg - angle_tolerance_deg:
                    break
                time.sleep(0.02)
        finally:
            if jog_started:
                try:
                    self.controller.move_jog("")
                except Exception as stop_exc:
                    self.get_logger().warning(f"Stopping user-frame Ry MoveJog returned: {stop_exc}")
            try:
                self.controller.set_speed_factor(int(self.get_parameter("joint_speed").value))
            except Exception as speed_exc:
                self.get_logger().warning(f"Restoring robot speed factor failed: {speed_exc}")

        self.controller.wait_until_idle(timeout_s=5.0)
        time.sleep(0.3)
        after_pose = self.controller.current_tcp_pose(user_index=user_index, tool_index=tool_index)
        after_xyz = np.array([after_pose.x, after_pose.y, after_pose.z], dtype=np.float64)
        drift_m = float(np.linalg.norm(after_xyz - fixed_xyz))
        after_rotation = SciPyRot.from_euler(
            "xyz", [after_pose.rx, after_pose.ry, after_pose.rz], degrees=True
        ).as_matrix()
        after_user_delta = after_rotation @ start_rotation.T
        final_signed_y_deg = math.degrees(
            math.atan2(after_user_delta[0, 2], after_user_delta[0, 0])
        )
        final_progress_deg = direction * final_signed_y_deg
        if final_progress_deg < target_deg - max(5.0, angle_tolerance_deg):
            raise RuntimeError(
                f"User-frame {axis_name} rotation insufficient: "
                f"progress={final_progress_deg:.1f}/{target_deg:.1f}deg"
            )
        if drift_m > tolerance_m:
            raise RuntimeError(
                f"User-frame {axis_name} rotation exceeded fixed-TCP tolerance: "
                f"drift={drift_m*1000:.2f}mm > {tolerance_m*1000:.2f}mm"
            )
        self._publish_status(
            f"user-frame {axis_name} face-up jog completed: "
            f"progress={final_progress_deg:.1f}deg, TCP XYZ drift={drift_m*1000:.2f}mm"
        )

    def destroy_node(self):
        self.running = False
        self.cycle_enabled = False
        try:
            # Close Modbus without issuing DHGripper.disconnect(), which would
            # open the gripper and could drop an object during a fault shutdown.
            if self.gripper.modbus_index is not None and self.controller.dashboard is not None:
                try:
                    self.controller.dashboard.ModbusClose(self.gripper.modbus_index)
                finally:
                    self.gripper.modbus_index = None
            self.controller.disconnect()
        finally:
            super().destroy_node()


class WorkerThread(QtCore.QThread):
    finished = Signal(bool, str, object)

    def __init__(self, function, *args):
        super().__init__()
        self.function = function
        self.args = args

    def run(self) -> None:
        try:
            result = self.function(*self.args)
        except Exception as exc:
            self.finished.emit(False, str(exc), None)
            return
        self.finished.emit(True, "OK", result)


class CosmeticBoxControlWindow(QMainWindow):
    def __init__(self, node: CosmeticBoxSingleArmNode):
        super().__init__()
        self.node = node
        self.worker: Optional[WorkerThread] = None
        self.recovery_worker: Optional[WorkerThread] = None
        self.running_workers: set[WorkerThread] = set()
        self.setWindowTitle("Nova5 101 化妆品盒抓取 / 扫码 / 放置")
        self.resize(1050, 820)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QHBoxLayout(content)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)

        self.joint_fields: dict[str, list[QDoubleSpinBox]] = {}
        self.pose_fields: dict[str, list[QDoubleSpinBox]] = {}
        self._build_status(left)
        self._build_joint_group(left, "startup_joint", "初始关节角度 (deg)", self.node._six_values("startup_joint"))
        self._build_joint_group(left, "transfer_joint", "中转检查点关节角度 (deg)", self.node._six_values("transfer_joint"))
        self._build_scan_xyz(left)
        self._build_place_pose(left)
        self._build_motion_parameters(right)
        self._build_barcode_parameters(right)
        self._build_actions(right)
        right.addStretch(1)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(500)
        self.refresh_status()

    def _build_status(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("运行状态")
        form = QFormLayout(box)
        self.robot_mode_label = QLabel("-")
        self.joint_feedback_label = QLabel("-")
        self.tcp_feedback_label = QLabel("-")
        self.vision_feedback_label = QLabel("尚未采样")
        self.cycle_status_label = QLabel("ready")
        self.cycle_status_label.setWordWrap(True)
        form.addRow("机械臂", QLabel("192.168.111.101（单臂）"))
        form.addRow("模式", self.robot_mode_label)
        form.addRow("当前关节", self.joint_feedback_label)
        form.addRow("Tool1 TCP", self.tcp_feedback_label)
        form.addRow("最新视觉", self.vision_feedback_label)
        form.addRow("流程", self.cycle_status_label)
        layout.addWidget(box)

    def _new_double(self, value: float, minimum=-360.0, maximum=360.0, decimals=3, step=1.0) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setSingleStep(step)
        field.setValue(float(value))
        return field

    def _build_joint_group(self, layout: QVBoxLayout, key: str, title: str, values: list[float]) -> None:
        box = QGroupBox(title)
        grid = QGridLayout(box)
        fields = []
        for index, value in enumerate(values):
            grid.addWidget(QLabel(f"J{index + 1}"), index // 3 * 2, index % 3)
            field = self._new_double(value)
            grid.addWidget(field, index // 3 * 2 + 1, index % 3)
            fields.append(field)
        self.joint_fields[key] = fields
        layout.addWidget(box)

    def _build_scan_xyz(self, layout: QVBoxLayout) -> None:
        values_m = [float(value) for value in self.node.get_parameter("scan_exit_user_xyz").value]
        box = QGroupBox("扫码后用户坐标 PTP 目标 (mm)")
        grid = QGridLayout(box)
        fields = []
        for index, (axis, value_m) in enumerate(zip(("X", "Y", "Z"), values_m)):
            grid.addWidget(QLabel(axis), 0, index)
            field = self._new_double(value_m * 1000.0, -2000.0, 2000.0, 1, 1.0)
            grid.addWidget(field, 1, index)
            fields.append(field)
        self.pose_fields["scan_exit_user_xyz"] = fields
        layout.addWidget(box)

    def _build_place_pose(self, layout: QVBoxLayout) -> None:
        values = self.node._six_values("place_pose")
        box = QGroupBox("最终放置位姿（XYZ mm，角度 deg）")
        grid = QGridLayout(box)
        fields = []
        for index, axis in enumerate(("X", "Y", "Z", "Rx", "Ry", "Rz")):
            grid.addWidget(QLabel(axis), index // 3 * 2, index % 3)
            display_value = values[index] * 1000.0 if index < 3 else values[index]
            field = self._new_double(display_value, -2000.0 if index < 3 else -360.0, 2000.0 if index < 3 else 360.0, 2, 1.0)
            grid.addWidget(field, index // 3 * 2 + 1, index % 3)
            fields.append(field)
        self.pose_fields["place_pose"] = fields
        layout.addWidget(box)

    def _build_motion_parameters(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("抓取与运动参数")
        form = QFormLayout(box)
        self.joint_speed = QSpinBox(); self.joint_speed.setRange(1, 100); self.joint_speed.setValue(int(self.node.get_parameter("joint_speed").value))
        self.linear_speed = QSpinBox(); self.linear_speed.setRange(1, 100); self.linear_speed.setValue(int(self.node.get_parameter("linear_speed").value))
        self.jog_speed = QSpinBox(); self.jog_speed.setRange(1, 100); self.jog_speed.setValue(int(self.node.get_parameter("jog_speed_factor").value))
        self.gripper_force = QSpinBox(); self.gripper_force.setRange(20, 100); self.gripper_force.setValue(int(self.node.get_parameter("dh_grasp_force").value))
        self.grasp_z_offset = self._new_double(float(self.node.get_parameter("grasp_z_offset_m").value) * 1000.0, -20.0, 20.0, 1, 0.5)
        self.minimum_safe_tcp_z = self._new_double(float(self.node.get_parameter("minimum_safe_tcp_z_m").value) * 1000.0, -50.0, 100.0, 1, 0.5)
        self.grasp_lift = self._new_double(float(self.node.get_parameter("grasp_lift_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        self.grasp_close_settle = self._new_double(float(self.node.get_parameter("grasp_close_settle_s").value), 0.0, 5.0, 2, 0.1)
        self.grasp_feedback_wait = self._new_double(float(self.node.get_parameter("grasp_feedback_wait_s").value), 0.1, 5.0, 2, 0.1)
        self.grasp_retry_limit = QSpinBox(); self.grasp_retry_limit.setRange(0, 20); self.grasp_retry_limit.setValue(int(self.node.get_parameter("single_cycle_grasp_retry_limit").value))
        self.feedback_required = QCheckBox("启用空抓检测与自动重试")
        self.feedback_required.setChecked(bool(self.node.get_parameter("grasp_feedback_required").value))
        form.addRow("关节速度 %", self.joint_speed)
        form.addRow("直线速度 %", self.linear_speed)
        form.addRow("用户 XYZ PTP 速度 %", self.jog_speed)
        form.addRow("夹爪抓取力", self.gripper_force)
        form.addRow("抓取 Z 修正 mm（正值更浅）", self.grasp_z_offset)
        form.addRow("TCP 最低安全 Z mm", self.minimum_safe_tcp_z)
        form.addRow("抓取后抬升 mm", self.grasp_lift)
        form.addRow("闭合后原地确认秒", self.grasp_close_settle)
        form.addRow("运动状态反馈等待秒", self.grasp_feedback_wait)
        form.addRow("单轮空抓重试次数", self.grasp_retry_limit)
        form.addRow("夹爪反馈保护", self.feedback_required)
        self.apply_button = QPushButton("应用以上全部参数")
        self.apply_button.clicked.connect(self.apply_parameters)
        form.addRow(self.apply_button)
        layout.addWidget(box)

    def _build_barcode_parameters(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("扫码与翻转参数")
        form = QFormLayout(box)
        self.barcode_hits = QSpinBox(); self.barcode_hits.setRange(1, 20); self.barcode_hits.setValue(int(self.node.get_parameter("barcode_stable_hits").value))
        self.barcode_rotations = QSpinBox(); self.barcode_rotations.setRange(4, 4); self.barcode_rotations.setValue(4)
        self.barcode_wait = self._new_double(float(self.node.get_parameter("barcode_face_wait_s").value), 0.1, 10.0, 1, 0.1)
        self.barcode_rz = self._new_double(float(self.node.get_parameter("barcode_flip_step_deg").value), -180.0, 180.0, 1, 5.0)
        self.face_up_user_ry = self._new_double(float(self.node.get_parameter("face_up_user_ry_deg").value), -180.0, 180.0, 1, 5.0)
        self.face_up_jog_tolerance = self._new_double(float(self.node.get_parameter("face_up_jog_tolerance_deg").value), 0.2, 10.0, 1, 0.2)
        form.addRow("同码稳定次数", self.barcode_hits)
        form.addRow("检查面数（旧逻辑固定）", self.barcode_rotations)
        form.addRow("每面等待秒", self.barcode_wait)
        form.addRow("找码 J6 步进 deg（旧逻辑）", self.barcode_rz)
        form.addRow("扫码后用户 Ry 旋转 deg", self.face_up_user_ry)
        form.addRow("Ry 点动停止容差 deg", self.face_up_jog_tolerance)
        layout.addWidget(box)

    def _build_actions(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("人工控制与流程执行")
        grid = QGridLayout(box)
        actions = [
            ("使能机械臂", self.node.enable_robot),
            ("清除错误", self.node.clear_robot_error),
            ("立即停止", self.node.stop_robot),
            ("回初始位并开夹爪", self.node.move_startup),
            ("夹爪打开", self.node.open_gripper),
            ("夹爪关闭测试", self.node.close_gripper),
            ("只采样视觉", self.node.sample_vision_only),
            ("执行完整一轮", self.node.execute_single_cycle),
        ]
        for index, (label, function) in enumerate(actions):
            button = QPushButton(label)
            if label == "立即停止":
                button.clicked.connect(lambda _checked=False, fn=function: self.run_priority_action(fn))
            elif label == "回初始位并开夹爪":
                # Recovery must remain available while an interrupted ordinary
                # GUI worker is still unwinding after Emergency Stop.
                button.clicked.connect(lambda _checked=False, fn=function: self.apply_and_run_recovery(fn))
            elif label in ("只采样视觉", "执行完整一轮"):
                button.clicked.connect(lambda _checked=False, fn=function: self.apply_and_run(fn))
            else:
                button.clicked.connect(lambda _checked=False, fn=function: self.run_action(fn))
            if label == "立即停止":
                button.setStyleSheet("background:#b71c1c;color:white;font-weight:bold")
            grid.addWidget(button, index // 2, index % 2)

        self.start_cycle_button = QPushButton("开始连续循环")
        self.stop_cycle_button = QPushButton("停止连续循环")
        self.start_cycle_button.clicked.connect(self.start_continuous)
        self.stop_cycle_button.clicked.connect(self.node.stop_continuous_cycle)
        grid.addWidget(self.start_cycle_button, 4, 0)
        grid.addWidget(self.stop_cycle_button, 4, 1)
        tip = QLabel("建议顺序：应用参数 → 只采样视觉 → 夹爪测试 → 执行完整一轮。连续循环最后再启用。")
        tip.setWordWrap(True)
        grid.addWidget(tip, 5, 0, 1, 2)
        layout.addWidget(box)

    def apply_parameters(self) -> None:
        startup = [field.value() for field in self.joint_fields["startup_joint"]]
        transfer = [field.value() for field in self.joint_fields["transfer_joint"]]
        scan_xyz = [field.value() / 1000.0 for field in self.pose_fields["scan_exit_user_xyz"]]
        place_display = [field.value() for field in self.pose_fields["place_pose"]]
        place = [value / 1000.0 if index < 3 else value for index, value in enumerate(place_display)]
        parameters = [
            Parameter("startup_joint", value=startup),
            Parameter("transfer_joint", value=transfer),
            Parameter("scan_exit_user_xyz", value=scan_xyz),
            Parameter("place_pose", value=place),
            Parameter("joint_speed", value=self.joint_speed.value()),
            Parameter("linear_speed", value=self.linear_speed.value()),
            Parameter("jog_speed_factor", value=self.jog_speed.value()),
            Parameter("dh_grasp_force", value=self.gripper_force.value()),
            Parameter("grasp_z_offset_m", value=self.grasp_z_offset.value() / 1000.0),
            Parameter("minimum_safe_tcp_z_m", value=self.minimum_safe_tcp_z.value() / 1000.0),
            Parameter("grasp_lift_m", value=self.grasp_lift.value() / 1000.0),
            Parameter("grasp_close_settle_s", value=self.grasp_close_settle.value()),
            Parameter("grasp_feedback_wait_s", value=self.grasp_feedback_wait.value()),
            Parameter("single_cycle_grasp_retry_limit", value=self.grasp_retry_limit.value()),
            Parameter("grasp_feedback_required", value=self.feedback_required.isChecked()),
            Parameter("barcode_stable_hits", value=self.barcode_hits.value()),
            Parameter("barcode_max_face_rotations", value=self.barcode_rotations.value()),
            Parameter("barcode_face_wait_s", value=self.barcode_wait.value()),
            Parameter("barcode_flip_step_deg", value=self.barcode_rz.value()),
            Parameter("face_up_user_ry_deg", value=self.face_up_user_ry.value()),
            Parameter("face_up_jog_tolerance_deg", value=self.face_up_jog_tolerance.value()),
        ]
        results = self.node.set_parameters(parameters)
        failures = [result.reason for result in results if not result.successful]
        if failures:
            self.cycle_status_label.setText("参数应用失败: " + "; ".join(failures))
            return
        self.node.controller.set_joint_profile(speed=self.joint_speed.value(), accel=int(self.node.get_parameter("joint_acc").value))
        self.node.controller.set_linear_profile(speed=self.linear_speed.value(), accel=int(self.node.get_parameter("linear_acc").value))
        self.node._publish_status("GUI parameters applied")

    def apply_and_run(self, function) -> None:
        self.apply_parameters()
        self.run_action(function)

    def apply_and_run_recovery(self, function) -> None:
        if self.recovery_worker is not None and self.recovery_worker.isRunning():
            self.cycle_status_label.setText("回初始位请求已经执行中，请等待机械臂恢复")
            return
        self.apply_parameters()
        self.run_recovery_action(function)

    def run_action(self, function) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.cycle_status_label.setText("已有操作正在运行；请等待或点击立即停止")
            return
        worker = WorkerThread(function)
        self.worker = worker
        self.running_workers.add(worker)
        worker.finished.connect(lambda ok, message, result, w=worker: self._action_finished(ok, message, result, w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def run_priority_action(self, function) -> None:
        worker = WorkerThread(function)
        self.running_workers.add(worker)
        worker.finished.connect(lambda ok, message, result, w=worker: self._action_finished(ok, message, result, w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def run_recovery_action(self, function) -> None:
        worker = WorkerThread(function)
        self.recovery_worker = worker
        self.running_workers.add(worker)
        worker.finished.connect(lambda ok, message, result, w=worker: self._action_finished(ok, message, result, w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _action_finished(self, ok: bool, message: str, result: object, worker: WorkerThread) -> None:
        self.running_workers.discard(worker)
        if self.worker is worker:
            self.worker = None
        if self.recovery_worker is worker:
            self.recovery_worker = None
        if ok:
            if isinstance(result, tuple) and len(result) == 3:
                target, width_m, height_m = result
                self.cycle_status_label.setText(
                    f"视觉成功 xyz=({target.x:.3f},{target.y:.3f},{target.z:.3f})m "
                    f"H={height_m*1000:.1f}mm W={width_m*1000:.1f}mm"
                )
            else:
                self.cycle_status_label.setText("操作完成")
        else:
            self.cycle_status_label.setText("操作失败: " + message)
            self.node.get_logger().error("GUI action failed: " + message)
        self.refresh_status()

    def start_continuous(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.cycle_status_label.setText("当前界面操作尚未完成，不能启动连续循环")
            return
        self.apply_parameters()
        try:
            self.node.start_continuous_cycle()
        except Exception as exc:
            self.cycle_status_label.setText(f"连续循环启动失败: {exc}")

    def refresh_status(self) -> None:
        self.cycle_status_label.setText(self.node.last_status)
        try:
            self.robot_mode_label.setText(self.node.controller.robot_mode_text())
            joints = self.node.controller.current_joint()
            self.joint_feedback_label.setText("  ".join(f"J{i+1}:{value:.1f}" for i, value in enumerate(joints)))
            tcp = self.node._current_command_pose()
            self.tcp_feedback_label.setText(
                f"{tcp.x*1000:.1f}, {tcp.y*1000:.1f}, {tcp.z*1000:.1f} mm | "
                f"{tcp.rx:.1f}, {tcp.ry:.1f}, {tcp.rz:.1f}°"
            )
        except Exception:
            pass
        target = self.node.last_accepted_target
        if target is not None:
            height = self.node.last_accepted_height_m or 0.0
            width = self.node.last_accepted_width_m or 0.0
            self.vision_feedback_label.setText(
                f"XYZ {target.x*1000:.1f}, {target.y*1000:.1f}, {target.z*1000:.1f} mm | "
                f"H {height*1000:.1f} mm | W {width*1000:.1f} mm"
            )

    def closeEvent(self, event) -> None:
        try:
            self.node.stop_robot()
        except Exception as exc:
            self.node.get_logger().error(f"Stop during GUI close failed: {exc}")
        event.accept()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CosmeticBoxSingleArmNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    app = QApplication.instance() or QApplication([])
    window = CosmeticBoxControlWindow(node)
    window.show()
    try:
        exec_function = getattr(app, "exec", None) or app.exec_
        exec_function()
    finally:
        node.stop_continuous_cycle()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
