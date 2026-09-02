"""Single Nova5 (192.168.111.101) cosmetic-box pick/scan/place cycle.

The state machine intentionally contains only the requested path:

startup -> trigger D405 -> grasp 75%-depth target -> transfer joint ->
consume any barcode seen during transfer, otherwise rotate wrist J6 -90 degrees
per barcode face with live scan monitoring and stop J6 as soon as one value is stable ->
single User-frame PTP combining XYZ=(557,200,320) mm, Ry-90 and Rz+50 ->
place -> startup.

The D405 pose represents the TCP-tip point 75% down from the measured top
surface. A small operator-visible Z correction compensates residual hand-eye
height bias, and an absolute TCP-Z floor prevents contact with the tabletop.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
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


def message_stamp_seconds(msg: PoseStamped) -> float:
    """Convert a ROS message timestamp to the host wall-clock seconds domain."""

    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def transform_to_pose(transform: np.ndarray) -> TcpPose:
    rx, ry, rz = SciPyRot.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    x, y, z = transform[:3, 3]
    return TcpPose(float(x), float(y), float(z), float(rx), float(ry), float(rz))


def circular_mean(values: list[float]) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


def compose_motion_percent(ratios: tuple[float, ...], scale_percent: float) -> int:
    """Return one Dobot command ratio equivalent to multiplied legacy ratios."""
    effective = 100.0
    for ratio in ratios:
        effective *= max(0.0, float(ratio)) / 100.0
    effective *= max(1.0, float(scale_percent)) / 100.0
    return max(1, min(100, int(math.floor(effective + 0.5))))


class RecoverableGraspError(RuntimeError):
    """A confirmed empty/lost grasp for which returning to startup is safe."""

    def __init__(self, stage: str, message: str, needs_vertical_retreat: bool = False):
        super().__init__(message)
        self.stage = stage
        self.needs_vertical_retreat = needs_vertical_retreat


@dataclass
class CycleTiming:
    """One vision-to-startup cycle with machine-readable stage durations."""

    cycle_id: str
    started_at: float = field(default_factory=time.monotonic)
    stages: list[dict[str, object]] = field(default_factory=list)

    def add_stage(self, name: str, duration_s: float, outcome: str) -> None:
        self.stages.append(
            {
                "stage": str(name),
                "duration_s": round(float(duration_s), 4),
                "outcome": str(outcome),
            }
        )

    def summary(self, outcome: str) -> dict[str, object]:
        total_s = time.monotonic() - self.started_at
        accounted_s = sum(float(stage["duration_s"]) for stage in self.stages)
        return {
            "event": "cycle_summary",
            "cycle_id": self.cycle_id,
            "outcome": str(outcome),
            "total_s": round(total_s, 4),
            "accounted_s": round(accounted_s, 4),
            "overhead_s": round(max(0.0, total_s - accounted_s), 4),
            "stages": list(self.stages),
        }


class CosmeticBoxSingleArmNode(Node):
    def __init__(self) -> None:
        super().__init__("nova5_cosmetic_box_single_arm_cycle")

        self.declare_parameter("robot_ip", "192.168.111.101")
        self.declare_parameter("dashboard_port", 29999)
        self.declare_parameter("feedback_port", 30004)
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("startup_joint", [14.0, 14.0, -115.0, 25.0, 83.0, 10.0])
        self.declare_parameter("transfer_joint", [14.0, -29.0, -99.0, 39.0, 88.0, 15.0])
        # self.declare_parameter("place_pose", [0.531, 0.328, 0.085, -179.0, -1.39, -85.18]) #放置物料小车的位置
        self.declare_parameter("place_pose", [0.531, 0.328, 0.215, -179.0, -1.39, -85.18]) #放置物料小车的位置
        # 扫码成功后的组合 PTP 目标位置，按 User 0 表达，单位为米。
        self.declare_parameter("scan_exit_user_xyz", [0.557, 0.200, 0.320])
        self.declare_parameter("user_index", 0)
        self.declare_parameter("flange_tool_index", 0)
        self.declare_parameter("command_tool_index", 1)
        # 以下运动参数保留原 GUI 的调节语义。控制器端不再把它们同时写入
        # SpeedFactor、VelJ/VelL 和单条指令，而是先在软件中合成为一个等效
        # 指令百分比。100 表示修改前的理论有效速度基线；当前抓取上方和
        # 抓取下降使用 400，抓取后的各阶段则使用下方独立有效百分比。
        self.declare_parameter("motion_speed_scale_percent", 400)
        # 普通关节动作：初始位、抓取上方、中转位、放置位及回初始位。
        # 加速度独立可调，短行程往往由加速度而不是最高速度决定耗时。
        self.declare_parameter("joint_speed", 65)
        self.declare_parameter("joint_acc", 55)
        # 抓取完成后的抬升、去中转位、扫码后翻转、放置和回初始位分别调速，
        # 避免为了加快后半段而把抓取上方/下降前的动作一起推得过快。
        # 后半段参数是“单条指令有效百分比”，不再与 joint_speed 重复相乘。
        self.declare_parameter("grasp_lift_speed_factor", 90)
        self.declare_parameter("grasp_lift_acc_factor", 85)
        self.declare_parameter("transfer_speed_factor", 100)
        self.declare_parameter("transfer_acc_factor", 100)
        self.declare_parameter("place_speed_factor", 100)
        self.declare_parameter("place_acc_factor", 100)
        self.declare_parameter("post_scan_acc_factor", 100)
        self.declare_parameter("return_startup_speed_factor", 100)
        self.declare_parameter("return_startup_acc_factor", 100)
        # 抓取下降使用的直线运动速度和加速度。
        self.declare_parameter("linear_speed", 60)
        self.declare_parameter("linear_acc", 50)
        # 扫码成功后，XYZ 与 User Ry/Rz 姿态一起变化的组合 PTP 速度。
        self.declare_parameter("jog_speed_factor", 100.0)
        # 扫码器靠近速度：盒子到达 transfer_joint 后，沿 User 0 X+ 自适应
        # 靠近扫码器时使用。它只影响这段 X+ 位移，不影响抓取、抬升或放置。
        self.declare_parameter("scanner_approach_speed_factor", 100)
        # 扫码完成后的安全退让速度：保持扫码完成姿态，沿 User 0 X- 原路
        # 退回实际靠近距离，远离扫码器后才允许执行大范围组合 PTP。
        self.declare_parameter("scanner_retreat_speed_factor", 100)
        self.declare_parameter("scanner_retreat_acc_factor", 100)
        # 回到原中转距离后继续沿 User X- 增加的安全余量，默认 30 mm。
        # 用于覆盖长盒子执行 Ry/Rz 时角点产生的额外旋转包络。
        self.declare_parameter("scanner_retreat_extra_m", 0.030)
        # J6 多面找码速度：只影响找码期间 J6 的连续点动，以及扫码成功后
        # 吸附到最近 90° 标准面的对齐动作。速度越高，停止超调通常越大。
        # MoveJog 没有单条 v 参数；200% 下该值会被合成为并钳位到 100%，
        # 因此继续提高统一比例不会再提高 J6 的硬件 Jog 速度。
        self.declare_parameter("barcode_j6_speed_factor", 100)
        self.declare_parameter("barcode_alignment_acc_factor", 100)
        # X+ 扫码靠近使用有界 RelMovJUser；收到条码时主动停止当前指令，
        # 没有条码时精确到达目标距离，避免高速 MoveJog 的刹停过冲。
        self.declare_parameter("scanner_approach_monitor_timeout_s", 20.0)
        self.declare_parameter("scanner_approach_monitor_period_s", 0.005)
        self.declare_parameter("scanner_approach_acc_factor", 100)
        # 条码已经在目标点前几毫米内出现时，让有界 RelMovJUser 自然完成，
        # 避免为极短剩余距离额外触发约 0.5s 的控制器 Stop 停机确认。
        self.declare_parameter("scanner_approach_natural_finish_margin_m", 0.005)
        # 旧版“固定 XYZ、单独 Ry 点动”的备用速度。当前生产流程已经改为
        # XYZ+Ry+Rz 单条组合 PTP，不再读取该参数。
        self.declare_parameter("face_up_rotation_speed_factor", 50)
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
        # D405 稳定目标后发布的后台预抓取位姿话题。机器人到达抓取上方
        # 后只消费这条话题的最新一帧，不增加单独视觉等待。
        self.declare_parameter("pregrasp_pose_topic", "/target_pose_cam_pregrasp")
        # 最新预抓取位姿允许的最大年龄，单位：秒。超过该时间视为视觉
        # 跟踪失效，机械臂不下压，直接回初始位重新检测。
        self.declare_parameter("pregrasp_pose_max_age_s", 0.50)
        # 目标相对初始稳定位姿的允许位置变化，单位：米。小于等于该值时
        # 直接使用最新位姿下压；超过该值时先在安全上方修正。
        self.declare_parameter("pregrasp_position_tolerance_m", 0.007)
        # 目标相对初始稳定位姿的允许姿态变化，单位：度。小于等于该值时
        # 直接下压；超过该值时先在安全上方修正。
        self.declare_parameter("pregrasp_angle_tolerance_deg", 8.0)
        # 单次安全上方位置修正的最大位移，单位：米。超过该值说明目标
        # 变化过大或跟踪可能跳变，不追踪移动，直接回初始位重新检测。
        self.declare_parameter("pregrasp_max_correction_m", 0.050)
        # 单次安全上方位置修正允许的最大姿态变化，单位：度。超过该值
        # 不执行修正和下压，直接回初始位重新检测。
        self.declare_parameter("pregrasp_max_correction_angle_deg", 30.0)
        # 目标与当前 TCP 安全上方位置之间必须保留的最小垂直间隙，单位：米。
        # 间隙不足时禁止下压，直接回初始位重新检测。
        self.declare_parameter("pregrasp_min_hover_clearance_m", 0.030)
        # D405 帧时间戳与机械臂反馈历史的最大允许边缘误差，单位：秒。
        # 正常情况下使用时间戳之间的历史反馈插值；只有帧落在历史首尾
        # 外侧时才使用这个容差，超出则丢弃该帧，避免再次套用错误的当前位姿。
        self.declare_parameter("vision_pose_max_time_skew_s", 0.30)
        # 夹爪张开命令：点云测得的盒子短边 + 预留开爪间隙，单位为米。
        self.declare_parameter("vision_width_topic", "/gripper_target_width")
        # 盒子长边尺寸：由 SAM2 分割点云的顶面三维包围盒计算，单位为米。
        # 该长度用于计算中转点处盒子向扫码器靠近的距离。
        self.declare_parameter("vision_length_topic", "/cosmetic_box_length")
        # 盒子高度：顶面与桌面之间的距离，单位为米。
        self.declare_parameter("vision_height_topic", "/cosmetic_box_height")
        self.declare_parameter("vision_trigger_topic", "/trigger_d405_vision")
        self.declare_parameter("handoff_state_topic", "/d405_handoff_zone_state")
        # 视觉节点连续发布 2 帧稳定结果；机器人取 2 帧做位置/角度一致性检查。
        # 相比原来的 3 帧少等待一次 FFS/SAM2 推理，同时仍保留跨帧校验。
        self.declare_parameter("vision_samples", 2)
        # 视觉请求内部会等待 102 从目标上方退出。延长的上限只影响真正
        # 被遮挡的请求；正常 CLEAR 流程仍在第二份新鲜点云到达后立即继续。
        self.declare_parameter("vision_timeout_s", 20.0)
        # 明确失败会由视觉结果话题立即返回；连续循环仅短暂停顿后重新检测。
        self.declare_parameter("vision_retry_delay_s", 0.1)
        self.declare_parameter("vision_result_topic", "/d405_vision_result")
        self.declare_parameter("vision_position_stability_m", 0.010)
        self.declare_parameter("vision_angle_stability_deg", 10.0)
        self.declare_parameter("min_box_height_m", 0.005)
        self.declare_parameter("max_box_height_m", 0.150)
        # 允许参与扫码距离计算的盒长范围，防止异常点云尺寸触发危险移动。
        self.declare_parameter("min_box_length_m", 0.020)  # 最小盒长 20 mm
        self.declare_parameter("max_box_length_m", 0.300)  # 最大盒长 300 mm
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
        self.declare_parameter("barcode_face_wait_s", 0.05)  # 每个标准面最多等待 0.05 秒
        self.declare_parameter("barcode_max_face_rotations", 4)
        # Each next-face search may rotate wrist J6 by as much as -90 degrees,
        # with a joint-limit guard. Live monitoring below stops it earlier when
        # the scanner decodes a barcode during that rotation.
        self.declare_parameter("barcode_flip_step_deg", -90.0)
        self.declare_parameter("barcode_flip_safe_joint_limit_deg", 355.0)
        self.declare_parameter("barcode_flip_watch_joint_index", 5)
        # J6 找码改用连续点动并实时监听扫码结果。到达目标角前一旦识别成功，
        # 立即停止点动，保留条码正对扫码器的姿态。
        self.declare_parameter("barcode_flip_jog_tolerance_deg", 1.0)
        self.declare_parameter("barcode_flip_jog_timeout_s", 60.0)
        # 扫码器可以在条码面斜对着它时提前解码。识别后不能直接保留任意
        # 中间角度，否则后续 User Ry -90° 会让条码面斜着朝上。停止 J6 后
        # 自动吸附到最近的 90° 标准面：前半程回上一面，后半程补到下一面。
        self.declare_parameter("barcode_snap_to_nearest_face", True)
        # 到达 transfer_joint 后，夹爪 TCP 中心沿 User 0 的 X+ 方向面对扫码器。
        # scanner_center_distance_m：此时 TCP 夹持中心到扫码器识读面的实测距离，
        # 默认 0.120 m（120 mm）。如果中转点或扫码器位置改变，需要重新实测此值。
        self.declare_parameter("scanner_center_distance_m", 0.120)
        # scanner_face_clearance_m：靠近完成后，盒子朝向扫码器的侧面与扫码器
        # 识读面之间保留的安全/识读间隙，默认 0.030 m（30 mm）。
        self.declare_parameter("scanner_face_clearance_m", 0.030)
        # 点云盒长和现场距离都有毫米级误差。若计算出的靠近量仅略微为负，
        # 说明 transfer_joint 已经足够接近，不再沿 X+ 前进即可；超过该容差
        # 仍然拒绝运动，避免长盒子撞向扫码器。
        self.declare_parameter("scanner_approach_negative_tolerance_m", 0.005)
        # 自动靠近量（沿 User 0 X+）：
        #   X移动量 = TCP中心到扫码器距离 - 盒长/2 - 盒侧面保留间隙
        # 示例：盒长 100 mm 时，120 - 100/2 - 30 = 40 mm。
        # 扫码成功时的姿态作为起点，组合目标先叠加 User Ry -90°，
        # 再叠加 User Rz +50°；两种旋转与 XYZ 在同一条 PTP 中同时完成。
        self.declare_parameter("face_up_user_ry_deg", -90.0)
        self.declare_parameter("post_scan_user_rz_deg", 50.0)
        self.declare_parameter("face_up_jog_tolerance_deg", 2.0)
        self.declare_parameter("face_up_jog_timeout_s", 60.0)
        self.declare_parameter("face_up_fixed_xyz_tolerance_m", 0.003)
        # 旧版单独 Ry 点动的备用稳定等待；当前组合 PTP 不使用。
        self.declare_parameter("face_up_settle_s", 0.0)

        self.declare_parameter("dh_max_opening_m", 0.095)
        self.declare_parameter("dh_force", 30)
        self.declare_parameter("dh_grasp_force", 30)
        self.declare_parameter("dh_slave_id", 1)
        self.declare_parameter("dh_tool_identify", 1)
        self.declare_parameter("dh_timeout_s", 10.0)
        # 放置时不再等待夹爪完全张到 95 mm；从实际夹持宽度额外张开
        # 15 mm 并确认到位即可释放盒子。所有机械臂点位保持不变。
        self.declare_parameter("place_release_clearance_m", 0.015)
        # close(wait=True) 已确认夹爪进入终态；只保留 50 ms 电气反馈稳定时间，
        # 后面仍执行两次独立 state/opening 检查，不取消空抓保护。
        self.declare_parameter("grasp_close_settle_s", 0.05)
        self.declare_parameter("grasp_confirm_samples", 2)
        self.declare_parameter("grasp_confirm_interval_s", 0.05)
        # 仅当抬升后反馈仍为“运动中”时使用，并非每轮固定等待。
        self.declare_parameter("grasp_feedback_wait_s", 0.7)
        self.declare_parameter("single_cycle_grasp_retry_limit", 3)
        self.declare_parameter("grasp_success_min_opening_m", 0.003)
        self.declare_parameter("grasp_feedback_required", True)
        self.declare_parameter("timing_enabled", True)
        self.declare_parameter("timing_topic", "/cosmetic_pick_cycle_timing")

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
        self.controller.enable_single_command_motion_scaling()
        self.gripper = self._initialize_gripper()

        self.data_lock = threading.Lock()
        self.pose_samples: deque[tuple[int, TcpPose]] = deque(maxlen=20)
        self.pose_count = 0
        self.pregrasp_pose_count = 0
        self.latest_pregrasp_pose: Optional[TcpPose] = None
        self.latest_pregrasp_pose_received_at = 0.0
        self.width_count = 0
        self.length_count = 0
        self.height_count = 0
        self.latest_width_m: Optional[float] = None
        self.latest_length_m: Optional[float] = None
        self.latest_height_m: Optional[float] = None
        self.vision_result_count = 0
        self.latest_vision_result = ""
        self.handoff_state_count = 0
        self.latest_handoff_state = "IDLE"
        self.latest_handoff_clear = False
        self.latest_handoff_candidate_points = 0
        self.latest_handoff_cluster_points = 0

        self.barcode_lock = threading.Lock()
        self.barcode_window_active = False
        self.barcode_value = ""
        self.barcode_hits = 0
        self.barcode_last_time = 0.0

        self.running = True
        self.cycle_enabled = False
        self.shutting_down = False
        self.worker: Optional[threading.Thread] = None
        self.active_timing: Optional[CycleTiming] = None
        # Serialize complete robot sequences. Emergency Stop intentionally does
        # not take this lock, so it can interrupt a blocking sequence; recovery
        # to startup waits for that interrupted sequence to unwind before it
        # sends a new movement command.
        self.action_lock = threading.RLock()
        self.last_status = "ready - waiting for operator"
        self.last_accepted_target: Optional[TcpPose] = None
        self.last_accepted_width_m: Optional[float] = None
        self.last_accepted_length_m: Optional[float] = None
        self.last_accepted_height_m: Optional[float] = None

        self.create_subscription(PoseStamped, str(self.get_parameter("vision_pose_topic").value), self._vision_pose_callback, 10)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("pregrasp_pose_topic").value),
            self._pregrasp_pose_callback,
            10,
        )
        self.create_subscription(Float32, str(self.get_parameter("vision_width_topic").value), self._vision_width_callback, 10)
        self.create_subscription(Float32, str(self.get_parameter("vision_length_topic").value), self._vision_length_callback, 10)
        self.create_subscription(Float32, str(self.get_parameter("vision_height_topic").value), self._vision_height_callback, 10)
        self.create_subscription(String, str(self.get_parameter("vision_result_topic").value), self._vision_result_callback, 10)
        self.create_subscription(String, str(self.get_parameter("handoff_state_topic").value), self._handoff_state_callback, 10)
        self.create_subscription(String, str(self.get_parameter("barcode_topic").value), self._barcode_callback, 20)
        self.create_subscription(Bool, "/cosmetic_pick_cycle_enable", self._cycle_enable_callback, 10)
        self.trigger_publisher = self.create_publisher(Bool, str(self.get_parameter("vision_trigger_topic").value), 10)
        self.status_publisher = self.create_publisher(String, "/cosmetic_pick_cycle_status", 10)
        self.timing_publisher = self.create_publisher(
            String,
            str(self.get_parameter("timing_topic").value),
            20,
        )

        self.start_timer = self.create_timer(1.0, self._start_automatically_once)
        self.get_logger().warning(
            "Using handeye_flange_to_cam parameter for robot 192.168.111.101 / D405 409122274792; "
            "verify this calibration on the real cell before enabling motion."
        )
        self.get_logger().info("Single-arm cosmetic-box controller connected; no 192.168.111.102 connection is created.")
        self._log_effective_motion_profile()

    def _six_values(self, name: str) -> list[float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != 6:
            raise ValueError(f"{name} must contain 6 values")
        return values

    def _composed_motion_percent(self, *ratios: float) -> int:
        """Collapse legacy Dobot ratio multiplication into one command value."""
        scale = max(1.0, float(self.get_parameter("motion_speed_scale_percent").value))
        return compose_motion_percent(tuple(ratios), scale)

    @staticmethod
    def _direct_motion_percent(value: float) -> int:
        """Clamp a stage-specific effective command percentage."""
        return max(1, min(100, int(math.floor(float(value) + 0.5))))

    def _motion_profile(self) -> dict[str, int]:
        joint_speed = float(self.get_parameter("joint_speed").value)
        joint_acc = float(self.get_parameter("joint_acc").value)
        linear_speed = float(self.get_parameter("linear_speed").value)
        linear_acc = float(self.get_parameter("linear_acc").value)
        grasp_lift_speed = float(self.get_parameter("grasp_lift_speed_factor").value)
        grasp_lift_acc = float(self.get_parameter("grasp_lift_acc_factor").value)
        transfer_speed = float(self.get_parameter("transfer_speed_factor").value)
        transfer_acc = float(self.get_parameter("transfer_acc_factor").value)
        place_speed = float(self.get_parameter("place_speed_factor").value)
        place_acc = float(self.get_parameter("place_acc_factor").value)
        post_scan_acc = float(self.get_parameter("post_scan_acc_factor").value)
        return_speed = float(self.get_parameter("return_startup_speed_factor").value)
        return_acc = float(self.get_parameter("return_startup_acc_factor").value)
        combined_speed = float(self.get_parameter("jog_speed_factor").value)
        approach_speed = float(self.get_parameter("scanner_approach_speed_factor").value)
        approach_acc = float(self.get_parameter("scanner_approach_acc_factor").value)
        retreat_speed = float(self.get_parameter("scanner_retreat_speed_factor").value)
        retreat_acc = float(self.get_parameter("scanner_retreat_acc_factor").value)
        barcode_speed = float(self.get_parameter("barcode_j6_speed_factor").value)
        face_up_speed = float(self.get_parameter("face_up_rotation_speed_factor").value)
        return {
            # Old replay speed: SpeedFactor(joint) * VelJ(joint) * command-v.
            "joint_speed": self._composed_motion_percent(
                joint_speed, joint_speed, joint_speed
            ),
            # Joint-coordinate MovJ previously omitted local ``a``.
            "joint_waypoint_acc": self._composed_motion_percent(
                joint_speed, joint_acc
            ),
            # Cartesian/relative MovJ used joint_acc both globally and locally.
            "joint_pose_acc": self._composed_motion_percent(
                joint_speed, joint_acc, joint_acc
            ),
            # These post-grasp values are already effective single-command
            # percentages; do not multiply them by joint_speed again.
            "grasp_lift_speed": self._direct_motion_percent(grasp_lift_speed),
            "grasp_lift_acc": self._direct_motion_percent(grasp_lift_acc),
            "transfer_speed": self._direct_motion_percent(transfer_speed),
            "transfer_acc": self._direct_motion_percent(transfer_acc),
            "place_speed": self._direct_motion_percent(place_speed),
            "place_acc": self._direct_motion_percent(place_acc),
            "post_scan_acc": self._direct_motion_percent(post_scan_acc),
            "return_startup_speed": self._direct_motion_percent(return_speed),
            "return_startup_acc": self._direct_motion_percent(return_acc),
            "linear_speed": self._composed_motion_percent(
                joint_speed, linear_speed, linear_speed
            ),
            "linear_acc": self._composed_motion_percent(
                joint_speed, linear_acc, linear_acc
            ),
            "post_scan_speed": self._direct_motion_percent(combined_speed),
            "scanner_approach_speed": self._direct_motion_percent(approach_speed),
            "scanner_approach_acc": self._direct_motion_percent(approach_acc),
            "scanner_retreat_speed": self._direct_motion_percent(retreat_speed),
            "scanner_retreat_acc": self._direct_motion_percent(retreat_acc),
            # MoveJog has no local v/a ratios; these are direct SpeedFactor
            # values and are capped by the controller at 100.
            "barcode_jog_speed": self._direct_motion_percent(barcode_speed),
            "barcode_alignment_speed": self._direct_motion_percent(barcode_speed),
            "barcode_alignment_acc": self._direct_motion_percent(
                float(self.get_parameter("barcode_alignment_acc_factor").value)
            ),
            "face_up_jog_speed": self._direct_motion_percent(face_up_speed),
        }

    def _log_effective_motion_profile(self) -> None:
        profile = self._motion_profile()
        scale = int(self.get_parameter("motion_speed_scale_percent").value)
        details = ", ".join(f"{name}={value}%" for name, value in profile.items())
        self.get_logger().info(
            f"Normalized motion scaling active: scale={scale}% (100%=legacy effective baseline); "
            f"SpeedFactor/VelJ/VelL/AccJ/AccL replay layers fixed at 100; {details}"
        )

    def _emit_timing(self, payload: dict[str, object]) -> None:
        if self.shutting_down or not rclpy.ok() or not bool(self.get_parameter("timing_enabled").value):
            return
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        message = String()
        message.data = encoded
        try:
            self.timing_publisher.publish(message)
        except Exception:
            # Ctrl-C can invalidate the ROS context while the cycle worker is
            # unwinding a blocking vision/motion stage. Timing must not keep
            # the process alive or mask the original shutdown.
            return
        self.get_logger().info(f"[timing] {encoded}")

    def _begin_cycle_timing(self, cycle_id: str) -> None:
        self.active_timing = CycleTiming(str(cycle_id))
        self._emit_timing({"event": "cycle_start", "cycle_id": str(cycle_id)})

    @contextmanager
    def _timed_stage(self, stage: str):
        timing = self.active_timing
        if timing is None or not bool(self.get_parameter("timing_enabled").value):
            yield
            return
        started_at = time.monotonic()
        outcome = "ok"
        try:
            yield
        except Exception:
            outcome = "error"
            raise
        finally:
            duration_s = time.monotonic() - started_at
            timing.add_stage(stage, duration_s, outcome)
            self._emit_timing(
                {
                    "event": "stage",
                    "cycle_id": timing.cycle_id,
                    "stage": str(stage),
                    "outcome": outcome,
                    "duration_s": round(duration_s, 4),
                    "elapsed_s": round(time.monotonic() - timing.started_at, 4),
                }
            )

    def _finish_cycle_timing(self, outcome: str) -> None:
        timing = self.active_timing
        if timing is None:
            return
        self._emit_timing(timing.summary(outcome))
        self.active_timing = None

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

    def _transform_vision_pose(self, msg: PoseStamped, source: str) -> Optional[TcpPose]:
        if msg.header.frame_id.strip() != str(self.get_parameter("camera_frame_id").value):
            self.get_logger().error(
                f"Ignoring {source} vision frame {msg.header.frame_id!r}"
            )
            return None
        try:
            frame_time_s = message_stamp_seconds(msg)
            user_index = int(self.get_parameter("user_index").value)
            flange_tool_index = int(self.get_parameter("flange_tool_index").value)
            max_time_skew_s = float(self.get_parameter("vision_pose_max_time_skew_s").value)
            try:
                flange_pose = self.controller.current_tcp_pose_at(
                    frame_time_s,
                    user_index=user_index,
                    tool_index=flange_tool_index,
                    max_skew_s=max_time_skew_s,
                )
            except RuntimeError as exact_history_error:
                # The feedback packet may report a different active tool than
                # the flange tool used by hand-eye calibration.  Reconstruct
                # the historical flange transform from the historical active
                # TCP motion and the current, exact GetPose(user, flange_tool)
                # result.  This preserves the old tool-coordinate behaviour
                # without losing the timestamp correction.
                historical_active, historical_user, historical_tool = (
                    self.controller.current_feedback_tcp_pose_at(
                        frame_time_s,
                        max_skew_s=max_time_skew_s,
                    )
                )
                current_active, current_user, current_tool = self.controller.current_feedback_tcp_pose()
                if (historical_user, historical_tool) != (current_user, current_tool):
                    raise RuntimeError(
                        "active User/Tool changed between the camera frame and now; "
                        "historical transform is ambiguous"
                    ) from exact_history_error
                if historical_user != user_index:
                    raise RuntimeError(
                        f"historical active User={historical_user} does not match "
                        f"configured User={user_index}; refusing an ambiguous transform"
                    ) from exact_history_error
                current_flange = self.controller.current_tcp_pose(
                    user_index=user_index,
                    tool_index=flange_tool_index,
                )
                base_to_flange_now = pose_to_transform(current_flange)
                base_to_active_now = pose_to_transform(current_active)
                flange_to_active = np.linalg.inv(base_to_flange_now) @ base_to_active_now
                base_to_flange_at = pose_to_transform(historical_active) @ np.linalg.inv(flange_to_active)
                flange_pose = transform_to_pose(base_to_flange_at)
            offset_angles = self._six_values_from_rotation("grasp_offset_rxyz_deg")
            target_to_grasp = np.eye(4, dtype=np.float64)
            target_to_grasp[:3, :3] = SciPyRot.from_euler("xyz", offset_angles, degrees=True).as_matrix()
            base_to_target = pose_to_transform(flange_pose) @ self.handeye_flange_to_cam @ message_to_transform(msg)
            command_pose = transform_to_pose(base_to_target @ target_to_grasp)
        except Exception as exc:
            self.get_logger().error(f"{source.capitalize()} vision pose transform failed: {exc}")
            return None
        return command_pose

    def _vision_pose_callback(self, msg: PoseStamped) -> None:
        command_pose = self._transform_vision_pose(msg, "stable")
        if command_pose is None:
            return
        with self.data_lock:
            self.pose_count += 1
            self.pose_samples.append((self.pose_count, command_pose))

    def _pregrasp_pose_callback(self, msg: PoseStamped) -> None:
        command_pose = self._transform_vision_pose(msg, "pregrasp")
        if command_pose is None:
            return
        with self.data_lock:
            self.pregrasp_pose_count += 1
            self.latest_pregrasp_pose = command_pose
            self.latest_pregrasp_pose_received_at = time.monotonic()

    def _six_values_from_rotation(self, name: str) -> list[float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != 3:
            raise ValueError(f"{name} must contain 3 values")
        return values

    def _vision_width_callback(self, msg: Float32) -> None:
        with self.data_lock:
            self.latest_width_m = float(msg.data)
            self.width_count += 1

    def _vision_length_callback(self, msg: Float32) -> None:
        with self.data_lock:
            self.latest_length_m = float(msg.data)
            self.length_count += 1

    def _vision_height_callback(self, msg: Float32) -> None:
        with self.data_lock:
            self.latest_height_m = float(msg.data)
            self.height_count += 1

    def _vision_result_callback(self, msg: String) -> None:
        result = msg.data.strip()
        if not result:
            return
        with self.data_lock:
            self.latest_vision_result = result
            self.vision_result_count += 1

    def _handoff_state_callback(self, msg: String) -> None:
        encoded = msg.data.strip()
        if not encoded:
            return
        try:
            payload = json.loads(encoded)
            state = str(payload.get("state", "UNKNOWN")).strip().upper()
            # Fail closed: only a literal JSON boolean true paired with the
            # exact CLEAR state can authorize accepting vision poses.
            clear = payload.get("clear") is True and state == "CLEAR"
            candidate_points = int(payload.get("candidate_points", 0))
            cluster_points = int(payload.get("largest_cluster_points", 0))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid D405 handoff state: {exc}")
            return
        with self.data_lock:
            self.latest_handoff_state = state
            self.latest_handoff_clear = clear
            self.latest_handoff_candidate_points = candidate_points
            self.latest_handoff_cluster_points = cluster_points
            self.handoff_state_count += 1

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
        if self.shutting_down or not rclpy.ok():
            return
        msg = String()
        msg.data = text
        try:
            self.status_publisher.publish(msg)
        except Exception:
            return
        self.get_logger().info(f"[cycle] {text}")

    def _cycle_worker(self) -> None:
        try:
            with self.action_lock:
                self._move_startup_and_open(require_cycle_active=True)
                cycle_index = 0
                while self.running and self.cycle_enabled:
                    cycle_index += 1
                    self._begin_cycle_timing(f"continuous-{cycle_index}")
                    self._publish_status(f"cycle {cycle_index}: detecting minimum-camera-X box")
                    with self._timed_stage("vision_detection"):
                        target_bundle = self._request_vision_target()
                    if target_bundle is None:
                        delay = max(0.1, float(self.get_parameter("vision_retry_delay_s").value))
                        self._publish_status(f"cycle {cycle_index}: no valid box; retrying in {delay:.1f}s")
                        with self._timed_stage("vision_retry_delay"):
                            time.sleep(delay)
                        self._finish_cycle_timing("no_target")
                        continue
                    target_pose, width_m, height_m, length_m = target_bundle
                    try:
                        self._execute_one_cycle(target_pose, width_m, height_m, length_m)
                    except RecoverableGraspError as exc:
                        with self._timed_stage("grasp_failure_recovery"):
                            self._recover_failed_grasp_for_retry(exc)
                        self._publish_status(
                            f"cycle {cycle_index}: grasp retry recovery completed; "
                            "requesting a fresh D405 target"
                        )
                        self._finish_cycle_timing("grasp_retry")
                        continue
                    # 放置点已经执行 open(wait=True)，回初始位后不再重复下发
                    # 一次相同的开爪命令和 Modbus 完成等待。
                    with self._timed_stage("return_startup"):
                        self._move_startup_and_open(
                            require_cycle_active=True,
                            open_gripper=False,
                        )
                    self._finish_cycle_timing("success")
            self._publish_status("cycle stopped")
        except Exception as exc:
            self._finish_cycle_timing("fault")
            self.cycle_enabled = False
            self._publish_status(f"FAULT: {exc}")
            self.get_logger().fatal(f"Automatic cycle stopped safely: {exc}")

    def _move_startup_and_open(
        self,
        require_cycle_active: bool = False,
        open_gripper: bool = True,
    ) -> None:
        self._publish_status("moving to startup joint")
        motion = self._motion_profile()
        self.controller.move_joint(
            self._six_values("startup_joint"),
            speed=motion["return_startup_speed"],
            accel=motion["return_startup_acc"],
        )
        if require_cycle_active:
            self._require_cycle_active("at startup joint")
        if open_gripper:
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

    def sample_vision_only(self) -> tuple[TcpPose, float, float, float]:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("continuous cycle is running")
        result = self._request_vision_target(require_cycle_enabled=False)
        if result is None:
            raise RuntimeError("no stable D405 target received")
        target, width_m, height_m, length_m = result
        self._publish_status(
            f"vision sampled: xyz=({target.x:.3f},{target.y:.3f},{target.z:.3f})m "
            f"length={length_m*1000:.1f}mm height={height_m*1000:.1f}mm "
            f"width={width_m*1000:.1f}mm"
        )
        return result

    def execute_single_cycle(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("continuous cycle is running")
        self.cycle_enabled = True
        try:
            with self.action_lock:
                # 单轮流程开始前必须确保夹爪已经打开。
                self._move_startup_and_open(require_cycle_active=True)
                retry_limit = max(
                    0, int(self.get_parameter("single_cycle_grasp_retry_limit").value)
                )
                grasp_failures = 0
                attempt_index = 0
                while True:
                    attempt_index += 1
                    self._begin_cycle_timing(f"single-{attempt_index}")
                    with self._timed_stage("vision_detection"):
                        result = self._request_vision_target()
                    if result is None:
                        self._finish_cycle_timing("no_target")
                        raise RuntimeError("no stable D405 target received")
                    try:
                        self._execute_one_cycle(*result)
                        break
                    except RecoverableGraspError as exc:
                        grasp_failures += 1
                        with self._timed_stage("grasp_failure_recovery"):
                            self._recover_failed_grasp_for_retry(exc)
                        self._finish_cycle_timing("grasp_retry")
                        if grasp_failures > retry_limit:
                            raise RuntimeError(
                                f"Grasp failed after {grasp_failures} attempts; "
                                "robot recovered to startup"
                            ) from exc
                        self._publish_status(
                            f"single cycle grasp retry {grasp_failures}/{retry_limit}: "
                            "requesting a fresh D405 target"
                        )
                # 放置点已经执行 open(wait=True)，回初始位后不再重复下发
                # 一次相同的开爪命令和 Modbus 完成等待。
                with self._timed_stage("return_startup"):
                    self._move_startup_and_open(
                        require_cycle_active=True,
                        open_gripper=False,
                    )
                self._finish_cycle_timing("success")
                self._publish_status("single cycle completed")
        except Exception:
            self._finish_cycle_timing("fault")
            raise
        finally:
            self.cycle_enabled = False

    def _request_vision_target(self, require_cycle_enabled: bool = True) -> Optional[tuple[TcpPose, float, float, float]]:
        with self.data_lock:
            previous_pose = self.pose_count
            previous_width = self.width_count
            previous_length = self.length_count
            previous_height = self.height_count
            previous_result = self.vision_result_count
            previous_handoff = self.handoff_state_count
        trigger = Bool()
        trigger.data = True
        self.trigger_publisher.publish(trigger)
        required_samples = max(1, int(self.get_parameter("vision_samples").value))
        deadline = time.monotonic() + max(0.1, float(self.get_parameter("vision_timeout_s").value))
        last_reported_handoff_state = ""
        while (
            self.running
            and (self.cycle_enabled or not require_cycle_enabled)
            and time.monotonic() < deadline
        ):
            status_update = ""
            with self.data_lock:
                if self.vision_result_count > previous_result:
                    result_text = self.latest_vision_result
                    if result_text.startswith("failure:"):
                        reason = result_text.split(":", 1)[1] or "unknown"
                        self.get_logger().warning(
                            f"D405 request rejected immediately: {reason}"
                        )
                        return None
                fresh_handoff_state = self.handoff_state_count > previous_handoff
                if fresh_handoff_state and self.latest_handoff_state != last_reported_handoff_state:
                    last_reported_handoff_state = self.latest_handoff_state
                    if self.latest_handoff_state == "BLOCKED":
                        status_update = (
                            "D405 handoff zone BLOCKED; waiting for 102 to retreat "
                            f"(points={self.latest_handoff_candidate_points}, "
                            f"cluster={self.latest_handoff_cluster_points})"
                        )
                    elif self.latest_handoff_state in ("VERIFYING_CLEAR", "WAIT_LIVE_CLOUD"):
                        status_update = "D405 handoff zone looks clear; confirming with a fresh FFS cloud"
                    elif self.latest_handoff_clear:
                        status_update = "D405 handoff zone CLEAR; accepting stable grasp samples"
                ready = (
                    self.pose_count >= previous_pose + required_samples
                    and self.width_count > previous_width
                    and self.length_count > previous_length
                    and self.height_count > previous_height
                    and fresh_handoff_state
                    and self.latest_handoff_clear
                )
                if ready:
                    samples = [pose for count, pose in self.pose_samples if count > previous_pose][-required_samples:]
                    width_m = self.latest_width_m
                    length_m = self.latest_length_m
                    height_m = self.latest_height_m
                    break
            if status_update:
                self._publish_status(status_update)
            time.sleep(0.02)
        else:
            self.get_logger().warning(
                "Timed out waiting for a fresh D405 target with a CLEAR handoff zone"
            )
            return None
        if len(samples) != required_samples or width_m is None or length_m is None or height_m is None:
            return None
        min_height = float(self.get_parameter("min_box_height_m").value)
        max_height = float(self.get_parameter("max_box_height_m").value)
        if not min_height <= height_m <= max_height:
            self.get_logger().error(f"Vision height {height_m:.4f}m outside [{min_height:.4f}, {max_height:.4f}]")
            return None
        min_length = float(self.get_parameter("min_box_length_m").value)
        max_length = float(self.get_parameter("max_box_length_m").value)
        if not min_length <= length_m <= max_length:
            self.get_logger().error(
                f"Vision length {length_m:.4f}m outside [{min_length:.4f}, {max_length:.4f}]"
            )
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
            f"length={length_m*1000:.1f}mm height={height_m*1000:.1f}mm "
            f"width_command={width_m*1000:.1f}mm"
        )
        self.last_accepted_target = averaged
        self.last_accepted_width_m = width_m
        self.last_accepted_length_m = length_m
        self.last_accepted_height_m = height_m
        return averaged, width_m, height_m, length_m

    @staticmethod
    def _pose_delta(first: TcpPose, second: TcpPose) -> tuple[float, float]:
        position_delta = math.sqrt(
            (second.x - first.x) ** 2
            + (second.y - first.y) ** 2
            + (second.z - first.z) ** 2
        )
        angle_delta = max(
            abs((second_angle - first_angle + 180.0) % 360.0 - 180.0)
            for first_angle, second_angle in (
                (first.rx, second.rx),
                (first.ry, second.ry),
                (first.rz, second.rz),
            )
        )
        return position_delta, angle_delta

    def _latest_fresh_pregrasp_pose(self, previous_count: int) -> tuple[TcpPose, int, float]:
        with self.data_lock:
            current_count = self.pregrasp_pose_count
            pose = self.latest_pregrasp_pose
            received_at = self.latest_pregrasp_pose_received_at
        age_s = time.monotonic() - received_at if received_at > 0.0 else float("inf")
        max_age_s = max(0.05, float(self.get_parameter("pregrasp_pose_max_age_s").value))
        if pose is None or current_count <= previous_count or age_s > max_age_s:
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    "no fresh D405 pregrasp pose after reaching the safe hover "
                    f"(new_samples={max(0, current_count - previous_count)}, age={age_s:.3f}s); "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )
        return pose, current_count, age_s

    def _apply_grasp_z_safety(
        self,
        pose: TcpPose,
        z_offset_m: float,
        minimum_safe_z: float,
        source: str,
    ) -> TcpPose:
        corrected_z = pose.z + z_offset_m
        if corrected_z < minimum_safe_z:
            self.get_logger().warning(
                f"Clamping {source} grasp TCP Z from {corrected_z:.4f}m "
                f"to safe floor {minimum_safe_z:.4f}m"
            )
            corrected_z = minimum_safe_z
        return TcpPose(pose.x, pose.y, corrected_z, pose.rx, pose.ry, pose.rz)

    def _revalidate_target_at_hover(
        self,
        target: TcpPose,
        motion: dict[str, int],
        z_offset_m: float,
        minimum_safe_z: float,
        previous_count: int,
    ) -> TcpPose:
        """Use the latest background-tracked target before any downward motion."""

        live_pose, live_count, age_s = self._latest_fresh_pregrasp_pose(previous_count)
        live_target = self._apply_grasp_z_safety(
            live_pose,
            z_offset_m,
            minimum_safe_z,
            "live pregrasp",
        )
        position_delta, angle_delta = self._pose_delta(target, live_target)
        position_tolerance = max(
            0.001,
            float(self.get_parameter("pregrasp_position_tolerance_m").value),
        )
        angle_tolerance = max(
            0.5,
            float(self.get_parameter("pregrasp_angle_tolerance_deg").value),
        )
        max_correction = max(
            position_tolerance,
            float(self.get_parameter("pregrasp_max_correction_m").value),
        )
        max_correction_angle = max(
            angle_tolerance,
            float(self.get_parameter("pregrasp_max_correction_angle_deg").value),
        )
        current = self._current_command_pose()
        min_hover_clearance = max(
            0.005,
            float(self.get_parameter("pregrasp_min_hover_clearance_m").value),
        )
        if current.z - live_target.z < min_hover_clearance:
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    f"safe hover clearance is only {(current.z - live_target.z)*1000:.1f}mm, "
                    f"below required {min_hover_clearance*1000:.1f}mm; "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )
        self.get_logger().info(
            f"Pregrasp target check: age={age_s*1000:.0f}ms, "
            f"position_delta={position_delta*1000:.1f}mm, angle_delta={angle_delta:.1f}deg"
        )

        if position_delta > max_correction or angle_delta > max_correction_angle:
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    f"D405 target change is outside safe hover-correction limits: "
                    f"position_delta={position_delta*1000:.1f}mm/{max_correction*1000:.1f}mm, "
                    f"angle_delta={angle_delta:.1f}deg/{max_correction_angle:.1f}deg; "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )

        if position_delta <= position_tolerance and angle_delta <= angle_tolerance:
            # Even without a correction move, use the latest pose for the final
            # descent so a small drift is not discarded.
            return live_target

        self._publish_status(
            f"pregrasp target shifted {position_delta*1000:.1f}mm/{angle_delta:.1f}deg; "
            "correcting at safe hover"
        )
        correction_hover = TcpPose(
            live_target.x,
            live_target.y,
            current.z,
            live_target.rx,
            live_target.ry,
            live_target.rz,
        )
        try:
            self.controller.inverse_kinematics(
                correction_hover,
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
                joint_near=self.controller.current_joint(),
            )
            self._require_cycle_active("before pregrasp hover correction")
            self.controller.move_joint_tcp(
                correction_hover,
                speed=motion["joint_speed"],
                accel=motion["joint_pose_acc"],
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
            )
            self._require_cycle_active("after pregrasp hover correction")
        except RecoverableGraspError:
            raise
        except Exception as exc:
            raise RecoverableGraspError(
                stage="pregrasp hover correction",
                message=f"could not safely correct the shifted target: {exc}",
                needs_vertical_retreat=False,
            ) from exc

        # A second update must arrive during the correction. If the object keeps
        # moving or tracking is lost, do not chase it downward; recover to startup.
        corrected_pose, _, corrected_age_s = self._latest_fresh_pregrasp_pose(live_count)
        corrected_target = self._apply_grasp_z_safety(
            corrected_pose,
            z_offset_m,
            minimum_safe_z,
            "corrected live pregrasp",
        )
        post_correction_delta, post_correction_angle = self._pose_delta(
            live_target,
            corrected_target,
        )
        if (
            post_correction_delta > position_tolerance
            or post_correction_angle > angle_tolerance
        ):
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    f"D405 target was not stable after hover correction: "
                    f"delta={post_correction_delta*1000:.1f}mm/{post_correction_angle:.1f}deg, "
                    f"age={corrected_age_s*1000:.0f}ms; "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )
        self._publish_status("pregrasp target confirmed after hover correction")
        return corrected_target

    def _execute_one_cycle(self, target: TcpPose, width_m: float, height_m: float, length_m: float) -> None:
        motion = self._motion_profile()
        z_offset_m = float(self.get_parameter("grasp_z_offset_m").value)
        z_offset_limit_m = abs(float(self.get_parameter("grasp_z_offset_limit_m").value))
        if abs(z_offset_m) > z_offset_limit_m:
            raise RuntimeError(
                f"grasp_z_offset_m={z_offset_m:.4f} exceeds limit {z_offset_limit_m:.4f}m"
            )
        vision_target_z = target.z
        minimum_safe_z = float(self.get_parameter("minimum_safe_tcp_z_m").value)
        target = self._apply_grasp_z_safety(
            target,
            z_offset_m,
            minimum_safe_z,
            "vision",
        )
        self._publish_status(
            f"grasp plan: vision_Z={vision_target_z*1000:.1f}mm, "
            f"Z_correction={z_offset_m*1000:+.1f}mm, command_Z={target.z*1000:.1f}mm, "
            f"box_height={height_m*1000:.1f}mm"
        )
        with self._timed_stage("grasp_prepare"):
            self._publish_status("commanding measured gripper width; pre-shaping during move-above")
            max_opening = float(self.get_parameter("dh_max_opening_m").value)
            width_m = max(0.0, min(max_opening, width_m))
            pre_shape_position = width_m / max_opening
            pre_shape_initial = self.gripper.read_position()
            # 夹爪预张开和机械臂前往目标上方互不冲突。先下发非阻塞命令，
            # 到达目标上方后再确认夹爪已经停止，从串行流程中隐藏约 0.6 秒。
            self.gripper.set_position(pre_shape_position, wait=False)
            self._require_cycle_active("after commanding gripper pre-shape")

            self.controller.inverse_kinematics(
                target,
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
                joint_near=self.controller.current_joint(),
            )
            current = self._current_command_pose()
            planar = TcpPose(target.x, target.y, current.z, target.rx, target.ry, target.rz)
        self._publish_status("moving above selected box")
        with self.data_lock:
            pregrasp_reference_count = self.pregrasp_pose_count
        with self._timed_stage("move_above"):
            self.controller.move_joint_tcp(
                planar,
                speed=motion["joint_speed"],
                accel=motion["joint_pose_acc"],
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
            )
        self._require_cycle_active("after move-above")
        with self._timed_stage("gripper_preshape_wait"):
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=pre_shape_position,
                initial_position=pre_shape_initial,
            )
        self._require_cycle_active("after confirming gripper pre-shape")
        with self._timed_stage("pregrasp_revalidation"):
            target = self._revalidate_target_at_hover(
                target,
                motion,
                z_offset_m,
                minimum_safe_z,
                pregrasp_reference_count,
            )
        self._publish_status("descending TCP tip to 75% box height")
        with self._timed_stage("grasp_descend"):
            self.controller.move_linear_tcp(
                target,
                speed=motion["linear_speed"],
                accel=motion["linear_acc"],
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
            )
        self._require_cycle_active("at grasp depth")
        self._publish_status("at grasp depth; closing gripper now")
        with self._timed_stage("gripper_close"):
            self.gripper.set_force(int(self.get_parameter("dh_grasp_force").value))
            self.gripper.close(wait=True)
        self._require_cycle_active("after gripper close")
        with self._timed_stage("grasp_confirm"):
            self._confirm_grasp_before_lift(max_opening)

        with self._timed_stage("grasp_lift"):
            self._relative_user_move(
                z=float(self.get_parameter("grasp_lift_m").value),
                label="lifting grasp",
                speed_factor=motion["grasp_lift_speed"],
                accel_factor=motion["grasp_lift_acc"],
            )
        self._require_cycle_active("after grasp lift")
        with self._timed_stage("post_lift_grasp_check"):
            self._validate_grasp_feedback("after lift", max_opening)

        # 扫码器可能在机械臂前往 transfer_joint 的途中就读到条码。必须在
        # 运动前开启窗口，否则这条比“到达中转点”早几十毫秒的消息会被回调丢弃。
        self._reset_barcode_window()
        self._publish_status("moving to barcode transfer joint; barcode window armed")
        with self._timed_stage("move_transfer"):
            self.controller.move_joint(
                self._six_values("transfer_joint"),
                speed=motion["transfer_speed"],
                accel=motion["transfer_acc"],
            )
        self._require_cycle_active("at barcode transfer joint")

        # 到达中转点后先检查“运动途中”捕获的码。已经扫到时无需再靠近
        # 扫码器，也无需转 J6；保留 scanner_approach_m=0，让后面的安全
        # 退让只执行额外 X- 余量。
        early_barcode = self._current_stable_barcode()
        if early_barcode:
            scanner_approach_m = 0.0
            self._publish_status(
                f"barcode acquired before scanner approach: {early_barcode}; "
                "skipping User X+ approach and J6 search"
            )
        else:
            self._publish_status("no barcode at transfer joint; approaching scanner adaptively")
            with self._timed_stage("scanner_approach"):
                scanner_approach_m = self._move_box_to_scanner(length_m)
            self._require_cycle_active("at adaptive barcode distance")
            self._publish_status("adaptive barcode distance reached; checking captured barcode")
        with self._timed_stage("barcode_acquisition"):
            barcode = self._rotate_until_stable_barcode()
        if barcode:
            self._publish_status(f"stable barcode acquired: {barcode}")
        else:
            self.get_logger().warning(
                "No stable barcode after checking all faces; continuing with placement"
            )
            self._publish_status(
                "no barcode acquired; continuing with scanner retreat and placement"
            )

        # 此时盒子侧面仍贴近扫码器，不能直接执行带 Ry/Rz 的关节 PTP，
        # 否则中间关节轨迹的旋转包络可能扫到扫码器。先沿 User X- 原路
        # 退出实际靠近距离，确认退让完成后才进入扫码后组合运动。
        with self._timed_stage("scanner_retreat"):
            self._retreat_box_from_scanner(scanner_approach_m)
        self._require_cycle_active("after scanner safety retreat")

        with self._timed_stage("post_scan_combined_ptp"):
            self._move_to_user_xyz_with_rotation(
                [float(value) for value in self.get_parameter("scan_exit_user_xyz").value],
                ry_delta_deg=float(self.get_parameter("face_up_user_ry_deg").value),
                rz_delta_deg=float(self.get_parameter("post_scan_user_rz_deg").value),
            )
        self._require_cycle_active("after combined post-scan XYZ/Ry/Rz PTP")
        with self._timed_stage("post_scan_grasp_check"):
            self._validate_grasp_feedback(
                "after combined post-scan pose motion",
                float(self.get_parameter("dh_max_opening_m").value),
            )

        place = self._six_values("place_pose")
        place_pose = TcpPose(*place)
        self._publish_status("moving to final place pose")
        with self._timed_stage("move_place"):
            self.controller.move_joint_tcp(
                place_pose,
                speed=motion["place_speed"],
                accel=motion["place_acc"],
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
            )
        self._require_cycle_active("at final place pose")
        with self._timed_stage("gripper_open_place"):
            max_opening = float(self.get_parameter("dh_max_opening_m").value)
            release_clearance = max(
                0.0,
                float(self.get_parameter("place_release_clearance_m").value),
            )
            current_position = self.gripper.read_position()
            current_opening = current_position * max_opening
            release_opening = min(max_opening, current_opening + release_clearance)
            release_position = release_opening / max_opening
            actual_clearance = release_opening - current_opening
            self._publish_status(
                f"releasing box at fixed place pose: opening "
                f"{current_opening*1000:.1f}->{release_opening*1000:.1f}mm "
                f"(clearance +{actual_clearance*1000:.1f}mm)"
            )
            self.gripper.set_position(release_position, wait=False)
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=release_position,
                initial_position=current_position,
            )
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

    def _relative_user_move(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        label="relative user move",
        speed_factor: int | None = None,
        accel_factor: int | None = None,
    ) -> None:
        # speed_factor 已经是合成后的单条指令速度，不会再与全局速度或
        # VelJ 重复相乘。扫码器靠近等独立阶段可显式传入自己的有效速度。
        motion = self._motion_profile()
        motion_speed = (
            motion["joint_speed"]
            if speed_factor is None
            else max(1, min(100, int(speed_factor)))
        )
        motion_accel = (
            motion["joint_pose_acc"]
            if accel_factor is None
            else max(1, min(100, int(accel_factor)))
        )
        self._publish_status(label)
        self.controller.rel_move_user_joint(
            TcpPose(float(x), float(y), float(z), float(rx), float(ry), float(rz)),
            speed=motion_speed,
            accel=motion_accel,
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )

    def _move_box_to_scanner(self, length_m: float) -> float:
        """按盒子长边自适应靠近扫码器，同时保持指定的侧面间隙。

        前提：机械臂已经到达 ``transfer_joint``，User 0 的 X+ 必须指向
        扫码器。``length_m`` 是夹持中心两侧的完整盒长，因此从中心到靠近
        扫码器的一侧是 ``length_m / 2``。返回实际沿 User X+ 移动的距离，
        供扫码成功后沿 X- 精确安全退让。
        """
        center_distance_m = float(self.get_parameter("scanner_center_distance_m").value)
        clearance_m = float(self.get_parameter("scanner_face_clearance_m").value)
        if center_distance_m <= 0.0:
            raise RuntimeError(
                f"scanner_center_distance_m must be positive, got {center_distance_m:.4f}m"
            )
        if clearance_m < 0.0 or clearance_m >= center_distance_m:
            raise RuntimeError(
                f"scanner_face_clearance_m={clearance_m:.4f}m must be in "
                f"[0, {center_distance_m:.4f})m"
            )
        if length_m <= 0.0:
            raise RuntimeError(f"Vision box length must be positive, got {length_m:.4f}m")

        # 从 TCP 中心到扫码器的距离中，扣除盒子的半长和要求保留的间隙，
        # 剩余值就是机械臂需要沿 User 0 X+ 前进的距离。
        approach_m = center_distance_m - 0.5 * length_m - clearance_m
        if approach_m < 0.0:
            negative_tolerance_m = max(
                0.0,
                float(self.get_parameter("scanner_approach_negative_tolerance_m").value),
            )
            if approach_m >= -negative_tolerance_m:
                self.get_logger().warning(
                    f"Scanner approach is {approach_m*1000:.1f}mm below the nominal clearance "
                    f"due to box-length/distance tolerance; clamping X+ approach to 0.0mm "
                    f"(allowed negative tolerance={negative_tolerance_m*1000:.1f}mm)"
                )
                approach_m = 0.0
            else:
                raise RuntimeError(
                    f"Unsafe scanner approach: center_distance={center_distance_m*1000:.1f}mm - "
                    f"length/2={0.5*length_m*1000:.1f}mm - clearance={clearance_m*1000:.1f}mm "
                    f"= {approach_m*1000:.1f}mm; X+ motion refused"
                )
        self._require_cycle_active("before adaptive scanner approach")
        formula = (
            f"scanner approach User X+: {center_distance_m*1000:.1f} - "
            f"{length_m*1000:.1f}/2 - {clearance_m*1000:.1f} "
            f"= {approach_m*1000:.1f}mm"
        )
        if approach_m <= 0.0005:
            self._publish_status(formula + "; already at requested clearance")
            return 0.0
        scanner_speed = self._motion_profile()["scanner_approach_speed"]
        actual_x_m, barcode_seen = self._monitored_scanner_approach(
            approach_m,
            scanner_speed,
            self._motion_profile()["scanner_approach_acc"],
        )
        self._require_cycle_active("after adaptive scanner approach")
        tolerance_m = max(0.001, float(self.get_parameter("jog_tolerance_m").value) * 1.5)
        # A barcode can intentionally stop the approach before the requested
        # endpoint, so displacement mismatch is only an error when no barcode
        # was captured during the monitored move.
        if not barcode_seen and abs(actual_x_m - approach_m) > tolerance_m:
            raise RuntimeError(
                f"Scanner approach X displacement mismatch: requested={approach_m*1000:.1f}mm, "
                f"actual={actual_x_m*1000:.1f}mm"
            )
        self._publish_status(
            f"scanner approach reached: User X moved {actual_x_m*1000:.1f}mm; "
            f"box-to-scanner clearance={clearance_m*1000:.1f}mm"
            + ("; barcode captured and approach stopped" if barcode_seen else "")
        )
        return float(actual_x_m)

    def _monitored_scanner_approach(
        self,
        target_distance_m: float,
        speed_percent: int,
        accel_percent: int,
    ) -> tuple[float, bool]:
        """Move a bounded User-X distance, interrupting it when a code arrives."""
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        timeout_s = max(
            1.0,
            float(self.get_parameter("scanner_approach_monitor_timeout_s").value),
        )
        period_s = max(
            0.005,
            min(0.05, float(self.get_parameter("scanner_approach_monitor_period_s").value)),
        )
        start_pose = self._current_command_pose()
        start_x = float(start_pose.x)
        barcode_seen = False
        motion_stopped = False
        natural_finish = False
        move_errors: list[BaseException] = []

        def run_bounded_move() -> None:
            try:
                self.controller.rel_move_user_joint(
                    TcpPose(target_distance_m, 0.0, 0.0, 0.0, 0.0, 0.0),
                    speed=int(speed_percent),
                    accel=int(accel_percent),
                    user_index=user_index,
                    tool_index=tool_index,
                )
            except BaseException as exc:
                move_errors.append(exc)

        move_thread = threading.Thread(
            target=run_bounded_move,
            name="scanner-approach-motion",
            daemon=True,
        )
        started_at = time.monotonic()
        self._publish_status(
            f"bounded scanner approach User X+: {target_distance_m*1000:.1f}mm; "
            f"speed={speed_percent}%, accel={accel_percent}%; "
            "stop immediately on barcode"
        )
        move_thread.start()
        try:
            while move_thread.is_alive():
                self._require_cycle_active("during monitored scanner approach")
                with self.barcode_lock:
                    if self.barcode_hits >= max(1, int(self.get_parameter("barcode_stable_hits").value)):
                        barcode_seen = True
                        try:
                            current_pose = self._current_command_pose()
                            progress_m = float(current_pose.x) - start_x
                            remaining_m = target_distance_m - progress_m
                        except Exception:
                            remaining_m = float("inf")
                        finish_margin_m = max(
                            0.0,
                            float(self.get_parameter("scanner_approach_natural_finish_margin_m").value),
                        )
                        if 0.0 <= remaining_m <= finish_margin_m:
                            natural_finish = True
                            self._publish_status(
                                f"barcode acquired during scanner approach: {self.barcode_value}; "
                                f"remaining={remaining_m*1000:.1f}mm, allowing bounded User X+ to finish"
                            )
                        else:
                            self._publish_status(
                                f"barcode acquired during scanner approach: {self.barcode_value}; "
                                "stopping bounded User X+ immediately"
                            )
                            self.controller.stop_motion()
                            motion_stopped = True
                        break
                if time.monotonic() - started_at > timeout_s:
                    raise RuntimeError(
                        f"Monitored scanner approach timed out: "
                        f"target={target_distance_m*1000:.1f}mm"
                    )
                time.sleep(period_s)
        finally:
            if move_thread.is_alive() and not motion_stopped and not natural_finish:
                try:
                    self.controller.stop_motion()
                    motion_stopped = True
                except Exception as stop_exc:
                    self.get_logger().warning(
                        f"Stopping bounded scanner approach returned: {stop_exc}"
                    )
            move_thread.join(timeout=2.0)

        if move_thread.is_alive():
            raise RuntimeError("Scanner approach motion thread did not stop after cancellation")
        if move_errors and not barcode_seen:
            raise move_errors[0]

        final_pose = self._current_command_pose()
        actual_x_m = max(0.0, float(final_pose.x) - start_x)
        return actual_x_m, barcode_seen

    def _retreat_box_from_scanner(self, actual_approach_m: float) -> None:
        """扫码后保持姿态沿 User X- 退回，给后续旋转留出安全空间。"""
        extra_retreat_m = float(self.get_parameter("scanner_retreat_extra_m").value)
        if extra_retreat_m < 0.0 or extra_retreat_m > 0.200:
            raise RuntimeError(
                f"scanner_retreat_extra_m must be in [0, 0.200]m, "
                f"got {extra_retreat_m:.4f}m"
            )
        retreat_m = abs(float(actual_approach_m)) + extra_retreat_m
        if retreat_m <= 0.0005:
            self._publish_status("scanner safety retreat not required; no X+ approach was made")
            return

        retreat_speed = self._motion_profile()["scanner_retreat_speed"]
        self._require_cycle_active("before scanner safety retreat")
        tcp_before = self._current_command_pose()
        self._relative_user_move(
            x=-retreat_m,
            label=(
                f"scanner safety retreat User X-: approach return "
                f"{abs(float(actual_approach_m))*1000:.1f} + extra "
                f"{extra_retreat_m*1000:.1f} = {retreat_m*1000:.1f}mm "
                f"at {retreat_speed}% before XYZ/Ry/Rz PTP"
            ),
            speed_factor=retreat_speed,
            accel_factor=self._motion_profile()["scanner_retreat_acc"],
        )
        self._require_cycle_active("during scanner safety retreat")
        tcp_after = self._current_command_pose()
        actual_retreat_m = tcp_after.x - tcp_before.x
        expected_retreat_m = -retreat_m
        tolerance_m = max(
            0.001,
            float(self.get_parameter("jog_tolerance_m").value) * 1.5,
        )
        if abs(actual_retreat_m - expected_retreat_m) > tolerance_m:
            raise RuntimeError(
                f"Scanner safety retreat X displacement mismatch: "
                f"requested={expected_retreat_m*1000:.1f}mm, "
                f"actual={actual_retreat_m*1000:.1f}mm; combined motion refused"
            )
        self._publish_status(
            f"scanner safety retreat completed: User X moved "
            f"{actual_retreat_m*1000:.1f}mm; combined motion is now permitted"
        )

    def _reset_barcode_window(self) -> None:
        with self.barcode_lock:
            self.barcode_window_active = True
            self.barcode_value = ""
            self.barcode_hits = 0
            self.barcode_last_time = 0.0

    def _current_stable_barcode(self) -> str:
        """无等待读取当前扫码窗口；用于决定是否跳过靠近和 J6 找码。"""
        required_hits = max(1, int(self.get_parameter("barcode_stable_hits").value))
        with self.barcode_lock:
            if self.barcode_window_active and self.barcode_hits >= required_hits:
                return self.barcode_value
        return ""

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

    def _rotate_barcode_flip_joint(
        self,
        delta_deg: float,
        face_index: int,
        required_hits: int,
        previous_face_anchor_deg: float,
        next_face_anchor_deg: float,
    ) -> str:
        """点动指定关节寻找条码，识别成功时立即停止并返回码值。"""
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
        start_joint_deg = float(current_joints[watch_index])
        target_joint_deg = float(next_face_anchor_deg)
        direction = 1.0 if delta_deg > 0.0 else -1.0
        axis_name = f"J{watch_index + 1}{'+' if direction > 0.0 else '-'}"
        target_progress_deg = abs(float(delta_deg))
        tolerance_deg = max(
            0.2,
            abs(float(self.get_parameter("barcode_flip_jog_tolerance_deg").value)),
        )
        timeout_s = max(1.0, float(self.get_parameter("barcode_flip_jog_timeout_s").value))
        motion = self._motion_profile()
        barcode_speed = motion["barcode_jog_speed"]
        self._publish_status(
            f"barcode face {face_index}: monitored {axis_name} jog "
            f"{start_joint_deg:.1f}->{target_joint_deg:.1f}deg at {barcode_speed}%; "
            f"stop immediately on scan"
        )

        captured_barcode = ""
        jog_started = False
        progress_deg = 0.0
        start_time = time.monotonic()
        try:
            self.controller.set_speed_factor(barcode_speed)
            self.controller.move_jog(
                axis_name,
                coord_type=1,
                user=int(self.get_parameter("user_index").value),
                tool=int(self.get_parameter("command_tool_index").value),
            )
            jog_started = True
            while True:
                self._require_cycle_active(f"during barcode {axis_name} jog")
                # 条码回调运行在 ROS executor 线程中；这里每 5 ms 检查一次，
                # 让扫码成功后的 J6 停止和后续阶段衔接更紧。
                # 一旦达到稳定次数，立即退出循环，并在 finally 中停止 J6。
                with self.barcode_lock:
                    if self.barcode_hits >= required_hits:
                        captured_barcode = self.barcode_value
                        break

                if time.monotonic() - start_time > timeout_s:
                    raise RuntimeError(
                        f"Barcode {axis_name} jog timed out: "
                        f"progress={progress_deg:.1f}/{target_progress_deg:.1f}deg"
                    )
                joints = self.controller.current_joint()
                if len(joints) != 6:
                    raise RuntimeError(
                        f"Current joint feedback must contain 6 values during jog, got {len(joints)}"
                    )
                current_joint_deg = float(joints[watch_index])
                progress_deg = direction * (current_joint_deg - start_joint_deg)
                safe_limit = abs(float(self.get_parameter("barcode_flip_safe_joint_limit_deg").value))
                if abs(current_joint_deg) > safe_limit:
                    raise RuntimeError(
                        f"Barcode {axis_name} jog exceeded safe joint limit: "
                        f"J{watch_index + 1}={current_joint_deg:.1f}deg"
                    )
                if progress_deg >= target_progress_deg - tolerance_deg:
                    break
                time.sleep(0.005)
        finally:
            if jog_started:
                try:
                    # Empty MoveJog stops only the active jog; unlike dashboard
                    # Stop(), this is not treated as an operator emergency stop.
                    self.controller.move_jog("")
                    self.controller.wait_until_idle(timeout_s=5.0)
                except Exception as stop_exc:
                    self.get_logger().warning(
                        f"Stopping barcode {axis_name} jog returned: {stop_exc}"
                    )
            try:
                # 避免 J6 专用速度残留并影响后续普通机械臂动作。
                self.controller.set_speed_factor(100)
            except Exception as speed_exc:
                self.get_logger().warning(
                    f"Restoring robot speed factor after barcode jog failed: {speed_exc}"
                )

        self._require_cycle_active(f"after barcode {axis_name} jog")
        final_joints = self.controller.current_joint()
        if len(final_joints) != 6:
            raise RuntimeError(
                f"Current joint feedback must contain 6 values after jog, got {len(final_joints)}"
            )
        stopped_joint_deg = float(final_joints[watch_index])

        # 只有真正扫到条码时才执行标准面对齐。没有扫码时不再浪费时间
        # 用 MovJ 修正点动的几度减速超调；下一轮会根据最初的标准面锚点
        # 重新计算剩余角度，因此超调不会逐轮累积，也不会影响最终精度。
        # 扫到码时比较停止角到前后两个标准锚点的距离，吸附到最近一面，
        # 使后续固定 User Ry -90° 能把条码面准确翻到正上方。
        snap_enabled = bool(self.get_parameter("barcode_snap_to_nearest_face").value)
        if captured_barcode and snap_enabled:
            distance_to_previous = abs(stopped_joint_deg - float(previous_face_anchor_deg))
            distance_to_next = abs(stopped_joint_deg - float(next_face_anchor_deg))
            if distance_to_previous <= distance_to_next:
                face_anchor = "previous"
                snap_target_deg = float(previous_face_anchor_deg)
            else:
                face_anchor = "next"
                snap_target_deg = float(next_face_anchor_deg)
        else:
            face_anchor = "not-scanned"
            snap_target_deg = stopped_joint_deg

        snap_correction_deg = snap_target_deg - stopped_joint_deg
        if abs(snap_correction_deg) > 0.2:
            self._publish_status(
                f"aligning barcode face to {face_anchor} 90deg anchor: "
                f"J{watch_index + 1} {stopped_joint_deg:.1f}->{snap_target_deg:.1f}deg "
                f"(correction {snap_correction_deg:+.1f}deg)"
            )
            aligned_joints = [float(value) for value in final_joints]
            aligned_joints[watch_index] = snap_target_deg
            self.controller.move_joint(
                aligned_joints,
                speed=motion["barcode_alignment_speed"],
                accel=motion["barcode_alignment_acc"],
            )
            self._require_cycle_active(f"after barcode face alignment to {face_anchor} anchor")
            final_joints = self.controller.current_joint()

        tcp_after = self._current_command_pose()
        xyz_shift_mm = 1000.0 * math.sqrt(
            (tcp_after.x - tcp_before.x) ** 2
            + (tcp_after.y - tcp_before.y) ** 2
            + (tcp_after.z - tcp_before.z) ** 2
        )
        final_joint_deg = float(final_joints[watch_index]) if len(final_joints) == 6 else float("nan")
        if captured_barcode:
            self._publish_status(
                f"barcode acquired during {axis_name} jog: {captured_barcode}; "
                f"raw stop={stopped_joint_deg:.1f}deg, aligned {face_anchor} face at "
                f"J{watch_index + 1}={final_joint_deg:.1f}deg"
            )
        else:
            self.get_logger().info(
                f"Barcode {axis_name} jog reached next face: "
                f"J{watch_index + 1}={final_joint_deg:.1f}deg"
            )
        self.get_logger().info(
            f"Monitored barcode rotation stopped: TCP XYZ shift={xyz_shift_mm:.2f}mm"
        )
        self._validate_grasp_feedback(
            f"after barcode face {face_index}",
            float(self.get_parameter("dh_max_opening_m").value),
        )
        return captured_barcode

    def _wait_for_current_barcode(self, required_hits: int, timeout_s: float, stage: str) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._require_cycle_active(stage)
            with self.barcode_lock:
                if self.barcode_hits >= required_hits:
                    return self.barcode_value
            time.sleep(0.005)
        return ""

    def _rotate_until_stable_barcode(self) -> str:
        required_hits = max(1, int(self.get_parameter("barcode_stable_hits").value))
        # Inspect the face already presented at the transfer joint, then make
        # at most three monitored J6 quarter turns. Each turn stops early when
        # a barcode is decoded. Keeping this fixed at four faces also prevents
        # an accidental mouse-wheel change in the GUI from shortening search.
        max_faces = 4
        wait_s = max(0.02, float(self.get_parameter("barcode_face_wait_s").value))
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
                self._publish_status(f"barcode already acquired before J6 search: {value}")
                return value

            watch_index = max(
                0,
                min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)),
            )
            anchor_joints = self.controller.current_joint()
            if len(anchor_joints) != 6:
                raise RuntimeError(
                    f"Current joint feedback must contain 6 values at barcode anchor, "
                    f"got {len(anchor_joints)}"
                )
            # 后续所有面的绝对锚点都基于这一初始 J6，防止点动停止时的
            # 几度超调被带入下一轮并逐步累积。
            first_face_anchor_deg = float(anchor_joints[watch_index])

            # max_faces includes the initial unrotated face, hence at most
            # max_faces - 1 wrist rotations, matching the old max_steps=3 loop.
            for flip_index in range(1, max_faces):
                self._require_cycle_active(f"before barcode face rotation {flip_index}")
                self._reset_barcode_window()
                previous_face_anchor_deg = first_face_anchor_deg + (flip_index - 1) * preferred_delta
                next_face_anchor_deg = first_face_anchor_deg + flip_index * preferred_delta
                current_joints = self.controller.current_joint()
                if len(current_joints) != 6:
                    raise RuntimeError(
                        f"Current joint feedback must contain 6 values before barcode rotation, "
                        f"got {len(current_joints)}"
                    )
                # 无扫码的上一轮不回正；这里直接扣除实际超调量，仍然朝
                # 最初中转姿态定义的下一个整数 90° 锚点转动。
                required_delta = next_face_anchor_deg - float(current_joints[watch_index])
                selected_delta = self._select_safe_barcode_flip_delta(required_delta)
                if selected_delta is None:
                    raise RuntimeError("No safe equivalent J6 barcode rotation is available")
                value = self._rotate_barcode_flip_joint(
                    selected_delta,
                    flip_index + 1,
                    required_hits,
                    previous_face_anchor_deg,
                    next_face_anchor_deg,
                )
                if value:
                    return value
                value = self._wait_for_current_barcode(
                    required_hits,
                    wait_s,
                    f"checking barcode after J6 rotation {flip_index}",
                )
                if value:
                    # 条码也可能在点动停止后的本面等待阶段才到达。此时
                    # 已确认扫码成功，才值得执行一次标准面对齐。
                    aligned_joints = self.controller.current_joint()
                    if len(aligned_joints) != 6:
                        raise RuntimeError(
                            f"Current joint feedback must contain 6 values before barcode alignment, "
                            f"got {len(aligned_joints)}"
                        )
                    correction_deg = next_face_anchor_deg - float(aligned_joints[watch_index])
                    if abs(correction_deg) > 0.2:
                        self._publish_status(
                            f"barcode acquired while waiting; aligning J{watch_index + 1} "
                            f"{aligned_joints[watch_index]:.1f}->{next_face_anchor_deg:.1f}deg"
                        )
                        aligned_joints[watch_index] = next_face_anchor_deg
                        self.controller.move_joint(
                            [float(joint) for joint in aligned_joints],
                            speed=self._motion_profile()["barcode_alignment_speed"],
                            accel=self._motion_profile()["barcode_alignment_acc"],
                        )
                        self._require_cycle_active("after waiting-phase barcode face alignment")
                    return value
            # A barcode is useful metadata, but it is not required to finish
            # the physical pick/place cycle.  The caller will still perform
            # the scanner safety retreat, post-scan motion, placement, and
            # return to startup.  Keep this as a normal return so an unread
            # label does not leave a gripped object stranded at the transfer
            # point.
            self.get_logger().warning(
                f"Barcode not stable after checking {max_faces} faces; "
                "continuing without barcode"
            )
            return ""
        finally:
            with self.barcode_lock:
                self.barcode_window_active = False

    def _move_to_user_xyz_with_rotation(
        self,
        target_xyz: list[float],
        ry_delta_deg: float,
        rz_delta_deg: float,
    ) -> None:
        """用一条 User/Tool PTP 同时完成 XYZ、User Ry 和 User Rz 变化。

        姿态组合顺序严格按现场要求：先绕固定 User Y 轴旋转 ``Ry``，再绕
        固定 User Z 轴旋转 ``Rz``。因此最终矩阵为 ``Rz @ Ry @ R_start``。
        """
        if len(target_xyz) != 3:
            raise ValueError("scan_exit_user_xyz must contain 3 values")
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        current = self._current_command_pose()
        start_rotation = SciPyRot.from_euler(
            "xyz",
            [current.rx, current.ry, current.rz],
            degrees=True,
        ).as_matrix()
        user_ry_rotation = SciPyRot.from_euler(
            "y", float(ry_delta_deg), degrees=True
        ).as_matrix()
        user_rz_rotation = SciPyRot.from_euler(
            "z", float(rz_delta_deg), degrees=True
        ).as_matrix()
        target_rotation = user_rz_rotation @ user_ry_rotation @ start_rotation
        target_rx, target_ry, target_rz = SciPyRot.from_matrix(
            target_rotation
        ).as_euler("xyz", degrees=True)
        target = TcpPose(
            float(target_xyz[0]),
            float(target_xyz[1]),
            float(target_xyz[2]),
            float(target_rx),
            float(target_ry),
            float(target_rz),
        )
        motion = self._motion_profile()
        combined_speed = motion["post_scan_speed"]
        self._publish_status(
            f"combined User PTP: XYZ=({target.x*1000:.0f},{target.y*1000:.0f},"
            f"{target.z*1000:.0f})mm, Ry={ry_delta_deg:+.1f}deg then "
            f"Rz={rz_delta_deg:+.1f}deg, target RPY=({target.rx:.1f},"
            f"{target.ry:.1f},{target.rz:.1f})deg, speed={combined_speed}%"
        )

        # 先验证包含位置和最终姿态的完整目标；无逆解时不会下发任何运动。
        self.controller.inverse_kinematics(
            target,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=self.controller.current_joint(),
        )
        self._require_cycle_active("before combined post-scan PTP")
        self.controller.move_joint_tcp(
            target,
            speed=combined_speed,
            accel=motion["post_scan_acc"],
            user_index=user_index,
            tool_index=tool_index,
        )
        self._require_cycle_active("at combined post-scan pose target")

        tolerance = max(0.0005, float(self.get_parameter("jog_tolerance_m").value))
        final = self._current_command_pose()
        errors = [abs(final.x - target_xyz[0]), abs(final.y - target_xyz[1]), abs(final.z - target_xyz[2])]
        if max(errors) > tolerance * 1.5:
            raise RuntimeError(
                f"Combined User PTP final XYZ error too large: "
                f"{[round(e, 4) for e in errors]}m"
            )
        final_rotation = SciPyRot.from_euler(
            "xyz",
            [final.rx, final.ry, final.rz],
            degrees=True,
        ).as_matrix()
        orientation_error_deg = math.degrees(
            SciPyRot.from_matrix(final_rotation @ target_rotation.T).magnitude()
        )
        orientation_tolerance_deg = max(
            1.0,
            float(self.get_parameter("face_up_jog_tolerance_deg").value),
        )
        if orientation_error_deg > orientation_tolerance_deg:
            raise RuntimeError(
                f"Combined User PTP final orientation error too large: "
                f"{orientation_error_deg:.2f}deg > {orientation_tolerance_deg:.2f}deg"
            )
        self._publish_status(
            f"combined User pose reached: XYZ=({final.x*1000:.1f},"
            f"{final.y*1000:.1f},{final.z*1000:.1f})mm, "
            f"RPY=({final.rx:.1f},{final.ry:.1f},{final.rz:.1f})deg"
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
        rotation_speed = self._motion_profile()["face_up_jog_speed"]

        self._publish_status(
            f"user-frame {axis_name} MoveJog to {total_delta_deg:+.1f}deg with fixed TCP XYZ; "
            f"User={user_index}, Tool={tool_index}, speed_factor="
            f"{rotation_speed}%, XYZ=({fixed_xyz[0]*1000:.1f},"
            f"{fixed_xyz[1]*1000:.1f},{fixed_xyz[2]*1000:.1f})mm"
        )
        jog_started = False
        directed_progress_deg = 0.0
        start_time = time.monotonic()
        last_log_progress = -10.0
        try:
            # 纯姿态旋转使用独立速度，不与普通关节运动、J6 找码或扫码器
            # 靠近速度联动；jog_speed_factor 仍只属于扫码后的 XYZ PTP。
            self.controller.set_speed_factor(rotation_speed)
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
                self.controller.set_speed_factor(100)
            except Exception as speed_exc:
                self.get_logger().warning(f"Restoring robot speed factor failed: {speed_exc}")

        self.controller.wait_until_idle(timeout_s=5.0)
        # 控制器已经确认 idle，只保留很短的可调反馈稳定时间，避免每轮
        # 无条件多等原来的 0.3 秒。
        face_up_settle_s = max(
            0.0,
            float(self.get_parameter("face_up_settle_s").value),
        )
        if face_up_settle_s > 0.0:
            time.sleep(face_up_settle_s)
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
        self.shutting_down = True
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
        self.motion_speed_scale = QSpinBox(); self.motion_speed_scale.setRange(50, 400); self.motion_speed_scale.setSingleStep(5); self.motion_speed_scale.setValue(int(self.node.get_parameter("motion_speed_scale_percent").value))
        self.joint_speed = QSpinBox(); self.joint_speed.setRange(1, 100); self.joint_speed.setValue(int(self.node.get_parameter("joint_speed").value))
        self.joint_acc = QSpinBox(); self.joint_acc.setRange(1, 100); self.joint_acc.setValue(int(self.node.get_parameter("joint_acc").value))
        self.grasp_lift_speed = QSpinBox(); self.grasp_lift_speed.setRange(1, 100); self.grasp_lift_speed.setValue(int(self.node.get_parameter("grasp_lift_speed_factor").value))
        self.grasp_lift_acc = QSpinBox(); self.grasp_lift_acc.setRange(1, 100); self.grasp_lift_acc.setValue(int(self.node.get_parameter("grasp_lift_acc_factor").value))
        self.transfer_speed = QSpinBox(); self.transfer_speed.setRange(1, 100); self.transfer_speed.setValue(int(self.node.get_parameter("transfer_speed_factor").value))
        self.transfer_acc = QSpinBox(); self.transfer_acc.setRange(1, 100); self.transfer_acc.setValue(int(self.node.get_parameter("transfer_acc_factor").value))
        self.post_scan_acc = QSpinBox(); self.post_scan_acc.setRange(1, 100); self.post_scan_acc.setValue(int(self.node.get_parameter("post_scan_acc_factor").value))
        self.place_speed = QSpinBox(); self.place_speed.setRange(1, 100); self.place_speed.setValue(int(self.node.get_parameter("place_speed_factor").value))
        self.place_acc = QSpinBox(); self.place_acc.setRange(1, 100); self.place_acc.setValue(int(self.node.get_parameter("place_acc_factor").value))
        self.return_startup_speed = QSpinBox(); self.return_startup_speed.setRange(1, 100); self.return_startup_speed.setValue(int(self.node.get_parameter("return_startup_speed_factor").value))
        self.return_startup_acc = QSpinBox(); self.return_startup_acc.setRange(1, 100); self.return_startup_acc.setValue(int(self.node.get_parameter("return_startup_acc_factor").value))
        self.linear_speed = QSpinBox(); self.linear_speed.setRange(1, 100); self.linear_speed.setValue(int(self.node.get_parameter("linear_speed").value))
        self.linear_acc = QSpinBox(); self.linear_acc.setRange(1, 100); self.linear_acc.setValue(int(self.node.get_parameter("linear_acc").value))
        self.jog_speed = QSpinBox(); self.jog_speed.setRange(1, 100); self.jog_speed.setValue(int(self.node.get_parameter("jog_speed_factor").value))
        self.gripper_force = QSpinBox(); self.gripper_force.setRange(20, 100); self.gripper_force.setValue(int(self.node.get_parameter("dh_grasp_force").value))
        self.grasp_z_offset = self._new_double(float(self.node.get_parameter("grasp_z_offset_m").value) * 1000.0, -20.0, 20.0, 1, 0.5)
        self.minimum_safe_tcp_z = self._new_double(float(self.node.get_parameter("minimum_safe_tcp_z_m").value) * 1000.0, -50.0, 100.0, 1, 0.5)
        self.grasp_lift = self._new_double(float(self.node.get_parameter("grasp_lift_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        self.place_release_clearance = self._new_double(float(self.node.get_parameter("place_release_clearance_m").value) * 1000.0, 1.0, 50.0, 1, 1.0)
        self.grasp_close_settle = self._new_double(float(self.node.get_parameter("grasp_close_settle_s").value), 0.0, 5.0, 2, 0.1)
        self.grasp_feedback_wait = self._new_double(float(self.node.get_parameter("grasp_feedback_wait_s").value), 0.1, 5.0, 2, 0.1)
        self.grasp_retry_limit = QSpinBox(); self.grasp_retry_limit.setRange(0, 20); self.grasp_retry_limit.setValue(int(self.node.get_parameter("single_cycle_grasp_retry_limit").value))
        self.feedback_required = QCheckBox("启用空抓检测与自动重试")
        self.feedback_required.setChecked(bool(self.node.get_parameter("grasp_feedback_required").value))
        self.effective_motion_label = QLabel()
        self.effective_motion_label.setWordWrap(True)
        self._refresh_effective_motion_label()
        form.addRow("统一提速比例 %（100=原有效速度）", self.motion_speed_scale)
        form.addRow("当前单指令有效值", self.effective_motion_label)
        form.addRow("关节速度兼容基准 %", self.joint_speed)
        form.addRow("关节加速度兼容基准 %", self.joint_acc)
        form.addRow("抓取后抬升速度兼容基准 %", self.grasp_lift_speed)
        form.addRow("抓取后抬升加速度兼容基准 %", self.grasp_lift_acc)
        form.addRow("抓取后中转位速度兼容基准 %", self.transfer_speed)
        form.addRow("抓取后中转位加速度兼容基准 %", self.transfer_acc)
        form.addRow("直线速度兼容基准 %", self.linear_speed)
        form.addRow("直线加速度兼容基准 %", self.linear_acc)
        form.addRow("扫码后组合 PTP 兼容基准 %", self.jog_speed)
        form.addRow("扫码后组合 PTP 加速度兼容基准 %", self.post_scan_acc)
        form.addRow("最终放置速度兼容基准 %", self.place_speed)
        form.addRow("最终放置加速度兼容基准 %", self.place_acc)
        form.addRow("回初始位速度兼容基准 %", self.return_startup_speed)
        form.addRow("回初始位加速度兼容基准 %", self.return_startup_acc)
        form.addRow("夹爪抓取力", self.gripper_force)
        form.addRow("抓取 Z 修正 mm（正值更浅）", self.grasp_z_offset)
        form.addRow("TCP 最低安全 Z mm", self.minimum_safe_tcp_z)
        form.addRow("抓取后抬升 mm", self.grasp_lift)
        form.addRow("放置释放额外开度 mm", self.place_release_clearance)
        form.addRow("闭合后原地确认秒", self.grasp_close_settle)
        form.addRow("运动状态反馈等待秒", self.grasp_feedback_wait)
        form.addRow("单轮空抓重试次数", self.grasp_retry_limit)
        form.addRow("夹爪反馈保护", self.feedback_required)
        self.apply_button = QPushButton("应用以上全部参数")
        self.apply_button.clicked.connect(self.apply_parameters)
        form.addRow(self.apply_button)
        layout.addWidget(box)

    def _refresh_effective_motion_label(self) -> None:
        if not hasattr(self, "effective_motion_label"):
            return
        profile = self.node._motion_profile()
        self.effective_motion_label.setText(
            "抓取上方关节 {joint_speed}% / 抬升 {grasp_lift_speed}% / "
            "中转 {transfer_speed}% / 组合 PTP {post_scan_speed}% / "
            "放置 {place_speed}% / 回初始 {return_startup_speed}% / "
            "直线 {linear_speed}%·{linear_acc}% / "
            "扫码靠近 {scanner_approach_speed}% / 退让 {scanner_retreat_speed}% / "
            "J6 点动 {barcode_jog_speed}%".format(**profile)
        )

    def _build_barcode_parameters(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("扫码与组合姿态参数")
        form = QFormLayout(box)
        self.barcode_hits = QSpinBox(); self.barcode_hits.setRange(1, 20); self.barcode_hits.setValue(int(self.node.get_parameter("barcode_stable_hits").value))
        self.barcode_rotations = QSpinBox(); self.barcode_rotations.setRange(4, 4); self.barcode_rotations.setValue(4)
        self.barcode_wait = self._new_double(float(self.node.get_parameter("barcode_face_wait_s").value), 0.02, 10.0, 2, 0.01)
        # 三段扫码相关运动分别调速，避免修改普通“关节速度”时全部一起变化。
        self.scanner_approach_speed = QSpinBox(); self.scanner_approach_speed.setRange(1, 100); self.scanner_approach_speed.setValue(int(self.node.get_parameter("scanner_approach_speed_factor").value))
        self.scanner_retreat_speed = QSpinBox(); self.scanner_retreat_speed.setRange(1, 100); self.scanner_retreat_speed.setValue(int(self.node.get_parameter("scanner_retreat_speed_factor").value))
        self.scanner_retreat_acc = QSpinBox(); self.scanner_retreat_acc.setRange(1, 100); self.scanner_retreat_acc.setValue(int(self.node.get_parameter("scanner_retreat_acc_factor").value))
        self.barcode_j6_speed = QSpinBox(); self.barcode_j6_speed.setRange(1, 100); self.barcode_j6_speed.setValue(int(self.node.get_parameter("barcode_j6_speed_factor").value))
        self.barcode_alignment_acc = QSpinBox(); self.barcode_alignment_acc.setRange(1, 100); self.barcode_alignment_acc.setValue(int(self.node.get_parameter("barcode_alignment_acc_factor").value))
        self.barcode_rz = self._new_double(float(self.node.get_parameter("barcode_flip_step_deg").value), -180.0, 180.0, 1, 5.0)
        self.face_up_user_ry = self._new_double(float(self.node.get_parameter("face_up_user_ry_deg").value), -180.0, 180.0, 1, 5.0)
        self.post_scan_user_rz = self._new_double(float(self.node.get_parameter("post_scan_user_rz_deg").value), -180.0, 180.0, 1, 5.0)
        self.face_up_jog_tolerance = self._new_double(float(self.node.get_parameter("face_up_jog_tolerance_deg").value), 0.2, 10.0, 1, 0.2)
        self.scanner_center_distance = self._new_double(float(self.node.get_parameter("scanner_center_distance_m").value) * 1000.0, 1.0, 500.0, 1, 1.0)
        self.scanner_face_clearance = self._new_double(float(self.node.get_parameter("scanner_face_clearance_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        self.scanner_negative_tolerance = self._new_double(float(self.node.get_parameter("scanner_approach_negative_tolerance_m").value) * 1000.0, 0.0, 20.0, 1, 1.0)
        self.scanner_retreat_extra = self._new_double(float(self.node.get_parameter("scanner_retreat_extra_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        form.addRow("同码稳定次数", self.barcode_hits)
        form.addRow("检查面数（旧逻辑固定）", self.barcode_rotations)
        form.addRow("每面等待秒", self.barcode_wait)
        form.addRow("中转 TCP 到扫码器距离 mm", self.scanner_center_distance)
        form.addRow("盒侧面扫码间隙 mm", self.scanner_face_clearance)
        form.addRow("负靠近量容差 mm", self.scanner_negative_tolerance)
        form.addRow("靠近扫码器 User X+ 速度 %", self.scanner_approach_speed)
        form.addRow("扫码后安全退让 User X- 速度 %", self.scanner_retreat_speed)
        form.addRow("扫码后安全退让 User X- 加速度 %", self.scanner_retreat_acc)
        form.addRow("回中转距离后额外退让 mm", self.scanner_retreat_extra)
        form.addRow("J6 多面找码/对齐速度 %", self.barcode_j6_speed)
        form.addRow("J6 标准面对齐加速度 %", self.barcode_alignment_acc)
        form.addRow("找码 J6 步进 deg（旧逻辑）", self.barcode_rz)
        form.addRow("组合目标 User Ry 增量 deg", self.face_up_user_ry)
        form.addRow("组合目标 User Rz 增量 deg", self.post_scan_user_rz)
        form.addRow("组合目标姿态到位容差 deg", self.face_up_jog_tolerance)
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
            Parameter("motion_speed_scale_percent", value=self.motion_speed_scale.value()),
            Parameter("joint_speed", value=self.joint_speed.value()),
            Parameter("joint_acc", value=self.joint_acc.value()),
            Parameter("grasp_lift_speed_factor", value=self.grasp_lift_speed.value()),
            Parameter("grasp_lift_acc_factor", value=self.grasp_lift_acc.value()),
            Parameter("transfer_speed_factor", value=self.transfer_speed.value()),
            Parameter("transfer_acc_factor", value=self.transfer_acc.value()),
            Parameter("linear_speed", value=self.linear_speed.value()),
            Parameter("linear_acc", value=self.linear_acc.value()),
            Parameter("jog_speed_factor", value=float(self.jog_speed.value())),
            Parameter("post_scan_acc_factor", value=self.post_scan_acc.value()),
            Parameter("place_speed_factor", value=self.place_speed.value()),
            Parameter("place_acc_factor", value=self.place_acc.value()),
            Parameter("return_startup_speed_factor", value=self.return_startup_speed.value()),
            Parameter("return_startup_acc_factor", value=self.return_startup_acc.value()),
            Parameter("dh_grasp_force", value=self.gripper_force.value()),
            Parameter("grasp_z_offset_m", value=self.grasp_z_offset.value() / 1000.0),
            Parameter("minimum_safe_tcp_z_m", value=self.minimum_safe_tcp_z.value() / 1000.0),
            Parameter("grasp_lift_m", value=self.grasp_lift.value() / 1000.0),
            Parameter("place_release_clearance_m", value=self.place_release_clearance.value() / 1000.0),
            Parameter("grasp_close_settle_s", value=self.grasp_close_settle.value()),
            Parameter("grasp_feedback_wait_s", value=self.grasp_feedback_wait.value()),
            Parameter("single_cycle_grasp_retry_limit", value=self.grasp_retry_limit.value()),
            Parameter("grasp_feedback_required", value=self.feedback_required.isChecked()),
            Parameter("barcode_stable_hits", value=self.barcode_hits.value()),
            Parameter("barcode_max_face_rotations", value=self.barcode_rotations.value()),
            Parameter("barcode_face_wait_s", value=self.barcode_wait.value()),
            Parameter("scanner_center_distance_m", value=self.scanner_center_distance.value() / 1000.0),
            Parameter("scanner_face_clearance_m", value=self.scanner_face_clearance.value() / 1000.0),
            Parameter("scanner_approach_negative_tolerance_m", value=self.scanner_negative_tolerance.value() / 1000.0),
            Parameter("scanner_approach_speed_factor", value=self.scanner_approach_speed.value()),
            Parameter("scanner_retreat_speed_factor", value=self.scanner_retreat_speed.value()),
            Parameter("scanner_retreat_acc_factor", value=self.scanner_retreat_acc.value()),
            Parameter("scanner_retreat_extra_m", value=self.scanner_retreat_extra.value() / 1000.0),
            Parameter("barcode_j6_speed_factor", value=self.barcode_j6_speed.value()),
            Parameter("barcode_alignment_acc_factor", value=self.barcode_alignment_acc.value()),
            Parameter("barcode_flip_step_deg", value=self.barcode_rz.value()),
            Parameter("face_up_user_ry_deg", value=self.face_up_user_ry.value()),
            Parameter("post_scan_user_rz_deg", value=self.post_scan_user_rz.value()),
            Parameter("face_up_jog_tolerance_deg", value=self.face_up_jog_tolerance.value()),
        ]
        results = self.node.set_parameters(parameters)
        failures = [result.reason for result in results if not result.successful]
        if failures:
            self.cycle_status_label.setText("参数应用失败: " + "; ".join(failures))
            return
        self.node.controller.enable_single_command_motion_scaling()
        self.node._log_effective_motion_profile()
        self._refresh_effective_motion_label()
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
            if isinstance(result, tuple) and len(result) == 4:
                target, width_m, height_m, length_m = result
                self.cycle_status_label.setText(
                    f"视觉成功 xyz=({target.x:.3f},{target.y:.3f},{target.z:.3f})m "
                    f"L={length_m*1000:.1f}mm H={height_m*1000:.1f}mm "
                    f"W={width_m*1000:.1f}mm"
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
            length = self.node.last_accepted_length_m or 0.0
            self.vision_feedback_label.setText(
                f"XYZ {target.x*1000:.1f}, {target.y*1000:.1f}, {target.z*1000:.1f} mm | "
                f"L {length*1000:.1f} mm | H {height*1000:.1f} mm | W {width*1000:.1f} mm"
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
