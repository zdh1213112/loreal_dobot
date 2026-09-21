"""V3 Nova5 (192.168.111.101) cosmetic-box pick/scan/place cycle.

This file was forked from the complete V2 cycle as the integration baseline
for turntable control. Shared controller, gripper and geometry modules remain
common dependencies so fixes in those low-level layers are not duplicated.

V3 pre-scans the placed material independently of the left-arm cycle.  After a
102 place-done event and clearance check, a D435 scans the rotating table.  A
hit stops the table and records a ready-to-pick material; Execute later consumes
that stopped state and starts D405 localization and the left-arm cycle.  A miss
stops the table and blocks the pick until the operator checks and retries the
scan.  The legacy HID-scanner approach and J6 face search remain in the file
only as compatibility helpers and are not used by the V3 automatic path.

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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
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

from .offset_grasp_geometry_v3 import camera_rotation_at_target_tcp, plan_offset

from .controller_v3 import DobotNova5Controller, TcpPose
from .dobot_dh_api_v3 import (
    GRIP_DROPPED,
    GRIP_GRIPPED,
    GRIP_IN_MOTION,
    GRIP_REACHED,
    DobotDHConfig,
    DHGripper,
    raise_if_error,
)
from .turntable_v3 import (
    PlacementRetreatTrigger,
    classify_barcode_face,
    nearest_face_anchor_deg,
    turntable_departure_target_z,
    validate_turntable_grasp_height,
)


# Keep look-ahead submission close to the configured TCP gate.  The feedback
# stream is faster than this on the Nova5, while the sleep still yields to the
# vision and ROS executor threads.
LOOKAHEAD_GATE_POLL_S = 0.002
# Vision callbacks are delivered by the ROS executor thread.  Polling at 5 ms
# keeps a newly published pose/handoff state from adding a visible scheduler
# gap without busy-spinning the request worker.
VISION_REQUEST_POLL_S = 0.005
# A brief missing 102 packet still fails the 50 ms safety gate.  Keep the
# read-only socket open while idle so a late packet can restore feedback;
# reconnect only if the reader dies or the stream has stopped for much longer.
SECONDARY_FEEDBACK_RECONNECT_STALE_S = 1.0


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


def grasp_feedback_is_plausible(
    grip_state: int,
    opening_m: float,
    commanded_preshape_m: float,
    minimum_opening_m: float,
    minimum_closure_m: float,
) -> bool:
    """Reject a false ``GRIPPED`` state caused by landing on a box top/edge."""

    closure_m = max(0.0, float(commanded_preshape_m) - float(opening_m))
    return (
        int(grip_state) == GRIP_GRIPPED
        and float(opening_m) > float(minimum_opening_m)
        and closure_m >= float(minimum_closure_m)
    )


def secondary_y_interlock_action(
    gap_m: float,
    robot_mode: int,
    protective_stop_m: float,
    emergency_retreat_m: float,
) -> str:
    """Classify the left-arm-only response to the live common-Y TCP gap."""

    gap_m = float(gap_m)
    protective_stop_m = float(protective_stop_m)
    emergency_retreat_m = float(emergency_retreat_m)
    if emergency_retreat_m <= 0.0 or protective_stop_m <= emergency_retreat_m:
        raise ValueError(
            "secondary protective stop distance must be greater than the "
            "emergency retreat distance"
        )
    if gap_m < emergency_retreat_m:
        return "retreat"
    if gap_m < protective_stop_m and int(robot_mode) in (7, 8, 10):
        return "stop"
    return "none"


def secondary_y_retreat_axis(left_y_m: float, right_common_y_m: float) -> str:
    """Return the User-Y jog direction that increases the current TCP gap."""

    return "Y+" if float(left_y_m) >= float(right_common_y_m) else "Y-"


class RecoverableGraspError(RuntimeError):
    """A confirmed empty/lost grasp for which returning to startup is safe."""

    def __init__(self, stage: str, message: str, needs_vertical_retreat: bool = False):
        super().__init__(message)
        self.stage = stage
        self.needs_vertical_retreat = needs_vertical_retreat


class SecondaryClearanceRetry(RuntimeError):
    """The left arm must retry a cycle because the right arm is too close."""


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


@dataclass(frozen=True)
class PregraspObservation:
    """One base-frame target observation with its original camera-frame time."""

    count: int
    pose: TcpPose
    frame_time_s: float
    received_at_monotonic: float
    tcp_linear_speed_mps: Optional[float] = None


class CosmeticBoxSingleArmNode(Node):
    def __init__(self) -> None:
        super().__init__("nova5_cosmetic_box_single_arm_cycle_v3")

        self.declare_parameter("robot_ip", "192.168.111.101")
        self.declare_parameter("dashboard_port", 29999)
        self.declare_parameter("feedback_port", 30004)
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("startup_joint", [23.0, 13.0, -120.0, 30.0, 80.0, 20.0]) #初始位置，关节角度
        # A completed cycle already ends at startup_joint.  Before the next
        # single-cycle request, use fresh joint feedback to avoid replaying the
        # same MovJ.  Gripper opening is still performed below.
        self.declare_parameter("startup_joint_skip_tolerance_deg", 1.0)
        self.declare_parameter("transfer_joint", [14.0, -29.0, -99.0, 39.0, 88.0, 15.0])
        # 101 is the left pick arm.  These parameters only read 102 feedback;
        # the left node never sends a motion command to the right/VLA arm.
        self.declare_parameter("secondary_collision_check_enabled", True)
        self.declare_parameter("secondary_robot_ip", "192.168.111.102")
        self.declare_parameter("secondary_dashboard_port", 29999)
        self.declare_parameter("secondary_feedback_port", 30004)
        self.declare_parameter("secondary_user_index", 0)
        self.declare_parameter("secondary_tool_index", 1)
        # 102 base is 725 mm on the -Y side of 101 base.  Convert 102 User-Y
        # to the common 101 User frame as y_102_common = y_102 - 0.725.
        self.declare_parameter("secondary_base_y_offset_m", -0.725)
        self.declare_parameter("secondary_tcp_max_age_s", 0.05)
        # Two-stage, left-arm-only interlock.  Below 165 mm an active 101
        # motion is cancelled and held.  Only if the live gap then falls below
        # 145 mm does 101 execute a monitored User-Y escape away from 102.
        # 102 remains feedback-only and never receives a command from here.
        self.declare_parameter("secondary_y_clearance_m", 0.160)
        self.declare_parameter("secondary_emergency_retreat_m", 0.142)
        self.declare_parameter("secondary_emergency_recover_m", 0.180)
        self.declare_parameter("secondary_motion_monitor_poll_s", 0.010)
        self.declare_parameter("secondary_emergency_retreat_speed", 100)
        self.declare_parameter("secondary_emergency_retreat_timeout_s", 3.0)
        self.declare_parameter("secondary_emergency_retreat_max_travel_m", 0.200)
        self.declare_parameter("secondary_clearance_wait_timeout_s", 20.0)
        self.declare_parameter("secondary_clearance_poll_s", 0.05)
        self.declare_parameter("secondary_connection_retry_s", 2.0)
        self.declare_parameter("secondary_clearance_log_period_s", 1.0)
        # 扫码后的组合 PTP 先把条码翻到朝上，并移动到放置区上方安全高度，
        # 按 User 0 表达，单位为米。侧面条码分支先移动到独立的固定放置位，
        # 再在那里执行 Rx 倾斜；旧版动态 Z 参数仍保留用于兼容配置。
        self.declare_parameter("scan_exit_user_xyz", [0.560, 0.375, 0.320])
        # 顶面条码分支保持抓取姿态，直接 PTP 到现场指定的固定放置位。
        self.declare_parameter("top_surface_barcode_place_xyz", [0.531, 0.328, 0.105])
        # 底面恢复分支原先复用顶面参数；独立保留其原有的 215 mm 放置高度。
        self.declare_parameter("bottom_barcode_place_xyz", [0.531, 0.328, 0.215])
        # 侧面条码分支使用更高的独立放置位，避免动态低 Z 使 J4/J5 接近桌面。
        self.declare_parameter("side_barcode_place_xyz", [0.531, 0.328, 0.180])
        self.declare_parameter("placement_surface_z_m", 0.060)
        self.declare_parameter(
            "placement_safety_margin_m",
            0.010,
        )
        self.declare_parameter("user_index", 0)
        self.declare_parameter("flange_tool_index", 0)
        self.declare_parameter("command_tool_index", 1)
        # 以下运动参数保留原 GUI 的调节语义。控制器端不再把它们同时写入
        # SpeedFactor、VelJ/VelL 和单条指令，而是先在软件中合成为一个等效
        # 指令百分比。100 表示修改前的理论有效速度基线；默认 400% 全局
        # 缩放会把可安全提速的抓取上方/下降参数合成为最多 100% 的单条指令，
        # 抓取后的各阶段则直接使用下方独立有效百分比。
        self.declare_parameter("motion_speed_scale_percent", 400)
        # Commissioning safety ceiling applied after every ordinary motion
        # speed/acceleration has been composed.  This covers both the legacy
        # scaled approach stages and the direct post-grasp stages.  The
        # collision-interlock emergency retreat is intentionally separate.
        self.declare_parameter("motion_command_cap_percent", 20)
        # 普通关节动作：初始位、抓取上方、中转位及回初始位。
        # 加速度独立可调，短行程往往由加速度而不是最高速度决定耗时。
        self.declare_parameter("joint_speed", 65)
        # With the default motion_speed_scale_percent=400, joint_speed=65
        # already composes to an effective 100% command speed.  Raising the
        # acceleration baseline to 65 makes the move-above/pregrasp PTP
        # acceleration reach the same 100% effective ceiling without changing
        # the nominal speed semantics.
        self.declare_parameter("joint_acc", 65)
        # 抓取完成后的抬升、去中转位、扫码后翻转/放置和回初始位分别调速，
        # 避免为了加快后半段而把抓取上方/下降前的动作一起推得过快。
        # 后半段参数是“单条指令有效百分比”，不再与 joint_speed 重复相乘。
        self.declare_parameter("grasp_lift_speed_factor", 100)
        self.declare_parameter("grasp_lift_acc_factor", 100)
        # Turntable cycles always use a blocking straight-line vertical
        # departure before any horizontal or joint-space motion.  This legacy
        # blend remains available only when turntable mode is disabled.
        self.declare_parameter("grasp_lift_transfer_blend_enabled", False)
        self.declare_parameter("grasp_lift_transfer_blend_cp", 20)
        self.declare_parameter("grasp_lift_transfer_queue_lead_m", 0.010)
        self.declare_parameter("grasp_lift_transfer_command_start_grace_s", 0.30)
        # The safe-height and fixed-placement PTPs are both known, collision-
        # checked waypoints after barcode handling is complete.  Queue only
        # this transition; all gripper and scanner feedback checkpoints remain
        # blocking as before.
        self.declare_parameter("post_scan_place_blend_enabled", True)
        self.declare_parameter("post_scan_place_blend_cp", 20)
        self.declare_parameter("post_scan_place_queue_lead_m", 0.020)
        self.declare_parameter("post_scan_place_command_start_grace_s", 0.30)
        self.declare_parameter("transfer_speed_factor", 100)
        self.declare_parameter("transfer_acc_factor", 100)
        self.declare_parameter("place_speed_factor", 100)
        self.declare_parameter("place_acc_factor", 100)
        self.declare_parameter("post_scan_acc_factor", 100)
        self.declare_parameter("return_startup_speed_factor", 100)
        self.declare_parameter("return_startup_acc_factor", 100)
        # 抓取下降使用的直线运动速度和加速度。默认 400% 全局缩放会把
        # 65/65 合成为 100% 单条指令；下降仍保持 MovL 和全部 Z 安全检查。
        self.declare_parameter("linear_speed", 65)
        self.declare_parameter("linear_acc", 65)
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
        # The scanner-retreat segment is a known straight User-X escape.  Once
        # the configured extra clearance has nearly been reached, the
        # collision-checked post-scan safe-height PTP can be queued with CP so
        # the controller does not stop between the two segments.  A separate
        # switch and lead keep this transition independently tunable from the
        # safe-height-to-place queue.
        self.declare_parameter("scanner_retreat_post_scan_blend_enabled", True)
        self.declare_parameter("scanner_retreat_post_scan_blend_cp", 20)
        self.declare_parameter("scanner_retreat_post_scan_queue_lead_m", 0.010)
        self.declare_parameter("scanner_retreat_post_scan_command_start_grace_s", 0.30)
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
        # If a barcode arrives near the already-safe endpoint, let the bounded
        # X+ move finish naturally instead of issuing Stop() and waiting for a
        # second controller idle transition.  The endpoint itself still
        # enforces the configured scanner clearance.
        self.declare_parameter("scanner_approach_natural_finish_margin_m", 0.015)
        # 旧版“固定 XYZ、单独 Ry 点动”的备用速度。当前生产流程已经改为
        # XYZ+Ry+Rz 单条组合 PTP，不再读取该参数。
        self.declare_parameter("face_up_rotation_speed_factor", 50)
        self.declare_parameter("jog_tolerance_m", 0.002)
        self.declare_parameter("jog_axis_timeout_s", 20.0)
        self.declare_parameter("offset_grasp_enabled", True)
        self.declare_parameter("offset_grasp_clearance_m", 0.020)
        self.declare_parameter("offset_finger_span_m", 0.060)
        # Reach the image-down offset while descending to this clearance above
        # grasp depth.  The remaining approach stays vertical and retains the
        # gripper/pregrasp checks at offset-high.
        self.declare_parameter("offset_high_clearance_m", 0.120)
        # A D435-confirmed side barcode does not need the offset/top-observation
        # path.  Move diagonally from startup to this height above the D405
        # grasp target, then use a vertical Cartesian descent.
        self.declare_parameter("side_barcode_direct_hover_clearance_m", 0.120)
        # The offset-high PTP and the following vertical descent are both
        # known safe once the gripper pre-shape and target identity checks have
        # passed.  Queue the MovL descent before the PTP reaches its endpoint
        # so the controller can blend the two segments without an idle stop.
        self.declare_parameter("offset_high_descent_blend_enabled", True)
        self.declare_parameter("offset_high_descent_blend_cp", 20)
        self.declare_parameter("offset_high_descent_queue_lead_m", 0.030)
        self.declare_parameter("offset_high_descent_command_start_grace_s", 0.30)
        self.declare_parameter("grasp_lift_m", 0.060)
        # Current real-cell trials show an approximately 10 mm vertical bias
        # between the transformed vision pose and the physical gripper tip.
        # Positive is shallower/safer and remains editable in the GUI.
        # V3 already applies the measured D405->User Z field correction.  Do
        # not retain V2's additional +10 mm shallow-grasp compensation, which
        # would lift the 75%-depth target above the intended side-grasp band.
        # V3 turntable trials showed the fingers passing just above the box at
        # the nominal 75%-depth point. Descend 4 mm deeper by default; the
        # turntable TCP floor still rejects an unsafe surface approach.
        self.declare_parameter("grasp_z_offset_m", -0.004)
        self.declare_parameter("grasp_z_offset_limit_m", 0.020)
        self.declare_parameter("minimum_safe_tcp_z_m", 0.010)

        self.declare_parameter("camera_frame_id", "camera_d405_link")
        self.declare_parameter("vision_pose_topic", "/target_pose_cam_fine")
        # D405 稳定目标后发布的后台预抓取位姿话题。机器人到达抓取上方
        # 后只消费这条话题中本次目标的已到达新鲜帧，不增加单独视觉等待。
        self.declare_parameter("pregrasp_pose_topic", "/target_pose_cam_pregrasp")
        # 最新预抓取位姿允许的最大年龄，单位：秒。超过该时间视为视觉
        # 跟踪失效，机械臂不下压，直接回初始位重新检测。
        self.declare_parameter("pregrasp_pose_max_age_s", 0.65)
        # 允许后台预抓取跟踪修正目标位置。修正不是增加一段等待：目标帧
        # 在 move-above 期间异步累积，到达悬停位后立即消费。
        self.declare_parameter("pregrasp_use_live_pose_for_descent", True)
        # 目标相对初始稳定位姿的位置变化阈值，单位：米。小变化继续使用
        # 首次稳定位置；超过该值且最新悬停附近实测帧有连续帧支持时，
        # 才在安全悬停高度做一次水平修正。
        self.declare_parameter("pregrasp_position_tolerance_m", 0.007)
        # 实时 OBB 姿态变化阈值，单位：度。姿态使用旋转矩阵的几何角度
        # 判断，不使用易跳变的 Euler 单轴差值。
        self.declare_parameter("pregrasp_angle_tolerance_deg", 8.0)
        # 单次安全上方位置修正的最大位移，单位：米。
        self.declare_parameter("pregrasp_max_correction_m", 0.050)
        # 单次姿态修正允许的最大几何角度，单位：度。超过该值不把实时
        # OBB 姿态带到 TCP，保留首次稳定姿态。
        self.declare_parameter("pregrasp_max_correction_angle_deg", 30.0)
        # 目标与当前 TCP 安全上方位置之间必须保留的最小垂直间隙，单位：米。
        # 间隙不足时禁止下压，直接回初始位重新检测。
        self.declare_parameter("pregrasp_min_hover_clearance_m", 0.030)
        # 兼容旧 launch 参数保留。实机数据证明眼在手相机运动期间的细小
        # 时序误差会形成平滑但虚假的“目标速度”，因此抓取位置不再外推。
        self.declare_parameter("pregrasp_prediction_horizon_s", 0.0)
        # TCP 线速度低于该值的视觉帧视为“悬停后稳定帧”。这些帧优先于
        # 机械臂运动中的帧，用于获得稳定的当前手眼变换结果。
        self.declare_parameter("pregrasp_settled_tcp_speed_mps", 0.015)
        # D405 通常约 4--5 Hz；允许使用到达悬停前约一个帧周期内的
        # 已采集稳定帧，不额外等待相机再拍一帧。
        self.declare_parameter("pregrasp_hover_frame_window_s", 0.25)
        # 连续位置帧聚类半径。单帧跳点不会触发水平修正。
        self.declare_parameter("pregrasp_position_consensus_m", 0.006)
        self.declare_parameter("pregrasp_position_consensus_samples", 2)
        # A large one-frame jump is target loss/mask drift, not a safe hover
        # correction.  It must be confirmed by the normal spatial consensus
        # before the robot is allowed to descend.
        self.declare_parameter("pregrasp_unconfirmed_shift_reject_m", 0.020)
        # 兼容旧参数保留；运动轨迹现在只写入诊断，不参与抓取点选择。
        self.declare_parameter("pregrasp_motion_min_displacement_m", 0.006)
        self.declare_parameter("pregrasp_motion_max_residual_m", 0.005)
        # 平放盒子默认锁定首次稳定姿态；若现场确实允许盒子在空中旋转，
        # 可显式打开，且仍必须满足至少 3 帧旋转共识和法向一致。
        self.declare_parameter("pregrasp_live_orientation_enabled", False)
        self.declare_parameter("pregrasp_orientation_consensus_samples", 3)
        self.declare_parameter("pregrasp_orientation_consensus_spread_deg", 4.0)
        self.declare_parameter("pregrasp_live_orientation_max_delta_deg", 20.0)
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
        # V3 turntable field calibration in User-0 Z.  Four stationary D405
        # observations placed the known Tool-1 support plane at 97.1--97.5 mm
        # while the physical plane is 126.0 mm, so compensate the repeatable
        # -28.7 mm offset after the hand-eye transform.  V2 remains untouched.
        self.declare_parameter("vision_user_z_bias_m", 0.0287)
        self.declare_parameter("grasp_offset_rxyz_deg", [180.0, 0.0, -90.0])

        # V3 turntable: the controller toggles run/stop on every tested
        # 0->1->0 pulse.  The D435 scans only while this node owns the active
        # four-second window; D405 localization is requested after stop/settle.
        self.declare_parameter("turntable_enabled", True)
        self.declare_parameter("turntable_do_index", 1)
        self.declare_parameter("turntable_pulse_ms", 300)
        self.declare_parameter("turntable_scan_timeout_s", 1.42)
        # Inspect the already-stopped face before rotating. D435 runs
        # continuously, so a correctly oriented placement should not cause an
        # unnecessary table revolution.
        self.declare_parameter("turntable_stationary_barcode_check_s", 1.5)
        self.declare_parameter("turntable_settle_s", 0.050)
        self.declare_parameter("turntable_assume_stopped_on_start", False)
        self.declare_parameter("turntable_require_place_done", True)
        self.declare_parameter("turntable_place_done_topic", "/turntable_place_done")
        self.declare_parameter("turntable_place_wait_timeout_s", 0.0)
        # Observe the VLA-controlled 102 arm through its existing read-only
        # 30004 stream.  No command or Dashboard connection is sent to 102.
        # One event first requires Y >= 400 mm in the place region, followed
        # by a stable Y < 400 mm retreat in 102 User 0 / Tool 1.  Z is not a
        # trigger condition; the legacy Z parameter remains declared only so
        # existing launch commands do not fail parameter validation.
        self.declare_parameter("turntable_auto_place_from_secondary_tcp", True)
        self.declare_parameter("turntable_secondary_place_y_m", 0.400)
        self.declare_parameter("turntable_secondary_safe_z_m", 0.200)
        self.declare_parameter("turntable_secondary_safe_z_stable_s", 0.200)
        self.declare_parameter(
            "turntable_barcode_trigger_topic", "/trigger_turntable_barcode"
        )
        self.declare_parameter(
            "d435_continuous_trigger_topic",
            "/trigger_d435_continuous_detection",
        )
        self.declare_parameter(
            "d435_continuous_result_topic",
            "/d435_continuous_barcode_result",
        )
        self.declare_parameter(
            "d435_continuous_presence_topic",
            "/d435_continuous_barcode_presence",
        )
        self.declare_parameter("d435_continuous_on_start", True)
        self.declare_parameter(
            "turntable_barcode_result_topic", "/turntable_barcode_result"
        )
        self.declare_parameter(
            "turntable_barcode_ready_topic", "/turntable_barcode_camera_ready"
        )
        self.declare_parameter("turntable_camera_ready_timeout_s", 20.0)
        # Measured on the current cell in User 0: the material-supporting top
        # surface of the turntable is Z=126 mm.  A negative replacement value
        # is still treated as unconfigured and blocks automatic descent.
        self.declare_parameter("turntable_height_safety_enabled", True)
        self.declare_parameter("turntable_surface_z_m", 0.126)
        self.declare_parameter("turntable_surface_tolerance_m", 0.020)
        self.declare_parameter("turntable_tcp_below_target_m", 0.0)
        self.declare_parameter("turntable_surface_clearance_m", 0.003)

        self.declare_parameter("barcode_topic", "/detected_barcodes")
        self.declare_parameter("top_surface_barcode_enabled", True)
        # Detection remains armed throughout offset-high, descent and insert.
        # Keep only a short dedicated low-pose observation hold between the
        # descent and insert so top-barcode scanning is retained without a
        # fixed half-second stop on every side-barcode cycle.
        self.declare_parameter("top_surface_barcode_wait_s", 0.20)
        self.declare_parameter("top_surface_barcode_topic", "/trigger_top_surface_barcode")
        self.declare_parameter(
            "top_surface_barcode_result_topic", "/top_surface_barcode_result"
        )
        # A HID scanner emits one complete decoded string per successful scan;
        # unlike frame-by-frame vision detections it need not be seen 3 times.
        self.declare_parameter("barcode_stable_hits", 1)
        self.declare_parameter("barcode_hit_gap_s", 0.7)
        self.declare_parameter("barcode_face_wait_s", 0.05)  # 每个标准面最多等待 0.05 秒
        # HID 条码常在 transfer_joint 到位后的几十毫秒内才进入 ROS 回调。
        # 先短暂保留当前姿态，可避免刚启动 X+ 靠近就因扫码成功而 Stop。
        self.declare_parameter("scanner_transfer_barcode_grace_s", 0.06)
        self.declare_parameter("barcode_max_face_rotations", 4)
        # Each next-face search may rotate wrist J6 by as much as -90 degrees,
        # with a joint-limit guard. Live monitoring below stops it earlier when
        # the scanner decodes a barcode during that rotation.
        self.declare_parameter("barcode_flip_step_deg", -90.0)
        self.declare_parameter("barcode_flip_safe_joint_limit_deg", 355.0)
        self.declare_parameter("barcode_flip_watch_joint_index", 5)
        # D435 side-barcode face alignment has its own 90-degree grid.  Do not
        # borrow transfer_joint J6: that waypoint also serves other paths.
        self.declare_parameter("d435_side_face_reference_joint_deg", 0.0)
        # J6 找码改用连续点动并实时监听扫码结果。到达目标角前一旦识别成功，
        # 立即停止点动，保留条码正对扫码器的姿态。默认把原来的 3 次 90°
        # 点动合并为一次连续 270° 扫描；如现场扫码器在运动中识别率不足，
        # 可关闭此开关回退到逐面停靠模式。
        self.declare_parameter("barcode_continuous_rotation", True)
        self.declare_parameter("barcode_flip_jog_tolerance_deg", 1.0)
        self.declare_parameter("barcode_flip_jog_timeout_s", 60.0)
        # 扫码器可以在条码面斜对着它时提前解码。识别后不能直接保留任意
        # 中间角度，否则后续 User Ry -90° 会让条码面斜着朝上。停止 J6 后
        # 自动吸附到最近的 90° 标准面：前半程回上一面，后半程补到下一面。
        self.declare_parameter("barcode_snap_to_nearest_face", True)
        # 顶面和四个侧面均未发现条码时，按“条码在底面”执行桌面翻转。
        # 第一次 User Ry- 目标为 45deg；若固定 TCP 位姿没有逆解，则每次
        # 减少 5deg，最低允许 45deg。当前现场要求固定使用 -45deg。
        self.declare_parameter("bottom_barcode_recovery_enabled", True)
        self.declare_parameter("bottom_flip_user_ry_target_deg", -45.0)
        self.declare_parameter("bottom_flip_user_ry_min_abs_deg", 45.0)
        self.declare_parameter("bottom_flip_user_ry_step_deg", 5.0)
        # Four-face search ends at J6 -270deg. Before leaving the scanner,
        # return +90deg so bottom recovery continues from a net -180deg pose.
        self.declare_parameter("bottom_flip_j6_pre_return_deg", 90.0)
        # 底面恢复下降量使用“首次抓取抬升量 - 本余量”。默认首次抬升
        # 60 mm、保留 10 mm，因此只下降 50 mm，且仍受最低 TCP Z 限制。
        self.declare_parameter("bottom_flip_table_z_offset_m", 0.020)
        # 扫码失败后先沿 User X- 再把盒子放回桌面，给 User-Ry 翻转留出
        # 夹爪和盒体的安全空间。数值是额外退回量，单位为米。
        self.declare_parameter("bottom_flip_table_retract_m", 0.050)
        self.declare_parameter("bottom_flip_lift_m", 0.160)
        # V3 turntable bottom-face recovery first grasps and clears the table,
        # snaps J6 to the nearest 90-degree face, applies User Rz +40 degrees,
        # and returns to the original grasp Z before the table flip.
        self.declare_parameter("bottom_center_first_rz_delta_deg", 40.0)
        self.declare_parameter("bottom_center_first_tool_rx_delta_deg", 70.0)
        self.declare_parameter("bottom_center_release_tool_rx_delta_deg", 70.0)
        self.declare_parameter("bottom_center_tracking_timeout_s", 2.0)
        # Lift 160 mm, rotate J6 +180 deg, then lower 120 mm before regrasp.
        self.declare_parameter("bottom_flip_j6_half_turn_deg", 180.0)
        self.declare_parameter("bottom_flip_post_turn_descent_m", 0.120)
        self.declare_parameter("bottom_flip_stall_timeout_s", 0.50)
        # After the bottom-face table flip (normally User Ry=-45deg), apply
        # one more User Ry- rotation while moving to the fixed place pose.
        # The field workflow therefore reaches a total User Ry of about -90deg
        # before the final User Rz=+50deg rotation.
        self.declare_parameter("bottom_barcode_place_ry_delta_deg", -45.0)
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
        # 侧面条码分支不再下降到按长度计算的低 Z；固定放置 XYZ 和
        # User-X 轴负向倾斜由同一条 PTP 完成。
        self.declare_parameter("side_barcode_place_rx_delta_deg", -20.0)
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
        # A correct side grasp starts with roughly 20 mm clearance and must
        # visibly close before state=GRIPPED is trusted.  If the fingers land
        # on the top/edge, DH can report GRIPPED while the opening hardly
        # changes (cycle 1398: 89.3 -> 89.1 mm).  Reject that condition before
        # sending any lift command; this uses existing feedback and adds no wait.
        self.declare_parameter("grasp_min_closure_from_preshape_m", 0.005)
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

        # 101 is the left pick arm.  Keep only a read-only feedback connection
        # for 102.  The VLA process remains the owner of 102's Dashboard/control
        # connection; this node must not open port 29999 on the right arm.
        self.secondary_controller: Optional[DobotNova5Controller] = None
        self.secondary_connection_lock = threading.Lock()
        self.secondary_next_connect_attempt = 0.0
        self.secondary_last_measurement: Optional[dict[str, object]] = None
        self.secondary_last_clearance_state: Optional[bool] = None
        self.secondary_last_clearance_log_at = 0.0
        self.secondary_safety_lock = threading.Lock()
        self.secondary_protective_stop_latched = threading.Event()
        self.secondary_retreat_active = threading.Event()
        self.secondary_retreat_attempted = threading.Event()
        # A hard Y-clearance retreat may interrupt a continuous production
        # cycle.  Remember that intent separately from ``cycle_enabled`` so
        # the old trajectory can stay cancelled while a fresh cycle is
        # started after the retreat reaches its recovery distance.
        self.secondary_auto_resume_requested = threading.Event()
        self.secondary_safety_shutdown = threading.Event()
        self.secondary_safety_reason = ""
        self.secondary_safety_thread: Optional[threading.Thread] = None
        self.secondary_resume_thread: Optional[threading.Thread] = None

        self.data_lock = threading.Lock()
        self.pose_samples: deque[tuple[int, TcpPose]] = deque(maxlen=20)
        self.pose_count = 0
        self.pregrasp_pose_count = 0
        self.latest_pregrasp_pose: Optional[TcpPose] = None
        self.latest_pregrasp_pose_received_at = 0.0
        self.pregrasp_observations: deque[PregraspObservation] = deque(maxlen=40)
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
        self.rgb_to_ir_rotation = None
        self.latest_handoff_clear = False
        self.latest_handoff_candidate_points = 0
        self.latest_handoff_cluster_points = 0
        self.latest_handoff_negative_side_clear = True
        self.latest_handoff_positive_side_clear = True

        self.barcode_lock = threading.Lock()
        self.barcode_window_active = False
        self.barcode_value = ""
        self.barcode_hits = 0
        self.barcode_last_time = 0.0
        # Net J6 travel used by the most recent four-face search.  The bottom
        # barcode recovery path reverses this exact accumulated travel after
        # releasing/regrasping the box on the temporary table position.
        self.barcode_search_net_delta_deg = 0.0
        self.top_surface_barcode_lock = threading.Lock()
        self.top_surface_barcode_window_active = False
        self.top_surface_barcode_value = ""
        self.top_surface_barcode_result_count = 0

        self.turntable_lock = threading.RLock()
        self.turntable_condition = threading.Condition(self.turntable_lock)
        self.turntable_state = (
            "STOPPED"
            if bool(self.get_parameter("turntable_assume_stopped_on_start").value)
            else "UNKNOWN"
        )
        self.turntable_barcode_window_active = False
        self.turntable_barcode_value = ""
        self.turntable_barcode_result_count = 0
        self.turntable_camera_ready = False
        self.d435_continuous_detection = bool(
            self.get_parameter("d435_continuous_on_start").value
        )
        self.d435_continuous_last_value = ""
        self.d435_continuous_presence = False
        self.turntable_place_done_count = 0
        self.turntable_place_done_consumed = 0
        self.turntable_place_done_duplicate_warned = False
        # V3 pre-scans material as soon as 102 finishes placing it.  The left
        # arm later consumes this ready state; it no longer owns turntable
        # scanning or waits for a fresh place event after Execute is clicked.
        self.turntable_waiting_for_place = True
        self.turntable_scan_in_progress = False
        self.turntable_material_ready = False
        self.turntable_ready_barcode = ""
        self.turntable_scan_error = ""
        self.turntable_scan_thread: Optional[threading.Thread] = None
        self.turntable_scan_cancel = threading.Event()
        self.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(
            place_y_m=float(
                self.get_parameter("turntable_secondary_place_y_m").value
            ),
            stable_s=float(
                self.get_parameter("turntable_secondary_safe_z_stable_s").value
            ),
        )

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
        self.create_subscription(
            String,
            str(self.get_parameter("top_surface_barcode_result_topic").value),
            self._top_surface_barcode_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("turntable_barcode_result_topic").value),
            self._turntable_barcode_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("d435_continuous_result_topic").value),
            self._d435_continuous_barcode_callback,
            10,
        )
        ready_qos = QoSProfile(depth=1)
        ready_qos.reliability = ReliabilityPolicy.RELIABLE
        ready_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool,
            str(self.get_parameter("d435_continuous_presence_topic").value),
            self._d435_continuous_presence_callback,
            ready_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("turntable_barcode_ready_topic").value),
            self._turntable_barcode_ready_callback,
            ready_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("turntable_place_done_topic").value),
            self._turntable_place_done_callback,
            10,
        )
        self.create_subscription(Bool, "/cosmetic_pick_cycle_enable", self._cycle_enable_callback, 10)
        self.trigger_publisher = self.create_publisher(Bool, str(self.get_parameter("vision_trigger_topic").value), 10)
        self.top_surface_barcode_trigger_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("top_surface_barcode_topic").value),
            10,
        )
        self.turntable_barcode_trigger_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("turntable_barcode_trigger_topic").value),
            10,
        )
        continuous_qos = QoSProfile(depth=1)
        continuous_qos.reliability = ReliabilityPolicy.RELIABLE
        continuous_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.d435_continuous_trigger_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("d435_continuous_trigger_topic").value),
            continuous_qos,
        )
        continuous_initial = Bool()
        continuous_initial.data = self.d435_continuous_detection
        self.d435_continuous_trigger_publisher.publish(continuous_initial)
        self.status_publisher = self.create_publisher(String, "/cosmetic_pick_cycle_status", 10)
        self.timing_publisher = self.create_publisher(
            String,
            str(self.get_parameter("timing_topic").value),
            20,
        )

        self.start_timer = self.create_timer(1.0, self._start_automatically_once)
        self._connect_secondary_safety_feedback()
        self.secondary_safety_thread = threading.Thread(
            target=self._secondary_motion_safety_loop,
            name="secondary-tcp-monitor",
            daemon=True,
        )
        self.secondary_safety_thread.start()
        self.get_logger().warning(
            "Using handeye_flange_to_cam parameter for robot 192.168.111.101 / D405 409122274792; "
            "verify this calibration on the real cell before enabling motion."
        )
        self.get_logger().warning(
            "V3 D405 User-Z field correction active: "
            f"{float(self.get_parameter('vision_user_z_bias_m').value) * 1000.0:+.1f}mm; "
            "the independent turntable surface-height interlock remains enabled"
        )
        self.get_logger().info(
            "V3 V2-compatible full-close grasp active: "
            f"Z correction={float(self.get_parameter('grasp_z_offset_m').value) * 1000.0:+.1f}mm; "
            "the gripper commands position=0 and GRIP_GRIPPED feedback remains mandatory before lift"
        )
        self.get_logger().info(
            "101 left-arm controller ready; monitoring 102 right-arm TCP feedback "
            "read-only for the Y-clearance interlock and passive turntable trigger."
        )
        self.get_logger().warning(
            "V3 turntable workflow: "
            f"DO={int(self.get_parameter('turntable_do_index').value)}, "
            f"pulse={int(self.get_parameter('turntable_pulse_ms').value)}ms, "
            f"D435 timeout={float(self.get_parameter('turntable_scan_timeout_s').value):.1f}s, "
            f"surface_Z={float(self.get_parameter('turntable_surface_z_m').value):.4f}m, "
            f"102_place_Y={float(self.get_parameter('turntable_secondary_place_y_m').value):.3f}m, "
            "102 retreat trigger=Y-only (Z ignored), "
            f"initial_state={self.turntable_state}. A negative surface Z blocks "
            "left-arm descent; the first valid place_done resolves UNKNOWN to "
            "STOPPED and immediately starts the independent D435 pre-scan."
        )
        self._log_effective_motion_profile()

    def _six_values(self, name: str) -> list[float]:
        values = [float(value) for value in self.get_parameter(name).value]
        if len(values) != 6:
            raise ValueError(f"{name} must contain 6 values")
        return values

    def _dynamic_placement_xyz(self, length_m: float) -> list[float]:
        """Return the User-frame XYZ target for the barcode-up placement.

        ``scan_exit_user_xyz`` supplies the placement-area X/Y coordinates and
        the safe approach height. The actual lower placement Z is calculated
        for every box as:

            placement surface Z + box length / 2 + safety margin

        The vision length is the full measured box length, so the half-length
        term places the box centre above the User-frame placement surface.
        """
        values = [
            float(value)
            for value in self.get_parameter("scan_exit_user_xyz").value
        ]
        if len(values) != 3:
            raise ValueError("scan_exit_user_xyz must contain 3 values")
        length_m = float(length_m)
        if length_m <= 0.0:
            raise ValueError(
                f"material length must be positive for placement, got {length_m:.4f}m"
            )
        surface_z_m = float(self.get_parameter("placement_surface_z_m").value)
        safety_margin_m = float(
            self.get_parameter("placement_safety_margin_m").value
        )
        if surface_z_m < 0.0:
            raise ValueError(
                f"placement_surface_z_m must be non-negative, got {surface_z_m:.4f}m"
            )
        if safety_margin_m < 0.0:
            raise ValueError(
                "placement_safety_margin_m must be non-negative, "
                f"got {safety_margin_m:.4f}m"
            )
        placement_z_m = surface_z_m + 0.5 * length_m + safety_margin_m
        self.get_logger().info(
            "dynamic barcode-up placement target: "
            f"XYZ=({values[0] * 1000.0:.1f},{values[1] * 1000.0:.1f},"
            f"{placement_z_m * 1000.0:.1f})mm; "
            f"surface={surface_z_m * 1000.0:.1f}mm, "
            f"length/2={0.5 * length_m * 1000.0:.1f}mm, "
            f"safety={safety_margin_m * 1000.0:.1f}mm"
        )
        return [values[0], values[1], placement_z_m]

    def _top_surface_barcode_place_xyz(self) -> list[float]:
        """Return the fixed XYZ used by the top-barcode placement branch."""

        values = [
            float(value)
            for value in self.get_parameter("top_surface_barcode_place_xyz").value
        ]
        if len(values) != 3:
            raise ValueError("top_surface_barcode_place_xyz must contain 3 values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "top_surface_barcode_place_xyz must contain finite XYZ values"
            )
        return values

    def _bottom_barcode_place_xyz(self) -> list[float]:
        """Return the independently configured bottom-recovery placement XYZ."""

        values = [
            float(value)
            for value in self.get_parameter("bottom_barcode_place_xyz").value
        ]
        if len(values) != 3:
            raise ValueError("bottom_barcode_place_xyz must contain 3 values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bottom_barcode_place_xyz must contain finite XYZ values")
        return values

    def _side_barcode_place_xyz(self) -> list[float]:
        """Return the fixed XYZ used by the side-barcode placement branch."""

        values = [
            float(value)
            for value in self.get_parameter("side_barcode_place_xyz").value
        ]
        if len(values) != 3:
            raise ValueError("side_barcode_place_xyz must contain 3 values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("side_barcode_place_xyz must contain finite XYZ values")
        return values

    def _bottom_flip_table_pose(self) -> TcpPose:
        """Descend slightly less than the lift used after the first grasp."""

        current = self._current_command_pose()
        lift_m = float(self.get_parameter("grasp_lift_m").value)
        clearance_m = float(
            self.get_parameter("bottom_flip_table_z_offset_m").value
        )
        if lift_m <= 0.0:
            raise ValueError(f"grasp_lift_m must be positive, got {lift_m:.4f}m")
        if clearance_m < 0.0 or clearance_m >= lift_m:
            raise ValueError(
                "bottom_flip_table_z_offset_m must be in [0, grasp_lift_m), "
                f"got {clearance_m:.4f}m for lift {lift_m:.4f}m"
            )
        requested_z = float(current.z) - (lift_m - clearance_m)
        minimum_z = float(self.get_parameter("minimum_safe_tcp_z_m").value)
        target_z = max(requested_z, minimum_z)
        if target_z > requested_z + 1e-6:
            self.get_logger().warning(
                "Bottom-flip descent clamped by minimum TCP Z: "
                f"requested={requested_z * 1000.0:.1f}mm, "
                f"clamped={target_z * 1000.0:.1f}mm"
            )
        return TcpPose(
            float(current.x),
            float(current.y),
            target_z,
            float(current.rx),
            float(current.ry),
            float(current.rz),
        )

    def _bottom_flip_ry_target_candidates(self) -> list[float]:
        target = float(self.get_parameter("bottom_flip_user_ry_target_deg").value)
        minimum = abs(float(self.get_parameter("bottom_flip_user_ry_min_abs_deg").value))
        step = max(0.1, abs(float(self.get_parameter("bottom_flip_user_ry_step_deg").value)))
        if target >= 0.0:
            raise ValueError("bottom_flip_user_ry_target_deg must be negative")
        minimum = min(abs(target), minimum)
        values = []
        angle = abs(target)
        while angle >= minimum - 1e-6:
            values.append(-angle)
            angle -= step
        return values

    def _try_bottom_flip_ry_at_table(
        self,
        table_pose: TcpPose,
    ) -> tuple[float, TcpPose]:
        """Try the configured User-Ry- angle while the box rests on the table.

        The returned angle is the measured signed User-Y rotation, rather than
        the requested command.  This lets the recovery stage compensate a few
        degrees of shortfall after the J6 return.
        """

        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        last_error = None
        for candidate in self._bottom_flip_ry_target_candidates():
            before = self._current_command_pose()
            rotation = SciPyRot.from_euler(
                "xyz", [before.rx, before.ry, before.rz], degrees=True
            ).as_matrix()
            # Use the candidate selected by the configured target/minimum
            # policy.  In the current field setting the candidate list is
            # exactly [-45.0], so no smaller fallback is attempted.
            requested_delta = float(candidate)
            ry = SciPyRot.from_euler("y", requested_delta, degrees=True).as_matrix()
            target_rotation = ry @ rotation
            rx, final_ry, rz = SciPyRot.from_matrix(target_rotation).as_euler(
                "xyz", degrees=True
            )
            target = TcpPose(
                table_pose.x, table_pose.y, table_pose.z, float(rx), float(final_ry), float(rz)
            )
            try:
                self.controller.inverse_kinematics(
                    target,
                    user_index=user_index,
                    tool_index=tool_index,
                    joint_near=self.controller.current_joint(),
                )
            except Exception as exc:
                last_error = exc
                self.get_logger().warning(
                    f"Bottom-barcode User Ry target {candidate:+.1f}deg unavailable; "
                    f"trying a smaller angle: {exc}"
                )
                continue
            self._publish_status(
                f"bottom-barcode table flip: User Ry target {candidate:+.1f}deg"
            )
            # Use a Cartesian linear move for the table flip.  MovJ reaches the
            # same endpoint but interpolates joints, so the TCP can dip in Z
            # while Ry changes even though both endpoint poses have identical
            # XYZ.  A MovL keeps the TCP XYZ fixed throughout the rotation.
            self.controller.move_linear_tcp(
                target,
                speed=self._motion_profile()["post_scan_speed"],
                accel=self._motion_profile()["post_scan_acc"],
                user_index=user_index,
                tool_index=tool_index,
            )
            self._require_cycle_active("at bottom-barcode table flip pose")
            after = self._current_command_pose()
            before_rotation = SciPyRot.from_euler(
                "xyz", [before.rx, before.ry, before.rz], degrees=True
            ).as_matrix()
            after_rotation = SciPyRot.from_euler(
                "xyz", [after.rx, after.ry, after.rz], degrees=True
            ).as_matrix()
            relative_rotation = after_rotation @ before_rotation.T
            actual_delta = math.degrees(
                math.atan2(relative_rotation[0, 2], relative_rotation[0, 0])
            )
            self._publish_status(
                f"bottom-barcode table flip User Ry requested {requested_delta:+.1f}deg, "
                f"measured {actual_delta:+.1f}deg"
            )
            return float(actual_delta), target
        raise RuntimeError(
            "No reachable User Ry- bottom-flip angle was found down to the configured minimum"
        ) from last_error

    def _lower_to_dynamic_placement_z(
        self,
        approach_xyz: list[float],
        placement_z_m: float,
    ) -> None:
        """Descend vertically in User Z after the safe placement approach."""
        if len(approach_xyz) != 3:
            raise ValueError("approach_xyz must contain 3 values")
        approach_z_m = float(approach_xyz[2])
        placement_z_m = float(placement_z_m)
        min_clearance_m = 0.005
        if placement_z_m >= approach_z_m - min_clearance_m:
            raise RuntimeError(
                "Dynamic placement Z must remain below the safe approach height: "
                f"approach={approach_z_m * 1000.0:.1f}mm, "
                f"placement={placement_z_m * 1000.0:.1f}mm, "
                f"required_clearance>={min_clearance_m * 1000.0:.1f}mm"
            )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        current = self._current_command_pose()
        target = TcpPose(
            float(approach_xyz[0]),
            float(approach_xyz[1]),
            placement_z_m,
            float(current.rx),
            float(current.ry),
            float(current.rz),
        )
        motion = self._motion_profile()
        self._publish_status(
            f"descending vertically in User Z: "
            f"{approach_z_m * 1000.0:.1f}->{placement_z_m * 1000.0:.1f}mm, "
            f"length-based placement with {motion['place_speed']}% speed"
        )
        self.controller.move_linear_tcp(
            target,
            speed=motion["place_speed"],
            accel=motion["place_acc"],
            user_index=user_index,
            tool_index=tool_index,
        )
        self._require_cycle_active("at dynamic placement Z")
        final = self._current_command_pose()
        xyz_error = [
            abs(final.x - target.x),
            abs(final.y - target.y),
            abs(final.z - target.z),
        ]
        tolerance_m = max(
            0.0005,
            float(self.get_parameter("jog_tolerance_m").value) * 1.5,
        )
        if max(xyz_error) > tolerance_m:
            raise RuntimeError(
                "Dynamic placement vertical descent endpoint error too large: "
                f"errors={[round(error, 4) for error in xyz_error]}m, "
                f"tolerance={tolerance_m:.4f}m"
            )

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
        profile = {
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
        command_cap = max(
            1,
            min(
                100,
                int(self.get_parameter("motion_command_cap_percent").value),
            ),
        )
        return {
            name: min(command_cap, value)
            for name, value in profile.items()
        }

    def _log_effective_motion_profile(self) -> None:
        profile = self._motion_profile()
        scale = int(self.get_parameter("motion_speed_scale_percent").value)
        command_cap = int(
            self.get_parameter("motion_command_cap_percent").value
        )
        details = ", ".join(f"{name}={value}%" for name, value in profile.items())
        self.get_logger().info(
            f"Normalized motion scaling active: scale={scale}% (100%=legacy effective baseline), "
            f"ordinary-command safety cap={command_cap}%; "
            f"SpeedFactor/VelJ/VelL/AccJ/AccL replay layers fixed at 100; {details}"
        )
        if bool(self.get_parameter("secondary_collision_check_enabled").value):
            self.get_logger().info(
                "Secondary TCP Y two-stage interlock active (101 commands only): "
                f"102={self.get_parameter('secondary_robot_ip').value}, "
                f"base_y_offset={float(self.get_parameter('secondary_base_y_offset_m').value) * 1000.0:+.1f}mm, "
                f"stop_101_gap<{float(self.get_parameter('secondary_y_clearance_m').value) * 1000.0:.1f}mm, "
                f"retreat_101_gap<{float(self.get_parameter('secondary_emergency_retreat_m').value) * 1000.0:.1f}mm, "
                f"recover_gap={float(self.get_parameter('secondary_emergency_recover_m').value) * 1000.0:.1f}mm, "
                f"feedback_max_age={float(self.get_parameter('secondary_tcp_max_age_s').value) * 1000.0:.0f}ms, "
                "continuous_auto_restart=fresh-startup-after-retreat"
            )
        else:
            self.get_logger().warning(
                "Secondary TCP Y interlock DISABLED: 101 will not connect to or "
                "check 102 feedback; use only when the two-arm collision risk is "
                "controlled by the operator"
            )
        if bool(self.get_parameter("turntable_enabled").value):
            safe_transfer_z = float(
                self.get_parameter("scan_exit_user_xyz").value[2]
            )
            self.get_logger().info(
                "Turntable-safe post-grasp departure active: D435 side face "
                "rises from the grasp pose and aligns J6 in one Cartesian MovL "
                f"to User-0/Tool-1 Z={safe_transfer_z * 1000.0:.1f}mm "
                "after kinematic preflight; other faces use straight lift; "
                "lift/transfer CP look-ahead remains disabled"
            )
        elif bool(self.get_parameter("grasp_lift_transfer_blend_enabled").value):
            self.get_logger().info(
                "Post-grasp lift/transfer CP look-ahead active: "
                f"cp={max(1, min(100, int(round(float(self.get_parameter('grasp_lift_transfer_blend_cp').value)))))}%, "
                f"queue_lead={max(0.0, float(self.get_parameter('grasp_lift_transfer_queue_lead_m').value)) * 1000.0:.1f}mm; "
                "grasp confirmation remains before lift and after transfer"
            )
        else:
            self.get_logger().info(
                "Post-grasp lift/transfer CP look-ahead disabled; using blocking motion stages"
            )
        if bool(self.get_parameter("post_scan_place_blend_enabled").value):
            self.get_logger().info(
                "Post-scan safe-height/place CP look-ahead active: "
                f"cp={max(1, min(100, int(round(float(self.get_parameter('post_scan_place_blend_cp').value)))))}%, "
                f"queue_lead={max(0.0, float(self.get_parameter('post_scan_place_queue_lead_m').value)) * 1000.0:.1f}mm; "
                "placement feedback remains before release"
            )
        else:
            self.get_logger().info(
                "Post-scan safe-height/place CP look-ahead disabled; "
                "using separate safe-height and placement PTPs"
            )
        if bool(
            self.get_parameter("scanner_retreat_post_scan_blend_enabled").value
        ):
            self.get_logger().info(
                "Scanner-retreat/post-scan CP look-ahead active: "
                f"cp={max(1, min(100, int(round(float(self.get_parameter('scanner_retreat_post_scan_blend_cp').value)))))}%, "
                f"queue_lead={max(0.0, float(self.get_parameter('scanner_retreat_post_scan_queue_lead_m').value)) * 1000.0:.1f}mm; "
                "scanner clearance is verified before post-scan rotation"
            )
        else:
            self.get_logger().info(
                "Scanner-retreat/post-scan CP look-ahead disabled; "
                "using a blocking scanner retreat"
            )
        if bool(self.get_parameter("offset_high_descent_blend_enabled").value):
            self.get_logger().info(
                "Offset-high/descent CP look-ahead active: "
                f"cp={max(1, min(100, int(round(float(self.get_parameter('offset_high_descent_blend_cp').value)))))}%, "
                f"queue_lead={max(0.0, float(self.get_parameter('offset_high_descent_queue_lead_m').value)) * 1000.0:.1f}mm; "
                "gripper pre-shape and target revalidation remain before descent"
            )
        else:
            self.get_logger().info(
                "Offset-high/descent CP look-ahead disabled; using separate blocking stages"
            )
        if bool(self.get_parameter("pregrasp_use_live_pose_for_descent").value):
            orientation_mode = (
                "orientation requires consensus"
                if bool(self.get_parameter("pregrasp_live_orientation_enabled").value)
                else "orientation locked to initial stable pose"
            )
            self.get_logger().info(
                "Pregrasp revalidation mode: latest near-hover measured consensus; "
                f"no motion extrapolation; {orientation_mode}; unconfirmed shifts "
                f">{float(self.get_parameter('pregrasp_unconfirmed_shift_reject_m').value) * 1000.0:.1f}mm "
                "are treated as target loss only when near-hover evidence exists; "
                "motion-only apparent shifts keep the initial stable target"
            )
        else:
            self.get_logger().info(
                "Pregrasp revalidation mode: initial-stable pose protected; "
                "live D405 pose is diagnostic only"
            )
        self.get_logger().info(
            "Grasp plausibility interlock active: minimum closure from pre-shape="
            f"{float(self.get_parameter('grasp_min_closure_from_preshape_m').value) * 1000.0:.1f}mm; "
            "false GRIPPED feedback is released before vertical retreat"
        )
        self.get_logger().info(
            "Vision/robot time alignment: feedback "
            f"{self.controller.feedback_timestamp_source} timestamp; "
            f"raw_TimeStamp={self.controller.feedback_raw_timestamp}; "
            "pregrasp tracking uses no extra hover wait"
        )
        if self.controller.feedback_timestamp_source == "host_receive":
            self.get_logger().warning(
                "101 feedback TimeStamp is not a usable Unix clock on this firmware; "
                "robot history uses packet receive time. Moving-camera trends are "
                "diagnostic only and cannot command a pregrasp correction."
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

    def _drop_secondary_safety_feedback(self) -> None:
        """Close the optional 102 feedback connection without touching its motion."""

        with self.secondary_connection_lock:
            controller = self.secondary_controller
            self.secondary_controller = None
        if controller is not None:
            try:
                controller.disconnect()
            except Exception as exc:
                self.get_logger().warning(f"Closing 102 safety feedback reported: {exc}")

    def _connect_secondary_safety_feedback(self) -> bool:
        """Connect to 102 for feedback only; never issue a command to that arm."""

        feedback_needed = bool(
            self.get_parameter("secondary_collision_check_enabled").value
        ) or bool(
            self.get_parameter("turntable_auto_place_from_secondary_tcp").value
        )
        if not feedback_needed:
            return False

        with self.secondary_connection_lock:
            controller = self.secondary_controller
            if controller is not None:
                try:
                    controller.current_feedback_tcp_pose_with_timestamp()
                except Exception:
                    self.secondary_controller = None
                    try:
                        controller.disconnect()
                    except Exception:
                        pass
                else:
                    return True

            now = time.monotonic()
            if now < self.secondary_next_connect_attempt:
                return False
            retry_s = max(
                0.5,
                float(self.get_parameter("secondary_connection_retry_s").value),
            )
            self.secondary_next_connect_attempt = now + retry_s
            controller = DobotNova5Controller(
                robot_ip=str(self.get_parameter("secondary_robot_ip").value),
                dashboard_port=int(self.get_parameter("secondary_dashboard_port").value),
                feedback_port=int(self.get_parameter("secondary_feedback_port").value),
                # The dashboard_port is retained as a parameter for config
                # compatibility, but feedback-only connect never opens it.
                startup_joint=[0.0] * 6,
                startup_speed=1,
            )
            try:
                controller.connect_feedback_only()
            except Exception as exc:
                try:
                    controller.disconnect()
                except Exception:
                    pass
                self.get_logger().warning(
                    f"102 read-only TCP feedback unavailable: {exc}; "
                    "left-arm interlocks and automatic turntable trigger remain closed"
                )
                return False

            self.secondary_controller = controller
            self.get_logger().info(
                "102 right-arm read-only TCP feedback connected: "
                f"ip={self.get_parameter('secondary_robot_ip').value}; "
                f"feedback_port={self.get_parameter('secondary_feedback_port').value}; "
                "Dashboard 29999 was not opened; "
                "no 102 motion commands are sent"
            )
            pose, active_user, active_tool, received_at = (
                controller.current_feedback_tcp_pose_with_timestamp()
            )
            feedback_age_ms = max(0.0, time.time() - float(received_at)) * 1000.0
            common_y_m = pose.y + float(
                self.get_parameter("secondary_base_y_offset_m").value
            )
            self.get_logger().info(
                "102 startup TCP pose: "
                f"XYZ=({pose.x * 1000.0:.1f}, {pose.y * 1000.0:.1f}, "
                f"{pose.z * 1000.0:.1f})mm, "
                f"RPY=({pose.rx:.1f}, {pose.ry:.1f}, {pose.rz:.1f})deg, "
                f"User={active_user}, Tool={active_tool}, "
                f"age={feedback_age_ms:.1f}ms, "
                f"common_Y_for_101={common_y_m * 1000.0:.1f}mm, "
                f"timestamp_source={controller.feedback_timestamp_source}, "
                f"raw_TimeStamp={controller.feedback_raw_timestamp}"
            )
            return True

    def _read_secondary_y_clearance(
        self,
        *,
        allow_reconnect: bool = True,
    ) -> dict[str, object]:
        """Measure the 101/102 TCP Y gap in the common 101 User frame."""

        collision_check_enabled = bool(
            self.get_parameter("secondary_collision_check_enabled").value
        )
        auto_place_enabled = bool(
            self.get_parameter("turntable_auto_place_from_secondary_tcp").value
        )
        if not collision_check_enabled and not auto_place_enabled:
            return {"enabled": False, "clear": True}
        if allow_reconnect and not self._connect_secondary_safety_feedback():
            raise RuntimeError("102 TCP feedback is not connected")

        with self.secondary_connection_lock:
            controller = self.secondary_controller
        if controller is None:
            raise RuntimeError("102 TCP feedback is not connected")

        expected_user = int(self.get_parameter("secondary_user_index").value)
        expected_tool = int(self.get_parameter("secondary_tool_index").value)
        feedback_pose, actual_user, actual_tool, received_at = (
            controller.current_feedback_tcp_pose_with_timestamp()
        )
        feedback_age_s = max(0.0, time.time() - float(received_at))
        source = "feedback"
        if (actual_user, actual_tool) == (expected_user, expected_tool):
            secondary_pose = feedback_pose
        else:
            # Feedback-only mode intentionally has no Dashboard fallback.  A
            # pose in another User/Tool frame cannot safely be compared with
            # 101's frame, so fail closed until the VLA selects the configured
            # pair (normally User 0 / Tool 1).
            raise RuntimeError(
                "102 feedback active User/Tool does not match the configured "
                f"pair: active=({actual_user},{actual_tool}), "
                f"expected=({expected_user},{expected_tool}); "
                "feedback-only safety connection cannot query GetPose"
            )

        max_age_s = max(
            0.05,
            float(self.get_parameter("secondary_tcp_max_age_s").value),
        )
        if feedback_age_s > max_age_s:
            reader_error = controller.feedback_last_read_error or "none"
            raw_timestamp = controller.feedback_raw_timestamp
            reader_state = controller.feedback_reader_diagnostics()
            if allow_reconnect and (
                not controller.feedback_reader_alive
                or feedback_age_s >= SECONDARY_FEEDBACK_RECONNECT_STALE_S
            ):
                self._drop_secondary_safety_feedback()
            raise RuntimeError(
                f"102 TCP feedback stale: age={feedback_age_s * 1000.0:.0f}ms "
                f"> {max_age_s * 1000.0:.0f}ms; "
                f"reader_error={reader_error}; last_raw_TimeStamp={raw_timestamp}; "
                f"{reader_state}"
            )

        left_pose = self._current_command_pose()
        base_y_offset_m = float(
            self.get_parameter("secondary_base_y_offset_m").value
        )
        right_common_y_m = secondary_pose.y + base_y_offset_m
        gap_y_m = abs(left_pose.y - right_common_y_m)
        threshold_m = max(
            0.001,
            float(self.get_parameter("secondary_y_clearance_m").value),
        )
        return {
            "enabled": collision_check_enabled,
            "clear": (gap_y_m >= threshold_m) if collision_check_enabled else True,
            "right_x_m": float(secondary_pose.x),
            "left_y_m": float(left_pose.y),
            "right_y_m": float(secondary_pose.y),
            "right_z_m": float(secondary_pose.z),
            "right_common_y_m": float(right_common_y_m),
            "gap_y_m": float(gap_y_m),
            "threshold_m": float(threshold_m),
            "feedback_age_s": float(feedback_age_s),
            "secondary_user": int(actual_user),
            "secondary_tool": int(actual_tool),
            "source": source,
        }

    def _update_turntable_place_from_secondary_tcp(
        self,
        measurement: dict[str, object],
    ) -> None:
        """Create one event from a passive 102 place-Y then retreat-Y sequence."""

        if not bool(
            self.get_parameter("turntable_auto_place_from_secondary_tcp").value
        ):
            return
        right_y_m = float(measurement["right_y_m"])
        now_s = time.monotonic()
        place_seen_now = False
        fired = False
        with self.turntable_lock:
            if not self.turntable_waiting_for_place:
                return
            if self.turntable_place_done_count > self.turntable_place_done_consumed:
                return
            was_place_seen = self.turntable_secondary_retreat_trigger.place_seen
            fired = self.turntable_secondary_retreat_trigger.update(
                right_y_m,
                now_s,
            )
            place_seen_now = (
                not was_place_seen
                and self.turntable_secondary_retreat_trigger.place_seen
            )

        if place_seen_now:
            self._publish_status(
                "102 TCP entered the turntable placement side: "
                f"Y={right_y_m * 1000.0:.1f}mm >= "
                f"{self.turntable_secondary_retreat_trigger.place_y_m * 1000.0:.1f}mm; "
                "waiting for a stable Y retreat; TCP Z is not used"
            )
        if fired:
            self._accept_turntable_place_done(
                "automatic 102 TCP place/retreat trigger "
                f"(Y={right_y_m * 1000.0:.1f}mm; Z ignored)"
            )

    def _wait_for_secondary_y_clearance(
        self,
        stage: str,
        *,
        require_cycle_active: bool = True,
    ) -> dict[str, object]:
        """Hold 101 before an approach until 102's common-frame Y gap is safe."""

        if not bool(self.get_parameter("secondary_collision_check_enabled").value):
            return {"enabled": False, "clear": True}

        timeout_s = max(
            0.1,
            float(self.get_parameter("secondary_clearance_wait_timeout_s").value),
        )
        poll_s = max(
            0.01,
            float(self.get_parameter("secondary_clearance_poll_s").value),
        )
        log_period_s = max(
            0.2,
            float(self.get_parameter("secondary_clearance_log_period_s").value),
        )
        deadline = time.monotonic() + timeout_s
        last_state = self.secondary_last_clearance_state
        last_error = "no measurement"

        while self.running and (self.cycle_enabled or not require_cycle_active):
            if not require_cycle_active and self.turntable_scan_cancel.is_set():
                raise RuntimeError(
                    f"turntable scan cancelled while waiting for 102 TCP Y "
                    f"clearance before {stage}"
                )
            try:
                measurement = self._read_secondary_y_clearance()
                self.secondary_last_measurement = measurement
                clear = bool(measurement.get("clear", False))
                now = time.monotonic()
                if clear:
                    if last_state is not True:
                        self._publish_status(
                            "102 TCP Y clearance CLEAR: "
                            f"101={float(measurement['left_y_m']) * 1000.0:.1f}mm, "
                            f"102(common)={float(measurement['right_common_y_m']) * 1000.0:.1f}mm, "
                            f"gap={float(measurement['gap_y_m']) * 1000.0:.1f}mm "
                            f">= {float(measurement['threshold_m']) * 1000.0:.1f}mm; "
                            f"continuing {stage}"
                        )
                    self.secondary_last_clearance_state = True
                    return measurement

                if (
                    last_state is not False
                    or now - self.secondary_last_clearance_log_at >= log_period_s
                ):
                    self._publish_status(
                        "102 TCP Y clearance BLOCKED: "
                        f"101={float(measurement['left_y_m']) * 1000.0:.1f}mm, "
                        f"102(common)={float(measurement['right_common_y_m']) * 1000.0:.1f}mm, "
                        f"gap={float(measurement['gap_y_m']) * 1000.0:.1f}mm "
                        f"< {float(measurement['threshold_m']) * 1000.0:.1f}mm; "
                        f"holding 101 before {stage}"
                    )
                    self.secondary_last_clearance_log_at = now
                self.secondary_last_clearance_state = False
                last_state = False
            except Exception as exc:
                last_error = str(exc)
                now = time.monotonic()
                if now - self.secondary_last_clearance_log_at >= log_period_s:
                    self._publish_status(
                        "102 TCP safety feedback unavailable; "
                        f"holding 101 before {stage}: {last_error}"
                    )
                    self.secondary_last_clearance_log_at = now
                last_state = None

            if time.monotonic() >= deadline:
                raise SecondaryClearanceRetry(
                    f"102 TCP Y clearance was not available for {stage} within "
                    f"{timeout_s:.1f}s; no left-arm approach was issued; "
                    f"last status: {last_error}"
                )
            time.sleep(poll_s)

        raise SecondaryClearanceRetry(
            f"cycle cancelled while waiting for 102 TCP Y clearance before {stage}"
        )

    def _secondary_interlock_distances(self) -> tuple[float, float, float]:
        protective_m = max(
            0.001,
            float(self.get_parameter("secondary_y_clearance_m").value),
        )
        retreat_m = max(
            0.001,
            float(self.get_parameter("secondary_emergency_retreat_m").value),
        )
        recover_m = max(
            retreat_m,
            float(self.get_parameter("secondary_emergency_recover_m").value),
        )
        if protective_m <= retreat_m:
            raise RuntimeError(
                "secondary_y_clearance_m must be greater than "
                "secondary_emergency_retreat_m"
            )
        if recover_m < protective_m:
            raise RuntimeError(
                "secondary_emergency_recover_m must be at least "
                "secondary_y_clearance_m"
            )
        return protective_m, retreat_m, recover_m

    def _latch_secondary_protective_stop(
        self,
        detail: str,
        robot_mode: int,
    ) -> None:
        """Cancel the active 101 cycle and stop only the left-arm controller."""

        with self.secondary_safety_lock:
            first_trigger = not self.secondary_protective_stop_latched.is_set()
            interrupted_continuous_cycle = (
                first_trigger
                and self.cycle_enabled
                and self.worker is not None
                and self.worker.is_alive()
            )
            if interrupted_continuous_cycle:
                self.secondary_auto_resume_requested.set()
            # Re-close the cycle on every unsafe sample.  This also handles an
            # enable request racing with a latch that has not cleared yet.
            self.cycle_enabled = False
            if first_trigger:
                self.secondary_protective_stop_latched.set()
                self.secondary_safety_reason = str(detail)

        if first_trigger:
            self.get_logger().warning(str(detail))
            self._publish_status(str(detail))
        if int(robot_mode) not in (7, 8, 10):
            return
        try:
            self.controller.stop_motion()
        except Exception as exc:
            self.get_logger().fatal(
                f"101 protective Stop failed after secondary TCP trigger: {exc}"
            )

    def _schedule_continuous_restart_after_secondary_retreat(self) -> None:
        """Start a fresh continuous worker after the cancelled worker exits."""

        with self.secondary_safety_lock:
            if not self.secondary_auto_resume_requested.is_set():
                return
            if (
                self.secondary_resume_thread is not None
                and self.secondary_resume_thread.is_alive()
            ):
                return
            resume_thread = threading.Thread(
                target=self._resume_continuous_after_secondary_retreat,
                name="secondary-y-continuous-restart",
                daemon=True,
            )
            self.secondary_resume_thread = resume_thread
        resume_thread.start()

    def _resume_continuous_after_secondary_retreat(self) -> None:
        """Restart at startup; never continue the interrupted trajectory."""

        current_thread = threading.current_thread()
        try:
            # The emergency retreat could acquire ``action_lock`` as soon as
            # the cancelled cycle left its robot sequence, while that old
            # Python thread still had a few log statements to finish.  Do not
            # let _ensure_worker mistake it for the replacement worker.
            while self.running and not self.secondary_safety_shutdown.is_set():
                if not self.secondary_auto_resume_requested.is_set():
                    return
                previous_worker = self.worker
                if previous_worker is None or not previous_worker.is_alive():
                    break
                time.sleep(0.01)
            else:
                return

            with self.secondary_safety_lock:
                if (
                    not self.running
                    or self.secondary_safety_shutdown.is_set()
                    or not self.secondary_auto_resume_requested.is_set()
                    or self.secondary_protective_stop_latched.is_set()
                    or self.secondary_retreat_active.is_set()
                ):
                    return
                if self.controller.robot_mode != 5:
                    self.secondary_auto_resume_requested.clear()
                    self._publish_status(
                        "101 retreat completed, but automatic restart was cancelled: "
                        f"controller is {self.controller.robot_mode_text()}"
                    )
                    return
                # Clearing the request and enabling the replacement worker are
                # atomic with respect to operator Stop/manual recovery methods.
                self.secondary_auto_resume_requested.clear()
                self.cycle_enabled = True

            self._publish_status(
                "101 retreat recovery complete; starting a fresh continuous "
                "cycle from the startup joint"
            )
            self._ensure_worker()
        finally:
            with self.secondary_safety_lock:
                if self.secondary_resume_thread is current_thread:
                    self.secondary_resume_thread = None

    def _wait_for_action_lock_during_emergency(self, deadline: float) -> bool:
        """Wait for the cancelled worker to unwind, re-stopping any late motion."""

        while self.running and not self.secondary_safety_shutdown.is_set():
            if self.action_lock.acquire(blocking=False):
                return True
            if self.controller.robot_mode in (7, 8, 10):
                try:
                    self.controller.stop_motion()
                except Exception as exc:
                    self.get_logger().fatal(
                        f"101 remained active while emergency retreat waited for control: {exc}"
                    )
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return False

    def _run_secondary_emergency_retreat(self) -> None:
        """Jog 101 directly away in User Y until the common-Y gap is recovered."""

        if self.secondary_retreat_active.is_set():
            return
        self.secondary_retreat_active.set()
        poll_s = max(
            0.005,
            min(
                0.05,
                float(self.get_parameter("secondary_motion_monitor_poll_s").value),
            ),
        )
        timeout_s = max(
            0.2,
            float(self.get_parameter("secondary_emergency_retreat_timeout_s").value),
        )
        deadline = time.monotonic() + timeout_s
        action_lock_acquired = False
        jog_started = False
        recovered = False
        retreat_cleanup_ok = True
        axis_name = ""
        start_left_y_m = 0.0
        last_gap_m = float("nan")
        try:
            action_lock_acquired = self._wait_for_action_lock_during_emergency(deadline)
            if not action_lock_acquired:
                raise RuntimeError(
                    "101 emergency retreat could not obtain robot control before timeout"
                )

            if self.controller.robot_mode in (7, 8, 10):
                self.controller.stop_motion()
                remaining_s = max(0.05, deadline - time.monotonic())
                self.controller.wait_until_idle(timeout_s=remaining_s)
            if self.controller.robot_mode != 5:
                raise RuntimeError(
                    "101 is not enabled and idle for emergency retreat: "
                    f"mode={self.controller.robot_mode_text()}"
                )

            _, retreat_trigger_m, recover_m = self._secondary_interlock_distances()
            measurement = self._read_secondary_y_clearance(allow_reconnect=False)
            last_gap_m = float(measurement["gap_y_m"])
            if last_gap_m >= recover_m:
                recovered = True
                self.secondary_retreat_attempted.clear()
                self._publish_status(
                    "101 emergency retreat no longer required: "
                    f"TCP Y gap recovered to {last_gap_m * 1000.0:.1f}mm"
                )
                return
            if last_gap_m >= retreat_trigger_m:
                # The hard threshold was crossed before this worker obtained
                # command ownership, but 102 has already moved back out.  Keep
                # 101 stopped; do not create an unnecessary new trajectory.
                self._publish_status(
                    "101 remains stopped after emergency trigger: "
                    f"TCP Y gap={last_gap_m * 1000.0:.1f}mm; "
                    f"waiting for {recover_m * 1000.0:.1f}mm before restart"
                )
                self.secondary_retreat_attempted.clear()
                return

            start_left_y_m = float(measurement["left_y_m"])
            right_common_y_m = float(measurement["right_common_y_m"])
            axis_name = secondary_y_retreat_axis(
                start_left_y_m,
                right_common_y_m,
            )
            retreat_speed = max(
                1,
                min(
                    100,
                    int(self.get_parameter("secondary_emergency_retreat_speed").value),
                ),
            )
            max_travel_m = max(
                0.001,
                float(
                    self.get_parameter(
                        "secondary_emergency_retreat_max_travel_m"
                    ).value
                ),
            )
            self._publish_status(
                "EMERGENCY 101 Y retreat starting: "
                f"gap={last_gap_m * 1000.0:.1f}mm "
                f"< {retreat_trigger_m * 1000.0:.1f}mm, "
                f"direction={axis_name}, target_gap={recover_m * 1000.0:.1f}mm, "
                f"speed={retreat_speed}%"
            )
            self.controller.set_speed_factor(retreat_speed)
            self.controller.move_jog(
                axis_name,
                coord_type=1,
                user=int(self.get_parameter("user_index").value),
                tool=int(self.get_parameter("command_tool_index").value),
            )
            jog_started = True
            clear_samples = 0
            while self.running and not self.secondary_safety_shutdown.is_set():
                measurement = self._read_secondary_y_clearance(allow_reconnect=False)
                last_gap_m = float(measurement["gap_y_m"])
                travel_m = abs(float(measurement["left_y_m"]) - start_left_y_m)
                if last_gap_m >= recover_m:
                    clear_samples += 1
                    if clear_samples >= 2:
                        recovered = True
                        break
                else:
                    clear_samples = 0
                if travel_m >= max_travel_m:
                    raise RuntimeError(
                        f"101 emergency retreat reached max travel "
                        f"{travel_m * 1000.0:.1f}mm before recovering gap"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"101 emergency retreat timed out with TCP Y gap "
                        f"{last_gap_m * 1000.0:.1f}mm"
                    )
                time.sleep(poll_s)
        except Exception as exc:
            with self.secondary_safety_lock:
                self.secondary_auto_resume_requested.clear()
            self.secondary_safety_reason = f"101 emergency retreat failed: {exc}"
            self.get_logger().fatal(self.secondary_safety_reason)
            self._publish_status(self.secondary_safety_reason)
        finally:
            if jog_started:
                try:
                    self.controller.move_jog("")
                    self.controller.wait_until_idle(timeout_s=5.0)
                except Exception as exc:
                    retreat_cleanup_ok = False
                    self.get_logger().fatal(
                        f"Stopping 101 emergency {axis_name} retreat failed: {exc}"
                    )
                try:
                    self.controller.set_speed_factor(100)
                except Exception as exc:
                    retreat_cleanup_ok = False
                    self.get_logger().warning(
                        f"Restoring 101 speed after emergency retreat failed: {exc}"
                    )
            if action_lock_acquired:
                self.action_lock.release()
            self.secondary_retreat_active.clear()

        if recovered and retreat_cleanup_ok:
            with self.secondary_safety_lock:
                self.secondary_protective_stop_latched.clear()
                self.secondary_retreat_attempted.clear()
                auto_restart = self.secondary_auto_resume_requested.is_set()
            if auto_restart:
                self._publish_status(
                    "101 emergency Y retreat completed: "
                    f"TCP Y gap={last_gap_m * 1000.0:.1f}mm; "
                    "returning to startup before automatic continuous restart"
                )
                self._schedule_continuous_restart_after_secondary_retreat()
            else:
                self._publish_status(
                    "101 emergency Y retreat completed: "
                    f"TCP Y gap={last_gap_m * 1000.0:.1f}mm; "
                    "manual operation remains stopped"
                )
        elif recovered:
            with self.secondary_safety_lock:
                self.secondary_auto_resume_requested.clear()
            self.secondary_safety_reason = (
                "101 reached the emergency recovery distance, but retreat cleanup "
                "failed; automatic restart is disabled"
            )
            self.get_logger().fatal(self.secondary_safety_reason)
            self._publish_status(self.secondary_safety_reason)

    def _secondary_motion_safety_loop(self) -> None:
        """Continuously enforce the two-stage common-Y policy for 101 only."""

        while self.running and not self.secondary_safety_shutdown.is_set():
            try:
                poll_s = max(
                    0.005,
                    min(
                        0.05,
                        float(
                            self.get_parameter(
                                "secondary_motion_monitor_poll_s"
                            ).value
                        ),
                    ),
                )
                collision_check_enabled = bool(
                    self.get_parameter("secondary_collision_check_enabled").value
                )
                auto_place_enabled = bool(
                    self.get_parameter(
                        "turntable_auto_place_from_secondary_tcp"
                    ).value
                )
                if not collision_check_enabled and not auto_place_enabled:
                    self.secondary_safety_shutdown.wait(poll_s)
                    continue
                robot_mode = self.controller.robot_mode
                monitoring_active = (
                    self.cycle_enabled
                    or self.secondary_protective_stop_latched.is_set()
                    or robot_mode in (7, 8, 10)
                    or auto_place_enabled
                )
                if not monitoring_active or self.secondary_retreat_active.is_set():
                    self.secondary_safety_shutdown.wait(poll_s)
                    continue

                try:
                    measurement = self._read_secondary_y_clearance(
                        # Reconnect only while 101 is idle.  During motion the
                        # safety path must fail fast instead of blocking on a
                        # new socket, while WAIT_PLACE may safely retry until
                        # the passive 102 trigger becomes available.
                        allow_reconnect=robot_mode not in (7, 8, 10)
                    )
                except Exception as exc:
                    if robot_mode in (7, 8, 10):
                        self._latch_secondary_protective_stop(
                            "102 TCP safety feedback failed while 101 was moving; "
                            f"stopping 101: {exc}",
                            robot_mode,
                        )
                    self.secondary_safety_shutdown.wait(poll_s)
                    continue

                self.secondary_last_measurement = measurement
                self._update_turntable_place_from_secondary_tcp(measurement)
                if not collision_check_enabled:
                    self.secondary_safety_shutdown.wait(poll_s)
                    continue
                protective_m, retreat_m, _ = self._secondary_interlock_distances()
                gap_m = float(measurement["gap_y_m"])
                action = secondary_y_interlock_action(
                    gap_m,
                    robot_mode,
                    protective_m,
                    retreat_m,
                )
                if action == "stop":
                    self._latch_secondary_protective_stop(
                        "101 protective stop: common-Y TCP gap "
                        f"{gap_m * 1000.0:.1f}mm "
                        f"< {protective_m * 1000.0:.1f}mm; "
                        "current left-arm trajectory cancelled; 102 remains read-only",
                        robot_mode,
                    )
                elif action == "retreat":
                    self._latch_secondary_protective_stop(
                        "101 emergency retreat trigger: common-Y TCP gap "
                        f"{gap_m * 1000.0:.1f}mm "
                        f"< {retreat_m * 1000.0:.1f}mm; "
                        "cancelling the cycle before left-arm-only Y retreat",
                        robot_mode,
                    )
                    if not self.secondary_retreat_attempted.is_set():
                        self.secondary_retreat_attempted.set()
                        self._run_secondary_emergency_retreat()
                elif (
                    self.secondary_protective_stop_latched.is_set()
                    and gap_m >= protective_m
                    and self.controller.robot_mode == 5
                ):
                    # The interrupted cycle remains disabled.  Clearing only
                    # the monitor latch prevents a later idle 102 motion from
                    # unexpectedly moving 101 after the incident is over.
                    self.secondary_protective_stop_latched.clear()
                    self.secondary_retreat_attempted.clear()
                    # A warning-band Stop that never required a physical 101
                    # retreat is intentionally not auto-resumed.
                    self.secondary_auto_resume_requested.clear()

                self.secondary_safety_shutdown.wait(poll_s)
            except Exception as exc:
                # A supervisor bug must not terminate monitoring silently.
                self.get_logger().error(f"Secondary motion safety loop error: {exc}")
                self.secondary_safety_shutdown.wait(0.05)

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
            user_z_bias_m = float(self.get_parameter("vision_user_z_bias_m").value)
            if not math.isfinite(user_z_bias_m) or abs(user_z_bias_m) > 0.050:
                raise ValueError(
                    "vision_user_z_bias_m must be finite and within +/-0.050m, "
                    f"got {user_z_bias_m!r}"
                )
            command_pose = TcpPose(
                command_pose.x,
                command_pose.y,
                command_pose.z + user_z_bias_m,
                command_pose.rx,
                command_pose.ry,
                command_pose.rz,
            )
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
        frame_time_s = message_stamp_seconds(msg)
        tcp_linear_speed_mps: Optional[float] = None
        try:
            tcp_linear_speed_mps = self.controller.current_tcp_linear_speed_at(
                frame_time_s,
                max_skew_s=float(self.get_parameter("vision_pose_max_time_skew_s").value),
            )
        except (RuntimeError, ValueError):
            # Speed is a quality hint only.  The pose transform has already
            # passed the strict timestamped feedback check above, so a
            # firmware variant without speed history must not discard it.
            pass
        with self.data_lock:
            self.pregrasp_pose_count += 1
            observation_count = self.pregrasp_pose_count
            self.latest_pregrasp_pose = command_pose
            self.latest_pregrasp_pose_received_at = time.monotonic()
            self.pregrasp_observations.append(
                PregraspObservation(
                    count=observation_count,
                    pose=command_pose,
                    frame_time_s=frame_time_s,
                    received_at_monotonic=self.latest_pregrasp_pose_received_at,
                    tcp_linear_speed_mps=tcp_linear_speed_mps,
                )
            )

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
            negative_side_clear = payload.get("negative_side_clear") is True
            positive_side_clear = payload.get("positive_side_clear") is True
            rgb_rotation = payload.get("rgb_to_ir_rotation")
            if rgb_rotation is not None:
                rgb_rotation = np.asarray(rgb_rotation, dtype=float).reshape(3,3)
                if (not np.all(np.isfinite(rgb_rotation)) or
                        not np.allclose(rgb_rotation.T @ rgb_rotation, np.eye(3), atol=1e-4) or
                        np.linalg.det(rgb_rotation) < 0.99):
                    raise ValueError("invalid RGB-to-IR calibration")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid D405 handoff state: {exc}")
            return
        with self.data_lock:
            self.rgb_to_ir_rotation = rgb_rotation
            self.latest_handoff_state = state
            self.latest_handoff_clear = clear
            self.latest_handoff_candidate_points = candidate_points
            self.latest_handoff_cluster_points = cluster_points
            self.latest_handoff_negative_side_clear = negative_side_clear
            self.latest_handoff_positive_side_clear = positive_side_clear
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

    def _top_surface_barcode_callback(self, msg: String) -> None:
        """Accept only confirmed top-surface results during the active cycle."""

        encoded = msg.data.strip()
        if not encoded.startswith("success:"):
            return
        value = encoded.split(":", 1)[1].strip()
        if not value:
            return
        with self.top_surface_barcode_lock:
            if not self.top_surface_barcode_window_active:
                return
            self.top_surface_barcode_value = value
            self.top_surface_barcode_result_count += 1
        self.get_logger().info(
            f"Top-surface barcode confirmed for current target: {value!r}"
        )

    def _turntable_barcode_callback(self, msg: String) -> None:
        """Accept one D435 side-barcode result from the active scan window."""

        encoded = msg.data.strip()
        if not encoded.startswith("success:"):
            return
        value = encoded.split(":", 1)[1].strip()
        if not value:
            return
        with self.turntable_lock:
            if not self.turntable_barcode_window_active:
                return
            self.turntable_barcode_value = value
            self.turntable_barcode_result_count += 1
        self.get_logger().info(
            f"D435 turntable side barcode confirmed: {value!r}"
        )

    def _turntable_barcode_ready_callback(self, msg: Bool) -> None:
        with self.turntable_lock:
            self.turntable_camera_ready = bool(msg.data)

    def _d435_continuous_barcode_callback(self, msg: String) -> None:
        encoded = msg.data.strip()
        if not encoded.startswith("success:"):
            return
        value = encoded.split(":", 1)[1].strip()
        if not value:
            return
        with self.turntable_lock:
            if not self.d435_continuous_detection:
                return
            self.d435_continuous_last_value = value
        self.get_logger().info(
            f"D435 continuous result received: {value!r}"
        )

    def _d435_continuous_presence_callback(self, msg: Bool) -> None:
        with self.turntable_condition:
            self.d435_continuous_presence = bool(msg.data)
            if not self.d435_continuous_presence:
                self.d435_continuous_last_value = ""
            self.turntable_condition.notify_all()

    def _turntable_place_done_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        self._accept_turntable_place_done("manual/topic place_done")

    def _accept_turntable_place_done(self, source: str) -> None:
        """Accept one placed material and immediately start its D435 scan.

        In the V3 cell contract a place_done event means the material has been
        released and the turntable is stationary.  This is therefore also the
        evidence that resolves an initial UNKNOWN software state to STOPPED.
        """

        direct_barcode = ""
        thread = None
        with self.turntable_condition:
            busy = self.turntable_scan_in_progress or self.turntable_material_ready
            if busy:
                pending_number = self.turntable_place_done_count
                should_warn = not self.turntable_place_done_duplicate_warned
                self.turntable_place_done_duplicate_warned = True
                accepted = False
            else:
                if self.turntable_state == "UNKNOWN":
                    self.turntable_state = "STOPPED"
                elif self.turntable_state != "STOPPED":
                    raise RuntimeError(
                        "place_done cannot start a new scan while turntable "
                        f"state={self.turntable_state}"
                    )
                self.turntable_place_done_count += 1
                event_number = self.turntable_place_done_count
                self.turntable_place_done_consumed = event_number
                self.turntable_place_done_duplicate_warned = False
                self.turntable_waiting_for_place = False
                self.turntable_scan_error = ""
                self.turntable_scan_cancel.clear()
                self.turntable_secondary_retreat_trigger.reset()
                if (
                    self.d435_continuous_detection
                    and self.d435_continuous_presence
                    and self.d435_continuous_last_value
                ):
                    direct_barcode = str(self.d435_continuous_last_value)
                    self.turntable_scan_in_progress = False
                    self.turntable_material_ready = True
                    self.turntable_ready_barcode = direct_barcode
                    self.turntable_scan_thread = None
                    self.turntable_condition.notify_all()
                else:
                    self.turntable_scan_in_progress = True
                    self.turntable_material_ready = False
                    self.turntable_ready_barcode = ""
                    thread = threading.Thread(
                        target=self._turntable_prescan_worker,
                        args=(event_number,),
                        name=f"turntable-prescan-{event_number}",
                        daemon=True,
                    )
                    self.turntable_scan_thread = thread
                accepted = True

        if not accepted:
            if should_warn:
                self._publish_status(
                    "ignoring duplicate 102 place_done; "
                    f"material event #{pending_number} is already scanning or "
                    "is stopped and ready for the left arm"
                )
            return
        if direct_barcode:
            self._publish_status(
                f"accepted material event #{event_number} from {source}; "
                f"D435 already sees {direct_barcode!r} on the stopped material, "
                "so no turntable pulse is needed and the left arm may pick"
            )
            return
        self._publish_status(
            f"accepted material event #{event_number} from {source}; "
            "place_done confirms the turntable is stopped, starting a fresh "
            "stopped-face D435 confirmation"
        )
        if thread is None:
            raise RuntimeError("turntable pre-scan worker was not created")
        thread.start()

    def _turntable_prescan_worker(self, event_number: int) -> None:
        barcode = ""
        error = ""
        cancelled = False
        try:
            barcode = self._scan_turntable_for_side_barcode(
                require_cycle_active=False,
            )
        except Exception as exc:
            error = str(exc)
            cancelled = self.turntable_scan_cancel.is_set()
            try:
                self._stop_turntable_if_running(
                    f"background scan failure for material #{event_number}"
                )
            except Exception as stop_exc:
                error = f"{error}; turntable stop also failed: {stop_exc}"
        finally:
            self._set_turntable_barcode_window(False)
            with self.turntable_condition:
                self.turntable_scan_in_progress = False
                if cancelled:
                    # An operator recovery owns the next state transition.  Do
                    # not let this obsolete worker publish ready/error state
                    # after the previous round has been abandoned.
                    self.turntable_material_ready = False
                    self.turntable_ready_barcode = ""
                    self.turntable_scan_error = ""
                elif not error:
                    self.turntable_material_ready = True
                    self.turntable_ready_barcode = barcode
                    self.turntable_scan_error = ""
                else:
                    self.turntable_material_ready = False
                    self.turntable_ready_barcode = ""
                    self.turntable_scan_error = error
                self.turntable_condition.notify_all()

        if cancelled:
            return
        if error:
            self._publish_status(
                f"material event #{event_number} scan failed: {error}; "
                "after checking the stopped table, click simulated place_done to retry"
            )
        elif not barcode:
            self._publish_status(
                f"material event #{event_number}: D435 found no side barcode "
                "within the scan window; turntable is stopped and the material "
                "is ready for 101 to grasp and check its top surface"
            )
        else:
            self._publish_status(
                f"material event #{event_number} barcode confirmed as {barcode!r}; "
                "turntable is stopped and the material is ready—click Execute once"
            )

    def _wait_for_scanned_turntable_material(self) -> str:
        """Wait for the independently scanned and stopped material.

        An empty barcode is a valid completed scan: D405 will inspect the
        material's top after the left arm starts its normal grasp cycle.
        """

        if not bool(self.get_parameter("turntable_enabled").value):
            return ""
        timeout_s = max(
            0.0,
            float(self.get_parameter("turntable_place_wait_timeout_s").value),
        )
        deadline = time.monotonic() + timeout_s if timeout_s > 0.0 else None
        announced = False
        while self.running and self.cycle_enabled:
            with self.turntable_condition:
                if self.turntable_material_ready:
                    barcode = self.turntable_ready_barcode
                    state = self.turntable_state
                    if state != "STOPPED":
                        raise RuntimeError(
                            "scanned material cannot be picked because turntable "
                            f"state={state}, expected=STOPPED"
                        )
                    return barcode
                scan_in_progress = self.turntable_scan_in_progress
                scan_error = self.turntable_scan_error
                if scan_error:
                    raise RuntimeError(scan_error)
                if not announced:
                    announced = True
                    self._publish_status(
                        "waiting for stopped, D435-scanned material; place_done "
                        "starts the turntable scan independently, and the left arm "
                        "will move after a side-barcode result or the configured "
                        "no-side-barcode timeout and stop"
                    )
                wait_s = 0.05
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        state_text = "scan still running" if scan_in_progress else "no material event"
                        raise TimeoutError(
                            f"no D435-ready turntable material within {timeout_s:.1f}s "
                            f"({state_text})"
                        )
                    wait_s = min(wait_s, remaining)
                self.turntable_condition.wait(timeout=wait_s)
        raise RuntimeError("cycle cancelled while waiting for D435-ready material")

    def _mark_turntable_material_removed(self) -> None:
        """Re-arm place detection once the lifted box has left the turntable."""

        with self.turntable_condition:
            if not self.turntable_material_ready:
                return
            self.turntable_material_ready = False
            self.turntable_ready_barcode = ""
            self.turntable_scan_error = ""
            self.turntable_waiting_for_place = True
            self.turntable_place_done_duplicate_warned = False
            self.turntable_secondary_retreat_trigger.reset()
            # Do not let the barcode on the box that is now in the left
            # gripper satisfy the next material event.  The D435 node will
            # publish a fresh presence edge after the old box leaves view.
            self.d435_continuous_last_value = ""
            self.d435_continuous_presence = False
            self.turntable_condition.notify_all()
        self._publish_status(
            "turntable material was removed by the left arm; waiting for the next 102 placement"
        )

    def _reset_turntable_round_for_operator_recovery(self) -> None:
        """Discard the previous material workflow after a startup recovery.

        The independent pre-scan thread can outlive a cancelled left-arm
        cycle, so it must be cancelled and joined before the ready/error flags
        are cleared.  Re-arming then requires 102 to be observed back on the
        retreat side before a new Y-entry/Y-retreat sequence can fire.
        """

        self.turntable_scan_cancel.set()
        with self.turntable_condition:
            self.turntable_condition.notify_all()

        # The scan worker also stops a running table in its finally path.  This
        # direct call covers camera/clearance waits that ended before rotation.
        self._stop_turntable_if_running("operator startup recovery")
        scan_thread = self.turntable_scan_thread
        if scan_thread is not None and scan_thread is not threading.current_thread():
            scan_thread.join(timeout=6.0)
            if scan_thread.is_alive():
                raise RuntimeError(
                    "previous turntable scan did not stop during startup recovery; "
                    "new material detection remains disabled"
                )

        self._set_turntable_barcode_window(False)
        self._set_top_surface_barcode_window(False)
        with self.turntable_condition:
            self.turntable_scan_in_progress = False
            self.turntable_material_ready = False
            self.turntable_ready_barcode = ""
            self.turntable_scan_error = ""
            self.turntable_waiting_for_place = True
            self.turntable_place_done_count = 0
            self.turntable_place_done_consumed = 0
            self.turntable_place_done_duplicate_warned = False
            self.turntable_scan_thread = None
            self.turntable_secondary_retreat_trigger.reset(
                require_fresh_entry=True
            )
            self.turntable_barcode_value = ""
            self.turntable_barcode_result_count = 0
            self.d435_continuous_last_value = ""
            self.d435_continuous_presence = False
            self.turntable_scan_cancel.clear()
            self.turntable_condition.notify_all()

        self.last_accepted_target = None
        self.last_accepted_width_m = None
        self.last_accepted_length_m = None
        self.last_accepted_height_m = None

    def notify_turntable_place_done(self) -> None:
        """GUI/manual equivalent of one rising place-done event."""

        message = Bool()
        message.data = True
        self._turntable_place_done_callback(message)

    def confirm_turntable_stopped(self) -> None:
        """Record an operator's physical stopped-state confirmation.

        GetDO can only confirm the electrical output returned low; it cannot
        prove motor speed.  This action is therefore intentionally explicit.
        """

        with self.turntable_lock:
            state = self.turntable_state
        if state not in ("UNKNOWN", "STOPPED"):
            raise RuntimeError(
                f"cannot confirm turntable stopped while software state is {state}"
            )
        index = int(self.get_parameter("turntable_do_index").value)
        electrical_state = self.controller.read_digital_output(index)
        if electrical_state != 0:
            raise RuntimeError(
                f"DO{index} is still high; return it to 0 before confirming stop"
            )
        with self.turntable_lock:
            self.turntable_state = "STOPPED"
        self._publish_status(
            f"operator confirmed turntable physically stopped; DO{index}=0"
        )

    def _set_turntable_barcode_window(self, active: bool) -> None:
        with self.turntable_lock:
            self.turntable_barcode_window_active = bool(active)
            if active:
                self.turntable_barcode_value = ""
                self.turntable_barcode_result_count = 0
        message = Bool()
        message.data = bool(active)
        self.turntable_barcode_trigger_publisher.publish(message)

    def set_d435_continuous_detection(self, enabled: bool) -> None:
        """Enable D435 inference without commanding the table or either arm."""

        enabled = bool(enabled)
        with self.turntable_lock:
            self.d435_continuous_detection = enabled
            self.d435_continuous_last_value = ""
            self.d435_continuous_presence = False
        message = Bool()
        message.data = enabled
        self.d435_continuous_trigger_publisher.publish(message)
        self._publish_status(
            f"D435 continuous detection {'enabled' if enabled else 'disabled'}; "
            "turntable and robot motion remain unchanged"
        )

    def _wait_for_turntable_camera_ready(
        self,
        *,
        require_cycle_active: bool = True,
    ) -> None:
        timeout_s = max(
            0.1,
            float(self.get_parameter("turntable_camera_ready_timeout_s").value),
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if require_cycle_active:
                self._require_cycle_active(
                    "waiting for D435 turntable scanner readiness"
                )
            elif not self.running or self.shutting_down:
                raise RuntimeError("node stopped while waiting for D435 readiness")
            elif self.turntable_scan_cancel.is_set():
                raise RuntimeError(
                    "D435 turntable scan cancelled by operator recovery"
                )
            with self.turntable_lock:
                if self.turntable_camera_ready:
                    return
            time.sleep(0.02)
        raise RuntimeError(
            f"D435 turntable scanner was not ready within {timeout_s:.1f}s"
        )

    def _toggle_turntable(self, expected: str, result: str, purpose: str) -> None:
        with self.turntable_lock:
            state = self.turntable_state
            if state != expected:
                raise RuntimeError(
                    f"turntable {purpose} refused: software state={state}, "
                    f"expected={expected}; physically verify stop and use the GUI "
                    "confirmation before continuing"
                )
            self.turntable_state = f"{purpose.upper()}_PULSE"
        index = int(self.get_parameter("turntable_do_index").value)
        pulse_ms = int(self.get_parameter("turntable_pulse_ms").value)
        try:
            self.controller.pulse_digital_output(index, pulse_ms)
        except Exception:
            with self.turntable_lock:
                self.turntable_state = "UNKNOWN"
            raise
        with self.turntable_lock:
            self.turntable_state = result
        self._publish_status(
            f"turntable {purpose} pulse completed on DO{index}; state={result}"
        )

    def _scan_turntable_for_side_barcode(
        self,
        *,
        require_cycle_active: bool = True,
    ) -> str:
        """Check the stopped face first, then rotate only as a fallback."""

        if not bool(self.get_parameter("turntable_enabled").value):
            return ""
        # The place event says the gripper has released, but the existing 102
        # feedback interlock remains the final software gate before rotation.
        self._wait_for_secondary_y_clearance(
            "turntable rotation",
            require_cycle_active=require_cycle_active,
        )
        self._wait_for_turntable_camera_ready(
            require_cycle_active=require_cycle_active,
        )
        self._set_turntable_barcode_window(True)
        barcode = ""
        turntable_was_started = False
        stationary_check_s = max(
            0.0,
            float(
                self.get_parameter("turntable_stationary_barcode_check_s").value
            ),
        )
        timeout_s = max(
            0.1,
            float(self.get_parameter("turntable_scan_timeout_s").value),
        )
        self._publish_status(
            "D435 side-barcode window armed while the turntable remains stopped; "
            f"checking the current face for up to {stationary_check_s:.1f}s"
        )

        def visible_barcode() -> str:
            with self.turntable_lock:
                workflow_value = str(self.turntable_barcode_value)
                continuous_value = (
                    str(self.d435_continuous_last_value)
                    if self.d435_continuous_detection
                    and self.d435_continuous_presence
                    else ""
                )
            return workflow_value or continuous_value

        try:
            try:
                stationary_deadline = time.monotonic() + stationary_check_s
                while time.monotonic() < stationary_deadline:
                    if require_cycle_active:
                        self._require_cycle_active(
                            "during stopped-face D435 barcode check"
                        )
                    elif not self.running or self.shutting_down:
                        raise RuntimeError(
                            "node stopped during stopped-face D435 barcode check"
                        )
                    if self.turntable_scan_cancel.is_set():
                        raise RuntimeError(
                            "D435 turntable scan cancelled by operator stop"
                        )
                    barcode = visible_barcode()
                    if barcode:
                        self._publish_status(
                            f"D435 detected barcode {barcode!r} on the already-visible "
                            "stopped face; skipping turntable rotation"
                        )
                        break
                    time.sleep(0.005)

                if not barcode:
                    self._publish_status(
                        "no barcode confirmed on the stopped face; starting turntable "
                        f"search for up to {timeout_s:.1f}s"
                    )
                    self._toggle_turntable("STOPPED", "RUNNING", "start")
                    turntable_was_started = True

                deadline = time.monotonic() + timeout_s
                while not barcode and time.monotonic() < deadline:
                    if require_cycle_active:
                        self._require_cycle_active(
                            "during D435 turntable side scan"
                        )
                    elif not self.running or self.shutting_down:
                        raise RuntimeError("node stopped during D435 turntable scan")
                    if self.turntable_scan_cancel.is_set():
                        raise RuntimeError(
                            "D435 turntable scan cancelled by operator stop"
                        )
                    barcode = visible_barcode()
                    if barcode:
                        self._publish_status(
                            f"D435 detected side barcode {barcode!r}; stopping turntable"
                        )
                        break
                    time.sleep(0.005)
                if not barcode and turntable_was_started:
                    self._publish_status(
                        f"no side barcode detected within {timeout_s:.1f}s; "
                        "stopping turntable but keeping D435 armed through settling"
                    )
            finally:
                with self.turntable_lock:
                    state = self.turntable_state
                if state == "RUNNING":
                    self._toggle_turntable("RUNNING", "STOPPED", "stop")

            settle_s = max(
                0.0,
                float(self.get_parameter("turntable_settle_s").value),
            )
            if turntable_was_started and settle_s > 0.0:
                self._publish_status(
                    f"waiting {settle_s:.2f}s for turntable/material to settle; "
                    "D435 confirmation remains armed"
                )
                deadline = time.monotonic() + settle_s
                while time.monotonic() < deadline:
                    if require_cycle_active:
                        self._require_cycle_active("waiting for turntable to settle")
                    elif not self.running or self.shutting_down:
                        raise RuntimeError("node stopped while turntable was settling")
                    if not barcode:
                        barcode = visible_barcode()
                    time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
            if not barcode:
                barcode = visible_barcode()
            if barcode and turntable_was_started:
                self._publish_status(
                    f"D435 barcode {barcode!r} confirmed before or during final stop/settle"
                )
            return barcode
        finally:
            self._set_turntable_barcode_window(False)

    def _stop_turntable_if_running(self, reason: str) -> None:
        # A GUI stop can arrive during either 0->1->0 pulse.  Wait for that
        # serialized Dashboard transaction to resolve, then stop exactly once
        # if it was a start pulse.  Never guess when state is UNKNOWN.
        pulse_s = max(
            0.05, int(self.get_parameter("turntable_pulse_ms").value) / 1000.0
        )
        deadline = time.monotonic() + 2.0 * pulse_s + 1.0
        while True:
            with self.turntable_lock:
                state = self.turntable_state
            if not state.endswith("_PULSE"):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"turntable pulse did not resolve during {reason}; state={state}"
                )
            time.sleep(0.01)
        if state != "RUNNING":
            return
        try:
            self._toggle_turntable("RUNNING", "STOPPED", f"stop ({reason})")
        except Exception as exc:
            self.get_logger().fatal(
                f"Could not stop running turntable during {reason}: {exc}"
            )
            raise

    def _set_top_surface_barcode_window(self, active: bool) -> None:
        """Open/close the v2 vision barcode ROI observation window."""

        enabled = bool(self.get_parameter("top_surface_barcode_enabled").value)
        active = bool(active and enabled)
        with self.top_surface_barcode_lock:
            self.top_surface_barcode_window_active = active
            if active:
                self.top_surface_barcode_value = ""
                self.top_surface_barcode_result_count = 0
        message = Bool()
        message.data = active
        self.top_surface_barcode_trigger_publisher.publish(message)
        self._publish_status(
            "top-surface barcode detection " + ("armed" if active else "disarmed")
        )

    def _current_top_surface_barcode(self) -> str:
        with self.top_surface_barcode_lock:
            return str(self.top_surface_barcode_value)

    def _wait_for_top_surface_barcode(self) -> str:
        """Allow a short hover window for the D405 detector to confirm the top label."""

        if not bool(self.get_parameter("top_surface_barcode_enabled").value):
            return ""
        with self.top_surface_barcode_lock:
            if not self.top_surface_barcode_window_active:
                return ""
        timeout_s = max(
            0.0, float(self.get_parameter("top_surface_barcode_wait_s").value)
        )
        deadline = time.monotonic() + timeout_s
        while self.running and self.cycle_enabled and time.monotonic() < deadline:
            value = self._current_top_surface_barcode()
            if value:
                return value
            time.sleep(0.01)
        return self._current_top_surface_barcode()

    def _cycle_enable_callback(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if not requested:
            with self.secondary_safety_lock:
                self.secondary_auto_resume_requested.clear()
                self.cycle_enabled = False
            self._set_top_surface_barcode_window(False)
            self._publish_status("cycle will stop after current blocking motion")
            return
        with self.secondary_safety_lock:
            protection_latched = self.secondary_protective_stop_latched.is_set()
            if protection_latched:
                self.cycle_enabled = False
            else:
                self.secondary_auto_resume_requested.clear()
                self.cycle_enabled = True
        if protection_latched:
            protective_m, _, _ = self._secondary_interlock_distances()
            self._publish_status(
                "cycle enable refused: 101/102 Y-clearance protection is still "
                f"latched; wait for an idle gap of at least "
                f"{protective_m * 1000.0:.1f}mm, then enable again"
            )
            return
        self._publish_status("cycle enabled")
        self._ensure_worker()

    def _start_automatically_once(self) -> None:
        self.start_timer.cancel()
        if bool(self.get_parameter("auto_start").value):
            with self.secondary_safety_lock:
                protection_latched = self.secondary_protective_stop_latched.is_set()
                if not protection_latched:
                    self.secondary_auto_resume_requested.clear()
                    self.cycle_enabled = True
            if protection_latched:
                self._publish_status(
                    "auto-start refused while 101/102 Y-clearance protection is latched"
                )
                return
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
                cycle_index = 0
                startup_prepared = False
                # None means this physical material has not yet been reserved
                # from the independent D435 ready queue.  The value survives
                # D405/grasp retries so the same box is never scanned again.
                pending_side_barcode: Optional[str] = None
                while self.running and self.cycle_enabled:
                    cycle_index += 1
                    self._begin_cycle_timing(f"continuous-{cycle_index}")
                    if pending_side_barcode is None:
                        with self._timed_stage("turntable_ready_wait"):
                            pending_side_barcode = (
                                self._wait_for_scanned_turntable_material()
                            )
                    if not startup_prepared:
                        self._wait_for_secondary_y_clearance("startup motion")
                        self._move_startup_and_open(require_cycle_active=True)
                        startup_prepared = True
                    # Do not let YOLO/SAM2 lock onto a box while 102 is still
                    # carrying or releasing it.  Waiting on the already-open
                    # read-only 102 feedback connection is cheap and avoids a
                    # slow discard/reacquire cycle after the target is placed.
                    try:
                        with self._timed_stage("prevision_secondary_clearance"):
                            self._wait_for_secondary_y_clearance("vision capture")
                    except SecondaryClearanceRetry as exc:
                        if not self.running or not self.cycle_enabled:
                            self._finish_cycle_timing("cancelled")
                            break
                        self._publish_status(str(exc))
                        self._finish_cycle_timing("prevision_secondary_clearance_retry")
                        continue
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
                        self._execute_one_cycle(
                            target_pose,
                            width_m,
                            height_m,
                            length_m,
                            pending_side_barcode,
                        )
                    except SecondaryClearanceRetry as exc:
                        if not self.running or not self.cycle_enabled:
                            self._finish_cycle_timing("cancelled")
                            break
                        self._publish_status(str(exc))
                        delay = max(0.1, float(self.get_parameter("vision_retry_delay_s").value))
                        with self._timed_stage("secondary_clearance_retry_delay"):
                            time.sleep(delay)
                        self._finish_cycle_timing("secondary_clearance_retry")
                        continue
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
                    pending_side_barcode = None
            self._publish_status("cycle stopped")
        except Exception as exc:
            self._set_top_surface_barcode_window(False)
            self._finish_cycle_timing("fault")
            self.cycle_enabled = False
            self._publish_status(f"FAULT: {exc}")
            self.get_logger().fatal(f"Automatic cycle stopped safely: {exc}")

    def _move_startup_and_open(
        self,
        require_cycle_active: bool = False,
        open_gripper: bool = True,
    ) -> None:
        motion = self._motion_profile()
        if require_cycle_active:
            self._require_cycle_active("before startup joint motion")
        startup_joint = self._six_values("startup_joint")
        tolerance_deg = max(
            0.0,
            float(self.get_parameter("startup_joint_skip_tolerance_deg").value),
        )
        startup_error_deg = math.inf
        try:
            current_joint = self.controller.current_joint()
            if len(current_joint) == len(startup_joint) and all(
                math.isfinite(float(value)) for value in current_joint
            ):
                startup_error_deg = max(
                    abs(float(current) - float(target))
                    for current, target in zip(current_joint, startup_joint)
                )
        except Exception as exc:
            self.get_logger().warning(
                f"Could not verify startup joint feedback; executing startup MovJ: {exc}"
            )

        if tolerance_deg > 0.0 and startup_error_deg <= tolerance_deg:
            self._publish_status(
                "startup joint already reached "
                f"(max error={startup_error_deg:.2f}deg <= {tolerance_deg:.2f}deg); "
                "skipping redundant MovJ"
            )
        else:
            self._publish_status("moving to startup joint")
            self.controller.move_joint(
                startup_joint,
                speed=motion["return_startup_speed"],
                accel=motion["return_startup_acc"],
            )
        if require_cycle_active:
            self._require_cycle_active("at startup joint")
        if open_gripper:
            self.gripper.set_force(int(self.get_parameter("dh_force").value))
            self.gripper.open(
                wait=True,
                cancel_check=self._cycle_cancel_requested
                if require_cycle_active
                else None,
            )

    def move_startup(self) -> None:
        """Recover to startup, forget the old round, and await a fresh place."""
        # This method is an explicit operator recovery request.  It overrides
        # any pending post-retreat automatic continuous restart.
        previous_worker = self.worker
        with self.secondary_safety_lock:
            self.secondary_auto_resume_requested.clear()
            self.cycle_enabled = False
        if self.secondary_protective_stop_latched.is_set():
            protective_m, _, _ = self._secondary_interlock_distances()
            raise RuntimeError(
                "startup motion refused while 101/102 Y-clearance protection "
                f"is latched; wait for an idle gap of at least "
                f"{protective_m * 1000.0:.1f}mm"
            )
        if bool(self.get_parameter("secondary_collision_check_enabled").value):
            try:
                measurement = self._read_secondary_y_clearance()
            except Exception as exc:
                raise RuntimeError(
                    "startup motion refused because 102 safety feedback is unavailable: "
                    f"{exc}"
                ) from exc
            protective_m, retreat_m, _ = self._secondary_interlock_distances()
            gap_m = float(measurement["gap_y_m"])
            if gap_m < protective_m:
                emergency_detail = (
                    "; the left-only emergency Y retreat will start"
                    if gap_m < retreat_m
                    else "; waiting for 102 to move away"
                )
                detail = (
                    "startup motion refused by Y-clearance protection: "
                    f"gap={gap_m * 1000.0:.1f}mm "
                    f"< {protective_m * 1000.0:.1f}mm"
                    f"{emergency_detail}"
                )
                self._latch_secondary_protective_stop(
                    detail,
                    self.controller.robot_mode,
                )
                raise RuntimeError(detail)
        self.turntable_scan_cancel.set()
        with self.turntable_condition:
            self.turntable_condition.notify_all()
        self._publish_status(
            "return-to-startup request accepted; cancelling and forgetting the "
            "previous material workflow"
        )

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
            self._reset_turntable_round_for_operator_recovery()

        # The cancelled worker can remain alive for a few final log/cleanup
        # statements after releasing action_lock.  Starting while it is still
        # alive would either reuse its local pending material or make
        # _ensure_worker incorrectly decide that no replacement is needed.
        deadline = time.monotonic() + 6.0
        while (
            previous_worker is not None
            and previous_worker is not threading.current_thread()
            and previous_worker.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if previous_worker is not None and previous_worker.is_alive():
            raise RuntimeError(
                "previous left-arm cycle did not exit after startup recovery; "
                "fresh-cycle waiting was not enabled"
            )

        with self.secondary_safety_lock:
            if self.secondary_protective_stop_latched.is_set():
                raise RuntimeError(
                    "startup recovery completed, but fresh-cycle waiting cannot "
                    "start while 101/102 Y-clearance protection is latched"
                )
            self.cycle_enabled = True
        self._ensure_worker()
        self._publish_status(
            "startup reached and gripper opened; previous round cleared; waiting "
            "for a fresh 102 Y-entry/Y-retreat placement event before any left-arm motion"
        )

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
        actual_raw = self.gripper.read_position_raw()
        target_raw = self.gripper.read_target_position_raw()
        position_m = max(0, min(1000, actual_raw)) / 1000.0 * float(
            self.get_parameter("dh_max_opening_m").value
        )
        state = self.gripper.read_grip_state()
        init_state = self.gripper.read_init_state()
        self._publish_status(
            f"gripper close test completed: command_raw={target_raw}, "
            f"actual_raw={actual_raw}, init_state={init_state}, "
            f"state={state}, opening={position_m*1000:.1f}mm"
        )

    def recalibrate_gripper(self) -> None:
        self._publish_status(
            "recalibrating AG-95 min/max stroke; keep the fingers clear"
        )
        self.gripper.recalibrate(
            timeout_s=max(15.0, float(self.get_parameter("dh_timeout_s").value)),
            open_after=True,
        )
        self._publish_status(
            "AG-95 recalibration completed; gripper opened and ready for close test"
        )

    def enable_robot(self) -> None:
        self.controller.enable_robot()
        self._publish_status("robot enabled")

    def clear_robot_error(self) -> None:
        self.controller.clear_error()
        self._publish_status("robot error cleared")

    def stop_robot(self) -> None:
        with self.secondary_safety_lock:
            self.secondary_auto_resume_requested.clear()
            self.cycle_enabled = False
        self.turntable_scan_cancel.set()
        with self.turntable_condition:
            self.turntable_condition.notify_all()
        turntable_error = None
        try:
            self._stop_turntable_if_running("operator stop")
        except Exception as exc:
            turntable_error = exc
        self.controller.stop_motion()
        self._publish_status("robot stopped; continuous cycle disabled")
        if turntable_error is not None:
            raise RuntimeError(
                "robot motion stopped, but turntable stop could not be confirmed: "
                f"{turntable_error}"
            ) from turntable_error

    def start_continuous_cycle(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("cycle worker is already running")
        with self.secondary_safety_lock:
            if self.secondary_protective_stop_latched.is_set():
                protective_m, _, _ = self._secondary_interlock_distances()
                raise RuntimeError(
                    "101/102 Y-clearance protection is latched; wait for an idle "
                    f"gap of at least {protective_m * 1000.0:.1f}mm before restarting"
                )
            self.secondary_auto_resume_requested.clear()
            self.cycle_enabled = True
        self._ensure_worker()
        self._publish_status("continuous cycle started")

    def stop_continuous_cycle(self) -> None:
        with self.secondary_safety_lock:
            self.secondary_auto_resume_requested.clear()
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
        with self.secondary_safety_lock:
            if self.secondary_protective_stop_latched.is_set():
                protective_m, _, _ = self._secondary_interlock_distances()
                raise RuntimeError(
                    "101/102 Y-clearance protection is latched; wait for an idle "
                    f"gap of at least {protective_m * 1000.0:.1f}mm before restarting"
                )
            self.secondary_auto_resume_requested.clear()
            self.cycle_enabled = True
        # Publish this before taking the action lock so a click that is waiting
        # for a previous action is visible immediately in the GUI/status topic.
        # The target-dependent motion still begins only after the fresh D405
        # pose is available; this status separates that vision wait from a
        # controller idle delay.
        self._publish_status(
            "single cycle request accepted; waiting for the already-scanned, "
            "stopped turntable material before moving the left arm"
        )
        try:
            turntable_side_barcode = self._wait_for_scanned_turntable_material()
            with self.action_lock:
                # 单轮流程开始前必须确保夹爪已经打开。
                self._wait_for_secondary_y_clearance("single-cycle startup motion")
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
                        self._execute_one_cycle(*result, turntable_side_barcode)
                        break
                    except SecondaryClearanceRetry as exc:
                        if not self.running or not self.cycle_enabled:
                            self._finish_cycle_timing("cancelled")
                            return
                        self._publish_status(str(exc))
                        delay = max(0.1, float(self.get_parameter("vision_retry_delay_s").value))
                        with self._timed_stage("secondary_clearance_retry_delay"):
                            time.sleep(delay)
                        self._finish_cycle_timing("secondary_clearance_retry")
                        continue
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
            self._set_top_surface_barcode_window(False)
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
        self._publish_status(
            "requesting fresh D405 target; robot remains at startup until the pose is verified"
        )
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
                        if (
                            not self.latest_handoff_negative_side_clear
                            or not self.latest_handoff_positive_side_clear
                        ):
                            status_update = (
                                "D405 finger corridor BLOCKED; waiting for obstacle "
                                "clearance "
                                f"(points={self.latest_handoff_candidate_points}, "
                                f"cluster={self.latest_handoff_cluster_points})"
                            )
                        else:
                            status_update = (
                                "D405 handoff zone BLOCKED; waiting for 102 to retreat "
                                f"(points={self.latest_handoff_candidate_points}, "
                                f"cluster={self.latest_handoff_cluster_points})"
                            )
                    elif self.latest_handoff_state == "VERIFYING_BLOCKED":
                        status_update = (
                            "D405 side corridor occupied in one LIVE cloud; "
                            "confirming spatial persistence before reacquire"
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
            time.sleep(VISION_REQUEST_POLL_S)
        else:
            self.get_logger().warning(
                "Timed out waiting for a fresh D405 target with a CLEAR handoff zone"
            )
            return None
        if (
            len(samples) != required_samples
            or width_m is None
            or length_m is None
            or height_m is None
        ):
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
        first_rotation = SciPyRot.from_euler(
            "xyz", [first.rx, first.ry, first.rz], degrees=True
        )
        second_rotation = SciPyRot.from_euler(
            "xyz", [second.rx, second.ry, second.rz], degrees=True
        )
        angle_delta = float(
            np.rad2deg((first_rotation.inv() * second_rotation).magnitude())
        )
        return position_delta, angle_delta

    def _fresh_pregrasp_observations(
        self,
        previous_count: int,
    ) -> tuple[list[PregraspObservation], float]:
        """Return this attempt's fresh tracked poses without waiting for a frame."""

        with self.data_lock:
            current_count = self.pregrasp_pose_count
            observations = list(self.pregrasp_observations)
        # ``previous_count`` delimits the current grasp attempt.  Even when a
        # post-correction diagnostic refresh is requested without waiting for a
        # new message, never mix observations from an earlier box/cycle.
        if previous_count >= 0:
            observations = [observation for observation in observations if observation.count > previous_count]
        now_monotonic = time.monotonic()
        max_age_s = max(0.05, float(self.get_parameter("pregrasp_pose_max_age_s").value))
        observations = [
            observation
            for observation in observations
            if now_monotonic - observation.received_at_monotonic <= max_age_s
        ]
        observations.sort(key=lambda observation: observation.frame_time_s)
        latest_age_s = (
            now_monotonic - observations[-1].received_at_monotonic
            if observations
            else float("inf")
        )
        if not observations:
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    "no fresh D405 pregrasp pose after reaching the safe hover "
                    f"(new_samples={max(0, current_count - previous_count)}, age={latest_age_s:.3f}s); "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )
        return observations, latest_age_s

    @staticmethod
    def _pose_position(pose: TcpPose) -> np.ndarray:
        return np.asarray([pose.x, pose.y, pose.z], dtype=np.float64)

    @staticmethod
    def _pose_rotation(pose: TcpPose) -> SciPyRot:
        return SciPyRot.from_euler(
            "xyz", [pose.rx, pose.ry, pose.rz], degrees=True
        )

    @staticmethod
    def _mean_rotation(rotations: list[SciPyRot]) -> SciPyRot:
        """Average quaternions while explicitly resolving their sign."""

        if not rotations:
            raise ValueError("Cannot average an empty rotation list")
        quaternions = np.asarray([rotation.as_quat() for rotation in rotations], dtype=np.float64)
        reference = quaternions[0]
        signs = np.where((quaternions @ reference) < 0.0, -1.0, 1.0)
        mean_quaternion = np.sum(quaternions * signs[:, None], axis=0)
        norm = np.linalg.norm(mean_quaternion)
        if norm <= 1e-9:
            return rotations[0]
        return SciPyRot.from_quat(mean_quaternion / norm)

    @staticmethod
    def _fit_pregrasp_xy_motion(
        observations: list[PregraspObservation],
    ) -> tuple[np.ndarray, float, float, float]:
        """Fit XY velocity; return velocity, max residual, displacement, span."""

        if len(observations) < 3:
            return np.zeros(2, dtype=np.float64), float("inf"), 0.0, 0.0
        times = np.asarray([observation.frame_time_s for observation in observations], dtype=np.float64)
        positions = np.asarray(
            [[observation.pose.x, observation.pose.y] for observation in observations],
            dtype=np.float64,
        )
        relative_times = times - times[-1]
        span_s = float(times[-1] - times[0])
        if span_s < 0.10 or not np.all(np.isfinite(positions)):
            return np.zeros(2, dtype=np.float64), float("inf"), 0.0, max(0.0, span_s)
        design = np.column_stack((relative_times, np.ones(len(relative_times))))
        coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
        velocity = np.asarray(coefficients[0], dtype=np.float64)
        fitted = design @ coefficients
        residuals = np.linalg.norm(positions - fitted, axis=1)
        displacement_m = float(np.linalg.norm(velocity * span_s))
        return velocity, float(np.max(residuals)), displacement_m, span_s

    def _estimate_pregrasp_target(
        self,
        reference: TcpPose,
        observations: list[PregraspObservation],
        now_s: float,
        hover_reached_at_s: float,
    ) -> dict[str, object]:
        """Select a no-wait target from the latest near-hover measured consensus.

        Eye-in-hand motion plus a small camera/feedback timestamp offset can look
        exactly like a smooth target trajectory.  Such a trajectory must never
        be extrapolated into a descent command.  A correction is accepted only
        when the newest near-hover measurement is itself outside tolerance and
        one or more immediately preceding measurements support that position.
        """

        recent = sorted(observations, key=lambda observation: observation.frame_time_s)[-8:]
        consensus_radius_m = max(
            0.002,
            float(self.get_parameter("pregrasp_position_consensus_m").value),
        )
        consensus_samples = max(
            2,
            int(self.get_parameter("pregrasp_position_consensus_samples").value),
        )
        settled_speed_mps = max(
            0.001,
            float(self.get_parameter("pregrasp_settled_tcp_speed_mps").value),
        )

        # Frames immediately around completion are preferred.  The window is
        # intentionally about one D405 period: this uses an already captured
        # frame when the camera/robot callback phases do not line up, without
        # inserting a settle sleep.
        hover_frame_window_s = max(
            0.08,
            float(self.get_parameter("pregrasp_hover_frame_window_s").value),
        )
        near_hover = [
            observation
            for observation in recent
            if observation.frame_time_s >= hover_reached_at_s - hover_frame_window_s
        ]

        def latest_suffix_consensus(
            candidates: list[PregraspObservation],
            require_settled: bool,
        ) -> Optional[list[PregraspObservation]]:
            """Return a consecutive consensus ending at the newest sample."""

            if len(candidates) < consensus_samples:
                return None
            newest = candidates[-1]
            newest_xy = np.asarray([newest.pose.x, newest.pose.y], dtype=np.float64)
            members_reversed: list[PregraspObservation] = []
            for observation in reversed(candidates):
                if require_settled and (
                    observation.tcp_linear_speed_mps is None
                    or observation.tcp_linear_speed_mps > settled_speed_mps
                ):
                    break
                observation_xy = np.asarray(
                    [observation.pose.x, observation.pose.y], dtype=np.float64
                )
                if np.linalg.norm(observation_xy - newest_xy) > consensus_radius_m:
                    break
                members_reversed.append(observation)
            if len(members_reversed) < consensus_samples:
                return None
            return list(reversed(members_reversed))

        # Keep the old fit only as evidence in logs.  It is deliberately never
        # used to choose or extrapolate a grasp point.
        velocity_xy, residual_m, displacement_m, span_s = self._fit_pregrasp_xy_motion(recent)
        _ = now_s
        reference_xy = np.asarray([reference.x, reference.y], dtype=np.float64)
        position_tolerance_m = max(
            0.001,
            float(self.get_parameter("pregrasp_position_tolerance_m").value),
        )
        consensus_members = None
        position_mode = "initial-stable-protected"
        selected_xy = reference_xy.copy()
        latest_xy_delta_m = float("inf")
        latest_speed_mps = None
        if near_hover:
            latest = near_hover[-1]
            latest_xy = np.asarray([latest.pose.x, latest.pose.y], dtype=np.float64)
            latest_xy_delta_m = float(np.linalg.norm(latest_xy - reference_xy))
            latest_speed_mps = latest.tcp_linear_speed_mps
            # A settled suffix has the best temporal quality.  If the feedback
            # speed hint is unavailable or the last frame was captured during
            # final deceleration, the same newest-ending measured consensus is
            # still usable; unlike the old estimator it is never extrapolated.
            consensus_members = latest_suffix_consensus(near_hover, require_settled=True)
            if consensus_members is not None:
                position_mode = "settled-hover-measured-consensus"
            else:
                consensus_members = latest_suffix_consensus(
                    near_hover, require_settled=False
                )
                if consensus_members is not None:
                    position_mode = "near-hover-measured-consensus"
        if consensus_members is not None:
            positions = np.asarray(
                [[observation.pose.x, observation.pose.y] for observation in consensus_members],
                dtype=np.float64,
            )
            selected_xy = np.median(positions, axis=0)
            selected_delta_m = float(np.linalg.norm(selected_xy - reference_xy))
            # The newest physical measurement is a mandatory gate.  This is
            # what prevents an older moving-camera trend from outvoting a fresh
            # frame that is already back at the original, correct box pose.
            if (
                latest_xy_delta_m <= position_tolerance_m
                or selected_delta_m <= position_tolerance_m
            ):
                consensus_members = None
                selected_xy = reference_xy.copy()
                position_mode = "initial-stable-protected"

        orientation_mode = "initial-stable-protected"
        selected_rotation = self._pose_rotation(reference)
        # Orientation is allowed to follow only when position evidence already
        # says that the target moved.  A stationary target with a persistent
        # but wrong OBB angle therefore cannot rewrite the TCP attitude.
        position_evidence_for_orientation = (
            consensus_members is not None
            and float(np.linalg.norm(selected_xy - reference_xy))
            > position_tolerance_m
        )
        if (
            bool(self.get_parameter("pregrasp_live_orientation_enabled").value)
            and position_evidence_for_orientation
        ):
            orientation_limit_deg = max(
                1.0,
                float(self.get_parameter("pregrasp_live_orientation_max_delta_deg").value),
            )
            orientation_required = max(
                3,
                int(self.get_parameter("pregrasp_orientation_consensus_samples").value),
            )
            orientation_spread_limit_deg = max(
                0.5,
                float(self.get_parameter("pregrasp_orientation_consensus_spread_deg").value),
            )
            orientation_candidates = []
            reference_rotation = self._pose_rotation(reference)
            reference_normal = reference_rotation.as_matrix()[:, 2]
            for observation in consensus_members or []:
                rotation = self._pose_rotation(observation.pose)
                relative_angle_deg = float(
                    np.rad2deg((reference_rotation.inv() * rotation).magnitude())
                )
                live_normal = rotation.as_matrix()[:, 2]
                normal_angle_deg = float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(float(np.dot(reference_normal, live_normal)), -1.0, 1.0)
                        )
                    )
                )
                # A flat box may yaw in its plane, but a sudden normal change
                # indicates an OBB flip/tilt and must not reach the TCP.
                if relative_angle_deg <= orientation_limit_deg and normal_angle_deg <= 5.0:
                    orientation_candidates.append((observation, rotation))
            if len(orientation_candidates) >= orientation_required:
                medoid_observation, _ = min(
                    orientation_candidates,
                    key=lambda candidate: sum(
                        float(
                            np.rad2deg(
                                (candidate[1].inv() * other[1]).magnitude()
                            )
                        )
                        for other in orientation_candidates
                    ),
                )
                medoid_rotation = next(
                    rotation
                    for observation, rotation in orientation_candidates
                    if observation is medoid_observation
                )
                consistent_rotations = [
                    rotation
                    for _, rotation in orientation_candidates
                    if float(
                        np.rad2deg((medoid_rotation.inv() * rotation).magnitude())
                    ) <= orientation_spread_limit_deg
                ]
                if len(consistent_rotations) >= orientation_required:
                    selected_rotation = self._mean_rotation(consistent_rotations)
                    orientation_mode = "live-planar-consensus"

        selected_rpy = selected_rotation.as_euler("xyz", degrees=True)
        selected_pose = TcpPose(
            float(selected_xy[0]),
            float(selected_xy[1]),
            reference.z,
            float(selected_rpy[0]),
            float(selected_rpy[1]),
            float(selected_rpy[2]),
        )
        position_delta, angle_delta = self._pose_delta(reference, selected_pose)
        return {
            "pose": selected_pose,
            "position_mode": position_mode,
            "orientation_mode": orientation_mode,
            "position_delta_m": float(position_delta),
            "angle_delta_deg": float(angle_delta),
            # Retained for compatibility with diagnostics.  Commanded target
            # velocity is always zero because grasp positions are not predicted.
            "velocity_xy_mps": np.zeros(2, dtype=np.float64),
            "apparent_velocity_xy_mps": velocity_xy,
            "motion_residual_m": float(residual_m),
            "motion_displacement_m": float(displacement_m),
            "motion_span_s": float(span_s),
            "observation_count": len(recent),
            "near_hover_count": len(near_hover),
            "consensus_count": len(consensus_members or []),
            "latest_xy_delta_m": float(latest_xy_delta_m),
            "latest_tcp_speed_mps": latest_speed_mps,
        }

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

    @staticmethod
    def _keep_reference_orientation(pose: TcpPose, reference: TcpPose) -> TcpPose:
        """Keep position from ``pose`` while preserving a known-safe RPY."""

        return TcpPose(
            pose.x,
            pose.y,
            pose.z,
            reference.rx,
            reference.ry,
            reference.rz,
        )

    def _revalidate_target_at_hover(
        self,
        target: TcpPose,
        motion: dict[str, int],
        z_offset_m: float,
        minimum_safe_z: float,
        previous_count: int,
        hover_reached_at_s: float,
        allow_hover_correction: bool = True,
    ) -> TcpPose:
        """Use synchronized, consensus-filtered tracking before descent."""

        observations, age_s = self._fresh_pregrasp_observations(previous_count)
        raw_live_target = self._apply_grasp_z_safety(
            observations[-1].pose,
            z_offset_m,
            minimum_safe_z,
            "live pregrasp",
        )
        raw_position_delta, raw_angle_delta = self._pose_delta(target, raw_live_target)
        estimate = self._estimate_pregrasp_target(
            target,
            observations,
            time.time(),
            hover_reached_at_s,
        )
        selected_target = estimate["pose"]
        if not isinstance(selected_target, TcpPose):
            raise RuntimeError("Pregrasp estimator returned an invalid TCP pose")
        position_delta = float(estimate["position_delta_m"])
        angle_delta = float(estimate["angle_delta_deg"])
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
        # Clearance is checked against the already accepted target pose.  A
        # noisy live depth sample must not create a new descent target or make
        # the safety check depend on a different frame.
        if current.z - target.z < min_hover_clearance:
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    f"safe hover clearance is only {(current.z - target.z)*1000:.1f}mm, "
                    f"below required {min_hover_clearance*1000:.1f}mm; "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )
        self.get_logger().info(
            f"Pregrasp target check: age={age_s*1000:.0f}ms, "
            f"position_delta={raw_position_delta*1000:.1f}mm, "
            f"angle_delta={raw_angle_delta:.1f}deg, "
            f"selected={position_delta*1000:.1f}mm/{angle_delta:.1f}deg, "
            f"position_source={estimate['position_mode']}, "
            f"orientation_source={estimate['orientation_mode']}, "
            f"samples={estimate['observation_count']}, "
            f"near_hover={estimate['near_hover_count']}, "
            f"consensus={estimate['consensus_count']}, "
            f"apparent_track={estimate['motion_displacement_m']*1000:.1f}mm "
            "(diagnostic-only)"
        )

        unconfirmed_shift_limit = max(
            position_tolerance,
            float(
                self.get_parameter("pregrasp_unconfirmed_shift_reject_m").value
            ),
        )
        required_consensus = max(
            1,
            int(self.get_parameter("pregrasp_position_consensus_samples").value),
        )
        near_hover_count = int(estimate["near_hover_count"])
        consensus_count = int(estimate["consensus_count"])
        if (
            raw_position_delta > unconfirmed_shift_limit
            and consensus_count < required_consensus
        ):
            if near_hover_count == 0:
                # During the offset-high motion, an eye-in-hand frame can show
                # a 16--25 mm apparent target displacement because camera and
                # robot feedback timestamps are not perfectly aligned.  With
                # no frame from the actual hover interval, that displacement
                # is not evidence that the stationary turntable material
                # moved.  Keep the already accepted stable pose; do not turn a
                # motion-only diagnostic sample into a startup retreat.
                self.get_logger().warning(
                    f"Ignoring motion-phase D405 apparent shift of "
                    f"{raw_position_delta*1000:.1f}mm: no near-hover sample is "
                    "available; keeping the initial stable grasp target"
                )
            else:
                raise RecoverableGraspError(
                    stage="pregrasp target identity",
                    message=(
                        f"D405 target jumped {raw_position_delta*1000:.1f}mm but only "
                        f"{consensus_count}/{required_consensus} near-hover samples "
                        "agree; treating this as target loss or SAM mask drift "
                        "and refusing descent"
                    ),
                    needs_vertical_retreat=False,
                )

        if not bool(self.get_parameter("pregrasp_use_live_pose_for_descent").value):
            if raw_position_delta > max_correction:
                self.get_logger().warning(
                    f"Live pregrasp position differs by {raw_position_delta*1000:.1f}mm; "
                    f"outside the {max_correction*1000:.1f}mm diagnostic range, "
                    "live correction is disabled; protected descent pose remains active"
                )
            return target

        # A 90-degree OBB axis flip is an orientation-quality failure, not a
        # reason to reject a valid translational target.  The estimator already
        # falls back to the initial orientation in that case.
        if position_delta > max_correction:
            raise RecoverableGraspError(
                stage="pregrasp revalidation",
                message=(
                    f"D405 target change is outside safe hover-correction limits: "
                    f"position_delta={position_delta*1000:.1f}mm/{max_correction*1000:.1f}mm, "
                    f"orientation_delta={angle_delta:.1f}deg/{max_correction_angle:.1f}deg; "
                    "will return to startup before descent"
                ),
                needs_vertical_retreat=False,
            )

        if position_delta <= position_tolerance:
            if angle_delta > angle_tolerance:
                # Translation is already within the safe grasp tolerance.  An
                # orientation-only update is not worth a second hover motion;
                # retain the first stable attitude so a 90-degree OBB flip
                # cannot make the TCP descend diagonally into a flat box.
                self.get_logger().warning(
                    f"Ignoring orientation-only live pregrasp change {angle_delta:.1f}deg; "
                    "translation is stable, keeping the initial grasp attitude"
                )
                return target
            # Never hide a small XY adjustment inside the vertical descent.
            # Changes within tolerance use the original stable target exactly.
            return target

        if angle_delta > max_correction_angle:
            self.get_logger().warning(
                f"Ignoring live pregrasp orientation change {angle_delta:.1f}deg; "
                f"safe limit is {max_correction_angle:.1f}deg"
            )
            selected_target = self._keep_reference_orientation(selected_target, target)
            position_delta, _ = self._pose_delta(target, selected_target)

        if not allow_hover_correction:
            # In offset-grasp mode the safe hover is beside the box.  Keep the
            # accepted target after validating identity and correction bounds;
            # moving to a corrected centre hover here would reintroduce the
            # redundant centre waypoint that the offset path is meant to skip.
            self._publish_status(
                "pregrasp target validated at offset-high; keeping the accepted "
                "target for direct offset descent"
            )
            return target

        self._publish_status(
            f"pregrasp target corrected by {position_delta*1000:.1f}mm; "
            f"source={estimate['position_mode']}, "
            "using synchronized target at safe hover"
        )
        correction_hover = TcpPose(
            selected_target.x,
            selected_target.y,
            current.z,
            selected_target.rx,
            selected_target.ry,
            selected_target.rz,
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

        # The correction already moved to the latest measured consensus at a
        # safe Z.  Do not reinterpret frames captured during this correction:
        # they contain the same moving-camera timestamp error that caused the
        # observed 16--25 mm overshoots.  Descend vertically at the measured XY.
        corrected_target = selected_target
        self._publish_status(
            f"pregrasp correction complete; descending with measured target "
            f"({corrected_target.x*1000:.1f},{corrected_target.y*1000:.1f})mm"
        )
        return corrected_target

    def _execute_offset_entry(
        self,
        target: TcpPose,
        length_m: float,
        after_offset_high=None,
    ) -> None:
        """Move image-down, descend beside the box, then insert to its centre."""
        user = int(self.get_parameter("user_index").value)
        tool = int(self.get_parameter("command_tool_index").value)
        current = self._current_command_pose()
        flange = self.controller.current_tcp_pose(
            user_index=user, tool_index=int(self.get_parameter("flange_tool_index").value))
        user_camera = pose_to_transform(flange) @ self.handeye_flange_to_cam
        planned_camera_rotation = camera_rotation_at_target_tcp(
            pose_to_transform(current)[:3, :3],
            user_camera[:3, :3],
            pose_to_transform(target)[:3, :3],
        )
        span = float(self.get_parameter("offset_finger_span_m").value)
        xyz = np.array([target.x, target.y, target.z])
        with self.data_lock:
            rgb_rotation = self.rgb_to_ir_rotation
        if rgb_rotation is None:
            raise RuntimeError("D405 RGB-to-IR calibration missing for image-down approach")
        low_xyz = plan_offset(
            xyz, planned_camera_rotation @ rgb_rotation, length_m, span,
            float(self.get_parameter("offset_grasp_clearance_m").value))
        high_clearance_m = float(self.get_parameter("offset_high_clearance_m").value)
        minimum_hover_clearance_m = max(
            0.0,
            float(self.get_parameter("pregrasp_min_hover_clearance_m").value),
        )
        if not math.isfinite(high_clearance_m) or high_clearance_m < minimum_hover_clearance_m:
            raise RuntimeError(
                f"offset_high_clearance_m={high_clearance_m:.4f}m must be at least "
                f"pregrasp_min_hover_clearance_m={minimum_hover_clearance_m:.4f}m"
            )
        # Never add an unexpected upward leg when a caller starts below the
        # configured high point.  Such a start is accepted only when it still
        # preserves the normal minimum hover clearance.
        high_z = min(float(current.z), float(low_xyz[2]) + high_clearance_m)
        actual_high_clearance_m = high_z - float(low_xyz[2])
        if actual_high_clearance_m < minimum_hover_clearance_m:
            raise RuntimeError(
                f"Current TCP leaves only {actual_high_clearance_m:.4f}m above grasp depth; "
                f"at least {minimum_hover_clearance_m:.4f}m is required before offset descent"
            )
        high_xyz = low_xyz.copy(); high_xyz[2] = high_z
        waypoints = [TcpPose(*v, target.rx, target.ry, target.rz) for v in (high_xyz, low_xyz, xyz)]
        for pose in waypoints:
            self.controller.inverse_kinematics(pose, user_index=user, tool_index=tool,
                                              joint_near=self.controller.current_joint())
        motion = self._motion_profile()
        self._publish_status(
            f"offset plan: low XYZ=({low_xyz[0]:.3f},{low_xyz[1]:.3f},{low_xyz[2]:.3f})m; "
            f"high Z={high_z:.3f}m, clearance={actual_high_clearance_m*1000:.1f}mm; "
            f"offset={np.linalg.norm(low_xyz-xyz)*1000:.1f}mm; "
            f"descent/insert speed={motion['linear_speed']}%, "
            f"accel={motion['linear_acc']}%"
        )

        blend_offset_high_descent = bool(
            self.get_parameter("offset_high_descent_blend_enabled").value
        )
        if blend_offset_high_descent:
            self._publish_status(
                "offset-high to descent continuous path enabled; "
                "descent will be queued before the high waypoint stops"
            )
            self._execute_offset_high_descent_blend(
                waypoints[0],
                waypoints[1],
                motion,
                after_offset_high=after_offset_high,
            )
            self._require_cycle_active("after offset-high/descent path")
            actual = self._current_command_pose()
            if np.linalg.norm(
                np.array(
                    [
                        actual.x - waypoints[1].x,
                        actual.y - waypoints[1].y,
                        actual.z - waypoints[1].z,
                    ],
                    dtype=np.float64,
                )
            ) > 0.003:
                raise RuntimeError("Offset waypoint position not reached: offset_descent")

            self._publish_status("top-barcode low offset observation")
            with self._timed_stage("top_barcode_low_observation"):
                self._wait_for_top_surface_barcode()

            self._require_cycle_active("before offset insert")
            self._publish_status("offset_insert")
            with self._timed_stage("offset_insert"):
                self.controller.move_linear_tcp(
                    waypoints[2],
                    speed=motion["linear_speed"],
                    accel=motion["linear_acc"],
                    user_index=user,
                    tool_index=tool,
                )
            self._require_cycle_active("after offset_insert")
            actual = self._current_command_pose()
            if np.linalg.norm(
                np.array(
                    [
                        actual.x - waypoints[2].x,
                        actual.y - waypoints[2].y,
                        actual.z - waypoints[2].z,
                    ],
                    dtype=np.float64,
                )
            ) > 0.003:
                raise RuntimeError("Offset waypoint position not reached: offset_insert")
            return

        for index, (stage, pose) in enumerate(zip(
                ("offset_high", "offset_descent", "offset_insert"), waypoints)):
            self._require_cycle_active(stage)
            self._publish_status(stage)
            with self._timed_stage(stage):
                if index == 0:
                    # The high waypoint is clear of the table.  One joint PTP
                    # can set the grasp attitude and reach the offset position
                    # together, avoiding the orientation-only stop.
                    self.controller.move_joint_tcp(
                        pose,
                        speed=motion["joint_speed"],
                        accel=motion["joint_pose_acc"],
                        user_index=user,
                        tool_index=tool,
                    )
                else:
                    self.controller.move_linear_tcp(
                        pose,
                        speed=motion["linear_speed"],
                        accel=motion["linear_acc"],
                        user_index=user,
                        tool_index=tool,
                    )
            self._require_cycle_active("after " + stage)
            actual = self._current_command_pose()
            if np.linalg.norm(np.array([actual.x-pose.x, actual.y-pose.y, actual.z-pose.z])) > 0.003:
                raise RuntimeError("Offset waypoint position not reached: " + stage)
            if index == 0 and after_offset_high is not None:
                after_offset_high()
            if index == 1:
                self._publish_status("top-barcode low offset observation")
                with self._timed_stage("top_barcode_low_observation"):
                    self._wait_for_top_surface_barcode()

    def _execute_offset_high_descent_blend(
        self,
        high_pose: TcpPose,
        low_pose: TcpPose,
        motion: dict[str, int],
        after_offset_high=None,
    ) -> bool:
        """Queue the vertical offset descent before the high PTP stops.

        The high waypoint is the first command because it establishes the
        grasp attitude and the lateral offset.  Once the TCP is within the
        configured lead distance, the existing gripper pre-shape and target
        revalidation callback runs while the controller is still completing
        that safe high segment.  A Cartesian ``MovL`` to ``low_pose`` is then
        queued with CP blending.  If the controller cannot expose an active
        queue or rejects ``MovL`` queuing, the helper completes the high move
        and falls back to the original blocking descent.
        """

        lead_m = float(
            self.get_parameter("offset_high_descent_queue_lead_m").value
        )
        if not math.isfinite(lead_m):
            raise RuntimeError(
                "offset_high_descent_queue_lead_m must be finite, "
                f"got {lead_m!r}"
            )
        lead_m = max(0.0, lead_m)
        cp = max(
            1,
            min(
                100,
                int(
                    round(
                        float(
                            self.get_parameter("offset_high_descent_blend_cp").value
                        )
                    )
                ),
            ),
        )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        queue_timeout_s = max(
            0.2,
            float(self.get_parameter("jog_axis_timeout_s").value),
        )
        command_start_grace_s = max(
            0.05,
            min(
                1.0,
                float(
                    self.get_parameter(
                        "offset_high_descent_command_start_grace_s"
                    ).value
                ),
            ),
        )

        self._require_cycle_active("before offset-high motion")
        self._publish_status(
            "offset_high; descent will be queued near the endpoint "
            f"(lead {lead_m * 1000.0:.1f}mm, CP={cp})"
        )
        with self._timed_stage("offset_high"):
            high_command = self.controller.submit_move_joint_tcp(
                high_pose,
                speed=motion["joint_speed"],
                accel=motion["joint_pose_acc"],
                cp=cp,
                user_index=user_index,
                tool_index=tool_index,
            )
            command_submitted_at = time.monotonic()
            gate_deadline = command_submitted_at + queue_timeout_s
            validated = False
            low_command = None
            fallback_reason = ""
            while True:
                self._require_cycle_active("while waiting to queue offset descent")
                try:
                    current_pose = self._current_command_pose()
                except Exception as exc:
                    if time.monotonic() >= gate_deadline:
                        self._stop_active_motion_for_queue_failure(
                            "offset-high descent gate feedback timeout"
                        )
                        raise RuntimeError(
                            "Could not observe TCP pose while waiting to queue "
                            f"offset descent: {exc}"
                        ) from exc
                    time.sleep(LOOKAHEAD_GATE_POLL_S)
                    continue

                distance_to_high_m = float(
                    np.linalg.norm(
                        np.array(
                            [
                                current_pose.x - high_pose.x,
                                current_pose.y - high_pose.y,
                                current_pose.z - high_pose.z,
                            ],
                            dtype=np.float64,
                        )
                    )
                )
                active = self.controller.motion_command_is_active(high_command)
                gate_reached = distance_to_high_m <= lead_m
                if gate_reached and active:
                    if not validated:
                        if after_offset_high is not None:
                            try:
                                after_offset_high()
                            except Exception:
                                # The high PTP is still active while the
                                # callback checks the gripper and target.  A
                                # failed check must stop that motion before
                                # the exception escapes; otherwise an old
                                # trajectory could continue after a retry.
                                self._stop_active_motion_for_queue_failure(
                                    "offset-high validation failure"
                                )
                                raise
                        validated = True
                    # The validation callback can take long enough for the
                    # high command to reach its endpoint.  Recheck activity
                    # before submitting the queued descent.
                    if not self.controller.motion_command_is_active(high_command):
                        fallback_reason = "high waypoint completed during revalidation"
                        break
                    if not hasattr(self.controller, "submit_move_linear_tcp"):
                        fallback_reason = "controller has no asynchronous MovL helper"
                        break
                    try:
                        low_command = self.controller.submit_move_linear_tcp(
                            low_pose,
                            speed=motion["linear_speed"],
                            accel=motion["linear_acc"],
                            cp=cp,
                            user_index=user_index,
                            tool_index=tool_index,
                        )
                    except Exception as exc:
                        if self.controller.motion_command_is_active(high_command):
                            fallback_reason = f"controller rejected queued MovL: {exc}"
                            break
                        raise
                    break

                if not active:
                    if time.monotonic() - command_submitted_at < command_start_grace_s:
                        time.sleep(LOOKAHEAD_GATE_POLL_S)
                        continue
                    fallback_reason = "high waypoint completed before descent queue gate"
                    break
                if time.monotonic() >= gate_deadline:
                    self._stop_active_motion_for_queue_failure(
                        "offset-high descent queue gate timeout"
                    )
                    raise TimeoutError(
                        "Timed out waiting to queue offset descent: "
                        f"distance={distance_to_high_m * 1000.0:.1f}mm, "
                        f"gate={lead_m * 1000.0:.1f}mm"
                    )
                time.sleep(LOOKAHEAD_GATE_POLL_S)

            if low_command is None:
                self.controller.wait_for_command(high_command, timeout_s=5.0)
                if not validated:
                    if after_offset_high is not None:
                        after_offset_high()
                    validated = True
                self._publish_status(
                    "offset-high reached before descent queue; using blocking descent "
                    + (f"({fallback_reason})" if fallback_reason else "")
                )

        self._require_cycle_active("before offset descent")
        if low_command is not None:
            with self._timed_stage("offset_descent"):
                try:
                    self.controller.wait_for_command(low_command, timeout_s=60.0)
                except Exception:
                    self._stop_active_motion_for_queue_failure(
                        "queued offset descent wait"
                    )
                    raise
            self._require_cycle_active("after blended offset-high/descent")
            self._publish_status("blended offset-high and descent reached")
            return True

        with self._timed_stage("offset_descent"):
            self.controller.move_linear_tcp(
                low_pose,
                speed=motion["linear_speed"],
                accel=motion["linear_acc"],
                user_index=user_index,
                tool_index=tool_index,
            )
        self._require_cycle_active("after blocking offset descent")
        return False

    def _execute_one_cycle(
        self,
        target: TcpPose,
        width_m: float,
        height_m: float,
        length_m: float,
        turntable_side_barcode: str = "",
    ) -> None:
        motion = self._motion_profile()
        offset_grasp_enabled = bool(self.get_parameter("offset_grasp_enabled").value)
        side_barcode_preconfirmed = bool(str(turntable_side_barcode).strip())
        # D435 has already classified this material when a side barcode is
        # present.  In that case D405 is needed only for target localization:
        # use a centre hover followed by a vertical descent instead of the
        # offset-high/descent/insert path used for top-surface observation.
        use_offset_grasp_entry = (
            offset_grasp_enabled and not side_barcode_preconfirmed
        )
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
        if (
            bool(self.get_parameter("turntable_enabled").value)
            and bool(self.get_parameter("turntable_height_safety_enabled").value)
        ):
            configured_surface_z_m = float(
                self.get_parameter("turntable_surface_z_m").value
            )
            if configured_surface_z_m < 0.0:
                raise RuntimeError(
                    "turntable_surface_z_m is not configured; measure the User-frame "
                    "turntable top surface and set the V3 launch argument before motion"
                )
            try:
                height_check = validate_turntable_grasp_height(
                    vision_target_z_m=vision_target_z,
                    command_target_z_m=target.z,
                    box_height_m=height_m,
                    configured_surface_z_m=configured_surface_z_m,
                    surface_tolerance_m=float(
                        self.get_parameter("turntable_surface_tolerance_m").value
                    ),
                    tcp_below_target_m=float(
                        self.get_parameter("turntable_tcp_below_target_m").value
                    ),
                    surface_clearance_m=float(
                        self.get_parameter("turntable_surface_clearance_m").value
                    ),
                )
            except ValueError as exc:
                raise RecoverableGraspError(
                    stage="turntable height validation",
                    message=str(exc),
                    needs_vertical_retreat=False,
                ) from exc
            self._publish_status(
                "turntable height validated: "
                f"estimated surface={height_check.estimated_surface_z_m*1000.0:.1f}mm, "
                f"configured={height_check.configured_surface_z_m*1000.0:.1f}mm, "
                f"command TCP Z={height_check.command_tcp_z_m*1000.0:.1f}mm, "
                f"floor={height_check.minimum_command_tcp_z_m*1000.0:.1f}mm"
            )
        self._publish_status(
            f"grasp plan: vision_Z={vision_target_z*1000:.1f}mm, "
            f"Z_correction={z_offset_m*1000:+.1f}mm, command_Z={target.z*1000:.1f}mm, "
            f"box_height={height_m*1000:.1f}mm"
        )
        with self._timed_stage("grasp_prepare"):
            if use_offset_grasp_entry:
                self._publish_status(
                    "commanding measured gripper width; pre-shaping during direct offset approach"
                )
            elif side_barcode_preconfirmed:
                self._publish_status(
                    "D435 side barcode already confirmed; pre-shaping while moving "
                    "directly above the D405 grasp centre"
                )
            else:
                self._publish_status(
                    "commanding measured gripper width; pre-shaping during move-above"
                )
            max_opening = float(self.get_parameter("dh_max_opening_m").value)
            width_m = max(0.0, min(max_opening, width_m))
            pre_shape_position = width_m / max_opening
            pre_shape_initial = self.gripper.read_position()
            # 夹爪预张开和机械臂接近目标互不冲突。先下发非阻塞命令，
            # 到达中心上方或偏置上方后再确认夹爪已经停止。
            self.gripper.set_position(pre_shape_position, wait=False)
            self._require_cycle_active("after commanding gripper pre-shape")

            self.controller.inverse_kinematics(
                target,
                user_index=int(self.get_parameter("user_index").value),
                tool_index=int(self.get_parameter("command_tool_index").value),
                joint_near=self.controller.current_joint(),
            )
            planar = None
            if not use_offset_grasp_entry:
                current = self._current_command_pose()
                hover_z = current.z
                if side_barcode_preconfirmed:
                    direct_clearance_m = float(
                        self.get_parameter(
                            "side_barcode_direct_hover_clearance_m"
                        ).value
                    )
                    minimum_hover_clearance_m = max(
                        0.005,
                        float(
                            self.get_parameter(
                                "pregrasp_min_hover_clearance_m"
                            ).value
                        ),
                    )
                    if (
                        not math.isfinite(direct_clearance_m)
                        or direct_clearance_m < minimum_hover_clearance_m
                    ):
                        raise RuntimeError(
                            "side_barcode_direct_hover_clearance_m="
                            f"{direct_clearance_m:.4f}m must be at least "
                            "pregrasp_min_hover_clearance_m="
                            f"{minimum_hover_clearance_m:.4f}m"
                        )
                    # Do not introduce an upward leg if this branch is entered
                    # from a TCP already below the configured hover height.
                    hover_z = min(current.z, target.z + direct_clearance_m)
                    actual_clearance_m = hover_z - target.z
                    if actual_clearance_m < minimum_hover_clearance_m:
                        raise RuntimeError(
                            "current TCP permits only "
                            f"{actual_clearance_m*1000.0:.1f}mm clearance above "
                            "the side-barcode grasp target; at least "
                            f"{minimum_hover_clearance_m*1000.0:.1f}mm is required"
                        )
                    self._publish_status(
                        "side-barcode direct approach: diagonal PTP from current "
                        f"pose to target XY at Z={hover_z*1000.0:.1f}mm "
                        f"({actual_clearance_m*1000.0:.1f}mm above grasp depth), "
                        "then vertical MovL descent"
                    )
                planar = TcpPose(
                    target.x, target.y, hover_z, target.rx, target.ry, target.rz
                )
        # The point-cloud handoff check is not enough when 102's TCP is not
        # visible in the D405 cloud.  Check the actual 102 feedback immediately
        # before issuing 101's approach command.  A blocked check leaves 101
        # at startup; once 102 retreats past the threshold, this same cycle
        # continues without manual intervention.
        with self._timed_stage("secondary_y_clearance"):
            self._wait_for_secondary_y_clearance(
                "offset-high" if use_offset_grasp_entry else "move-above"
            )
        self._require_cycle_active(
            "immediately before offset-high"
            if use_offset_grasp_entry
            else "immediately before move-above"
        )
        # A confirmed D435 side barcode already has priority over the D405
        # top barcode.  Skip the redundant detector and its low-pose wait for
        # this material, while keeping D405 target localization unchanged.
        observe_top_barcode = not side_barcode_preconfirmed
        self._set_top_surface_barcode_window(observe_top_barcode)
        with self.data_lock:
            pregrasp_reference_count = self.pregrasp_pose_count

        def confirm_preshape_and_revalidate(allow_hover_correction: bool) -> TcpPose:
            # Keep the same host-wall-clock domain as the D405 frame stamps.
            # The offset-high waypoint replaces the old centre hover when the
            # offset path is enabled, so its arrival time anchors revalidation.
            hover_reached_at_s = time.time()
            with self._timed_stage("gripper_preshape_wait"):
                self.gripper.wait_until_stopped(
                    timeout_s=float(self.get_parameter("dh_timeout_s").value),
                    target_position=pre_shape_position,
                    initial_position=pre_shape_initial,
                    cancel_check=self._cycle_cancel_requested,
                )
            self._require_cycle_active("after confirming gripper pre-shape")
            with self._timed_stage("pregrasp_revalidation"):
                return self._revalidate_target_at_hover(
                    target,
                    motion,
                    z_offset_m,
                    minimum_safe_z,
                    pregrasp_reference_count,
                    hover_reached_at_s,
                    allow_hover_correction=allow_hover_correction,
                )

        if use_offset_grasp_entry:
            self._publish_status(
                "moving to offset-high while aligning grasp orientation; "
                "center translation remains skipped"
            )

            def validate_at_offset_high() -> None:
                confirm_preshape_and_revalidate(allow_hover_correction=False)

            self._execute_offset_entry(
                target,
                length_m,
                after_offset_high=validate_at_offset_high,
            )
        else:
            if side_barcode_preconfirmed:
                self._publish_status(
                    "D435 side barcode already confirmed; skipping offset-high, "
                    "offset descent, offset insert and top-surface observation; "
                    "moving directly above the D405 grasp centre"
                )
            else:
                self._publish_status("moving above selected box")
            with self._timed_stage("move_above"):
                self.controller.move_joint_tcp(
                    planar,
                    speed=motion["joint_speed"],
                    accel=motion["joint_pose_acc"],
                    user_index=int(self.get_parameter("user_index").value),
                    tool_index=int(self.get_parameter("command_tool_index").value),
                )
            self._require_cycle_active("after move-above")
            target = confirm_preshape_and_revalidate(allow_hover_correction=True)
            if observe_top_barcode:
                self._publish_status("top-barcode hover observation")
                with self._timed_stage("top_barcode_hover_observation"):
                    self._wait_for_top_surface_barcode()
                self._publish_status("descending TCP tip to 75% box height")
            else:
                self._publish_status(
                    "descending vertically to the D405 grasp centre for the "
                    "preconfirmed side-barcode material"
                )
            self._require_cycle_active("immediately before grasp descent")
            with self._timed_stage("grasp_descend"):
                self.controller.move_linear_tcp(
                    target,
                    speed=motion["linear_speed"],
                    accel=motion["linear_acc"],
                    user_index=int(self.get_parameter("user_index").value),
                    tool_index=int(self.get_parameter("command_tool_index").value),
                )
        self._require_cycle_active("at grasp depth")
        # Keep the D405 top-surface ROI detector armed through the descent. At
        # the distant hover pose a portrait label can be only a few pixels wide;
        # the closer grasp-depth frames provide the resolution needed by YOLO.
        top_surface_barcode = (
            self._current_top_surface_barcode() if observe_top_barcode else ""
        )
        self._set_top_surface_barcode_window(False)
        barcode_face = classify_barcode_face(
            turntable_side_barcode,
            top_surface_barcode,
        )
        if observe_top_barcode and not top_surface_barcode and bool(
            self.get_parameter("top_surface_barcode_enabled").value
        ):
            self._publish_status(
                "no top-surface barcode confirmed during approach/descent in YOLO/SAM target region; "
                "continuing bottom-face classification after the grasp"
            )
        actual_grasp_pose = self._current_command_pose()
        grasp_position_error_m = math.sqrt(
            (actual_grasp_pose.x - target.x) ** 2
            + (actual_grasp_pose.y - target.y) ** 2
            + (actual_grasp_pose.z - target.z) ** 2
        )
        turntable_enabled = bool(self.get_parameter("turntable_enabled").value)
        if turntable_enabled and barcode_face == "bottom":
            if not bool(self.get_parameter("bottom_barcode_recovery_enabled").value):
                raise RuntimeError(
                    "D435 found no side barcode and D405 found no top barcode, "
                    "but bottom-barcode recovery is disabled"
                )
            self._publish_status(
                "no side or top barcode; starting the staged pick and table-flip "
                "sequence at the D405 grasp centre"
            )
            self._execute_turntable_bottom_center_recovery(
                actual_grasp_pose,
                pre_shape_position,
                width_m,
                max_opening,
                motion,
            )
            self._mark_turntable_material_removed()
            self._place_as_top_barcode_box()
            return
        self._publish_status(
            "at grasp depth; commanding V2-compatible full close: "
            f"pre-shape={width_m*1000.0:.1f}mm, target=0.0mm; "
            f"TCP=({actual_grasp_pose.x*1000.0:.1f},"
            f"{actual_grasp_pose.y*1000.0:.1f},"
            f"{actual_grasp_pose.z*1000.0:.1f})mm, "
            f"position_error={grasp_position_error_m*1000.0:.1f}mm"
        )
        with self._timed_stage("gripper_close"):
            self.gripper.set_force(int(self.get_parameter("dh_grasp_force").value))
            self.gripper.close(
                wait=True,
                cancel_check=self._cycle_cancel_requested,
            )
        self._require_cycle_active("after gripper close")
        with self._timed_stage("grasp_confirm"):
            self._confirm_grasp_before_lift(max_opening, width_m)

        transfer_precompleted = False
        blend_lift_transfer = (
            not turntable_enabled
            and barcode_face != "top"
            and bool(self.get_parameter("grasp_lift_transfer_blend_enabled").value)
        )
        face_snap_precompleted = False
        side_face_snap = (
            turntable_enabled
            and barcode_face == "side"
            and bool(self.get_parameter("barcode_snap_to_nearest_face").value)
        )
        if turntable_enabled:
            if side_face_snap:
                with self._timed_stage("turntable_lift_face_snap"):
                    face_snap_precompleted = self._execute_turntable_lift_face_snap(motion)
            else:
                with self._timed_stage("turntable_safe_departure_lift"):
                    self._execute_turntable_safe_departure_lift(motion)
        elif blend_lift_transfer:
            with self._timed_stage("grasp_lift_transfer"):
                transfer_precompleted = self._execute_grasp_lift_transfer(
                    motion,
                    max_opening,
                )
        else:
            with self._timed_stage("grasp_lift"):
                self._require_cycle_active("immediately before grasp lift")
                self._relative_user_move(
                    z=float(self.get_parameter("grasp_lift_m").value),
                    label="lifting grasp",
                    speed_factor=motion["grasp_lift_speed"],
                    accel_factor=motion["grasp_lift_acc"],
                )
        self._require_cycle_active("after grasp lift")
        with self._timed_stage("post_lift_grasp_check"):
            self._validate_grasp_feedback(
                (
                    "at turntable safe departure height"
                    if turntable_enabled
                    else "after lift-transfer"
                    if transfer_precompleted
                    else "after lift"
                ),
                max_opening,
            )
        if side_face_snap:
            if not face_snap_precompleted:
                with self._timed_stage("d435_side_nearest_face_snap"):
                    self._snap_turntable_side_to_nearest_face(motion)
                with self._timed_stage("post_face_snap_grasp_check"):
                    self._validate_grasp_feedback(
                        "after D435 side-barcode nearest-90deg J6 alignment",
                        max_opening,
                    )
        # At this point the gripper has retained state=GRIPPED through the
        # complete safe departure (and any non-turntable transfer), and the
        # box is physically clear of the turntable. Re-arm the next place
        # event here instead of waiting for
        # placement/return-startup to finish. A later placement fault or an
        # operator Stop must not leave the old D435-ready latch blocking the
        # next material event.
        self._mark_turntable_material_removed()

        if barcode_face == "top":
            self._publish_status(
                f"top-surface barcode {top_surface_barcode!r} detected; "
                "skipping scanner approach, J6 rotation, Ry/Rz flip and dynamic Z descent"
            )
            self._place_as_top_barcode_box()
            return

        if barcode_face == "side":
            self._publish_status(
                f"D435 side barcode {turntable_side_barcode!r} has priority; "
                "skipping the legacy fixed scanner and J6 face search"
            )
        else:
            self._publish_status(
                "D435 found no side barcode and D405 found no top barcode; "
                "classifying this material as bottom-barcode"
            )

        # A D435-confirmed side barcode no longer visits the legacy transfer
        # joint.  The TCP has already departed vertically to the absolute safe
        # height and can proceed to the equally high placement-area waypoint.
        # Bottom recovery retains its dedicated transfer staging sequence.
        if barcode_face == "side" and turntable_enabled:
            self._publish_status(
                "straight turntable departure reached; skipping the legacy "
                "transfer joint and proceeding at safe height"
            )
        elif transfer_precompleted:
            self._publish_status(
                "at barcode transfer joint; lift/transfer path already completed"
            )
        else:
            self._publish_status("moving to barcode transfer joint")
            self._require_cycle_active("immediately before barcode transfer motion")
            with self._timed_stage("move_transfer"):
                self.controller.move_joint(
                    self._six_values("transfer_joint"),
                    speed=motion["transfer_speed"],
                    accel=motion["transfer_acc"],
                )
        self._require_cycle_active("after safe post-lift staging")
        self._reset_barcode_search_travel()

        if barcode_face == "bottom":
            if not bool(
                self.get_parameter("bottom_barcode_recovery_enabled").value
            ):
                raise RuntimeError(
                    "D435 found no side barcode and D405 found no top barcode, "
                    "but bottom-barcode recovery is disabled"
                )
            # Reproduce the old bottom-recovery staging location without ever
            # approaching the legacy scanner: its zero-approach retreat adds
            # only the already-tested extra User-X clearance.
            with self._timed_stage("scanner_retreat"):
                self._retreat_box_from_scanner(0.0)
            self._require_cycle_active("after bottom-recovery staging retreat")
            with self._timed_stage("bottom_barcode_recovery"):
                self._execute_bottom_barcode_recovery(
                    target,
                    pre_shape_position,
                    height_m,
                )
            return

        # First move above the placement area at the previous safe height while
        # applying the barcode-up orientation.  The side-barcode branch then
        # combines its fixed XYZ and final User-Rx tilt in one PTP; it does not
        # descend to the length-based low Z, which can put J4/J5 into the table.
        approach_xyz = [
            float(value)
            for value in self.get_parameter("scan_exit_user_xyz").value
        ]
        fixed_place_xyz = self._side_barcode_place_xyz()
        side_rx_delta_deg = float(
            self.get_parameter("side_barcode_place_rx_delta_deg").value
        )
        post_scan_place_precompleted = False
        # The box never approached the legacy scanner, so there is no scanner
        # collision envelope to retreat from.  Start the unchanged barcode-up
        # placement orientation directly from the tested transfer joint.
        if bool(self.get_parameter("post_scan_place_blend_enabled").value):
            with self._timed_stage("post_scan_safe_height_place"):
                post_scan_place_precompleted = self._execute_post_scan_safe_place_blend(
                    approach_xyz,
                    fixed_place_xyz,
                    side_rx_delta_deg,
                    motion,
                )
        else:
            with self._timed_stage("post_scan_safe_height_ptp"):
                self._move_to_user_xyz_with_rotation(
                    approach_xyz,
                    ry_delta_deg=float(self.get_parameter("face_up_user_ry_deg").value),
                    rz_delta_deg=float(self.get_parameter("post_scan_user_rz_deg").value),
                )

        self._require_cycle_active(
            "at fixed placement pose"
            if post_scan_place_precompleted
            else "at safe height above placement area"
        )
        if not post_scan_place_precompleted:
            with self._timed_stage("side_barcode_fixed_place_ptp"):
                self._move_to_user_xyz_with_rotation(
                    fixed_place_xyz,
                    ry_delta_deg=0.0,
                    rz_delta_deg=0.0,
                    rx_delta_deg=side_rx_delta_deg,
                )
        self._require_cycle_active(
            "at side-barcode fixed placement pose with User Rx tilt"
        )
        with self._timed_stage("placement_grasp_check"):
            self._validate_grasp_feedback(
                "at side-barcode fixed placement pose",
                float(self.get_parameter("dh_max_opening_m").value),
            )
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
                f"releasing box at side-barcode fixed placement pose: opening "
                f"{current_opening*1000:.1f}->{release_opening*1000:.1f}mm "
                f"(clearance +{actual_clearance*1000:.1f}mm)"
            )
            self.gripper.set_position(release_position, wait=False)
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=release_position,
                initial_position=current_position,
                cancel_check=self._cycle_cancel_requested,
            )
        self._publish_status(
            "placed side-barcode box at fixed XYZ with User Rx tilt; "
            "returning to startup"
        )

    @staticmethod
    def _signed_user_y_delta_deg(before: TcpPose, after: TcpPose) -> float:
        """Return the signed User-Y component of an orientation change."""

        before_rotation = SciPyRot.from_euler(
            "xyz", [before.rx, before.ry, before.rz], degrees=True
        ).as_matrix()
        after_rotation = SciPyRot.from_euler(
            "xyz", [after.rx, after.ry, after.rz], degrees=True
        ).as_matrix()
        relative_rotation = after_rotation @ before_rotation.T
        return float(
            math.degrees(
                math.atan2(relative_rotation[0, 2], relative_rotation[0, 0])
            )
        )

    def _return_j6_before_bottom_recovery(self) -> float:
        """Return J6 by one face before retreating from the scanner."""

        return_delta = float(
            self.get_parameter("bottom_flip_j6_pre_return_deg").value
        )
        if return_delta <= 0.0:
            raise RuntimeError(
                "bottom_flip_j6_pre_return_deg must be positive, "
                f"got {return_delta:+.1f}deg"
            )
        net_before = float(self.barcode_search_net_delta_deg)
        if net_before >= -0.2:
            self._publish_status(
                "bottom-barcode recovery: no negative J6 search travel to partially return"
            )
            return 0.0
        if return_delta > abs(net_before) + 0.2:
            raise RuntimeError(
                f"J6 pre-return {return_delta:+.1f}deg exceeds failed search travel "
                f"{net_before:+.1f}deg"
            )

        watch_index = max(
            0,
            min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)),
        )
        before_joints = self.controller.current_joint()
        if len(before_joints) != 6:
            raise RuntimeError(
                "Current joint feedback must contain 6 values before bottom-barcode "
                f"J6 pre-return, got {len(before_joints)}"
            )
        if not self._is_barcode_flip_joint_safe(before_joints, return_delta):
            raise RuntimeError(
                f"Unsafe J6 pre-return before bottom recovery: "
                f"current={before_joints[watch_index]:.1f}deg, "
                f"delta={return_delta:+.1f}deg"
            )
        target_joints = [float(value) for value in before_joints]
        target_joints[watch_index] += return_delta
        motion = self._motion_profile()
        self._publish_status(
            f"bottom-barcode recovery: returning J{watch_index + 1} one face "
            f"{return_delta:+.1f}deg before scanner retreat; "
            f"net {net_before:+.1f}->{net_before + return_delta:+.1f}deg"
        )
        self._require_cycle_active("before bottom-barcode J6 pre-return")
        self.controller.move_joint(
            target_joints,
            speed=motion["barcode_alignment_speed"],
            accel=motion["barcode_alignment_acc"],
        )
        self._require_cycle_active("after bottom-barcode J6 pre-return")
        after_joints = self.controller.current_joint()
        if len(after_joints) != 6:
            raise RuntimeError(
                "Current joint feedback must contain 6 values after bottom-barcode "
                f"J6 pre-return, got {len(after_joints)}"
            )
        actual_delta = float(after_joints[watch_index]) - float(
            before_joints[watch_index]
        )
        tolerance_deg = max(
            3.0,
            abs(float(self.get_parameter("barcode_flip_jog_tolerance_deg").value))
            * 2.0,
        )
        if abs(actual_delta - return_delta) > tolerance_deg:
            raise RuntimeError(
                f"Bottom-barcode J6 pre-return mismatch: "
                f"requested={return_delta:+.1f}deg, actual={actual_delta:+.1f}deg"
            )
        self._record_barcode_search_travel(return_delta)
        self._publish_status(
            f"bottom-barcode recovery: J{watch_index + 1} pre-return completed; "
            f"net search travel={self.barcode_search_net_delta_deg:+.1f}deg"
        )
        return actual_delta

    def _reverse_barcode_search_j6(self) -> float:
        """Undo the exact J6 travel used by this cycle's unsuccessful search."""

        reverse_delta = -float(self.barcode_search_net_delta_deg)
        if abs(reverse_delta) <= 0.2:
            self._publish_status("bottom-barcode recovery: no J6 search travel to reverse")
            return 0.0

        watch_index = max(
            0,
            min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)),
        )
        before_joints = self.controller.current_joint()
        if len(before_joints) != 6:
            raise RuntimeError(
                f"Current joint feedback must contain 6 values before J6 recovery, "
                f"got {len(before_joints)}"
            )
        if not self._is_barcode_flip_joint_safe(before_joints, reverse_delta):
            raise RuntimeError(
                f"Unsafe reverse J6 travel during bottom-barcode recovery: "
                f"current={before_joints[watch_index]:.1f}deg, "
                f"delta={reverse_delta:+.1f}deg"
            )
        target_joints = [float(value) for value in before_joints]
        target_joints[watch_index] += reverse_delta
        motion = self._motion_profile()
        self._publish_status(
            f"bottom-barcode recovery: reversing J{watch_index + 1} search travel "
            f"{reverse_delta:+.1f}deg (net search={self.barcode_search_net_delta_deg:+.1f}deg)"
        )
        self._require_cycle_active("before reverse barcode J6 travel")
        self.controller.move_joint(
            target_joints,
            speed=motion["barcode_alignment_speed"],
            accel=motion["barcode_alignment_acc"],
        )
        self._require_cycle_active("after reverse barcode J6 travel")
        after_joints = self.controller.current_joint()
        if len(after_joints) != 6:
            raise RuntimeError(
                f"Current joint feedback must contain 6 values after J6 recovery, "
                f"got {len(after_joints)}"
            )
        actual_delta = float(after_joints[watch_index]) - float(before_joints[watch_index])
        tolerance_deg = max(
            3.0,
            abs(float(self.get_parameter("barcode_flip_jog_tolerance_deg").value)) * 2.0,
        )
        if abs(actual_delta - reverse_delta) > tolerance_deg:
            raise RuntimeError(
                f"Reverse J6 travel mismatch: requested={reverse_delta:+.1f}deg, "
                f"actual={actual_delta:+.1f}deg"
            )
        self._publish_status(
            f"bottom-barcode recovery: J{watch_index + 1} returned "
            f"{actual_delta:+.1f}deg"
        )
        return actual_delta

    def _place_as_top_barcode_box(self) -> None:
        """Use the same fixed XYZ and release checks for either top-facing path."""

        with self._timed_stage("top_barcode_fixed_place_ptp"):
            self._move_to_user_xyz_with_rotation(
                self._top_surface_barcode_place_xyz(),
                ry_delta_deg=0.0,
                rz_delta_deg=0.0,
            )
        self._require_cycle_active("at top-barcode fixed placement pose")
        with self._timed_stage("top_barcode_placement_grasp_check"):
            self._validate_grasp_feedback(
                "at top-barcode fixed placement pose",
                float(self.get_parameter("dh_max_opening_m").value),
            )
        with self._timed_stage("top_barcode_gripper_open_place"):
            max_opening = float(self.get_parameter("dh_max_opening_m").value)
            release_clearance = max(
                0.0,
                float(self.get_parameter("place_release_clearance_m").value),
            )
            current_position = self.gripper.read_position()
            current_opening = current_position * max_opening
            release_opening = min(max_opening, current_opening + release_clearance)
            release_position = release_opening / max_opening
            self._publish_status(
                "releasing top-facing box at fixed XYZ placement pose: opening "
                f"{current_opening*1000:.1f}->{release_opening*1000:.1f}mm"
            )
            self.gripper.set_position(release_position, wait=False)
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=release_position,
                initial_position=current_position,
                cancel_check=self._cycle_cancel_requested,
            )
        self._publish_status("top-facing box placed at fixed XYZ; returning to startup")

    def _bottom_center_linear_z(
        self,
        target_z: float,
        motion: dict[str, int],
        label: str,
        *,
        lifting: bool,
        target_xy: tuple[float, float] | None = None,
    ) -> None:
        """Move to a User Z target, normally preserving current feedback X/Y."""

        current = self._current_command_pose()
        tolerance_m = max(0.001, float(self.get_parameter("jog_tolerance_m").value))
        if not math.isfinite(target_z):
            raise RuntimeError(f"{label}: target Z is not finite")
        target_x, target_y = (
            (current.x, current.y)
            if target_xy is None
            else (float(target_xy[0]), float(target_xy[1]))
        )
        if not all(math.isfinite(value) for value in (target_x, target_y)):
            raise RuntimeError(f"{label}: target X/Y is not finite")
        minimum_z = float(self.get_parameter("minimum_safe_tcp_z_m").value)
        if target_z < minimum_z - 1e-6:
            raise RuntimeError(
                f"{label}: target Z={target_z*1000:.1f}mm is below the configured "
                f"minimum TCP Z={minimum_z*1000:.1f}mm"
            )
        target = TcpPose(
            target_x, target_y, target_z,
            current.rx, current.ry, current.rz,
        )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        self.controller.inverse_kinematics(
            target,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=self.controller.current_joint(),
        )
        xy_text = (
            "preserving current XY"
            if target_xy is None
            else "using latest D405 tracked target XY"
        )
        self._publish_status(
            f"{label}: {xy_text}=({target_x*1000:.1f},{target_y*1000:.1f})mm, "
            f"TCP Z {current.z*1000:.1f}->{target_z*1000:.1f}mm"
        )
        self._require_cycle_active(f"before {label}")
        self.controller.move_linear_tcp(
            target,
            speed=motion["grasp_lift_speed" if lifting else "linear_speed"],
            accel=motion["grasp_lift_acc" if lifting else "linear_acc"],
            user_index=user_index,
            tool_index=tool_index,
        )
        self._require_cycle_active(f"after {label}")
        final = self._current_command_pose()
        xyz_error = max(
            abs(final.x - target.x),
            abs(final.y - target.y),
            abs(final.z - target_z),
        )
        if xyz_error > tolerance_m * 1.5:
            raise RuntimeError(
                f"{label}: final TCP XYZ error={xyz_error*1000:.1f}mm"
            )

    def _bottom_center_tool_rx(self, delta_deg: float, label: str) -> None:
        """Rotate around the current Tool X axis at a fixed TCP XYZ using MovL."""

        current = self._current_command_pose()
        start_rotation = SciPyRot.from_euler(
            "xyz",
            [current.rx, current.ry, current.rz],
            degrees=True,
        )
        # A tool-frame rotation is intrinsic/local, so it post-multiplies the
        # current Tool orientation. User-axis rotations elsewhere pre-multiply.
        target_rotation = start_rotation * SciPyRot.from_euler(
            "x", float(delta_deg), degrees=True
        )
        target_rx, target_ry, target_rz = target_rotation.as_euler(
            "xyz", degrees=True
        )
        target = TcpPose(
            current.x,
            current.y,
            current.z,
            float(target_rx),
            float(target_ry),
            float(target_rz),
        )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        motion = self._motion_profile()
        self.controller.inverse_kinematics(
            target,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=self.controller.current_joint(),
        )
        self._publish_status(
            f"{label}: fixed XYZ=({current.x*1000:.1f},{current.y*1000:.1f},"
            f"{current.z*1000:.1f})mm, Tool Rx={delta_deg:+.1f}deg, "
            f"target RPY=({target.rx:.1f},{target.ry:.1f},{target.rz:.1f})deg"
        )
        self._require_cycle_active(f"before {label}")
        self.controller.move_linear_tcp(
            target,
            speed=motion["post_scan_speed"],
            accel=motion["post_scan_acc"],
            user_index=user_index,
            tool_index=tool_index,
        )
        self._require_cycle_active(f"after {label}")
        final = self._current_command_pose()
        tolerance_m = max(0.0005, float(self.get_parameter("jog_tolerance_m").value))
        xyz_error = max(
            abs(final.x - target.x),
            abs(final.y - target.y),
            abs(final.z - target.z),
        )
        final_rotation = SciPyRot.from_euler(
            "xyz", [final.rx, final.ry, final.rz], degrees=True
        )
        rotation_error_deg = math.degrees(
            (target_rotation.inv() * final_rotation).magnitude()
        )
        if xyz_error > tolerance_m * 1.5 or rotation_error_deg > 3.0:
            raise RuntimeError(
                f"{label}: final XYZ error={xyz_error*1000:.1f}mm, "
                f"orientation error={rotation_error_deg:.1f}deg"
            )

    def _wait_for_bottom_center_tracked_pose(self) -> TcpPose:
        """Wait for one fresh D405 target published after the second Tool-Rx."""

        timeout_s = max(
            0.1,
            float(self.get_parameter("bottom_center_tracking_timeout_s").value),
        )
        max_age_s = max(
            0.05,
            float(self.get_parameter("pregrasp_pose_max_age_s").value),
        )
        with self.data_lock:
            previous_count = self.pregrasp_pose_count
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._require_cycle_active(
                "waiting for latest D405 target before final bottom regrasp"
            )
            with self.data_lock:
                current_count = self.pregrasp_pose_count
                pose = self.latest_pregrasp_pose
                received_at = self.latest_pregrasp_pose_received_at
            age_s = time.monotonic() - received_at
            if (
                current_count > previous_count
                and pose is not None
                and age_s <= max_age_s
            ):
                current = self._current_command_pose()
                xy_shift_m = math.hypot(pose.x - current.x, pose.y - current.y)
                max_shift_m = max(
                    0.001,
                    float(self.get_parameter("pregrasp_max_correction_m").value),
                )
                if xy_shift_m > max_shift_m:
                    raise RuntimeError(
                        "Latest D405 bottom-regrasp target is too far from the current "
                        f"TCP: shift={xy_shift_m*1000:.1f}mm, "
                        f"limit={max_shift_m*1000:.1f}mm"
                    )
                self._publish_status(
                    "fresh D405 target accepted for final bottom regrasp: "
                    f"XY=({pose.x*1000:.1f},{pose.y*1000:.1f})mm, "
                    f"age={age_s*1000:.0f}ms, shift={xy_shift_m*1000:.1f}mm"
                )
                return pose
            time.sleep(0.01)
        raise RuntimeError(
            "No fresh D405 tracked target arrived after the second Tool-Rx; "
            "final bottom regrasp descent was not commanded"
        )

    def _snap_bottom_center_to_nearest_face(
        self,
        motion: dict[str, int],
    ) -> None:
        """Snap J6 to the nearest 90-degree grid after a short table-clear lift."""

        watch_index = max(
            0,
            min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)),
        )
        joints = [float(value) for value in self.controller.current_joint()]
        if len(joints) != 6:
            raise RuntimeError("Bottom recovery face snap requires 6 joint values")
        current_deg = joints[watch_index]
        target_deg = nearest_face_anchor_deg(
            current_deg,
            float(self.get_parameter("d435_side_face_reference_joint_deg").value),
            float(self.get_parameter("barcode_flip_step_deg").value),
            abs(float(self.get_parameter("barcode_flip_safe_joint_limit_deg").value)),
        )
        correction_deg = target_deg - current_deg
        if abs(correction_deg) <= 0.2:
            self._publish_status(
                f"bottom recovery J6 already at nearest 90deg face: {current_deg:.1f}deg"
            )
            return
        if not self._is_barcode_flip_joint_safe(joints, correction_deg):
            raise RuntimeError(
                "Bottom recovery nearest 90deg face exceeds the configured joint limit"
            )
        target_joints = list(joints)
        target_joints[watch_index] = target_deg
        self._publish_status(
            f"bottom recovery: snapping J{watch_index + 1} to nearest 90deg face "
            f"{current_deg:.1f}->{target_deg:.1f}deg"
        )
        self._require_cycle_active("before bottom recovery nearest-face snap")
        self.controller.move_joint(
            target_joints,
            speed=motion["barcode_alignment_speed"],
            accel=motion["barcode_alignment_acc"],
        )
        self._require_cycle_active("after bottom recovery nearest-face snap")
        final_joints = self.controller.current_joint()
        tolerance_deg = max(
            1.0,
            abs(float(self.get_parameter("barcode_flip_jog_tolerance_deg").value)),
        )
        if (
            len(final_joints) != 6
            or abs(float(final_joints[watch_index]) - target_deg) > tolerance_deg
        ):
            raise RuntimeError("Bottom recovery did not reach the nearest 90deg face")

    def _execute_turntable_bottom_center_recovery(
        self,
        grasp_pose: TcpPose,
        pre_shape_position: float,
        width_m: float,
        max_opening: float,
        motion: dict[str, int],
    ) -> None:
        """Reorient a presumed bottom label through two table regrasp stages."""

        first_rz_deg = float(
            self.get_parameter("bottom_center_first_rz_delta_deg").value
        )
        first_tool_rx_deg = float(
            self.get_parameter("bottom_center_first_tool_rx_delta_deg").value
        )
        release_tool_rx_deg = float(
            self.get_parameter("bottom_center_release_tool_rx_delta_deg").value
        )
        initial_lift_m = float(self.get_parameter("grasp_lift_m").value)
        lift_m = float(self.get_parameter("bottom_flip_lift_m").value)
        half_turn_deg = float(self.get_parameter("bottom_flip_j6_half_turn_deg").value)
        descent_m = float(self.get_parameter("bottom_flip_post_turn_descent_m").value)
        if abs(first_rz_deg - 40.0) > 1e-6:
            raise RuntimeError(
                "V3 bottom recovery requires User Rz=+40deg before the first release"
            )
        if abs(first_tool_rx_deg - 70.0) > 1e-6:
            raise RuntimeError(
                "V3 bottom recovery requires Tool Rx=+70deg after the first release"
            )
        if abs(release_tool_rx_deg - 70.0) > 1e-6:
            raise RuntimeError(
                "V3 bottom recovery requires Tool Rx=+70deg after the second release"
            )
        if not (0.0 < initial_lift_m <= 0.200):
            raise RuntimeError("grasp_lift_m must be in (0, 0.200]m")
        if not (0.0 < lift_m <= 0.300):
            raise RuntimeError("bottom_flip_lift_m must be in (0, 0.300]m")
        if abs(half_turn_deg - 180.0) > 1e-6:
            raise RuntimeError("V3 in-place bottom recovery requires J6 +180deg")
        if not (0.0 < descent_m < lift_m):
            raise RuntimeError(
                "bottom_flip_post_turn_descent_m must be positive and less "
                "than bottom_flip_lift_m"
            )
        if not (0.0 < pre_shape_position <= 1.0 and max_opening > 0.0):
            raise RuntimeError("invalid gripper pre-shape for bottom recovery")

        def open_to_preshape() -> None:
            current_position = self.gripper.read_position()
            self.gripper.set_position(pre_shape_position, wait=False)
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=pre_shape_position,
                initial_position=current_position,
                cancel_check=self._cycle_cancel_requested,
            )

        # First pick: clear the turntable only by the normal 60 mm grasp lift.
        with self._timed_stage("bottom_center_initial_close"):
            self.gripper.set_force(int(self.get_parameter("dh_grasp_force").value))
            self.gripper.close(wait=True, cancel_check=self._cycle_cancel_requested)
        with self._timed_stage("bottom_center_initial_grasp_confirm"):
            self._confirm_grasp_before_lift(max_opening, width_m)
        with self._timed_stage("bottom_center_initial_table_clear_lift"):
            self._bottom_center_linear_z(
                grasp_pose.z + initial_lift_m,
                motion,
                "bottom recovery initial table-clear lift",
                lifting=True,
            )
        with self._timed_stage("bottom_center_nearest_90_face_snap"):
            self._validate_grasp_feedback("before nearest-90deg face snap", max_opening)
            self._snap_bottom_center_to_nearest_face(motion)
            self._validate_grasp_feedback("after nearest-90deg face snap", max_opening)

        # Apply Rz +40 while clear, return to the original grasp Z, and release.
        with self._timed_stage("bottom_center_rz_plus_40"):
            current = self._current_command_pose()
            self._move_to_user_xyz_with_rotation(
                [current.x, current.y, current.z],
                ry_delta_deg=0.0,
                rz_delta_deg=first_rz_deg,
                linear_tcp=True,
            )
        with self._timed_stage("bottom_center_first_return_grasp_z"):
            self._bottom_center_linear_z(
                grasp_pose.z,
                motion,
                "bottom recovery first return to initial grasp Z",
                lifting=False,
            )
        with self._timed_stage("bottom_center_first_release"):
            open_to_preshape()

        # Keep Rz unchanged and rotate only around the current Tool X axis.
        with self._timed_stage("bottom_center_tool_rx_plus_70"):
            self._bottom_center_tool_rx(
                first_tool_rx_deg,
                "bottom recovery first Tool-Rx rotation",
            )
        with self._timed_stage("bottom_center_first_regrasp"):
            self.gripper.close(wait=True, cancel_check=self._cycle_cancel_requested)
        with self._timed_stage("bottom_center_first_regrasp_confirm"):
            self._confirm_grasp_before_lift(max_opening, width_m)
        with self._timed_stage("bottom_center_lift_160"):
            self._bottom_center_linear_z(
                grasp_pose.z + lift_m,
                motion,
                "bottom recovery 160mm lift",
                lifting=True,
            )

        with self._timed_stage("bottom_center_j6_plus_180"):
            self._validate_grasp_feedback("before bottom recovery J6 turn", max_opening)
            joints = self.controller.current_joint()
            safe_limit_deg = abs(
                float(self.get_parameter("barcode_flip_safe_joint_limit_deg").value)
            )
            if (
                len(joints) != 6
                or abs(float(joints[5]) + half_turn_deg) > safe_limit_deg
            ):
                raise RuntimeError(
                    "bottom recovery J6 +180deg exceeds the configured J6 limit"
                )
            target_joints = [float(value) for value in joints]
            target_joints[5] += half_turn_deg
            self._publish_status(
                f"bottom recovery: J6 {joints[5]:.1f}->{target_joints[5]:.1f}deg"
            )
            self._require_cycle_active("before bottom recovery J6 +180deg")
            self.controller.move_joint(
                target_joints,
                speed=motion["barcode_alignment_speed"],
                accel=motion["barcode_alignment_acc"],
            )
            self._require_cycle_active("after bottom recovery J6 +180deg")
            final_joints = self.controller.current_joint()
            tolerance_deg = max(
                1.0, float(self.get_parameter("barcode_flip_jog_tolerance_deg").value)
            )
            if (
                len(final_joints) != 6
                or abs(final_joints[5] - target_joints[5]) > tolerance_deg
            ):
                raise RuntimeError("bottom recovery J6 +180deg target was not reached")
            self._validate_grasp_feedback("after bottom recovery J6 turn", max_opening)
            post_turn_pose = self._current_command_pose()
            xy_shift_m = math.hypot(
                post_turn_pose.x - grasp_pose.x,
                post_turn_pose.y - grasp_pose.y,
            )
            self._publish_status(
                "bottom recovery: accepting J6-induced TCP XY shift "
                f"of {xy_shift_m*1000:.1f}mm; subsequent Z moves preserve current XY"
            )

        # Put the box down, release, apply Tool Rx +70 after J6 has turned
        # 180 degrees, and regrasp at the original Z.
        with self._timed_stage("bottom_center_descent_120"):
            self._bottom_center_linear_z(
                grasp_pose.z + lift_m - descent_m,
                motion,
                "bottom recovery second descent",
                lifting=False,
            )
        with self._timed_stage("bottom_center_second_release"):
            open_to_preshape()
        with self._timed_stage("bottom_center_second_tool_rx_plus_70"):
            self._bottom_center_tool_rx(
                release_tool_rx_deg,
                "bottom recovery second Tool-Rx rotation after J6 half-turn",
            )
        with self._timed_stage("bottom_center_latest_target_xy"):
            latest_target = self._wait_for_bottom_center_tracked_pose()
        with self._timed_stage("bottom_center_second_return_grasp_z"):
            self._bottom_center_linear_z(
                grasp_pose.z,
                motion,
                "bottom recovery tracked-XY return to initial grasp Z",
                lifting=False,
                target_xy=(latest_target.x, latest_target.y),
            )
        with self._timed_stage("bottom_center_final_close"):
            self.gripper.close(wait=True, cancel_check=self._cycle_cancel_requested)
        with self._timed_stage("bottom_center_final_grasp_confirm"):
            self._confirm_grasp_before_lift(max_opening, width_m)
        with self._timed_stage("turntable_safe_departure_lift"):
            self._execute_turntable_safe_departure_lift(motion)
        with self._timed_stage("post_lift_grasp_check"):
            self._validate_grasp_feedback("after bottom recovery departure", max_opening)

    def _execute_bottom_barcode_recovery(
        self,
        grasp_target: TcpPose,
        pre_shape_position: float,
        height_m: float,
    ) -> None:
        """Recover a box whose barcode is presumed to be on its bottom face.

        The box is returned to the table while still held, released to the
        original pre-shape width, flipped by User Ry=-45 degrees, and gripped
        again and lifted 160 mm. J6 is then returned by the remaining net
        -180-degree search travel. A small measured Ry shortfall is compensated before the
        fixed placement PTP; the final placement adds User Ry=-45 degrees and
        User Rz=+50 degrees, producing about -90 degrees total User Ry.
        """

        if not bool(self.get_parameter("bottom_barcode_recovery_enabled").value):
            raise RuntimeError(
                "No barcode found and bottom-barcode recovery is disabled; "
                "the held box was not released"
            )
        max_opening = float(self.get_parameter("dh_max_opening_m").value)
        if max_opening <= 0.0:
            raise RuntimeError(f"dh_max_opening_m must be positive, got {max_opening}")
        pre_shape_position = max(0.0, min(1.0, float(pre_shape_position)))
        target_ry_deg = float(
            self.get_parameter("bottom_flip_user_ry_target_deg").value
        )
        if target_ry_deg >= 0.0:
            raise RuntimeError(
                f"bottom_flip_user_ry_target_deg must be negative, got {target_ry_deg:+.1f}deg"
            )

        retract_m = float(self.get_parameter("bottom_flip_table_retract_m").value)
        if retract_m < 0.0 or retract_m > 0.200:
            raise RuntimeError(
                f"bottom_flip_table_retract_m must be in [0, 0.200]m, got {retract_m:.4f}m"
            )
        motion = self._motion_profile()
        self._publish_status(
            "no barcode on top or four side faces; entering bottom-barcode recovery "
            f"with User Ry target {target_ry_deg:+.1f}deg"
        )

        if retract_m > 0.0005:
            with self._timed_stage("bottom_flip_x_retract"):
                self._relative_user_move(
                    x=-retract_m,
                    label=(
                        "bottom-barcode recovery: User X- table clearance "
                        f"{retract_m * 1000.0:.1f}mm"
                    ),
                    speed_factor=motion["scanner_retreat_speed"],
                    accel_factor=motion["scanner_retreat_acc"],
                )
                self._require_cycle_active("after bottom-barcode X- clearance")

        # Reverse the first grasp lift, but retain the configured clearance.
        # This avoids deriving another low Z from the measured box height.
        table_pose = self._bottom_flip_table_pose()
        current_pose = self._current_command_pose()
        table_z_delta = float(table_pose.z) - float(current_pose.z)
        if abs(table_z_delta) > 0.0005:
            with self._timed_stage("bottom_flip_table_descent"):
                configured_lift_m = float(self.get_parameter("grasp_lift_m").value)
                retained_clearance_m = float(
                    self.get_parameter("bottom_flip_table_z_offset_m").value
                )
                self._relative_user_move(
                    z=table_z_delta,
                    label=(
                        "bottom-barcode recovery: reversing first lift "
                        f"{configured_lift_m * 1000.0:.1f}mm minus "
                        f"{retained_clearance_m * 1000.0:.1f}mm clearance; "
                        f"actual descent={-table_z_delta * 1000.0:.1f}mm, "
                        f"target Z={table_pose.z * 1000.0:.1f}mm"
                    ),
                    speed_factor=motion["place_speed"],
                    accel_factor=motion["place_acc"],
                )
                self._require_cycle_active("at bottom-barcode table pose")

        with self._timed_stage("bottom_flip_open_preshape"):
            current_position = self.gripper.read_position()
            self.gripper.set_position(pre_shape_position, wait=False)
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=pre_shape_position,
                initial_position=current_position,
                cancel_check=self._cycle_cancel_requested,
            )
            self._publish_status(
                "bottom-barcode recovery: released box on table at the original "
                f"pre-shape opening {pre_shape_position * max_opening * 1000.0:.1f}mm"
            )

        with self._timed_stage("bottom_flip_user_ry"):
            before_flip = self._current_command_pose()
            measured_ry_deg, _ = self._try_bottom_flip_ry_at_table(
                table_pose,
            )
            after_flip = self._current_command_pose()
            xyz_error_m = max(
                abs(after_flip.x - table_pose.x),
                abs(after_flip.y - table_pose.y),
                abs(after_flip.z - table_pose.z),
            )
            xyz_tolerance_m = max(
                0.001,
                float(self.get_parameter("face_up_fixed_xyz_tolerance_m").value),
            )
            if xyz_error_m > xyz_tolerance_m:
                raise RuntimeError(
                    "Bottom-barcode table flip changed TCP XYZ by too much: "
                    f"error={xyz_error_m * 1000.0:.1f}mm"
                )
            # Prefer the measured value returned by the motion helper, but keep
            # this local pose pair as a diagnostic if the controller reports a
            # wrapped Euler representation.
            measured_pose_delta = self._signed_user_y_delta_deg(before_flip, after_flip)
            if abs(measured_pose_delta - measured_ry_deg) > 5.0:
                self.get_logger().warning(
                    "Bottom-barcode User Ry feedback differs between command and "
                    f"pose samples: helper={measured_ry_deg:+.1f}deg, "
                    f"pose_delta={measured_pose_delta:+.1f}deg"
                )
                measured_ry_deg = measured_pose_delta

        with self._timed_stage("bottom_flip_close_preshape"):
            self.gripper.set_force(int(self.get_parameter("dh_grasp_force").value))
            # The fingers were already opened to the original pre-shape width
            # before the table rotation.  Close from that same starting width;
            # issuing set_position(pre_shape_position) here would merely open
            # the fingers again and could never regrip the box.
            self.gripper.close(
                wait=True,
                cancel_check=self._cycle_cancel_requested,
            )
            self._validate_grasp_feedback(
                "after bottom-barcode table regrasp",
                max_opening,
            )

        with self._timed_stage("bottom_flip_lift"):
            bottom_lift_m = float(
                self.get_parameter("bottom_flip_lift_m").value
            )
            if bottom_lift_m <= 0.0 or bottom_lift_m > 0.300:
                raise RuntimeError(
                    f"bottom_flip_lift_m must be in (0, 0.300]m, "
                    f"got {bottom_lift_m:.4f}m"
                )
            self._relative_user_move(
                z=bottom_lift_m,
                label=(
                    "bottom-barcode recovery: lifting regrasped box from table "
                    f"{bottom_lift_m * 1000.0:.1f}mm"
                ),
                speed_factor=motion["grasp_lift_speed"],
                accel_factor=motion["grasp_lift_acc"],
            )
            self._require_cycle_active("after bottom-barcode table lift")

        with self._timed_stage("bottom_flip_j6_reverse"):
            self._reverse_barcode_search_j6()

        # The table flip is commanded as -45 degrees.  If feedback shows a
        # shortfall, add only the missing User-Y amount after J6 has returned;
        # this keeps the bottom face as close to perfectly upward as the robot
        # can reach without trying another large table rotation.
        remaining_ry_deg = target_ry_deg - measured_ry_deg
        ry_tolerance_deg = max(
            1.0,
            abs(float(self.get_parameter("face_up_jog_tolerance_deg").value)),
        )
        for attempt in range(2):
            if abs(remaining_ry_deg) <= ry_tolerance_deg:
                break
            current = self._current_command_pose()
            before_compensation = current
            self._publish_status(
                f"bottom-barcode recovery: compensating remaining User Ry "
                f"{remaining_ry_deg:+.1f}deg (attempt {attempt + 1}/2)"
            )
            self._move_to_user_xyz_with_rotation(
                [current.x, current.y, current.z],
                ry_delta_deg=remaining_ry_deg,
                rz_delta_deg=0.0,
            )
            after_compensation = self._current_command_pose()
            applied_ry_deg = self._signed_user_y_delta_deg(
                before_compensation,
                after_compensation,
            )
            measured_ry_deg += applied_ry_deg
            remaining_ry_deg = target_ry_deg - measured_ry_deg
        if abs(remaining_ry_deg) > ry_tolerance_deg:
            raise RuntimeError(
                "Bottom-barcode User Ry compensation incomplete: "
                f"target={target_ry_deg:+.1f}deg, measured={measured_ry_deg:+.1f}deg"
            )
        self._publish_status(
            f"bottom-barcode recovery orientation ready: total User Ry "
            f"{measured_ry_deg:+.1f}deg"
        )

        fixed_place_xyz = self._bottom_barcode_place_xyz()
        place_ry_delta_deg = float(
            self.get_parameter("bottom_barcode_place_ry_delta_deg").value
        )
        with self._timed_stage("bottom_barcode_fixed_place_ptp"):
            self._move_to_user_xyz_with_rotation(
                fixed_place_xyz,
                ry_delta_deg=place_ry_delta_deg,
                rz_delta_deg=float(self.get_parameter("post_scan_user_rz_deg").value),
            )
            self._require_cycle_active("at bottom-barcode fixed placement pose")
            self._validate_grasp_feedback(
                "at bottom-barcode fixed placement pose",
                max_opening,
            )

        with self._timed_stage("bottom_barcode_gripper_open_place"):
            current_position = self.gripper.read_position()
            current_opening = current_position * max_opening
            release_clearance = max(
                0.0,
                float(self.get_parameter("place_release_clearance_m").value),
            )
            release_opening = min(max_opening, current_opening + release_clearance)
            release_position = release_opening / max_opening
            self._publish_status(
                "bottom-barcode box placed after additional User Ry "
                f"{place_ry_delta_deg:+.1f}deg and User Rz "
                f"{float(self.get_parameter('post_scan_user_rz_deg').value):+.1f}deg; "
                f"opening {current_opening * 1000.0:.1f}->{release_opening * 1000.0:.1f}mm"
            )
            self.gripper.set_position(release_position, wait=False)
            self.gripper.wait_until_stopped(
                timeout_s=float(self.get_parameter("dh_timeout_s").value),
                target_position=release_position,
                initial_position=current_position,
                cancel_check=self._cycle_cancel_requested,
            )
        self._publish_status(
            "bottom-barcode box placed at fixed XYZ; returning to startup"
        )

    def _require_cycle_active(self, stage: str) -> None:
        if not self.running or not self.cycle_enabled:
            if self.secondary_protective_stop_latched.is_set():
                raise RuntimeError(
                    f"Cycle stopped by 102 safety interlock at {stage}: "
                    f"{self.secondary_safety_reason}; no subsequent motion was issued"
                )
            raise RuntimeError(f"Cycle cancelled by operator at {stage}; no subsequent motion was issued")

    def _cycle_cancel_requested(self) -> bool:
        """Let gripper waits release the robot sequence lock after a trip."""

        return not self.running or not self.cycle_enabled

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
        gripper_opened_for_retreat = False
        if exc.needs_vertical_retreat:
            # A false GRIPPED state can mean one or both fingers are resting on
            # the box top.  Open in place before the vertical escape so the
            # recovery cannot drag or launch the box.
            self.gripper.open(
                wait=True,
                cancel_check=self._cycle_cancel_requested,
            )
            gripper_opened_for_retreat = True
            self._relative_user_move(
                z=float(self.get_parameter("grasp_lift_m").value),
                label="invalid grasp released; retreating vertically before startup",
            )
            self._require_cycle_active("after empty-grasp vertical retreat")
        self._move_startup_and_open(
            require_cycle_active=True,
            open_gripper=not gripper_opened_for_retreat,
        )

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
                    cancel_check=self._cycle_cancel_requested,
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

    def _confirm_grasp_before_lift(
        self,
        max_opening: float,
        commanded_preshape_m: float,
    ) -> None:
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
        minimum_closure = max(
            0.0,
            float(self.get_parameter("grasp_min_closure_from_preshape_m").value),
        )
        confirmed = 0
        last_state = GRIP_IN_MOTION
        last_opening_m = 0.0
        for sample_index in range(required):
            self._require_cycle_active("while confirming grasp before lift")
            last_opening_m = self.gripper.read_position() * max_opening
            last_state = self.gripper.read_grip_state()
            state_text = self._grip_state_text(last_state)
            closure_m = max(0.0, commanded_preshape_m - last_opening_m)
            self._publish_status(
                f"grasp confirmation {sample_index + 1}/{required}: "
                f"state={last_state} ({state_text}), opening={last_opening_m*1000:.1f}mm, "
                f"closure_from_preshape={closure_m*1000:.1f}mm"
            )
            if grasp_feedback_is_plausible(
                last_state,
                last_opening_m,
                commanded_preshape_m,
                minimum_opening,
                minimum_closure,
            ):
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
            f"opening={last_opening_m:.4f}m, "
            f"closure_from_preshape="
            f"{max(0.0, commanded_preshape_m-last_opening_m):.4f}m "
            f"(required>={minimum_closure:.4f}m); TCP remains at grasp depth"
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

    def _execute_turntable_safe_departure_lift(
        self,
        motion: dict[str, int],
    ) -> None:
        """Lift straight up to an absolute safe Z before leaving the table.

        A queued ``MovJ`` can make the TCP dip even when both joint-space
        endpoints look safe.  Turntable pickup therefore uses one blocking
        Cartesian ``MovL`` with unchanged X/Y/orientation.  Only after this
        command reaches and verifies the safe Z may any transfer start.
        """

        approach_xyz = [
            float(value)
            for value in self.get_parameter("scan_exit_user_xyz").value
        ]
        if len(approach_xyz) != 3:
            raise RuntimeError("scan_exit_user_xyz must contain 3 values")
        current = self._current_command_pose()
        target_z = turntable_departure_target_z(
            current.z,
            float(self.get_parameter("grasp_lift_m").value),
            approach_xyz[2],
        )
        target = TcpPose(
            current.x,
            current.y,
            target_z,
            current.rx,
            current.ry,
            current.rz,
        )
        self._require_cycle_active("immediately before turntable-safe departure")
        self._publish_status(
            "lifting grasp vertically clear of turntable with blocking MovL: "
            f"User-0/Tool-1 Z {current.z * 1000.0:.1f}->"
            f"{target_z * 1000.0:.1f}mm; no transfer command is queued"
        )
        self.controller.move_linear_tcp(
            target,
            speed=motion["grasp_lift_speed"],
            accel=motion["grasp_lift_acc"],
            user_index=int(self.get_parameter("user_index").value),
            tool_index=int(self.get_parameter("command_tool_index").value),
        )
        self._require_cycle_active("after turntable-safe departure")
        final = self._current_command_pose()
        tolerance_m = max(
            0.001,
            float(self.get_parameter("jog_tolerance_m").value),
        )
        xy_error_m = math.hypot(final.x - current.x, final.y - current.y)
        if final.z < target_z - tolerance_m or xy_error_m > tolerance_m * 1.5:
            raise RuntimeError(
                "Turntable-safe vertical departure did not reach its verified "
                "clearance corridor: "
                f"target_Z={target_z * 1000.0:.1f}mm, "
                f"actual_Z={final.z * 1000.0:.1f}mm, "
                f"XY_drift={xy_error_m * 1000.0:.1f}mm"
            )
        self._publish_status(
            "turntable-safe vertical departure verified: "
            f"TCP Z={final.z * 1000.0:.1f}mm, "
            f"XY drift={xy_error_m * 1000.0:.1f}mm"
        )

    def _execute_turntable_lift_face_snap(
        self,
        motion: dict[str, int],
    ) -> bool:
        """Rise and align the held D435 side face in one Cartesian MovL.

        A failed read-only kinematic preflight uses the original straight
        320 mm lift followed by the separate J6 snap.  A commanded motion
        failure is never retried automatically.
        """

        current = self._current_command_pose()
        safe_z = turntable_departure_target_z(
            current.z,
            float(self.get_parameter("grasp_lift_m").value),
            float(self.get_parameter("scan_exit_user_xyz").value[2]),
        )
        if not math.isfinite(safe_z) or safe_z < current.z + 0.010:
            raise RuntimeError(
                "Turntable departure target must rise at least 10mm above the grasp pose"
            )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        tolerance_m = max(0.001, float(self.get_parameter("jog_tolerance_m").value))
        start_pose = current

        watch_index = max(
            0, min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value))
        )
        joints = [float(value) for value in self.controller.current_joint()]
        if len(joints) != 6:
            raise RuntimeError("Current joint feedback must contain 6 values before face snap")
        target_deg = nearest_face_anchor_deg(
            joints[watch_index],
            float(self.get_parameter("d435_side_face_reference_joint_deg").value),
            float(self.get_parameter("barcode_flip_step_deg").value),
            abs(float(self.get_parameter("barcode_flip_safe_joint_limit_deg").value)),
        )
        correction_deg = target_deg - joints[watch_index]
        if abs(correction_deg) <= 0.2:
            self._execute_turntable_safe_departure_lift(motion)
            self._publish_status("D435 side face already aligned; straight lift completed")
            return True
        if not self._is_barcode_flip_joint_safe(joints, correction_deg):
            raise RuntimeError(
                f"Nearest V2 face anchor is outside the configured J{watch_index + 1} "
                f"limit: current={joints[watch_index]:.1f}deg, target={target_deg:.1f}deg"
            )

        target_joints = list(joints)
        target_joints[watch_index] = target_deg
        final_target = None
        try:
            fk_now = self.controller.forward_kinematics(
                joints, user_index=user_index, tool_index=tool_index
            )
            fk_target = self.controller.forward_kinematics(
                target_joints, user_index=user_index, tool_index=tool_index
            )
            fk_position_error = math.sqrt(
                (fk_now.x - start_pose.x) ** 2
                + (fk_now.y - start_pose.y) ** 2
                + (fk_now.z - start_pose.z) ** 2
            )
            pure_j6_shift = math.sqrt(
                (fk_target.x - fk_now.x) ** 2
                + (fk_target.y - fk_now.y) ** 2
                + (fk_target.z - fk_now.z) ** 2
            )
            if (
                not math.isfinite(fk_position_error)
                or not math.isfinite(pure_j6_shift)
                or fk_position_error > 0.005
                or pure_j6_shift > 0.005
            ):
                raise RuntimeError(
                    f"FK pose mismatch={fk_position_error * 1000.0:.1f}mm "
                    f"or J6-only TCP shift={pure_j6_shift * 1000.0:.1f}mm"
                )

            def angle_delta(a: float, b: float) -> float:
                return (b - a + 180.0) % 360.0 - 180.0

            if max(
                abs(angle_delta(a, b))
                for a, b in zip(
                    (fk_now.rx, fk_now.ry, fk_now.rz),
                    (start_pose.rx, start_pose.ry, start_pose.rz),
                )
            ) > 3.0:
                raise RuntimeError("FK orientation differs from live TCP feedback")

            # Preflight the same fixed-XY, ascending-Z orientation sweep with
            # near-joint IK.  The controller still executes one Cartesian MovL.
            previous_joints = joints
            for fraction in (0.25, 0.5, 0.75, 1.0):
                candidate = TcpPose(
                    start_pose.x,
                    start_pose.y,
                    start_pose.z + (safe_z - start_pose.z) * fraction,
                    start_pose.rx + angle_delta(start_pose.rx, fk_target.rx) * fraction,
                    start_pose.ry + angle_delta(start_pose.ry, fk_target.ry) * fraction,
                    start_pose.rz + angle_delta(start_pose.rz, fk_target.rz) * fraction,
                )
                solved_joints = self.controller.inverse_kinematics(
                    candidate,
                    user_index=user_index,
                    tool_index=tool_index,
                    joint_near=previous_joints,
                )
                if len(solved_joints) != 6 or any(
                    not math.isfinite(float(a))
                    or abs(float(a) - float(b)) > 45.0
                    for a, b in zip(solved_joints, previous_joints)
                ):
                    raise RuntimeError("IK sweep changes joint branch")
                if not (
                    min(joints[watch_index], target_deg) - 5.0
                    <= float(solved_joints[watch_index])
                    <= max(joints[watch_index], target_deg) + 5.0
                ):
                    raise RuntimeError("IK sweep turns J6 beyond the nearest-face range")
                previous_joints = [float(value) for value in solved_joints]
                final_target = candidate
            if abs(previous_joints[watch_index] - target_deg) > 3.0:
                raise RuntimeError(
                    f"IK endpoint J{watch_index + 1}={previous_joints[watch_index]:.1f}deg "
                    f"misses {target_deg:.1f}deg anchor"
                )
        except Exception as exc:
            self._require_cycle_active("after lift/J6 kinematic preflight")
            self.get_logger().warning(
                f"Turntable lift/J6 overlap preflight rejected ({exc}); "
                "using straight lift then separate J6 snap"
            )
            self._execute_turntable_safe_departure_lift(motion)
            return False

        assert final_target is not None
        self._require_cycle_active("before combined turntable lift/J6 alignment")
        self._publish_status(
            "combining ascending MovL and nearest-90deg J6 face alignment: "
            f"TCP Z {start_pose.z * 1000.0:.1f}->{safe_z * 1000.0:.1f}mm, "
            f"J{watch_index + 1} {joints[watch_index]:.1f}->{target_deg:.1f}deg"
        )
        self.controller.move_linear_tcp(
            final_target,
            speed=min(motion["grasp_lift_speed"], motion["barcode_alignment_speed"]),
            accel=min(motion["grasp_lift_acc"], motion["barcode_alignment_acc"]),
            user_index=user_index,
            tool_index=tool_index,
        )
        self._require_cycle_active("after combined turntable lift/J6 alignment")
        final_pose = self._current_command_pose()
        xy_drift = math.hypot(final_pose.x - start_pose.x, final_pose.y - start_pose.y)
        if final_pose.z < safe_z - tolerance_m or xy_drift > tolerance_m * 1.5:
            raise RuntimeError(
                "Combined turntable lift/J6 alignment missed safe corridor: "
                f"required_Z={safe_z * 1000.0:.1f}mm, "
                f"actual_Z={final_pose.z * 1000.0:.1f}mm, "
                f"XY_drift={xy_drift * 1000.0:.1f}mm"
            )
        final_joints = self.controller.current_joint()
        if len(final_joints) != 6:
            raise RuntimeError("Current joint feedback must contain 6 values after face snap")
        final_deg = float(final_joints[watch_index])
        tolerance_deg = max(
            1.0, abs(float(self.get_parameter("barcode_flip_jog_tolerance_deg").value))
        )
        if abs(final_deg - target_deg) > tolerance_deg:
            self._publish_status(
                f"combined lift reached safe Z but J{watch_index + 1} is "
                f"{final_deg:.1f}deg; completing the face snap at safe height"
            )
            self._snap_turntable_side_to_nearest_face(motion)
        else:
            self._publish_status(
                "combined turntable departure and nearest face verified: "
                f"TCP Z={final_pose.z * 1000.0:.1f}mm, "
                f"J{watch_index + 1}={final_deg:.1f}deg, "
                f"XY drift={xy_drift * 1000.0:.1f}mm"
            )
        return True

    def _snap_turntable_side_to_nearest_face(
        self,
        motion: dict[str, int],
    ) -> None:
        """Align a D435-confirmed side face to V2's nearest 90-degree J6 anchor.

        D435 confirms the side barcode before the left arm moves, so the V3
        fast path deliberately skips V2's scanner sweep.  Perform only the
        final V2 face snap here, after the blocking vertical departure has put
        the held box safely above the turntable.
        """

        watch_index = max(
            0,
            min(5, int(self.get_parameter("barcode_flip_watch_joint_index").value)),
        )
        current_joints = self.controller.current_joint()
        if len(current_joints) != 6:
            raise RuntimeError(
                "Current joint feedback must contain 6 values before D435 "
                f"face alignment, got {len(current_joints)}"
            )
        current_deg = float(current_joints[watch_index])
        reference_deg = float(
            self.get_parameter("d435_side_face_reference_joint_deg").value
        )
        safe_limit_deg = abs(
            float(self.get_parameter("barcode_flip_safe_joint_limit_deg").value)
        )
        target_deg = nearest_face_anchor_deg(
            current_deg,
            reference_deg,
            float(self.get_parameter("barcode_flip_step_deg").value),
            safe_limit_deg,
        )
        correction_deg = target_deg - current_deg
        if abs(correction_deg) <= 0.2:
            self._publish_status(
                f"D435 side face already at the nearest V2 90deg anchor: "
                f"J{watch_index + 1}={current_deg:.1f}deg"
            )
            return
        if not self._is_barcode_flip_joint_safe(current_joints, correction_deg):
            raise RuntimeError(
                f"Nearest V2 face anchor is outside the configured J{watch_index + 1} "
                f"limit: current={current_deg:.1f}deg, target={target_deg:.1f}deg"
            )

        before_pose = self._current_command_pose()
        target_joints = [float(value) for value in current_joints]
        target_joints[watch_index] = target_deg
        self._publish_status(
            "D435 side barcode confirmed; aligning the held box to the nearest "
            f"V2 90deg face at safe height: J{watch_index + 1} "
            f"{current_deg:.1f}->{target_deg:.1f}deg "
            f"(correction {correction_deg:+.1f}deg, reference={reference_deg:.1f}deg)"
        )
        self._require_cycle_active("before D435 nearest-face alignment")
        self.controller.move_joint(
            target_joints,
            speed=motion["barcode_alignment_speed"],
            accel=motion["barcode_alignment_acc"],
        )
        self._require_cycle_active("after D435 nearest-face alignment")

        final_joints = self.controller.current_joint()
        if len(final_joints) != 6:
            raise RuntimeError(
                "Current joint feedback must contain 6 values after D435 "
                f"face alignment, got {len(final_joints)}"
            )
        final_deg = float(final_joints[watch_index])
        tolerance_deg = max(
            1.0,
            abs(float(self.get_parameter("barcode_flip_jog_tolerance_deg").value)),
        )
        if abs(final_deg - target_deg) > tolerance_deg:
            raise RuntimeError(
                f"D435 nearest-face alignment did not reach its J{watch_index + 1} "
                f"anchor: target={target_deg:.1f}deg, actual={final_deg:.1f}deg"
            )

        final_pose = self._current_command_pose()
        safe_transfer_z_m = float(
            self.get_parameter("scan_exit_user_xyz").value[2]
        )
        z_tolerance_m = max(
            0.001,
            float(self.get_parameter("jog_tolerance_m").value),
        )
        if final_pose.z < safe_transfer_z_m - z_tolerance_m:
            raise RuntimeError(
                "D435 nearest-face alignment left the turntable-safe height: "
                f"required_Z={safe_transfer_z_m * 1000.0:.1f}mm, "
                f"actual_Z={final_pose.z * 1000.0:.1f}mm"
            )
        xyz_shift_mm = 1000.0 * math.sqrt(
            (final_pose.x - before_pose.x) ** 2
            + (final_pose.y - before_pose.y) ** 2
            + (final_pose.z - before_pose.z) ** 2
        )
        self._publish_status(
            f"nearest V2 90deg face aligned: J{watch_index + 1}={final_deg:.1f}deg, "
            f"TCP Z={final_pose.z * 1000.0:.1f}mm, XYZ shift={xyz_shift_mm:.1f}mm"
        )

    def _stop_active_motion_for_queue_failure(self, reason: str) -> None:
        """Stop a queued look-ahead motion before propagating its failure.

        The normal cycle recovery opens the gripper and may perform a vertical
        escape.  That recovery is only safe after the controller has stopped
        the active lift/transfer queue, so keep this cleanup in one place.
        """

        try:
            mode = self.controller.robot_mode
        except Exception:
            mode = -1
        if mode not in (7, 8, 10):
            return
        try:
            self.controller.stop_motion()
        except Exception as stop_exc:
            self.get_logger().warning(
                f"Could not stop active lift/transfer motion after {reason}: {stop_exc}"
            )

    def _execute_grasp_lift_transfer(
        self,
        motion: dict[str, int],
        max_opening: float,
    ) -> bool:
        """Blend the confirmed grasp lift into the transfer-joint move.

        The lift remains the first command and starts only after the existing
        two-sample grasp confirmation.  Once feedback shows that the TCP is
        within the configured lead distance of the safe lift height, the
        transfer ``MovJ`` is submitted with the same CP value.  The controller
        can then blend the two command segments instead of decelerating to an
        idle mode between them.  ``False`` means the lift completed before the
        look-ahead gate (or the firmware rejected queueing), so the caller must
        use the ordinary blocking transfer path.
        """

        lift_m = float(self.get_parameter("grasp_lift_m").value)
        if not math.isfinite(lift_m) or lift_m <= 0.0:
            raise RuntimeError(f"grasp_lift_m must be positive, got {lift_m!r}")

        lead_m = float(
            self.get_parameter("grasp_lift_transfer_queue_lead_m").value
        )
        if not math.isfinite(lead_m):
            raise RuntimeError(
                "grasp_lift_transfer_queue_lead_m must be finite, "
                f"got {lead_m!r}"
            )
        lead_m = max(0.0, lead_m)
        # Do not allow a queue request before the lift has made meaningful
        # vertical progress.  A value equal to the full lift is therefore a
        # safe configuration error with a clear legacy fallback.
        if lead_m >= lift_m:
            self.get_logger().warning(
                "Lift/transfer look-ahead disabled for this cycle: "
                f"queue lead={lead_m * 1000.0:.1f}mm is not below lift="
                f"{lift_m * 1000.0:.1f}mm; using the blocking transfer path"
            )
            self._require_cycle_active("before fallback grasp lift")
            self._relative_user_move(
                z=lift_m,
                label="lifting grasp",
                speed_factor=motion["grasp_lift_speed"],
                accel_factor=motion["grasp_lift_acc"],
            )
            return False

        cp = max(
            1,
            min(
                100,
                int(
                    round(
                        float(
                            self.get_parameter("grasp_lift_transfer_blend_cp").value
                        )
                    )
                ),
            ),
        )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        start_pose = self._current_command_pose()
        lift_target_z = float(start_pose.z) + lift_m
        queue_gate_z = lift_target_z - lead_m
        queue_timeout_s = max(
            0.2,
            float(self.get_parameter("jog_axis_timeout_s").value),
        )

        self._require_cycle_active("immediately before blended grasp lift")
        self._publish_status(
            "lifting grasp; transfer joint will be queued near "
            f"{queue_gate_z * 1000.0:.1f}mm TCP Z "
            f"(lead {lead_m * 1000.0:.1f}mm, CP={cp})"
        )
        lift_command = self.controller.submit_rel_move_user_joint(
            TcpPose(0.0, 0.0, lift_m, 0.0, 0.0, 0.0),
            speed=motion["grasp_lift_speed"],
            accel=motion["grasp_lift_acc"],
            cp=cp,
            user_index=user_index,
            tool_index=tool_index,
        )

        # The dashboard reply can arrive before the first feedback packet that
        # changes RobotMode/CurrentCommandId.  Treat that short handoff as a
        # pending command rather than immediately falling back to a blocking
        # lift; otherwise every real cycle can miss the queue gate.
        command_start_grace_s = max(
            0.05,
            min(
                1.0,
                float(
                    self.get_parameter(
                        "grasp_lift_transfer_command_start_grace_s"
                    ).value
                ),
            ),
        )
        command_submitted_at = time.monotonic()
        gate_deadline = time.monotonic() + queue_timeout_s
        while True:
            self._require_cycle_active("while waiting to queue transfer")
            try:
                current_pose = self._current_command_pose()
            except Exception as exc:
                # A transient dashboard/feedback read should not cause a
                # premature queue.  The feedback stream is still monitored by
                # the controller; retry until the command or gate is resolved.
                if time.monotonic() >= gate_deadline:
                    self._stop_active_motion_for_queue_failure("TCP gate feedback timeout")
                    raise RuntimeError(
                        "Could not observe TCP Z while waiting to queue transfer: "
                        f"{exc}"
                    ) from exc
                time.sleep(LOOKAHEAD_GATE_POLL_S)
                continue
            if float(current_pose.z) >= queue_gate_z:
                if self.controller.motion_command_is_active(lift_command):
                    break
                # If the endpoint is already visible while the command is no
                # longer active, the lift really completed before we could
                # queue the next segment; use the safe legacy path.
                self.controller.wait_for_command(lift_command, timeout_s=5.0)
                self._publish_status(
                    "grasp lift reached its endpoint before the queue gate; "
                    "using the blocking transfer path"
                )
                return False
            if not self.controller.motion_command_is_active(lift_command):
                if time.monotonic() - command_submitted_at < command_start_grace_s:
                    time.sleep(LOOKAHEAD_GATE_POLL_S)
                    continue
                # The first command has already reached its terminal state, so
                # there is no remaining segment to blend into.  Confirm its
                # completion and let the caller issue the normal transfer.
                self.controller.wait_for_command(lift_command, timeout_s=5.0)
                self._publish_status(
                    "grasp lift reached its endpoint before the queue gate; "
                    "using the blocking transfer path"
                )
                return False
            if time.monotonic() >= gate_deadline:
                self._stop_active_motion_for_queue_failure("queue gate timeout")
                raise TimeoutError(
                    "Timed out waiting for the grasp-lift TCP Z queue gate: "
                    f"current={float(current_pose.z) * 1000.0:.1f}mm, "
                    f"gate={queue_gate_z * 1000.0:.1f}mm"
                )
            time.sleep(LOOKAHEAD_GATE_POLL_S)

        self._require_cycle_active("at the grasp-lift transfer queue gate")
        # Preserve a feedback checkpoint before any horizontal/joint transfer
        # motion is allowed.  If the gripper reports an empty or implausible
        # hold, stop the still-active lift before the normal retry recovery
        # opens the fingers and retreats vertically.
        try:
            self._validate_grasp_feedback("before lift-transfer queue", max_opening)
        except RecoverableGraspError:
            self._stop_active_motion_for_queue_failure("pre-transfer grasp feedback failure")
            raise
        if not self.controller.motion_command_is_active(lift_command):
            self.controller.wait_for_command(lift_command, timeout_s=5.0)
            self._publish_status(
                "grasp lift completed while checking feedback; "
                "using the blocking transfer path"
            )
            return False
        # Keep the existing barcode semantics: callbacks are accepted from the
        # transfer motion onward, including the short CP-blended segment.
        self._reset_barcode_window()
        self._publish_status("queueing transfer joint before lift endpoint")
        try:
            transfer_command = self.controller.submit_move_joint(
                self._six_values("transfer_joint"),
                speed=motion["transfer_speed"],
                accel=motion["transfer_acc"],
                cp=cp,
            )
        except Exception as exc:
            # Some controller firmware accepts only one command at a time.  If
            # it rejected the second command while the lift is still active,
            # wait for the lift and let the caller use the safe legacy path.
            if self.controller.motion_command_is_active(lift_command):
                self.get_logger().warning(
                    "Controller rejected the lift/transfer queue while the lift "
                    f"was active; completing the lift then falling back: {exc}"
                )
                self.controller.wait_for_command(lift_command, timeout_s=5.0)
                return False
            raise

        try:
            self.controller.wait_for_command(transfer_command, timeout_s=60.0)
        except Exception:
            self._stop_active_motion_for_queue_failure("queued transfer wait")
            raise
        self._require_cycle_active("after blended grasp lift and transfer")
        self._publish_status("blended grasp lift and transfer joint reached")
        return True

    def _execute_post_scan_safe_place_blend(
        self,
        approach_xyz: list[float],
        fixed_place_xyz: list[float],
        side_rx_delta_deg: float,
        motion: dict[str, int],
        allow_pending_queue: bool = False,
    ) -> bool:
        """Queue the fixed placement PTP while the safe-height PTP finishes.

        This transition is deliberately narrower than the lift/transfer
        queue.  Barcode acquisition has finished, and both waypoints are
        predetermined, collision-checked PTP targets.  In the normal call the
        scanner retreat has also finished.  A scanner look-ahead caller may
        set ``allow_pending_queue`` while that known straight retreat is still
        in the controller queue; the safe-height command is then allowed to
        remain pending until its TCP gate is reached.  The helper validates the
        final placement pose before the gripper release stage.
        """

        if len(approach_xyz) != 3 or len(fixed_place_xyz) != 3:
            raise ValueError("safe-height and placement XYZ targets must contain 3 values")

        lead_m = float(self.get_parameter("post_scan_place_queue_lead_m").value)
        if not math.isfinite(lead_m):
            raise RuntimeError(
                "post_scan_place_queue_lead_m must be finite, "
                f"got {lead_m!r}"
            )
        lead_m = max(0.0, lead_m)
        cp = max(
            1,
            min(
                100,
                int(
                    round(
                        float(
                            self.get_parameter("post_scan_place_blend_cp").value
                        )
                    )
                ),
            ),
        )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)

        # The scanner-retreat command does not change orientation, so the
        # measured post-retreat pose is a stable reference for both targets.
        start_pose = self._current_command_pose()
        approach_pose = self._compose_user_target_pose(
            start_pose,
            approach_xyz,
            ry_delta_deg=float(self.get_parameter("face_up_user_ry_deg").value),
            rz_delta_deg=float(self.get_parameter("post_scan_user_rz_deg").value),
        )
        place_pose = self._compose_user_target_pose(
            approach_pose,
            fixed_place_xyz,
            ry_delta_deg=0.0,
            rz_delta_deg=0.0,
            rx_delta_deg=side_rx_delta_deg,
        )

        current_joint = self.controller.current_joint()
        safe_joint = self.controller.inverse_kinematics(
            approach_pose,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=current_joint,
        )
        self.controller.inverse_kinematics(
            place_pose,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=safe_joint,
        )
        self._require_cycle_active("before post-scan safe-height motion")

        combined_speed = motion["post_scan_speed"]
        combined_acc = motion["post_scan_acc"]
        self._publish_status(
            "moving to post-scan safe height; fixed placement will be queued "
            f"near the endpoint (lead {lead_m * 1000.0:.1f}mm, CP={cp})"
        )
        safe_command = self.controller.submit_move_joint_tcp(
            approach_pose,
            speed=combined_speed,
            accel=combined_acc,
            cp=cp,
            user_index=user_index,
            tool_index=tool_index,
        )

        command_start_grace_s = max(
            0.05,
            min(
                1.0,
                float(
                    self.get_parameter(
                        "post_scan_place_command_start_grace_s"
                    ).value
                ),
            ),
        )
        command_submitted_at = time.monotonic()
        queue_timeout_s = max(
            0.2,
            float(self.get_parameter("jog_axis_timeout_s").value),
        )
        gate_deadline = time.monotonic() + queue_timeout_s
        while True:
            self._require_cycle_active("while waiting to queue fixed placement")
            try:
                current_pose = self._current_command_pose()
            except Exception as exc:
                if time.monotonic() >= gate_deadline:
                    self._stop_active_motion_for_queue_failure(
                        "safe-height gate feedback timeout"
                    )
                    raise RuntimeError(
                        "Could not observe TCP pose while waiting to queue fixed placement: "
                        f"{exc}"
                    ) from exc
                time.sleep(LOOKAHEAD_GATE_POLL_S)
                continue

            distance_to_safe_m = float(
                np.linalg.norm(
                    np.array(
                        [
                            current_pose.x - approach_pose.x,
                            current_pose.y - approach_pose.y,
                            current_pose.z - approach_pose.z,
                        ],
                        dtype=np.float64,
                    )
                )
            )
            active = self.controller.motion_command_is_active(safe_command)
            # CurrentCommandId remains on the preceding scanner-retreat
            # command until that command reaches its endpoint.  When the safe
            # PTP was deliberately submitted behind that known predecessor,
            # ``motion_command_is_active(safe_command)`` is therefore false
            # even though the command is valid and waiting in the firmware
            # queue.  Keep polling the measured TCP gate in that narrow case.
            pending = False
            if allow_pending_queue and not active:
                try:
                    pending = self.controller.robot_mode in (7, 8)
                except Exception:
                    pending = False
            if distance_to_safe_m <= lead_m:
                if active or pending:
                    break
                if time.monotonic() - command_submitted_at < command_start_grace_s:
                    time.sleep(LOOKAHEAD_GATE_POLL_S)
                    continue
                self.controller.wait_for_command(safe_command, timeout_s=5.0)
                self._publish_status(
                    "post-scan safe-height PTP completed before the queue gate; "
                    "using the blocking placement path"
                )
                return False
            if not active and pending:
                if time.monotonic() >= gate_deadline:
                    self._stop_active_motion_for_queue_failure(
                        "safe-height pending queue gate timeout"
                    )
                    raise TimeoutError(
                        "Timed out waiting for the queued post-scan safe-height "
                        "motion to reach its placement gate: "
                        f"distance={distance_to_safe_m * 1000.0:.1f}mm, "
                        f"gate={lead_m * 1000.0:.1f}mm"
                    )
                time.sleep(LOOKAHEAD_GATE_POLL_S)
                continue
            if not active:
                if time.monotonic() - command_submitted_at < command_start_grace_s:
                    time.sleep(LOOKAHEAD_GATE_POLL_S)
                    continue
                self.controller.wait_for_command(safe_command, timeout_s=5.0)
                self._publish_status(
                    "post-scan safe-height PTP completed before the queue gate; "
                    "using the blocking placement path"
                )
                return False
            if time.monotonic() >= gate_deadline:
                self._stop_active_motion_for_queue_failure(
                    "safe-height placement queue gate timeout"
                )
                raise TimeoutError(
                    "Timed out waiting to queue fixed placement: "
                    f"distance={distance_to_safe_m * 1000.0:.1f}mm, "
                    f"gate={lead_m * 1000.0:.1f}mm"
                )
            time.sleep(LOOKAHEAD_GATE_POLL_S)

        self._require_cycle_active("at post-scan safe-height queue gate")
        self._publish_status("queueing fixed placement PTP before safe-height endpoint")
        try:
            place_command = self.controller.submit_move_joint_tcp(
                place_pose,
                speed=combined_speed,
                accel=combined_acc,
                cp=cp,
                user_index=user_index,
                tool_index=tool_index,
            )
        except Exception as exc:
            if self.controller.motion_command_is_active(safe_command):
                self.get_logger().warning(
                    "Controller rejected the safe-height/fixed-placement queue while "
                    f"the safe-height move was active; completing it then falling back: {exc}"
                )
                self.controller.wait_for_command(safe_command, timeout_s=5.0)
                return False
            raise

        try:
            self.controller.wait_for_command(place_command, timeout_s=60.0)
        except Exception:
            self._stop_active_motion_for_queue_failure("queued fixed placement wait")
            raise
        self._require_cycle_active("after blended safe-height and fixed placement")

        final = self._current_command_pose()
        tolerance = max(0.0005, float(self.get_parameter("jog_tolerance_m").value))
        position_errors = [
            abs(final.x - place_pose.x),
            abs(final.y - place_pose.y),
            abs(final.z - place_pose.z),
        ]
        if max(position_errors) > tolerance * 1.5:
            raise RuntimeError(
                "Blended fixed-placement final XYZ error too large: "
                f"{[round(error, 4) for error in position_errors]}m"
            )
        final_rotation = SciPyRot.from_euler(
            "xyz", [final.rx, final.ry, final.rz], degrees=True
        )
        target_rotation = SciPyRot.from_euler(
            "xyz", [place_pose.rx, place_pose.ry, place_pose.rz], degrees=True
        )
        orientation_error_deg = math.degrees(
            SciPyRot.from_matrix(
                final_rotation.as_matrix() @ target_rotation.as_matrix().T
            ).magnitude()
        )
        orientation_tolerance_deg = max(
            1.0,
            float(self.get_parameter("face_up_jog_tolerance_deg").value),
        )
        if orientation_error_deg > orientation_tolerance_deg:
            raise RuntimeError(
                "Blended fixed-placement final orientation error too large: "
                f"{orientation_error_deg:.2f}deg > {orientation_tolerance_deg:.2f}deg"
            )
        self._publish_status(
            "blended post-scan safe-height and fixed-placement pose reached: "
            f"XYZ=({final.x * 1000.0:.1f},{final.y * 1000.0:.1f},"
            f"{final.z * 1000.0:.1f})mm"
        )
        return True

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

    def _scanner_retreat_distance(self, actual_approach_m: float) -> float:
        """Return the User-X distance required to clear the scanner."""

        extra_retreat_m = float(self.get_parameter("scanner_retreat_extra_m").value)
        if not math.isfinite(extra_retreat_m) or extra_retreat_m < 0.0 or extra_retreat_m > 0.200:
            raise RuntimeError(
                f"scanner_retreat_extra_m must be finite and in [0, 0.200]m, "
                f"got {extra_retreat_m:.4f}m"
            )
        actual_approach_m = float(actual_approach_m)
        if not math.isfinite(actual_approach_m):
            raise RuntimeError(
                f"actual scanner approach must be finite, got {actual_approach_m!r}"
            )
        return abs(actual_approach_m) + extra_retreat_m

    def _validate_scanner_retreat_displacement(
        self,
        tcp_before: TcpPose,
        tcp_after: TcpPose,
        retreat_m: float,
    ) -> float:
        """Verify that the commanded User-X safety retreat was reached."""

        actual_retreat_m = float(tcp_after.x) - float(tcp_before.x)
        expected_retreat_m = -float(retreat_m)
        tolerance_m = max(
            0.001,
            float(self.get_parameter("jog_tolerance_m").value) * 1.5,
        )
        if abs(actual_retreat_m - expected_retreat_m) > tolerance_m:
            raise RuntimeError(
                f"Scanner safety retreat X displacement mismatch: "
                f"requested={expected_retreat_m * 1000.0:.1f}mm, "
                f"actual={actual_retreat_m * 1000.0:.1f}mm; combined motion refused"
            )
        return actual_retreat_m

    def _execute_scanner_retreat_post_scan_blend(
        self,
        actual_approach_m: float,
        approach_xyz: list[float],
        fixed_place_xyz: list[float],
        side_rx_delta_deg: float,
        motion: dict[str, int],
    ) -> bool:
        """Blend scanner clearance into the post-scan safe-height/place path.

        The scanner retreat is a straight User-X move with a known extra
        clearance.  Near its endpoint, queue the already IK-checked safe-height
        PTP; the existing helper then queues fixed placement near the next
        endpoint.  If the firmware does not accept a command while the retreat
        is active, wait for and validate the retreat and run the ordinary
        post-scan helper.
        """

        retreat_m = self._scanner_retreat_distance(actual_approach_m)
        if retreat_m <= 0.0005:
            self._publish_status(
                "scanner safety retreat not required; no X+ approach was made"
            )
            return self._execute_post_scan_safe_place_blend(
                approach_xyz,
                fixed_place_xyz,
                side_rx_delta_deg,
                motion,
            )

        required_methods = (
            "submit_rel_move_user_joint",
            "motion_command_is_active",
            "wait_for_command",
        )
        if not all(hasattr(self.controller, method) for method in required_methods):
            self._publish_status(
                "scanner-retreat look-ahead unavailable; using blocking scanner retreat"
            )
            self._retreat_box_from_scanner(actual_approach_m)
            return self._execute_post_scan_safe_place_blend(
                approach_xyz,
                fixed_place_xyz,
                side_rx_delta_deg,
                motion,
            )

        lead_m = float(
            self.get_parameter("scanner_retreat_post_scan_queue_lead_m").value
        )
        if not math.isfinite(lead_m):
            raise RuntimeError(
                "scanner_retreat_post_scan_queue_lead_m must be finite, "
                f"got {lead_m!r}"
            )
        # Keep a small amount of actual retreat progress before allowing the
        # next PTP to be queued.  The default 10 mm lead still leaves 20 mm of
        # the default extra scanner clearance before any large rotation path.
        lead_m = min(
            max(0.0, lead_m),
            max(0.0, retreat_m - 0.005),
        )
        cp = max(
            1,
            min(
                100,
                int(
                    round(
                        float(
                            self.get_parameter(
                                "scanner_retreat_post_scan_blend_cp"
                            ).value
                        )
                    )
                ),
            ),
        )
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        retreat_speed = self._motion_profile()["scanner_retreat_speed"]
        retreat_acc = self._motion_profile()["scanner_retreat_acc"]
        tcp_before = self._current_command_pose()
        start_x = float(tcp_before.x)

        self._require_cycle_active("before scanner-retreat look-ahead")
        self._publish_status(
            "scanner safety retreat User X-; post-scan safe height will be queued "
            f"near the endpoint (lead {lead_m * 1000.0:.1f}mm, CP={cp})"
        )
        try:
            retreat_command = self.controller.submit_rel_move_user_joint(
                TcpPose(-retreat_m, 0.0, 0.0, 0.0, 0.0, 0.0),
                speed=retreat_speed,
                accel=retreat_acc,
                cp=cp,
                user_index=user_index,
                tool_index=tool_index,
            )
        except Exception as exc:
            self.get_logger().warning(
                "Controller rejected scanner-retreat look-ahead submission; "
                f"using blocking retreat: {exc}"
            )
            self._retreat_box_from_scanner(actual_approach_m)
            return self._execute_post_scan_safe_place_blend(
                approach_xyz,
                fixed_place_xyz,
                side_rx_delta_deg,
                motion,
            )

        command_start_grace_s = max(
            0.05,
            min(
                1.0,
                float(
                    self.get_parameter(
                        "scanner_retreat_post_scan_command_start_grace_s"
                    ).value
                ),
            ),
        )
        command_submitted_at = time.monotonic()
        queue_timeout_s = max(
            0.2,
            float(self.get_parameter("jog_axis_timeout_s").value),
        )
        gate_deadline = time.monotonic() + queue_timeout_s

        while True:
            self._require_cycle_active("while waiting to queue post-scan motion")
            try:
                current_pose = self._current_command_pose()
            except Exception as exc:
                if time.monotonic() >= gate_deadline:
                    self._stop_active_motion_for_queue_failure(
                        "scanner-retreat gate feedback timeout"
                    )
                    raise RuntimeError(
                        "Could not observe TCP pose while waiting to queue post-scan "
                        f"motion: {exc}"
                    ) from exc
                time.sleep(LOOKAHEAD_GATE_POLL_S)
                continue

            progress_m = start_x - float(current_pose.x)
            remaining_m = retreat_m - progress_m
            active = self.controller.motion_command_is_active(retreat_command)
            if remaining_m <= lead_m and progress_m >= 0.0:
                if active:
                    break
                if time.monotonic() - command_submitted_at < command_start_grace_s:
                    time.sleep(LOOKAHEAD_GATE_POLL_S)
                    continue
                self.controller.wait_for_command(retreat_command, timeout_s=5.0)
                tcp_after = self._current_command_pose()
                actual_retreat = self._validate_scanner_retreat_displacement(
                    tcp_before,
                    tcp_after,
                    retreat_m,
                )
                self._publish_status(
                    "scanner retreat reached its endpoint before post-scan queue; "
                    f"User X moved {actual_retreat * 1000.0:.1f}mm; using blocking post-scan path"
                )
                return self._execute_post_scan_safe_place_blend(
                    approach_xyz,
                    fixed_place_xyz,
                    side_rx_delta_deg,
                    motion,
                )
            if not active:
                if time.monotonic() - command_submitted_at < command_start_grace_s:
                    time.sleep(LOOKAHEAD_GATE_POLL_S)
                    continue
                self.controller.wait_for_command(retreat_command, timeout_s=5.0)
                tcp_after = self._current_command_pose()
                actual_retreat = self._validate_scanner_retreat_displacement(
                    tcp_before,
                    tcp_after,
                    retreat_m,
                )
                self._publish_status(
                    "scanner retreat completed before the post-scan queue gate; "
                    f"User X moved {actual_retreat * 1000.0:.1f}mm; using blocking post-scan path"
                )
                return self._execute_post_scan_safe_place_blend(
                    approach_xyz,
                    fixed_place_xyz,
                    side_rx_delta_deg,
                    motion,
                )
            if time.monotonic() >= gate_deadline:
                self._stop_active_motion_for_queue_failure(
                    "scanner-retreat post-scan queue gate timeout"
                )
                raise TimeoutError(
                    "Timed out waiting to queue post-scan motion after scanner retreat: "
                    f"remaining={remaining_m * 1000.0:.1f}mm, "
                    f"gate={lead_m * 1000.0:.1f}mm"
                )
            time.sleep(LOOKAHEAD_GATE_POLL_S)

        self._require_cycle_active("at scanner-retreat post-scan queue gate")
        self._publish_status(
            "queueing post-scan safe-height PTP before scanner-retreat endpoint"
        )
        try:
            post_scan_completed = self._execute_post_scan_safe_place_blend(
                approach_xyz,
                fixed_place_xyz,
                side_rx_delta_deg,
                motion,
                allow_pending_queue=True,
            )
        except Exception as exc:
            # If the first queued PTP was rejected while the straight retreat
            # is still active, finish and validate that retreat, then retry the
            # post-scan helper through its normal blocking/CP path.
            if self.controller.motion_command_is_active(retreat_command):
                self.get_logger().warning(
                    "Controller rejected the scanner-retreat/post-scan queue while "
                    f"retreat was active; completing retreat then falling back: {exc}"
                )
                self.controller.wait_for_command(retreat_command, timeout_s=5.0)
                tcp_after = self._current_command_pose()
                self._validate_scanner_retreat_displacement(
                    tcp_before,
                    tcp_after,
                    retreat_m,
                )
                return self._execute_post_scan_safe_place_blend(
                    approach_xyz,
                    fixed_place_xyz,
                    side_rx_delta_deg,
                    motion,
                )
            raise

        # A CP queue normally makes the retreat command terminal as the safe
        # PTP starts.  If the firmware reports it as still active after the
        # post-scan helper returns, wait for that command before releasing the
        # gripper or entering any recovery path.
        if self.controller.motion_command_is_active(retreat_command):
            self.controller.wait_for_command(retreat_command, timeout_s=5.0)
        return post_scan_completed

    def _retreat_box_from_scanner(self, actual_approach_m: float) -> None:
        """扫码后保持姿态沿 User X- 退回，给后续旋转留出安全空间。"""
        extra_retreat_m = float(self.get_parameter("scanner_retreat_extra_m").value)
        retreat_m = self._scanner_retreat_distance(actual_approach_m)
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
        actual_retreat_m = self._validate_scanner_retreat_displacement(
            tcp_before,
            tcp_after,
            retreat_m,
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

    def _reset_barcode_search_travel(self) -> None:
        self.barcode_search_net_delta_deg = 0.0

    def _record_barcode_search_travel(self, delta_deg: float) -> None:
        self.barcode_search_net_delta_deg += float(delta_deg)

    def _current_stable_barcode(self) -> str:
        """无等待读取当前扫码窗口；用于决定是否跳过靠近和 J6 找码。"""
        required_hits = max(1, int(self.get_parameter("barcode_stable_hits").value))
        with self.barcode_lock:
            if self.barcode_window_active and self.barcode_hits >= required_hits:
                return self.barcode_value
        return ""

    def _barcode_after_transfer_grace(self) -> str:
        """Catch a decode that arrives just after the transfer motion completes."""
        value = self._current_stable_barcode()
        if value:
            return value

        grace_s = max(
            0.0,
            float(self.get_parameter("scanner_transfer_barcode_grace_s").value),
        )
        if grace_s <= 0.0:
            return ""

        required_hits = max(1, int(self.get_parameter("barcode_stable_hits").value))
        self._publish_status(
            f"waiting up to {grace_s * 1000.0:.0f}ms at transfer joint for barcode callback"
        )
        value = self._wait_for_current_barcode(
            required_hits,
            grace_s,
            "waiting for transfer-joint barcode callback",
        )
        if value:
            self._publish_status(
                f"barcode acquired during transfer grace window: {value}"
            )
        return value

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
        face_anchor_candidates: Optional[list[float]] = None,
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
            # In continuous mode the stop can occur anywhere in a 270° sweep,
            # so the nearest face may be any of the four 90° anchors, not only
            # the two anchors surrounding the current 90° segment.
            anchors = [
                float(anchor)
                for anchor in (face_anchor_candidates or [
                    previous_face_anchor_deg,
                    next_face_anchor_deg,
                ])
            ]
            snap_target_deg = min(anchors, key=lambda anchor: abs(stopped_joint_deg - anchor))
            anchor_index = anchors.index(snap_target_deg)
            if face_anchor_candidates:
                face_anchor = f"face-{anchor_index + 1}/{len(anchors)}"
            elif anchor_index == 0:
                face_anchor = "previous"
            else:
                face_anchor = "next"
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
        """Acquire a barcode using continuous or per-face J6 search.

        Continuous mode performs one monitored 270° sweep.  The scan window
        remains armed for the whole sweep, so a HID decode received at any
        intermediate angle stops J6 immediately and is then aligned to the
        nearest 90° face.  The legacy segmented implementation remains
        available as a field fallback when continuous motion is disabled or
        when the full sweep would violate the configured J6 safety limit.
        """
        self._reset_barcode_search_travel()
        if bool(self.get_parameter("barcode_continuous_rotation").value):
            return self._rotate_until_stable_barcode_continuous()
        return self._rotate_until_stable_barcode_segmented()

    def _rotate_until_stable_barcode_continuous(self) -> str:
        required_hits = max(1, int(self.get_parameter("barcode_stable_hits").value))
        max_faces = 4
        wait_s = max(0.02, float(self.get_parameter("barcode_face_wait_s").value))
        preferred_delta = float(self.get_parameter("barcode_flip_step_deg").value)
        sweep_delta = preferred_delta * float(max_faces - 1)
        try:
            with self.barcode_lock:
                window_active = self.barcode_window_active
            if not window_active:
                self._reset_barcode_window()

            # Keep the existing fast path: a code received during transfer or
            # while arriving at the scanner face avoids any J6 movement.
            value = self._wait_for_current_barcode(
                required_hits,
                wait_s,
                "checking barcode at transfer joint",
            )
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
            first_face_anchor_deg = float(anchor_joints[watch_index])
            face_anchors = [
                first_face_anchor_deg + float(index) * preferred_delta
                for index in range(max_faces)
            ]

            # Do not silently replace a 270° sweep with an equivalent 90°/630°
            # winding: that would leave one or more faces unscanned.  If the
            # exact sweep is outside the safe joint range, use the tested
            # segmented fallback instead.
            if not self._is_barcode_flip_joint_safe(anchor_joints, sweep_delta):
                self.get_logger().warning(
                    f"Continuous barcode sweep {sweep_delta:+.1f}deg is outside the "
                    "configured J6 safety range; falling back to segmented face search"
                )
                return self._rotate_until_stable_barcode_segmented()

            self._require_cycle_active("before continuous barcode face sweep")
            self._reset_barcode_window()
            self._publish_status(
                f"barcode continuous J{watch_index + 1} sweep: "
                f"{face_anchors[0]:.1f}->{face_anchors[-1]:.1f}deg "
                f"({sweep_delta:+.1f}deg across {max_faces - 1} faces); "
                "stop immediately on scan"
            )
            self._record_barcode_search_travel(sweep_delta)
            value = self._rotate_barcode_flip_joint(
                sweep_delta,
                max_faces - 1,
                required_hits,
                face_anchors[0],
                face_anchors[-1],
                face_anchor_candidates=face_anchors,
            )
            if value:
                return value

            # A decode can arrive during the controller's final deceleration
            # or just after the sweep reaches the last face.  Give it the same
            # short confirmation window as the segmented implementation and
            # snap to whichever standard face is actually closest.
            value = self._wait_for_current_barcode(
                required_hits,
                wait_s,
                "checking barcode after continuous J6 sweep",
            )
            if value:
                aligned_joints = self.controller.current_joint()
                if len(aligned_joints) != 6:
                    raise RuntimeError(
                        f"Current joint feedback must contain 6 values before barcode alignment, "
                        f"got {len(aligned_joints)}"
                    )
                stopped_joint_deg = float(aligned_joints[watch_index])
                snap_target_deg = min(
                    face_anchors,
                    key=lambda anchor: abs(stopped_joint_deg - anchor),
                )
                correction_deg = snap_target_deg - stopped_joint_deg
                if abs(correction_deg) > 0.2:
                    self._publish_status(
                        f"barcode acquired after continuous sweep; aligning J{watch_index + 1} "
                        f"{stopped_joint_deg:.1f}->{snap_target_deg:.1f}deg "
                        f"(correction {correction_deg:+.1f}deg)"
                    )
                    aligned_joints[watch_index] = snap_target_deg
                    motion = self._motion_profile()
                    self.controller.move_joint(
                        [float(joint) for joint in aligned_joints],
                        speed=motion["barcode_alignment_speed"],
                        accel=motion["barcode_alignment_acc"],
                    )
                    self._require_cycle_active(
                        "after final-face continuous barcode alignment"
                    )
                return value

            self.get_logger().warning(
                f"Barcode not stable after continuous {sweep_delta:+.1f}deg sweep; "
                "continuing without barcode"
            )
            return ""
        finally:
            with self.barcode_lock:
                self.barcode_window_active = False

    def _rotate_until_stable_barcode_segmented(self) -> str:
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
                self._record_barcode_search_travel(selected_delta)
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

    def _compose_user_target_pose(
        self,
        current: TcpPose,
        target_xyz: list[float],
        ry_delta_deg: float,
        rz_delta_deg: float,
        rx_delta_deg: float = 0.0,
    ) -> TcpPose:
        """Compose a User-axis rotation with a measured TCP pose."""

        if len(target_xyz) != 3:
            raise ValueError("target XYZ must contain 3 values")
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
        user_rx_rotation = SciPyRot.from_euler(
            "x", float(rx_delta_deg), degrees=True
        ).as_matrix()
        target_rotation = (
            user_rx_rotation
            @ user_rz_rotation
            @ user_ry_rotation
            @ start_rotation
        )
        target_rx, target_ry, target_rz = SciPyRot.from_matrix(
            target_rotation
        ).as_euler("xyz", degrees=True)
        return TcpPose(
            float(target_xyz[0]),
            float(target_xyz[1]),
            float(target_xyz[2]),
            float(target_rx),
            float(target_ry),
            float(target_rz),
        )

    def _move_to_user_xyz_with_rotation(
        self,
        target_xyz: list[float],
        ry_delta_deg: float,
        rz_delta_deg: float,
        rx_delta_deg: float = 0.0,
        linear_tcp: bool = False,
    ) -> None:
        """用一条 User/Tool PTP 同时完成 XYZ、User Ry/Rz/Rx 变化。

        姿态组合顺序严格按现场要求：先绕固定 User Y 轴旋转 ``Ry``，再绕
        固定 User Z 轴旋转 ``Rz``，最后绕固定 User X 轴旋转 ``Rx``。
        因此最终矩阵为 ``Rx @ Rz @ Ry @ R_start``。
        """
        if len(target_xyz) != 3:
            raise ValueError("scan_exit_user_xyz must contain 3 values")
        user_index = int(self.get_parameter("user_index").value)
        tool_index = int(self.get_parameter("command_tool_index").value)
        current = self._current_command_pose()
        target = self._compose_user_target_pose(
            current,
            target_xyz,
            ry_delta_deg=ry_delta_deg,
            rz_delta_deg=rz_delta_deg,
            rx_delta_deg=rx_delta_deg,
        )
        target_rotation = SciPyRot.from_euler(
            "xyz", [target.rx, target.ry, target.rz], degrees=True
        ).as_matrix()
        motion = self._motion_profile()
        combined_speed = motion["post_scan_speed"]
        rotation_text = (
            f"Ry={ry_delta_deg:+.1f}deg then Rz={rz_delta_deg:+.1f}deg"
        )
        if abs(float(rx_delta_deg)) > 1e-6:
            rotation_text += f" then Rx={rx_delta_deg:+.1f}deg"
        motion_name = "User MovL" if linear_tcp else "combined User PTP"
        self._publish_status(
            f"{motion_name}: XYZ=({target.x*1000:.0f},{target.y*1000:.0f},"
            f"{target.z*1000:.0f})mm, {rotation_text}, "
            f"target RPY=({target.rx:.1f},{target.ry:.1f},{target.rz:.1f})deg, "
            f"speed={combined_speed}%"
        )

        # 先验证包含位置和最终姿态的完整目标；无逆解时不会下发任何运动。
        self.controller.inverse_kinematics(
            target,
            user_index=user_index,
            tool_index=tool_index,
            joint_near=self.controller.current_joint(),
        )
        self._require_cycle_active("before combined post-scan PTP")
        move_command = (
            self.controller.move_linear_tcp
            if linear_tcp
            else self.controller.move_joint_tcp
        )
        move_command(
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

    def _rotate_face_up_about_user_y_jog(
        self,
        total_delta_deg: float,
        *,
        allow_shortfall: bool = False,
        minimum_progress_deg: float = 45.0,
    ) -> float:
        if abs(total_delta_deg) <= 1e-6:
            return 0.0
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
                    if allow_shortfall and directed_progress_deg >= max(
                        0.0, min(target_deg, float(minimum_progress_deg))
                    ):
                        self.get_logger().warning(
                            f"User-frame {axis_name} MoveJog timed out after a "
                            f"usable partial rotation: progress={directed_progress_deg:.1f}/"
                            f"{target_deg:.1f}deg"
                        )
                        break
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
            if allow_shortfall and final_progress_deg >= max(
                0.0, min(target_deg, float(minimum_progress_deg))
            ):
                self.get_logger().warning(
                    f"User-frame {axis_name} rotation reached only "
                    f"{final_progress_deg:.1f}/{target_deg:.1f}deg; accepting a "
                    "partial turn and compensating later"
                )
            else:
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
        return float(direction * final_progress_deg)

    def destroy_node(self):
        self.shutting_down = True
        self.running = False
        self.cycle_enabled = False
        self.turntable_scan_cancel.set()
        with self.turntable_condition:
            self.turntable_condition.notify_all()
        self.secondary_auto_resume_requested.clear()
        self.secondary_safety_shutdown.set()
        try:
            try:
                self._set_turntable_barcode_window(False)
                self._stop_turntable_if_running("node shutdown")
            except Exception as exc:
                self.get_logger().fatal(
                    f"Turntable shutdown request failed; use the hardware stop: {exc}"
                )
            scan_thread = self.turntable_scan_thread
            if scan_thread is not None and scan_thread is not threading.current_thread():
                scan_thread.join(timeout=6.0)
                if scan_thread.is_alive():
                    self.get_logger().warning(
                        "Turntable pre-scan thread did not exit before shutdown"
                    )
            safety_thread = self.secondary_safety_thread
            if safety_thread is not None and safety_thread is not threading.current_thread():
                safety_thread.join(timeout=6.0)
                if safety_thread.is_alive():
                    self.get_logger().warning(
                        "Secondary motion safety thread did not exit before shutdown"
                    )
            resume_thread = self.secondary_resume_thread
            if resume_thread is not None and resume_thread is not threading.current_thread():
                resume_thread.join(timeout=2.0)
                if resume_thread.is_alive():
                    self.get_logger().warning(
                        "Secondary automatic restart thread did not exit before shutdown"
                    )
            # Close Modbus without issuing DHGripper.disconnect(), which would
            # open the gripper and could drop an object during a fault shutdown.
            if self.gripper.modbus_index is not None and self.controller.dashboard is not None:
                try:
                    self.controller.dashboard.ModbusClose(self.gripper.modbus_index)
                finally:
                    self.gripper.modbus_index = None
            self._drop_secondary_safety_feedback()
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
        self.turntable_status_label = QLabel("-")
        self.cycle_status_label = QLabel("ready")
        self.cycle_status_label.setWordWrap(True)
        form.addRow("机械臂", QLabel("192.168.111.101（单臂）"))
        form.addRow("模式", self.robot_mode_label)
        form.addRow("当前关节", self.joint_feedback_label)
        form.addRow("Tool1 TCP", self.tcp_feedback_label)
        form.addRow("最新视觉", self.vision_feedback_label)
        form.addRow("转盘 / D435", self.turntable_status_label)
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
        box = QGroupBox("扫码后放置区用户坐标 PTP 目标 (mm)")
        box_layout = QVBoxLayout(box)
        grid = QGridLayout()
        box_layout.addLayout(grid)
        fields = []
        for index, (axis, value_m) in enumerate(zip(("X", "Y", "Z"), values_m)):
            grid.addWidget(QLabel(axis), 0, index)
            field = self._new_double(value_m * 1000.0, -2000.0, 2000.0, 1, 1.0)
            if axis == "Z":
                field.setToolTip(
                    "先以此安全高度移动到放置区，再执行侧面条码的 User Rx 倾斜"
                )
            grid.addWidget(field, 1, index)
            fields.append(field)
        self.pose_fields["scan_exit_user_xyz"] = fields
        surface_z_m = float(self.node.get_parameter("placement_surface_z_m").value)
        safety_margin_m = float(self.node.get_parameter("placement_safety_margin_m").value)
        self.placement_surface_z = self._new_double(
            surface_z_m * 1000.0, 0.0, 1000.0, 1, 1.0
        )
        self.placement_safety_margin = self._new_double(
            safety_margin_m * 1000.0, 0.0, 200.0, 1, 1.0
        )
        form = QFormLayout()
        form.addRow(
            "放置区上方安全高度 Z",
            QLabel("使用上方 XYZ 参数中的 Z 值"),
        )
        form.addRow("旧动态放置面 User Z mm（兼容参数）", self.placement_surface_z)
        form.addRow("旧动态放置安全余量 mm（兼容参数）", self.placement_safety_margin)
        form.addRow(
            "当前侧面条码放置 Z",
            QLabel("侧面条码固定放置位 Z=180 mm；不使用物料长度动态下降"),
        )
        box_layout.addLayout(form)
        layout.addWidget(box)

    def _build_motion_parameters(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("抓取与运动参数")
        form = QFormLayout(box)
        self.motion_speed_scale = QSpinBox(); self.motion_speed_scale.setRange(50, 400); self.motion_speed_scale.setSingleStep(5); self.motion_speed_scale.setValue(int(self.node.get_parameter("motion_speed_scale_percent").value))
        self.motion_command_cap = QSpinBox(); self.motion_command_cap.setRange(1, 100); self.motion_command_cap.setSingleStep(5); self.motion_command_cap.setValue(int(self.node.get_parameter("motion_command_cap_percent").value))
        self.joint_speed = QSpinBox(); self.joint_speed.setRange(1, 100); self.joint_speed.setValue(int(self.node.get_parameter("joint_speed").value))
        self.joint_acc = QSpinBox(); self.joint_acc.setRange(1, 100); self.joint_acc.setValue(int(self.node.get_parameter("joint_acc").value))
        self.grasp_lift_speed = QSpinBox(); self.grasp_lift_speed.setRange(1, 100); self.grasp_lift_speed.setValue(int(self.node.get_parameter("grasp_lift_speed_factor").value))
        self.grasp_lift_acc = QSpinBox(); self.grasp_lift_acc.setRange(1, 100); self.grasp_lift_acc.setValue(int(self.node.get_parameter("grasp_lift_acc_factor").value))
        self.grasp_lift_transfer_blend = QCheckBox(
            "非转盘流程：抬升接近终点时连续衔接中转位（CP）"
        )
        self.grasp_lift_transfer_blend.setChecked(
            bool(self.node.get_parameter("grasp_lift_transfer_blend_enabled").value)
        )
        if bool(self.node.get_parameter("turntable_enabled").value):
            self.grasp_lift_transfer_blend.setChecked(False)
            self.grasp_lift_transfer_blend.setEnabled(False)
        self.grasp_lift_transfer_cp = QSpinBox()
        self.grasp_lift_transfer_cp.setRange(1, 100)
        self.grasp_lift_transfer_cp.setValue(
            int(self.node.get_parameter("grasp_lift_transfer_blend_cp").value)
        )
        self.grasp_lift_transfer_queue_lead = self._new_double(
            float(
                self.node.get_parameter("grasp_lift_transfer_queue_lead_m").value
            )
            * 1000.0,
            0.0,
            50.0,
            1,
            1.0,
        )
        self.post_scan_place_blend = QCheckBox(
            "扫码后安全高度连续衔接固定放置位（CP）"
        )
        self.post_scan_place_blend.setChecked(
            bool(self.node.get_parameter("post_scan_place_blend_enabled").value)
        )
        self.post_scan_place_cp = QSpinBox()
        self.post_scan_place_cp.setRange(1, 100)
        self.post_scan_place_cp.setValue(
            int(self.node.get_parameter("post_scan_place_blend_cp").value)
        )
        self.post_scan_place_queue_lead = self._new_double(
            float(
                self.node.get_parameter("post_scan_place_queue_lead_m").value
            )
            * 1000.0,
            0.0,
            100.0,
            1,
            1.0,
        )
        self.scanner_retreat_post_scan_blend = QCheckBox(
            "扫码退让连续衔接扫码后移动（CP）"
        )
        self.scanner_retreat_post_scan_blend.setChecked(
            bool(
                self.node.get_parameter(
                    "scanner_retreat_post_scan_blend_enabled"
                ).value
            )
        )
        self.scanner_retreat_post_scan_cp = QSpinBox()
        self.scanner_retreat_post_scan_cp.setRange(1, 100)
        self.scanner_retreat_post_scan_cp.setValue(
            int(
                self.node.get_parameter(
                    "scanner_retreat_post_scan_blend_cp"
                ).value
            )
        )
        self.scanner_retreat_post_scan_queue_lead = self._new_double(
            float(
                self.node.get_parameter(
                    "scanner_retreat_post_scan_queue_lead_m"
                ).value
            )
            * 1000.0,
            0.0,
            100.0,
            1,
            1.0,
        )
        self.offset_high_descent_blend = QCheckBox(
            "偏置高位连续衔接下降（CP）"
        )
        self.offset_high_descent_blend.setChecked(
            bool(
                self.node.get_parameter(
                    "offset_high_descent_blend_enabled"
                ).value
            )
        )
        self.offset_high_descent_cp = QSpinBox()
        self.offset_high_descent_cp.setRange(1, 100)
        self.offset_high_descent_cp.setValue(
            int(
                self.node.get_parameter(
                    "offset_high_descent_blend_cp"
                ).value
            )
        )
        self.offset_high_descent_queue_lead = self._new_double(
            float(
                self.node.get_parameter(
                    "offset_high_descent_queue_lead_m"
                ).value
            )
            * 1000.0,
            0.0,
            100.0,
            1,
            1.0,
        )
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
        self.secondary_collision_check = QCheckBox("启用 102 机械臂 TCP Y 间距联动保护")
        self.secondary_collision_check.setChecked(
            bool(self.node.get_parameter("secondary_collision_check_enabled").value)
        )
        self.top_surface_barcode_enabled = QCheckBox(
            "启用目标顶面条形码检测（命中后跳过扫码器/J6/Ry/Rz）"
        )
        self.top_surface_barcode_enabled.setChecked(
            bool(self.node.get_parameter("top_surface_barcode_enabled").value)
        )
        self.grasp_z_offset = self._new_double(float(self.node.get_parameter("grasp_z_offset_m").value) * 1000.0, -20.0, 20.0, 1, 0.5)
        self.minimum_safe_tcp_z = self._new_double(float(self.node.get_parameter("minimum_safe_tcp_z_m").value) * 1000.0, -50.0, 100.0, 1, 0.5)
        self.grasp_lift = self._new_double(float(self.node.get_parameter("grasp_lift_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        self.place_release_clearance = self._new_double(float(self.node.get_parameter("place_release_clearance_m").value) * 1000.0, 1.0, 50.0, 1, 1.0)
        self.grasp_close_settle = self._new_double(float(self.node.get_parameter("grasp_close_settle_s").value), 0.0, 5.0, 2, 0.1)
        self.grasp_min_closure = self._new_double(
            float(self.node.get_parameter("grasp_min_closure_from_preshape_m").value)
            * 1000.0,
            0.0,
            30.0,
            1,
            0.5,
        )
        self.grasp_feedback_wait = self._new_double(float(self.node.get_parameter("grasp_feedback_wait_s").value), 0.1, 5.0, 2, 0.1)
        self.grasp_retry_limit = QSpinBox(); self.grasp_retry_limit.setRange(0, 20); self.grasp_retry_limit.setValue(int(self.node.get_parameter("single_cycle_grasp_retry_limit").value))
        self.feedback_required = QCheckBox("启用空抓检测与自动重试")
        self.feedback_required.setChecked(bool(self.node.get_parameter("grasp_feedback_required").value))
        self.effective_motion_label = QLabel()
        self.effective_motion_label.setWordWrap(True)
        self._refresh_effective_motion_label()
        form.addRow("统一提速比例 %（100=原有效速度）", self.motion_speed_scale)
        form.addRow("全流程速度/加速度安全上限 %", self.motion_command_cap)
        form.addRow("当前单指令有效值", self.effective_motion_label)
        form.addRow("关节速度兼容基准 %", self.joint_speed)
        form.addRow("关节加速度兼容基准 %", self.joint_acc)
        form.addRow("抓取后抬升速度兼容基准 %", self.grasp_lift_speed)
        form.addRow("抓取后抬升加速度兼容基准 %", self.grasp_lift_acc)
        form.addRow("抬升→中转连续队列", self.grasp_lift_transfer_blend)
        form.addRow("抬升→中转 CP 平滑比例 %", self.grasp_lift_transfer_cp)
        form.addRow("抬升终点前排队距离 mm", self.grasp_lift_transfer_queue_lead)
        form.addRow("安全高度→固定放置连续队列", self.post_scan_place_blend)
        form.addRow("安全高度→固定放置 CP 平滑比例 %", self.post_scan_place_cp)
        form.addRow("安全高度终点前排队距离 mm", self.post_scan_place_queue_lead)
        form.addRow(
            "扫码退让→扫码后移动连续队列",
            self.scanner_retreat_post_scan_blend,
        )
        form.addRow(
            "扫码退让→扫码后移动 CP 平滑比例 %",
            self.scanner_retreat_post_scan_cp,
        )
        form.addRow(
            "扫码退让终点前排队距离 mm",
            self.scanner_retreat_post_scan_queue_lead,
        )
        form.addRow("偏置高位→下降连续队列", self.offset_high_descent_blend)
        form.addRow("偏置高位→下降 CP 平滑比例 %", self.offset_high_descent_cp)
        form.addRow("偏置高位终点前排队距离 mm", self.offset_high_descent_queue_lead)
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
        form.addRow("102 联动保护", self.secondary_collision_check)
        form.addRow("抓取 Z 修正 mm（正值更浅）", self.grasp_z_offset)
        form.addRow("TCP 最低安全 Z mm", self.minimum_safe_tcp_z)
        form.addRow("抓取后抬升 mm", self.grasp_lift)
        form.addRow("放置释放额外开度 mm", self.place_release_clearance)
        form.addRow("闭合后原地确认秒", self.grasp_close_settle)
        form.addRow("有效抓取最小闭合量 mm", self.grasp_min_closure)
        form.addRow("运动状态反馈等待秒", self.grasp_feedback_wait)
        form.addRow("单轮空抓重试次数", self.grasp_retry_limit)
        form.addRow("夹爪反馈保护", self.feedback_required)
        form.addRow("顶面条码分支", self.top_surface_barcode_enabled)
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
        self.turntable_do_index = QSpinBox()
        self.turntable_do_index.setRange(1, 8)
        self.turntable_do_index.setValue(
            int(self.node.get_parameter("turntable_do_index").value)
        )
        self.turntable_pulse_ms = QSpinBox()
        self.turntable_pulse_ms.setRange(50, 5000)
        self.turntable_pulse_ms.setValue(
            int(self.node.get_parameter("turntable_pulse_ms").value)
        )
        self.turntable_pulse_ms.setSuffix(" ms")
        self.turntable_scan_timeout = self._new_double(
            float(self.node.get_parameter("turntable_scan_timeout_s").value),
            1.0,
            10.0,
            1,
            0.5,
        )
        self.turntable_settle = self._new_double(
            float(self.node.get_parameter("turntable_settle_s").value),
            0.0,
            10.0,
            2,
            0.1,
        )
        self.turntable_surface_z = self._new_double(
            float(self.node.get_parameter("turntable_surface_z_m").value) * 1000.0,
            -1000.0,
            1000.0,
            1,
            1.0,
        )
        self.vision_user_z_bias = self._new_double(
            float(self.node.get_parameter("vision_user_z_bias_m").value) * 1000.0,
            -50.0,
            50.0,
            1,
            0.1,
        )
        self.turntable_surface_tolerance = self._new_double(
            float(self.node.get_parameter("turntable_surface_tolerance_m").value)
            * 1000.0,
            1.0,
            100.0,
            1,
            1.0,
        )
        self.turntable_tcp_below_target = self._new_double(
            float(self.node.get_parameter("turntable_tcp_below_target_m").value)
            * 1000.0,
            0.0,
            300.0,
            1,
            1.0,
        )
        self.turntable_surface_clearance = self._new_double(
            float(self.node.get_parameter("turntable_surface_clearance_m").value)
            * 1000.0,
            0.0,
            100.0,
            1,
            1.0,
        )
        self.barcode_hits = QSpinBox(); self.barcode_hits.setRange(1, 20); self.barcode_hits.setValue(int(self.node.get_parameter("barcode_stable_hits").value))
        self.barcode_rotations = QSpinBox(); self.barcode_rotations.setRange(4, 4); self.barcode_rotations.setValue(4)
        self.barcode_wait = self._new_double(float(self.node.get_parameter("barcode_face_wait_s").value), 0.02, 10.0, 2, 0.01)
        self.scanner_transfer_barcode_grace = self._new_double(
            float(self.node.get_parameter("scanner_transfer_barcode_grace_s").value),
            0.0,
            0.5,
            2,
            0.01,
        )
        self.top_surface_barcode_wait = self._new_double(
            float(self.node.get_parameter("top_surface_barcode_wait_s").value),
            0.0,
            5.0,
            2,
            0.05,
        )
        self.barcode_continuous_rotation = QCheckBox("连续旋转 270°，扫码后吸附最近 90° 面")
        self.barcode_continuous_rotation.setChecked(
            bool(self.node.get_parameter("barcode_continuous_rotation").value)
        )
        # 三段扫码相关运动分别调速，避免修改普通“关节速度”时全部一起变化。
        self.scanner_approach_speed = QSpinBox(); self.scanner_approach_speed.setRange(1, 100); self.scanner_approach_speed.setValue(int(self.node.get_parameter("scanner_approach_speed_factor").value))
        self.scanner_retreat_speed = QSpinBox(); self.scanner_retreat_speed.setRange(1, 100); self.scanner_retreat_speed.setValue(int(self.node.get_parameter("scanner_retreat_speed_factor").value))
        self.scanner_retreat_acc = QSpinBox(); self.scanner_retreat_acc.setRange(1, 100); self.scanner_retreat_acc.setValue(int(self.node.get_parameter("scanner_retreat_acc_factor").value))
        self.barcode_j6_speed = QSpinBox(); self.barcode_j6_speed.setRange(1, 100); self.barcode_j6_speed.setValue(int(self.node.get_parameter("barcode_j6_speed_factor").value))
        self.barcode_alignment_acc = QSpinBox(); self.barcode_alignment_acc.setRange(1, 100); self.barcode_alignment_acc.setValue(int(self.node.get_parameter("barcode_alignment_acc_factor").value))
        self.barcode_rz = self._new_double(float(self.node.get_parameter("barcode_flip_step_deg").value), -180.0, 180.0, 1, 5.0)
        self.d435_side_face_reference = self._new_double(
            float(self.node.get_parameter("d435_side_face_reference_joint_deg").value),
            -355.0,
            355.0,
            1,
            1.0,
        )
        self.face_up_user_ry = self._new_double(float(self.node.get_parameter("face_up_user_ry_deg").value), -180.0, 180.0, 1, 5.0)
        self.post_scan_user_rz = self._new_double(float(self.node.get_parameter("post_scan_user_rz_deg").value), -180.0, 180.0, 1, 5.0)
        self.side_barcode_place_rx = self._new_double(
            float(self.node.get_parameter("side_barcode_place_rx_delta_deg").value),
            -180.0,
            180.0,
            1,
            1.0,
        )
        self.face_up_jog_tolerance = self._new_double(float(self.node.get_parameter("face_up_jog_tolerance_deg").value), 0.2, 10.0, 1, 0.2)
        self.bottom_barcode_recovery_enabled = QCheckBox(
            "无顶面/侧面条码时启用底面翻转恢复"
        )
        self.bottom_barcode_recovery_enabled.setChecked(
            bool(self.node.get_parameter("bottom_barcode_recovery_enabled").value)
        )
        self.bottom_flip_user_ry = self._new_double(
            float(self.node.get_parameter("bottom_flip_user_ry_target_deg").value),
            -180.0,
            -1.0,
            1,
            5.0,
        )
        self.bottom_flip_table_retract = self._new_double(
            float(self.node.get_parameter("bottom_flip_table_retract_m").value) * 1000.0,
            0.0,
            200.0,
            1,
            1.0,
        )
        self.scanner_center_distance = self._new_double(float(self.node.get_parameter("scanner_center_distance_m").value) * 1000.0, 1.0, 500.0, 1, 1.0)
        self.scanner_face_clearance = self._new_double(float(self.node.get_parameter("scanner_face_clearance_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        self.scanner_negative_tolerance = self._new_double(float(self.node.get_parameter("scanner_approach_negative_tolerance_m").value) * 1000.0, 0.0, 20.0, 1, 1.0)
        self.scanner_retreat_extra = self._new_double(float(self.node.get_parameter("scanner_retreat_extra_m").value) * 1000.0, 0.0, 200.0, 1, 1.0)
        self.scanner_natural_finish_margin = self._new_double(
            float(self.node.get_parameter("scanner_approach_natural_finish_margin_m").value)
            * 1000.0,
            0.0,
            100.0,
            1,
            1.0,
        )
        form.addRow("转盘控制 DO", self.turntable_do_index)
        form.addRow("转盘 0→1→0 脉冲", self.turntable_pulse_ms)
        form.addRow("D435 四周码超时秒（固定）", self.turntable_scan_timeout)
        form.addRow("转盘停止后停稳等待秒", self.turntable_settle)
        form.addRow("转盘表面 User Z mm（负值=未标定/禁止下降）", self.turntable_surface_z)
        form.addRow("V3 D405→User Z 标定补偿 mm", self.vision_user_z_bias)
        form.addRow("转盘表面视觉允许误差 mm", self.turntable_surface_tolerance)
        form.addRow("TCP 以下夹爪伸出量 mm", self.turntable_tcp_below_target)
        form.addRow("夹爪距转盘表面安全余量 mm", self.turntable_surface_clearance)
        form.addRow("旧 HID 同码稳定次数（兼容）", self.barcode_hits)
        form.addRow("连续找码模式", self.barcode_continuous_rotation)
        form.addRow("检查面数（分段模式固定）", self.barcode_rotations)
        form.addRow("每面等待秒", self.barcode_wait)
        form.addRow("中转位扫码回调宽限秒", self.scanner_transfer_barcode_grace)
        form.addRow("顶面条码悬停检测等待秒", self.top_surface_barcode_wait)
        form.addRow("中转 TCP 到扫码器距离 mm", self.scanner_center_distance)
        form.addRow("盒侧面扫码间隙 mm", self.scanner_face_clearance)
        form.addRow("负靠近量容差 mm", self.scanner_negative_tolerance)
        form.addRow("扫码近端自然完成余量 mm", self.scanner_natural_finish_margin)
        form.addRow("靠近扫码器 User X+ 速度 %", self.scanner_approach_speed)
        form.addRow("扫码后安全退让 User X- 速度 %", self.scanner_retreat_speed)
        form.addRow("扫码后安全退让 User X- 加速度 %", self.scanner_retreat_acc)
        form.addRow("回中转距离后额外退让 mm", self.scanner_retreat_extra)
        form.addRow("J6 多面找码/对齐速度 %", self.barcode_j6_speed)
        form.addRow("J6 标准面对齐加速度 %", self.barcode_alignment_acc)
        form.addRow("找码 J6 步进 deg（旧逻辑）", self.barcode_rz)
        form.addRow("D435 侧码最近 90° 基准 J6 deg", self.d435_side_face_reference)
        form.addRow("组合目标 User Ry 增量 deg", self.face_up_user_ry)
        form.addRow("组合目标 User Rz 增量 deg", self.post_scan_user_rz)
        form.addRow("四周条码固定放置 User Rx 增量 deg", self.side_barcode_place_rx)
        form.addRow("组合目标姿态到位容差 deg", self.face_up_jog_tolerance)
        form.addRow("底面条码翻转恢复", self.bottom_barcode_recovery_enabled)
        form.addRow("底面恢复 User Ry 目标 deg", self.bottom_flip_user_ry)
        form.addRow("底面恢复 User X- 额外退让 mm", self.bottom_flip_table_retract)
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
            ("夹爪重新标定（会运动）", self.node.recalibrate_gripper),
            ("夹爪关闭测试", self.node.close_gripper),
            ("只采样视觉", self.node.sample_vision_only),
            ("执行完整一轮", self.node.execute_single_cycle),
            ("确认转盘当前已停止", self.node.confirm_turntable_stopped),
            ("模拟右臂放料完成", self.node.notify_turntable_place_done),
        ]
        for index, (label, function) in enumerate(actions):
            button = QPushButton(label)
            if label == "立即停止":
                button.clicked.connect(lambda _checked=False, fn=function: self.run_priority_action(fn))
            elif label == "模拟右臂放料完成":
                # This signal must remain usable while a single-cycle worker
                # is waiting in WAIT_PLACE.
                button.clicked.connect(
                    lambda _checked=False, fn=function: fn()
                )
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
        next_row = (len(actions) + 1) // 2
        grid.addWidget(self.start_cycle_button, next_row, 0)
        grid.addWidget(self.stop_cycle_button, next_row, 1)
        self.d435_continuous_button = QPushButton("D435 连续检测：关闭")
        self.d435_continuous_button.setCheckable(True)
        continuous_enabled = bool(self.node.d435_continuous_detection)
        self.d435_continuous_button.setChecked(continuous_enabled)
        self.d435_continuous_button.setText(
            f"D435 连续检测：{'开启' if continuous_enabled else '关闭'}"
        )
        self.d435_continuous_button.setStyleSheet(
            "background:#2e7d32;color:white;font-weight:bold"
            if continuous_enabled
            else ""
        )
        self.d435_continuous_button.toggled.connect(
            self.toggle_d435_continuous_detection
        )
        grid.addWidget(self.d435_continuous_button, next_row + 1, 0, 1, 2)
        tip = QLabel(
            "转盘实机顺序：填写转盘表面Z → 应用参数 → 现场确认转盘停止并点击确认 "
            "→ 点击执行完整一轮或开始连续循环 → 右臂放料完成。"
            "模拟放料按钮仅用于单机联调。D435默认持续检测；放料完成后先检查"
            "静止物料当前面，已有条码就直接等待左臂抓取，未检测到才旋转转盘。"
        )
        tip.setWordWrap(True)
        grid.addWidget(tip, next_row + 2, 0, 1, 2)
        layout.addWidget(box)

    def apply_parameters(self) -> bool:
        if self.node.secondary_retreat_active.is_set():
            self.cycle_status_label.setText(
                "101 正在执行安全退让，参数将在退让完成后才能应用"
            )
            return False
        with self.node.turntable_lock:
            turntable_state = self.node.turntable_state
        if turntable_state not in ("STOPPED", "UNKNOWN"):
            self.cycle_status_label.setText(
                f"转盘状态为 {turntable_state}，停止后才能修改联动参数"
            )
            return False
        try:
            robot_mode = self.node.controller.robot_mode
        except Exception as exc:
            self.cycle_status_label.setText(f"无法确认机械臂状态，参数未应用: {exc}")
            return False
        if robot_mode in (7, 8, 10):
            self.cycle_status_label.setText(
                "101 正在运动或暂停，参数未应用；请先停止并等待机械臂空闲"
            )
            return False
        startup = [field.value() for field in self.joint_fields["startup_joint"]]
        transfer = [field.value() for field in self.joint_fields["transfer_joint"]]
        scan_xyz = [field.value() / 1000.0 for field in self.pose_fields["scan_exit_user_xyz"]]
        previous_turntable_do = int(
            self.node.get_parameter("turntable_do_index").value
        )
        parameters = [
            Parameter("startup_joint", value=startup),
            Parameter("transfer_joint", value=transfer),
            Parameter("scan_exit_user_xyz", value=scan_xyz),
            Parameter(
                "placement_surface_z_m",
                value=self.placement_surface_z.value() / 1000.0,
            ),
            Parameter(
                "placement_safety_margin_m",
                value=self.placement_safety_margin.value() / 1000.0,
            ),
            Parameter("motion_speed_scale_percent", value=self.motion_speed_scale.value()),
            Parameter("motion_command_cap_percent", value=self.motion_command_cap.value()),
            Parameter("joint_speed", value=self.joint_speed.value()),
            Parameter("joint_acc", value=self.joint_acc.value()),
            Parameter("grasp_lift_speed_factor", value=self.grasp_lift_speed.value()),
            Parameter("grasp_lift_acc_factor", value=self.grasp_lift_acc.value()),
            Parameter(
                "grasp_lift_transfer_blend_enabled",
                value=self.grasp_lift_transfer_blend.isChecked(),
            ),
            Parameter(
                "grasp_lift_transfer_blend_cp",
                value=self.grasp_lift_transfer_cp.value(),
            ),
            Parameter(
                "grasp_lift_transfer_queue_lead_m",
                value=self.grasp_lift_transfer_queue_lead.value() / 1000.0,
            ),
            Parameter(
                "post_scan_place_blend_enabled",
                value=self.post_scan_place_blend.isChecked(),
            ),
            Parameter(
                "post_scan_place_blend_cp",
                value=self.post_scan_place_cp.value(),
            ),
            Parameter(
                "post_scan_place_queue_lead_m",
                value=self.post_scan_place_queue_lead.value() / 1000.0,
            ),
            Parameter(
                "scanner_retreat_post_scan_blend_enabled",
                value=self.scanner_retreat_post_scan_blend.isChecked(),
            ),
            Parameter(
                "scanner_retreat_post_scan_blend_cp",
                value=self.scanner_retreat_post_scan_cp.value(),
            ),
            Parameter(
                "scanner_retreat_post_scan_queue_lead_m",
                value=self.scanner_retreat_post_scan_queue_lead.value() / 1000.0,
            ),
            Parameter(
                "offset_high_descent_blend_enabled",
                value=self.offset_high_descent_blend.isChecked(),
            ),
            Parameter(
                "offset_high_descent_blend_cp",
                value=self.offset_high_descent_cp.value(),
            ),
            Parameter(
                "offset_high_descent_queue_lead_m",
                value=self.offset_high_descent_queue_lead.value() / 1000.0,
            ),
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
            Parameter(
                "secondary_collision_check_enabled",
                value=self.secondary_collision_check.isChecked(),
            ),
            Parameter(
                "top_surface_barcode_enabled",
                value=self.top_surface_barcode_enabled.isChecked(),
            ),
            Parameter("grasp_z_offset_m", value=self.grasp_z_offset.value() / 1000.0),
            Parameter("minimum_safe_tcp_z_m", value=self.minimum_safe_tcp_z.value() / 1000.0),
            Parameter("grasp_lift_m", value=self.grasp_lift.value() / 1000.0),
            Parameter("place_release_clearance_m", value=self.place_release_clearance.value() / 1000.0),
            Parameter("grasp_close_settle_s", value=self.grasp_close_settle.value()),
            Parameter(
                "grasp_min_closure_from_preshape_m",
                value=self.grasp_min_closure.value() / 1000.0,
            ),
            Parameter("grasp_feedback_wait_s", value=self.grasp_feedback_wait.value()),
            Parameter("single_cycle_grasp_retry_limit", value=self.grasp_retry_limit.value()),
            Parameter("grasp_feedback_required", value=self.feedback_required.isChecked()),
            Parameter("turntable_do_index", value=self.turntable_do_index.value()),
            Parameter("turntable_pulse_ms", value=self.turntable_pulse_ms.value()),
            Parameter(
                "turntable_scan_timeout_s", value=self.turntable_scan_timeout.value()
            ),
            Parameter("turntable_settle_s", value=self.turntable_settle.value()),
            Parameter(
                "turntable_surface_z_m",
                value=self.turntable_surface_z.value() / 1000.0,
            ),
            Parameter(
                "vision_user_z_bias_m",
                value=self.vision_user_z_bias.value() / 1000.0,
            ),
            Parameter(
                "turntable_surface_tolerance_m",
                value=self.turntable_surface_tolerance.value() / 1000.0,
            ),
            Parameter(
                "turntable_tcp_below_target_m",
                value=self.turntable_tcp_below_target.value() / 1000.0,
            ),
            Parameter(
                "turntable_surface_clearance_m",
                value=self.turntable_surface_clearance.value() / 1000.0,
            ),
            Parameter("barcode_stable_hits", value=self.barcode_hits.value()),
            Parameter(
                "barcode_continuous_rotation",
                value=self.barcode_continuous_rotation.isChecked(),
            ),
            Parameter("barcode_max_face_rotations", value=self.barcode_rotations.value()),
            Parameter("barcode_face_wait_s", value=self.barcode_wait.value()),
            Parameter(
                "scanner_transfer_barcode_grace_s",
                value=self.scanner_transfer_barcode_grace.value(),
            ),
            Parameter(
                "top_surface_barcode_wait_s", value=self.top_surface_barcode_wait.value()
            ),
            Parameter("scanner_center_distance_m", value=self.scanner_center_distance.value() / 1000.0),
            Parameter("scanner_face_clearance_m", value=self.scanner_face_clearance.value() / 1000.0),
            Parameter("scanner_approach_negative_tolerance_m", value=self.scanner_negative_tolerance.value() / 1000.0),
            Parameter(
                "scanner_approach_natural_finish_margin_m",
                value=self.scanner_natural_finish_margin.value() / 1000.0,
            ),
            Parameter("scanner_approach_speed_factor", value=self.scanner_approach_speed.value()),
            Parameter("scanner_retreat_speed_factor", value=self.scanner_retreat_speed.value()),
            Parameter("scanner_retreat_acc_factor", value=self.scanner_retreat_acc.value()),
            Parameter("scanner_retreat_extra_m", value=self.scanner_retreat_extra.value() / 1000.0),
            Parameter("barcode_j6_speed_factor", value=self.barcode_j6_speed.value()),
            Parameter("barcode_alignment_acc_factor", value=self.barcode_alignment_acc.value()),
            Parameter("barcode_flip_step_deg", value=self.barcode_rz.value()),
            Parameter(
                "d435_side_face_reference_joint_deg",
                value=self.d435_side_face_reference.value(),
            ),
            Parameter("face_up_user_ry_deg", value=self.face_up_user_ry.value()),
            Parameter("post_scan_user_rz_deg", value=self.post_scan_user_rz.value()),
            Parameter(
                "side_barcode_place_rx_delta_deg",
                value=self.side_barcode_place_rx.value(),
            ),
            Parameter("face_up_jog_tolerance_deg", value=self.face_up_jog_tolerance.value()),
            Parameter(
                "bottom_barcode_recovery_enabled",
                value=self.bottom_barcode_recovery_enabled.isChecked(),
            ),
            Parameter(
                "bottom_flip_user_ry_target_deg",
                value=self.bottom_flip_user_ry.value(),
            ),
            Parameter(
                "bottom_flip_table_retract_m",
                value=self.bottom_flip_table_retract.value() / 1000.0,
            ),
        ]
        results = self.node.set_parameters(parameters)
        failures = [result.reason for result in results if not result.successful]
        if failures:
            self.cycle_status_label.setText("参数应用失败: " + "; ".join(failures))
            return False
        if self.turntable_do_index.value() != previous_turntable_do:
            with self.node.turntable_lock:
                self.node.turntable_state = "UNKNOWN"
            self.node._publish_status(
                "turntable DO index changed; physical stopped state must be "
                "confirmed again before the next pulse"
            )
        try:
            # AccJ/VelJ/AccL/VelL are controller-global settings and some Nova5
            # firmware rejects them with -1 while MoveJog is active.  Recheck
            # immediately before issuing them and turn a race into a clean GUI
            # refusal instead of an uncaught Qt callback traceback.
            if (
                self.node.secondary_retreat_active.is_set()
                or self.node.controller.robot_mode in (7, 8, 10)
            ):
                self.cycle_status_label.setText(
                    "101 在参数应用期间开始运动，控制器运动参数未写入；空闲后请重试"
                )
                return False
            self.node.controller.enable_single_command_motion_scaling()
        except Exception as exc:
            message = f"控制器运动参数应用失败: {exc}"
            self.cycle_status_label.setText(message)
            self.node.get_logger().warning(message)
            return False
        self.node._log_effective_motion_profile()
        self._refresh_effective_motion_label()
        self.node._publish_status("GUI parameters applied")
        return True

    def apply_and_run(self, function) -> None:
        if self.worker is not None and self.worker.isRunning():
            function_name = getattr(function, "__name__", "unknown")
            message = (
                "已有操作正在运行；本次请求已拒绝，参数未重新应用。"
                "请等待当前操作结束或点击立即停止"
            )
            self.cycle_status_label.setText(message)
            self.node._publish_status(
                f"GUI action {function_name} refused: another GUI action is still running; "
                "parameters were not reapplied"
            )
            return
        if not self.apply_parameters():
            return
        self.run_action(function)

    def apply_and_run_recovery(self, function) -> None:
        if self.recovery_worker is not None and self.recovery_worker.isRunning():
            self.cycle_status_label.setText("回初始位请求已经执行中，请等待机械臂恢复")
            return
        if self.node.secondary_retreat_active.is_set():
            self.cycle_status_label.setText(
                "101 正在自动退让到 200 mm，无需再次点击回初始位；请等待退让完成"
            )
            return
        if self.node.secondary_protective_stop_latched.is_set():
            protective_m, _, _ = self.node._secondary_interlock_distances()
            self.cycle_status_label.setText(
                "Y 间距保护仍处于闭锁状态；请等待间距恢复到 "
                f"{protective_m * 1000.0:.1f} mm 后再回初始位"
            )
            return
        ordinary_action_running = (
            (self.worker is not None and self.worker.isRunning())
            or (self.node.worker is not None and self.node.worker.is_alive())
            or self.node.controller.robot_mode in (7, 8, 10)
        )
        # Recovery is allowed to interrupt an ordinary action.  In that case
        # retain the already-applied profile: writing global AccJ/VelJ settings
        # while motion is active is rejected by the controller.
        if not ordinary_action_running and not self.apply_parameters():
            return
        self.run_recovery_action(function)

    def run_action(self, function) -> None:
        if self.worker is not None and self.worker.isRunning():
            function_name = getattr(function, "__name__", "unknown")
            message = "已有操作正在运行；请等待或点击立即停止"
            self.cycle_status_label.setText(message)
            self.node._publish_status(
                f"GUI action {function_name} refused: another GUI action is still running"
            )
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
            expected_stop = (
                "cancelled by Stop" in message
                or "cancelled by safety interlock" in message
                or "Y-clearance protection" in message
            )
            if expected_stop:
                self.cycle_status_label.setText("操作已由停止/安全联锁取消: " + message)
                self.node.get_logger().warning("GUI action cancelled: " + message)
            else:
                self.cycle_status_label.setText("操作失败: " + message)
                self.node.get_logger().error("GUI action failed: " + message)
        self.refresh_status()

    def start_continuous(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.cycle_status_label.setText("当前界面操作尚未完成，不能启动连续循环")
            return
        if not self.apply_parameters():
            return
        try:
            self.node.start_continuous_cycle()
        except Exception as exc:
            self.cycle_status_label.setText(f"连续循环启动失败: {exc}")

    def toggle_d435_continuous_detection(self, checked: bool) -> None:
        enabled = bool(checked)
        self.node.set_d435_continuous_detection(enabled)
        self.d435_continuous_button.setText(
            f"D435 连续检测：{'开启' if enabled else '关闭'}"
        )
        self.d435_continuous_button.setStyleSheet(
            "background:#2e7d32;color:white;font-weight:bold" if enabled else ""
        )

    def refresh_status(self) -> None:
        self.cycle_status_label.setText(self.node.last_status)
        with self.node.turntable_lock:
            turntable_state = self.node.turntable_state
            camera_ready = self.node.turntable_camera_ready
            continuous_detection = self.node.d435_continuous_detection
            continuous_value = self.node.d435_continuous_last_value
            scan_in_progress = self.node.turntable_scan_in_progress
            material_ready = self.node.turntable_material_ready
            ready_barcode = self.node.turntable_ready_barcode
            scan_error = self.node.turntable_scan_error
        material_state = (
            "扫码中"
            if scan_in_progress
            else "待左臂抓取并查顶面"
            if material_ready and not ready_barcode
            else "待左臂抓取"
            if material_ready
            else "扫码失败"
            if scan_error
            else "等待右臂放料"
        )
        self.turntable_status_label.setText(
            f"状态={turntable_state} | D435={'就绪' if camera_ready else '未就绪'} "
            f"| 连续检测={'开' if continuous_detection else '关'} "
            f"| 最近结果={continuous_value or '-'} "
            f"| 物料={material_state} "
            f"| 待抓条码={ready_barcode or '-'}"
        )
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
