"""D405 cosmetic-box fine localization for the single-arm pick/scan/place cell.

Differences from ``d405_local_ransac_coarse_lock_wait_lock_manual_roi_exposure_panel.py``:

* binds D405 serial 409122274792;
* does not subscribe to or wait for a coarse target; an optional persistent
  mouse-drawn ROI constrains which YOLO candidates may be selected;
* when YOLO sees several boxes, chooses the valid candidate whose segmented
  stereo point cloud has the smallest camera-optical X coordinate;
* requires side-depth evidence below the dominant top plane, estimates box
  height from it, and places the published grasp point 75% down from the top;
* publishes a short burst of stable poses, then keeps SAM2 display tracking
  active until the next ``/trigger_d405_vision`` request.
"""

import json
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rclpy
import torch
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from scipy.spatial.transform import Rotation as SciPyRot
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, String

from dobot_nova5_driver.handoff_clearance import (
    evaluate_camera_right_handoff_clearance,
)

SAM2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Camera/model parameters
CAMERA_SERIAL = "409122274792"
FFS_MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"
YOLO_MODEL_PATH = "/home/zdh/yolo_one/yolo_train_web_github/outputs/train/obb_demo-8/weights/best.pt"
VALID_ITERS = 6
MAX_DISP = 192
ZNEAR = 0.16
ZFAR = 5.0
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
IR_PROJECTOR_ON = True
# 触发后清除相机管线中可能滞留的旧帧。自动曝光在等待触发期间一直工作，
# 眼在手相机到达初始关节位后仍需丢弃少量运动尾帧；2 帧约为 67 ms，
# 可以避免使用回程中的图像，同时比原来的 15 帧减少约 0.43 秒等待（30 FPS）。
CAPTURE_FLUSH_FRAMES = 2
# 稳定目标发布后，继续用 SAM2/FFS 后台跟踪一小段时间，供机械臂到达
# 抓取上方后读取最新目标位姿。该跟踪不增加单独等待步骤。
PREGRASP_POSE_TOPIC = "/target_pose_cam_pregrasp"
PREGRASP_TRACKING_WINDOW_S = 5.0

# Cosmetic-box geometry parameters
MAX_PLANES = 3
MIN_PLANE_POINTS = 20
RANSAC_DISTANCE_M = 0.005
SIDE_START_DEPTH_M = 0.003
MIN_SIDE_EVIDENCE_POINTS = 20
MIN_EDGE_SIDE_EVIDENCE_POINTS = 12
EDGE_EVIDENCE_KERNEL_PX = 21
TABLE_RING_KERNEL_PX = 41
TABLE_EXCLUSION_KERNEL_PX = 7
MIN_TABLE_PLANE_POINTS = 30
TABLE_RANSAC_DISTANCE_M = 0.003
HEIGHT_PERCENTILE = 98.0
MIN_BOX_HEIGHT_M = 0.005
MAX_BOX_HEIGHT_M = 0.150
GRASP_DEPTH_RATIO = 0.75
GRIP_CLEARANCE_M = 0.020
MAX_GRIPPER_OPENING_M = 0.095
OBB_SMOOTH = 0.65
# 连续两帧有效几何结果用于机器人端稳定性检查；相比原来的三帧少一次
# FFS/SAM2/平面拟合，但不退化成不做跨帧验证的单帧抓取。
PUBLISH_FRAMES_BEFORE_RESET = 2
MAX_TRACKING_FRAMES_WITHOUT_HEIGHT = 18
YOLO_BOX_DISPLAY_TTL_S = 0.8

# 102 从 D405 图像右侧把盒子放到交接区并沿右侧退出。101 在发布抓取
# 目标前，使用完整 FFS 场景点云检查目标正上方和右边缘外的小型通道。
# 第一份锁定点云可用于快速初检，但必须再有一份 LIVE FFS 点云确认；
# 障碍消失后同样连续两份独立点云 CLEAR 才放行。
HANDOFF_CLEARANCE_ENABLED = True
HANDOFF_CLEARANCE_CLEAR_FRAMES = 2
HANDOFF_CLEARANCE_RIGHT_EXTENSION_M = 0.080
HANDOFF_CLEARANCE_SIDE_MARGIN_M = 0.030
HANDOFF_CLEARANCE_VERTICAL_GAP_M = 0.020
HANDOFF_CLEARANCE_CHECK_HEIGHT_M = 0.180
HANDOFF_CLEARANCE_VOXEL_SIZE_M = 0.015
HANDOFF_CLEARANCE_MIN_CLUSTER_POINTS = 30

# Camera/panel parameters
AUTO_EXPOSURE = True
MANUAL_EXPOSURE = 11000.0
MANUAL_GAIN = 8.0
AUTO_WHITE_BALANCE = True
# 机械臂回到初始位后的第一次视觉触发，立即保存两张抓取前 RGB 帧；
# 本次触发最终得到稳定抓取目标时，再保存两张稳定目标 RGB 帧。
# 同一初始位的连续无目标重试不会重复保存；完成一次抓取后下次回到初始位会重新保存。
# 设置为 False 时不保存，也不会自动创建 d405_output 文件夹。
SAVE_INITIAL_POSITION_FRAMES = True
INITIAL_POSITION_FRAME_COUNT = 2
STABLE_TARGET_FRAME_COUNT = 2
D405_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "d405_output")
D405_OUTPUT_IMAGE_SUFFIX = ".png"
# 机械臂节点发布最终 FAULT 或可恢复抓取故障状态时，保存一份当前 D405
# 组合可视化画面。
# 每个故障保存一张 <序号>.png 和一份同编号 <序号>.json；JSON 中包含故障
# 状态、失败阶段、周期计时、ROI、目标尺寸、点云及当前显示状态等分析信息。
# 设置为 False 时不创建 faults 文件夹，也不保存故障快照。
SAVE_FAULT_SNAPSHOTS = True
# 故障快照目录与 d405_output 同级，避免混入正常采集帧。
FAULT_SNAPSHOT_OUTPUT_DIR = os.path.join(
    os.path.dirname(D405_OUTPUT_DIR), "d405_fault_snapshots"
)
FAULT_SNAPSHOT_STATUS_TOPIC = "/cosmetic_pick_cycle_status"
FAULT_SNAPSHOT_TIMING_TOPIC = "/cosmetic_pick_cycle_timing"
FAULT_SNAPSHOT_IMAGE_SUFFIX = ".png"
# 等待同一故障的 timing summary 和故障 status 都到达后再写盘，保证
# JSON 尽量包含完整的 cycle_summary；不影响机械臂动作线程。
FAULT_SNAPSHOT_METADATA_WAIT_S = 0.10
# 可恢复抓取错误的状态消息和 cycle_summary 之间可能包含一次回初始位运动；
# 用较宽的窗口把它们合并成同一份故障快照。
FAULT_SNAPSHOT_EVENT_MERGE_WINDOW_S = 10.0
MASK_ALPHA = 0.5
MASK_COLOR_BGR = np.array([75, 70, 203], dtype=np.uint8)
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.float64) / 255.0
PANEL_JPEG_QUALITY = 80
CLOUD_VIEW_W = 720
CLOUD_VIEW_H = 540
ENABLE_LOCAL_WINDOWS = True
LOCAL_WINDOW_NAME = "D405 Cosmetic Box RGB + Point Cloud"
LOCAL_DIVIDER_WIDTH_PX = 4
MIN_ROI_SIZE_PX = 20

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_grad_enabled(False)

rclpy.init()
ros_node = rclpy.create_node("d405_cosmetic_box_height75_node")
executor = SingleThreadedExecutor()
executor.add_node(ros_node)


def spin_ros() -> None:
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


spin_thread = threading.Thread(target=spin_ros, daemon=True)
spin_thread.start()

pose_pub = ros_node.create_publisher(PoseStamped, "/target_pose_cam_fine", 10)
# 点云包围盒短边加开爪余量，供 DH 夹爪预张开使用。
width_pub = ros_node.create_publisher(Float32, "/gripper_target_width", 10)
# 点云包围盒长边的真实尺寸（不加夹爪余量），供扫码靠近距离计算使用。
length_pub = ros_node.create_publisher(Float32, "/cosmetic_box_length", 10)
# 顶面与桌面之间的盒子高度，供 75% 下爪深度计算使用。
height_pub = ros_node.create_publisher(Float32, "/cosmetic_box_height", 10)
# 每次触发的明确成功/失败结果。机械臂据此在 YOLO/ROI 已明确失败时
# 立即结束等待，而不是固定等满 vision_timeout_s。
vision_result_pub = ros_node.create_publisher(String, "/d405_vision_result", 10)
handoff_clear_pub = ros_node.create_publisher(Bool, "/d405_handoff_zone_clear", 10)
handoff_state_pub = ros_node.create_publisher(String, "/d405_handoff_zone_state", 10)
rgb_pub = ros_node.create_publisher(CompressedImage, "/vision_panel/d405_local_rgb/image/compressed", 1)
cloud_pub = ros_node.create_publisher(CompressedImage, "/vision_panel/d405_local_cloud/image/compressed", 1)
# 稳定目标后的后台预抓取跟踪位姿；与初始稳定采样话题分开，避免污染
# 机器人端下一次视觉请求的 2 帧稳定性计数。
pregrasp_pose_pub = ros_node.create_publisher(PoseStamped, PREGRASP_POSE_TOPIC, 10)

trigger_requested = False
quit_requested = False
reset_requested = False
clear_target_display_requested = False
initial_position_frames_saved_for_startup = False
stable_target_frames = deque(maxlen=STABLE_TARGET_FRAME_COUNT)
manual_roi = None
roi_drawing = False
roi_start = (-1, -1)
roi_end = (-1, -1)
roi_lock = threading.Lock()
vision_request_active = False

next_initial_position_frame_index = 1
initial_position_frame_output_ready = False
next_fault_snapshot_index = 1
fault_snapshot_output_ready = False
fault_snapshot_lock = threading.Lock()
latest_cycle_status = ""
latest_cycle_timing = None
latest_cycle_timing_received_at = 0.0
pending_fault_snapshots = deque(maxlen=32)
saved_fault_cycle_ids = set()
latest_barcode = ""
latest_barcode_received_at = 0.0
latest_panel_image = None
latest_panel_info = None
latest_panel_captured_at_epoch = 0.0


def initialize_initial_position_frame_output() -> None:
    """Create the output directory and continue numbering existing images."""

    global next_initial_position_frame_index, initial_position_frame_output_ready
    if not SAVE_INITIAL_POSITION_FRAMES:
        return

    try:
        os.makedirs(D405_OUTPUT_DIR, exist_ok=True)
        largest_index = 0
        image_name_pattern = re.compile(r"^(\d+)\.(?:png|jpg|jpeg|bmp)$", re.IGNORECASE)
        for entry in os.scandir(D405_OUTPUT_DIR):
            if not entry.is_file():
                continue
            match = image_name_pattern.fullmatch(entry.name)
            if match is not None:
                largest_index = max(largest_index, int(match.group(1)))
        next_initial_position_frame_index = largest_index + 1
        initial_position_frame_output_ready = True
        logging.info(
            f"[D405] Initial-position frame recording enabled: "
            f"directory={D405_OUTPUT_DIR}, next={next_initial_position_frame_index:04d}"
        )
    except OSError as exc:
        initial_position_frame_output_ready = False
        logging.error(
            f"[D405] Cannot initialize initial-position frame directory "
            f"{D405_OUTPUT_DIR}: {exc}"
        )


def save_d405_output_frame(color_bgr: np.ndarray, stage: str) -> None:
    """Save one raw RGB-camera frame and continue the global sequence."""

    global next_initial_position_frame_index
    if not SAVE_INITIAL_POSITION_FRAMES:
        return
    if not initial_position_frame_output_ready:
        initialize_initial_position_frame_output()
    if not initial_position_frame_output_ready:
        return

    image_index = next_initial_position_frame_index
    while True:
        filename = f"{image_index:04d}{D405_OUTPUT_IMAGE_SUFFIX}"
        output_path = os.path.join(D405_OUTPUT_DIR, filename)
        if not os.path.exists(output_path):
            break
        image_index += 1

    try:
        if not cv2.imwrite(output_path, np.ascontiguousarray(color_bgr)):
            logging.error(f"[D405] Failed to save {stage} frame: {output_path}")
            return
    except (OSError, cv2.error) as exc:
        logging.error(f"[D405] Failed to save {stage} frame {output_path}: {exc}")
        return

    next_initial_position_frame_index = image_index + 1
    logging.info(f"[D405] Saved {stage} frame: {output_path}")


def save_d405_output_frames(frames, stage: str, expected_count: int) -> None:
    """Save exactly the requested number of frames from one capture stage."""

    if not SAVE_INITIAL_POSITION_FRAMES:
        return
    frames_to_save = list(frames)[-expected_count:]
    if len(frames_to_save) != expected_count:
        logging.warning(
            f"[D405] {stage} capture contained {len(frames_to_save)} frame(s); "
            f"expected {expected_count}."
        )
    for frame in frames_to_save:
        save_d405_output_frame(frame, stage)


def initialize_fault_snapshot_output() -> None:
    """Create the fault directory and continue its independent numbering."""

    global next_fault_snapshot_index, fault_snapshot_output_ready
    if not SAVE_FAULT_SNAPSHOTS:
        return

    try:
        os.makedirs(FAULT_SNAPSHOT_OUTPUT_DIR, exist_ok=True)
        largest_index = 0
        image_name_pattern = re.compile(r"^(\d+)\.png$", re.IGNORECASE)
        for entry in os.scandir(FAULT_SNAPSHOT_OUTPUT_DIR):
            if not entry.is_file():
                continue
            match = image_name_pattern.fullmatch(entry.name)
            if match is not None:
                largest_index = max(largest_index, int(match.group(1)))
        next_fault_snapshot_index = largest_index + 1
        fault_snapshot_output_ready = True
        logging.info(
            f"[D405] Fault snapshot recording enabled: "
            f"directory={FAULT_SNAPSHOT_OUTPUT_DIR}, next={next_fault_snapshot_index:04d}"
        )
    except OSError as exc:
        fault_snapshot_output_ready = False
        logging.error(
            f"[D405] Cannot initialize fault snapshot directory "
            f"{FAULT_SNAPSHOT_OUTPUT_DIR}: {exc}"
        )


def json_safe(value):
    """Convert NumPy/scalar values into JSON-safe diagnostic values."""

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def timing_cycle_id(timing_payload) -> str:
    if not isinstance(timing_payload, dict):
        return ""
    value = timing_payload.get("cycle_id", "")
    return "" if value is None else str(value)


def is_fault_status(status: str) -> bool:
    """Recognize fatal and recoverable grasp-failure status messages."""

    normalized = str(status or "").strip().lower()
    return (
        normalized.startswith("fault:")
        or "recoverable grasp failure" in normalized
        or "grasp retry" in normalized
    )


def is_fault_timing(timing_payload) -> bool:
    """Recognize final faults and recoverable grasp retries in cycle timing."""

    if not isinstance(timing_payload, dict):
        return False
    return (
        timing_payload.get("event") == "cycle_summary"
        and timing_payload.get("outcome") in ("fault", "grasp_retry")
    )


def queue_fault_snapshot(*, status: str = "", timing_payload=None, source: str = "") -> None:
    """Queue one robot grasp failure for saving by the camera/render thread."""

    if not SAVE_FAULT_SNAPSHOTS:
        return
    now = time.time()
    timing_copy = json_safe(timing_payload) if isinstance(timing_payload, dict) else None
    cycle_id = timing_cycle_id(timing_copy)
    status_text = str(status or "")
    with fault_snapshot_lock:
        snapshot_image = None if latest_panel_image is None else latest_panel_image.copy()
        snapshot_info = None if latest_panel_info is None else json_safe(latest_panel_info)
        snapshot_captured_at = float(latest_panel_captured_at_epoch or 0.0)
        # The timing summary and the status message describe the same failure.
        # Merge them instead of saving two identical screenshots.
        for event in pending_fault_snapshots:
            event_cycle_id = timing_cycle_id(event.get("timing"))
            same_cycle = bool(cycle_id and event_cycle_id == cycle_id)
            recent_unidentified = (
                not event_cycle_id
                and now - float(event.get("queued_at", now))
                <= FAULT_SNAPSHOT_EVENT_MERGE_WINDOW_S
            )
            if not (same_cycle or recent_unidentified):
                continue
            if status_text:
                event["status"] = status_text
            if timing_copy is not None:
                event["timing"] = timing_copy
            if snapshot_image is not None:
                event["panel_image"] = snapshot_image
                event["panel_info"] = snapshot_info
                event["panel_captured_at_epoch"] = snapshot_captured_at
            if source:
                event.setdefault("sources", []).append(str(source))
            event["ready_at"] = now + FAULT_SNAPSHOT_METADATA_WAIT_S
            return

        if cycle_id and cycle_id in saved_fault_cycle_ids:
            return
        pending_fault_snapshots.append(
            {
                "queued_at": now,
                "ready_at": now + FAULT_SNAPSHOT_METADATA_WAIT_S,
                "status": status_text,
                "timing": timing_copy,
                "sources": [str(source)] if source else [],
                "panel_image": snapshot_image,
                "panel_info": snapshot_info,
                "panel_captured_at_epoch": snapshot_captured_at,
            }
        )


def cycle_status_callback(msg: String) -> None:
    """Receive robot status and arm snapshots for all grasp failures."""

    global latest_cycle_status
    status = str(msg.data)
    with fault_snapshot_lock:
        latest_cycle_status = status
        timing_payload = (
            None
            if not isinstance(latest_cycle_timing, dict)
            or not is_fault_timing(latest_cycle_timing)
            else dict(latest_cycle_timing)
        )
    if is_fault_status(status):
        queue_fault_snapshot(status=status, timing_payload=timing_payload, source="status")


def cycle_timing_callback(msg: String) -> None:
    """Keep the latest machine-readable cycle timing and detect fault summaries."""

    global latest_cycle_timing, latest_cycle_timing_received_at
    try:
        payload = json.loads(msg.data)
    except (TypeError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    received_at = time.time()
    with fault_snapshot_lock:
        latest_cycle_timing = json_safe(payload)
        latest_cycle_timing_received_at = received_at
        status = latest_cycle_status if is_fault_status(latest_cycle_status) else ""
    if is_fault_timing(payload):
        queue_fault_snapshot(status=status, timing_payload=payload, source="timing")


def barcode_callback(msg: String) -> None:
    """Keep the latest scanner value for the fault JSON diagnostics."""

    global latest_barcode, latest_barcode_received_at
    latest_barcode = str(msg.data)
    latest_barcode_received_at = time.time()


def build_fault_panel_info(
    *,
    display_status: str,
    fps: float,
    exposure_status: str,
    active_roi,
    cloud_source: str,
    frame_stamp,
    point_count: int,
    tracked_target_count: int,
) -> dict:
    """Collect every value represented by the RGB/point-cloud panel."""

    with fault_snapshot_lock:
        robot_status = latest_cycle_status
        robot_timing = None if latest_cycle_timing is None else json_safe(latest_cycle_timing)
        robot_timing_received_at = latest_cycle_timing_received_at

    frame_stamp_ns = int(frame_stamp.sec) * 1_000_000_000 + int(frame_stamp.nanosec)
    target_corners = None if locked_target_corners is None else json_safe(locked_target_corners)
    yolo_boxes = None if last_yolo_obbs is None else json_safe(last_yolo_obbs)
    roi_value = None if active_roi is None else list(active_roi)
    barcode_value = latest_barcode
    return json_safe(
        {
            "captured_at_epoch": time.time(),
            "panel": {
                "status": display_status,
                "fps": float(fps),
                "exposure": exposure_status,
                "roi": roi_value,
                "cloud_source": cloud_source,
                "point_count": int(point_count),
                "sam_initialized": bool(sam_initialized),
                "sam_tracked_target_points": int(tracked_target_count),
                "handoff_state": str(last_handoff_state),
                "handoff_clearance_passed": bool(handoff_clearance_passed),
                "handoff_candidate_points": int(last_handoff_candidate_points),
                "handoff_cluster_points": int(last_handoff_cluster_points),
                "handoff_clear_streak": int(handoff_clear_streak),
                "handoff_required_clear_frames": int(HANDOFF_CLEARANCE_CLEAR_FRAMES),
                "height_m": last_height_m,
                "length_m": None if smooth_extent is None else smooth_extent[0],
                "side_planes": int(last_side_planes),
                "side_points": int(last_side_points),
                "height_evidence": str(last_height_evidence_mode),
                "target_ready": bool(locked_target_ready),
                "target_id": int(locked_target_id),
                "target_camera_x_m": locked_target_camera_x,
                "target_height_m": locked_target_height_m,
                "target_corners_px": target_corners,
                "yolo_boxes": yolo_boxes,
                "smooth_grasp_m": smooth_grasp,
                "smooth_box_center_m": smooth_box_center,
                "smooth_extent_m": smooth_extent,
                "smooth_rotation": smooth_rotation,
            },
            "camera": {
                "serial": CAMERA_SERIAL,
                "resolution": [IMG_WIDTH, IMG_HEIGHT],
                "pcd_stride": int(PCD_STRIDE),
                "frame_stamp_ns": frame_stamp_ns,
            },
            "scanner": {
                "latest_barcode": barcode_value,
                "latest_barcode_received_at_epoch": latest_barcode_received_at or None,
            },
            "robot": {
                "latest_status": robot_status,
                "latest_timing": robot_timing,
                "latest_timing_received_at_epoch": robot_timing_received_at or None,
            },
            "configuration": {
                "save_fault_snapshots": bool(SAVE_FAULT_SNAPSHOTS),
                "fault_snapshot_output_dir": FAULT_SNAPSHOT_OUTPUT_DIR,
                "grasp_depth_ratio": float(GRASP_DEPTH_RATIO),
                "pregrasp_tracking_window_s": float(PREGRASP_TRACKING_WINDOW_S),
                "handoff_clearance_enabled": bool(HANDOFF_CLEARANCE_ENABLED),
            },
        }
    )


def add_fault_footer(panel_image: np.ndarray, image_index: int, event: dict, panel_info: dict) -> np.ndarray:
    """Append a readable fault banner without hiding the original panel."""

    footer_height = 94
    output = np.full(
        (panel_image.shape[0] + footer_height, panel_image.shape[1], 3),
        18,
        dtype=np.uint8,
    )
    output[: panel_image.shape[0]] = panel_image
    cv2.rectangle(
        output,
        (0, panel_image.shape[0]),
        (panel_image.shape[1] - 1, output.shape[0] - 1),
        (0, 0, 150),
        -1,
    )
    timing_payload = event.get("timing") if isinstance(event.get("timing"), dict) else {}
    stages = timing_payload.get("stages") if isinstance(timing_payload.get("stages"), list) else []
    last_stage = stages[-1].get("stage", "unknown") if stages and isinstance(stages[-1], dict) else "unknown"
    status = str(event.get("status") or panel_info.get("robot", {}).get("latest_status") or "FAULT")
    cycle_id = str(timing_payload.get("cycle_id", "unknown"))
    outcome = str(timing_payload.get("outcome", "fault"))
    lines = (
        f"FAULT SNAPSHOT {image_index:04d} | {status}",
        f"cycle={cycle_id} outcome={outcome} failed_stage={last_stage}",
        f"saved={datetime.now().astimezone().isoformat(timespec='milliseconds')}",
    )
    for line_index, line in enumerate(lines):
        max_chars = max(20, output.shape[1] // 10)
        if len(line) > max_chars:
            line = line[: max_chars - 3] + "..."
        cv2.putText(
            output,
            line,
            (14, panel_image.shape[0] + 25 + line_index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 if line_index == 0 else 0.50,
            (255, 255, 255),
            2 if line_index == 0 else 1,
            cv2.LINE_AA,
        )
    return output


def save_pending_fault_snapshots(panel_image: np.ndarray, panel_info: dict) -> None:
    """Write queued fault panels once the current render frame is complete."""

    global next_fault_snapshot_index
    if not SAVE_FAULT_SNAPSHOTS:
        return
    now = time.time()
    with fault_snapshot_lock:
        ready_events = []
        remaining_events = deque(maxlen=pending_fault_snapshots.maxlen)
        for event in pending_fault_snapshots:
            if float(event.get("ready_at", 0.0)) <= now:
                ready_events.append(dict(event))
            else:
                remaining_events.append(event)
        pending_fault_snapshots.clear()
        pending_fault_snapshots.extend(remaining_events)

    if not ready_events:
        return
    if not fault_snapshot_output_ready:
        initialize_fault_snapshot_output()
    if not fault_snapshot_output_ready:
        return

    for event in ready_events:
        timing_payload = event.get("timing")
        cycle_id = timing_cycle_id(timing_payload)
        if cycle_id:
            with fault_snapshot_lock:
                if cycle_id in saved_fault_cycle_ids:
                    continue
                saved_fault_cycle_ids.add(cycle_id)

        image_index = next_fault_snapshot_index
        while True:
            image_name = f"{image_index:04d}{FAULT_SNAPSHOT_IMAGE_SUFFIX}"
            json_name = f"{image_index:04d}.json"
            image_path = os.path.join(FAULT_SNAPSHOT_OUTPUT_DIR, image_name)
            json_path = os.path.join(FAULT_SNAPSHOT_OUTPUT_DIR, json_name)
            if not os.path.exists(image_path) and not os.path.exists(json_path):
                break
            image_index += 1

        event_panel_image = event.get("panel_image")
        if not isinstance(event_panel_image, np.ndarray):
            event_panel_image = panel_image
        event_panel_info = event.get("panel_info")
        if not isinstance(event_panel_info, dict):
            event_panel_info = panel_info
        fault_image = add_fault_footer(event_panel_image, image_index, event, event_panel_info)
        event_metadata = {
            key: value
            for key, value in event.items()
            if key not in ("panel_image", "panel_info")
        }
        try:
            if not cv2.imwrite(image_path, np.ascontiguousarray(fault_image)):
                logging.error(f"[D405] Failed to save fault snapshot image: {image_path}")
                continue
            record = {
                "event": "fault_snapshot",
                "snapshot_index": image_index,
                "saved_at_local": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "saved_at_epoch": time.time(),
                "image_file": image_name,
                "fault": json_safe(event_metadata),
                "panel_info": json_safe(event_panel_info),
                "panel_frame_age_s": round(
                    max(
                        0.0,
                        time.time() - float(event.get("panel_captured_at_epoch", 0.0)),
                    ),
                    4,
                )
                if float(event.get("panel_captured_at_epoch", 0.0)) > 0.0
                else None,
            }
            with open(json_path, "w", encoding="utf-8") as info_file:
                json.dump(record, info_file, ensure_ascii=False, indent=2, allow_nan=False)
        except (OSError, cv2.error, TypeError, ValueError) as exc:
            logging.error(f"[D405] Failed to save fault snapshot {image_index:04d}: {exc}")
            continue

        next_fault_snapshot_index = image_index + 1
        logging.info(
            f"[D405] Saved fault snapshot: image={image_path}, info={json_path}"
        )


def publish_vision_result(success: bool, reason: str) -> None:
    """每个触发请求最多发布一次最终结果。"""
    global vision_request_active
    if not vision_request_active:
        return
    if not success:
        # 同一初始位的失败重试不清除已保存的初始位图片；下一次触发
        # 仍属于同一轮等待，直到稳定目标成功或机械臂重新回到初始位。
        stable_target_frames.clear()
    message = String()
    message.data = f"{'success' if success else 'failure'}:{reason}"
    vision_result_pub.publish(message)
    vision_request_active = False


def publish_handoff_clearance(
    state: str,
    *,
    clear: bool,
    candidate_points: int = 0,
    largest_cluster_points: int = 0,
    clear_streak: int = 0,
) -> None:
    """Publish a fail-closed handoff-zone state for monitoring and logging."""

    clear_message = Bool()
    clear_message.data = bool(clear)
    handoff_clear_pub.publish(clear_message)
    state_message = String()
    state_message.data = json.dumps(
        {
            "state": str(state),
            "clear": bool(clear),
            "candidate_points": int(candidate_points),
            "largest_cluster_points": int(largest_cluster_points),
            "clear_streak": int(clear_streak),
            "required_clear_frames": int(HANDOFF_CLEARANCE_CLEAR_FRAMES),
            "stamp_ns": int(ros_node.get_clock().now().nanoseconds),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    handoff_state_pub.publish(state_message)


def publish_pregrasp_pose(frame_stamp) -> None:
    """Publish the current-frame grasp pose during the pre-grasp window."""

    if current_frame_grasp is None or current_frame_rotation is None:
        return
    pose_msg = PoseStamped()
    # The robot may receive this message after FFS/SAM2 processing and after
    # it has already moved.  Preserve the stamp of the frame that produced
    # current_frame_grasp so the driver can use the matching historical TCP pose.
    pose_msg.header.stamp = frame_stamp
    pose_msg.header.frame_id = "camera_d405_link"
    pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = map(
        float, current_frame_grasp
    )
    quat = SciPyRot.from_matrix(current_frame_rotation).as_quat()
    pose_msg.pose.orientation.x = float(quat[0])
    pose_msg.pose.orientation.y = float(quat[1])
    pose_msg.pose.orientation.z = float(quat[2])
    pose_msg.pose.orientation.w = float(quat[3])
    pregrasp_pose_pub.publish(pose_msg)


def trigger_callback(msg: Bool) -> None:
    global trigger_requested
    if msg.data:
        trigger_requested = True
        logging.info("[D405] New cosmetic-box detection requested.")


def clamp_image_point(x: int, y: int) -> tuple[int, int]:
    return (
        max(0, min(IMG_WIDTH - 1, int(x))),
        max(0, min(IMG_HEIGHT - 1, int(y))),
    )


def normalize_manual_roi(start, end):
    x0, y0 = clamp_image_point(*start)
    x1, y1 = clamp_image_point(*end)
    left, right = min(x0, x1), max(x0, x1) + 1
    top, bottom = min(y0, y1), max(y0, y1) + 1
    if right - left <= MIN_ROI_SIZE_PX or bottom - top <= MIN_ROI_SIZE_PX:
        return None
    return left, top, right, bottom


def get_manual_roi():
    with roi_lock:
        return None if manual_roi is None else tuple(manual_roi)


def clear_manual_roi() -> None:
    global manual_roi, roi_drawing
    with roi_lock:
        had_roi = manual_roi is not None or roi_drawing
        manual_roi = None
        roi_drawing = False
    if had_roi:
        logging.info("[ROI] Cleared; full-frame detection restored.")


def mouse_callback(event, x, y, flags, param) -> None:
    del flags, param
    global manual_roi, roi_drawing, roi_start, roi_end
    point = clamp_image_point(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        with roi_lock:
            roi_drawing = True
            roi_start = point
            roi_end = point
    elif event == cv2.EVENT_MOUSEMOVE:
        with roi_lock:
            if roi_drawing:
                roi_end = point
    elif event == cv2.EVENT_LBUTTONUP:
        with roi_lock:
            if not roi_drawing:
                return
            roi_drawing = False
            roi_end = point
            new_roi = normalize_manual_roi(roi_start, roi_end)
            if new_roi is not None:
                manual_roi = new_roi
        if new_roi is not None:
            logging.info(
                f"[ROI] Locked {new_roi}; it will constrain the next and subsequent detections."
            )
        else:
            logging.warning(
                f"[ROI] Drag is too small; minimum size is greater than "
                f"{MIN_ROI_SIZE_PX}x{MIN_ROI_SIZE_PX}px. Previous ROI is unchanged."
            )


def combined_window_mouse_callback(event, x, y, flags, param) -> None:
    """Route ROI gestures from only the RGB half of the combined window."""

    if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
        mouse_callback(event, x, y, flags, param)
    elif event == cv2.EVENT_LBUTTONUP:
        # A drag that starts on RGB and ends over the cloud should still close
        # cleanly at the RGB boundary instead of leaving ROI drawing active.
        mouse_callback(
            event,
            max(0, min(IMG_WIDTH - 1, x)),
            max(0, min(IMG_HEIGHT - 1, y)),
            flags,
            param,
        )


def compose_local_window(rgb_image: np.ndarray, cloud_image: np.ndarray) -> np.ndarray:
    """Place RGB and point-cloud views side by side in one local window."""

    cloud_width = max(1, int(round(cloud_image.shape[1] * IMG_HEIGHT / cloud_image.shape[0])))
    cloud_scaled = cv2.resize(
        cloud_image,
        (cloud_width, IMG_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    divider = np.full(
        (IMG_HEIGHT, LOCAL_DIVIDER_WIDTH_PX, 3),
        48,
        dtype=np.uint8,
    )
    return np.hstack((rgb_image, divider, cloud_scaled))


def draw_manual_roi(image: np.ndarray) -> None:
    with roi_lock:
        drawing = roi_drawing
        start = tuple(roi_start)
        end = tuple(roi_end)
        roi = None if manual_roi is None else tuple(manual_roi)
    color = (0, 200, 255)
    if drawing:
        preview = normalize_manual_roi(start, end)
        if preview is not None:
            left, top, right, bottom = preview
            cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(image, start, end, color, 1, cv2.LINE_AA)
    elif roi is not None:
        left, top, right, bottom = roi
        cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "ROI LOCKED",
            (left, max(62, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )


def handle_key(key: int) -> None:
    global quit_requested, reset_requested, clear_target_display_requested
    if key in (ord("q"), ord("Q")):
        quit_requested = True
    elif key in (ord("r"), ord("R")):
        clear_manual_roi()
        reset_requested = True
        clear_target_display_requested = True
    elif key in (ord("a"), ord("A")):
        toggle_fn = globals().get("toggle_auto_exposure")
        if toggle_fn is not None:
            toggle_fn()
    elif key == ord("["):
        adjust_manual_exposure(exposure_delta=-50.0)
    elif key == ord("]"):
        adjust_manual_exposure(exposure_delta=50.0)
    elif key == ord("-"):
        adjust_manual_exposure(gain_delta=-1.0)
    elif key in (ord("="), ord("+")):
        adjust_manual_exposure(gain_delta=1.0)


def panel_event_callback(msg: String) -> None:
    try:
        event = json.loads(msg.data)
    except json.JSONDecodeError:
        return
    event_type = event.get("type")
    if event_type == "mouse":
        mouse_callback(
            int(event.get("event", -1)),
            int(event.get("x", 0)),
            int(event.get("y", 0)),
            int(event.get("flags", 0)),
            None,
        )
    elif event_type == "key":
        handle_key(int(event.get("key", -1)))


ros_node.create_subscription(Bool, "/trigger_d405_vision", trigger_callback, 10)
ros_node.create_subscription(String, "/vision_panel/d405_local_rgb/event", panel_event_callback, 10)
ros_node.create_subscription(String, FAULT_SNAPSHOT_STATUS_TOPIC, cycle_status_callback, 10)
ros_node.create_subscription(String, FAULT_SNAPSHOT_TIMING_TOPIC, cycle_timing_callback, 20)
ros_node.create_subscription(String, "/detected_barcodes", barcode_callback, 20)


def publish_jpeg(pub, image: np.ndarray, frame_id: str) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), PANEL_JPEG_QUALITY])
    if not ok:
        return
    msg = CompressedImage()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = encoded.tobytes()
    pub.publish(msg)


def render_cloud(
    points,
    colors,
    center=None,
    extent=None,
    rotation=None,
    source_label="LIVE",
    tracked_target_points=None,
) -> np.ndarray:
    canvas = np.full((CLOUD_VIEW_H, CLOUD_VIEW_W, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, "D405 COSMETIC BOX CLOUD", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(canvas, f"source: {source_label}", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1)
    if points is None or len(points) == 0:
        cv2.putText(canvas, "NO VALID STEREO POINTS", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
        return canvas
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.float64)
    stride = max(1, len(pts) // 22000)
    pts, cols = pts[::stride], cols[::stride]
    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]
    view_center = np.array([0.0, 0.0, 0.65])
    yaw, pitch = np.deg2rad(-35.0), np.deg2rad(-22.0)
    r_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    r_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    view_rotation = r_yaw @ r_pitch
    view = (pts - view_center) @ view_rotation.T
    scale = 520.0
    px = (CLOUD_VIEW_W * 0.52 + view[:, 0] * scale).astype(np.int32)
    py = (CLOUD_VIEW_H * 0.58 - view[:, 1] * scale).astype(np.int32)
    valid = (px >= 0) & (px < CLOUD_VIEW_W) & (py >= 0) & (py < CLOUD_VIEW_H)
    bgr = np.clip(cols[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
    for idx in np.argsort(view[:, 2])[::-1]:
        if valid[idx]:
            canvas[py[idx], px[idx]] = bgr[idx]
    target_count = 0
    if tracked_target_points is not None:
        target_pts = np.asarray(tracked_target_points, dtype=np.float64)
        target_pts = target_pts[np.isfinite(target_pts).all(axis=1)]
        target_count = len(target_pts)
        target_stride = max(1, target_count // 5000)
        target_pts = target_pts[::target_stride]
        target_view = (target_pts - view_center) @ view_rotation.T
        target_px = (CLOUD_VIEW_W * 0.52 + target_view[:, 0] * scale).astype(np.int32)
        target_py = (CLOUD_VIEW_H * 0.58 - target_view[:, 1] * scale).astype(np.int32)
        target_valid = (
            (target_px >= 0)
            & (target_px < CLOUD_VIEW_W)
            & (target_py >= 0)
            & (target_py < CLOUD_VIEW_H)
        )
        for x_px, y_px in zip(target_px[target_valid], target_py[target_valid]):
            cv2.circle(canvas, (int(x_px), int(y_px)), 2, (0, 0, 255), -1, cv2.LINE_AA)
    if center is not None and extent is not None and rotation is not None:
        local = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]) * extent
        corners = local @ rotation.T + center
        view_c = (corners - view_center) @ view_rotation.T
        pts2 = np.column_stack((CLOUD_VIEW_W * 0.52 + view_c[:, 0] * scale, CLOUD_VIEW_H * 0.58 - view_c[:, 1] * scale)).astype(np.int32)
        for a, b in [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]:
            cv2.line(canvas, tuple(pts2[a]), tuple(pts2[b]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"points: {len(pts)}", (18, CLOUD_VIEW_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if target_count:
        cv2.putText(
            canvas,
            f"SAM2 tracked target: {target_count} points (RED)",
            (18, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def load_ffs_model():
    model = torch.load(FFS_MODEL_DIR, map_location="cpu", weights_only=False)
    model.args.valid_iters = VALID_ITERS
    model.args.max_disp = MAX_DISP
    return model.cuda().eval()


def warm_up_ffs_model(model) -> None:
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
    warm_padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
    warm0, warm1 = warm_padder.pad(dummy, dummy)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        model.forward(
            warm0,
            warm1,
            iters=VALID_ITERS,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    del dummy, warm0, warm1
    torch.cuda.empty_cache()


logging.info("Loading FFS model...")
with open(os.path.join(os.path.dirname(FFS_MODEL_DIR), "cfg.yaml"), "r", encoding="utf-8") as cfg_file:
    ffs_cfg = yaml.safe_load(cfg_file)
ffs_cfg["valid_iters"] = VALID_ITERS
ffs_cfg["max_disp"] = MAX_DISP
ffs_model = load_ffs_model()

from ultralytics import YOLO

logging.info("Loading YOLO-OBB model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0

logging.info(f"Starting D405 serial {CAMERA_SERIAL}...")
pipeline = rs.pipeline()
camera_config = rs.config()
camera_config.enable_device(CAMERA_SERIAL)
camera_config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
camera_config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
camera_config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
profile = pipeline.start(camera_config)
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if IR_PROJECTOR_ON else 0.0)

exposure_sensors = []


def set_option_safe(sensor, option, value) -> bool:
    try:
        option_range = sensor.get_option_range(option)
        sensor.set_option(option, max(option_range.min, min(option_range.max, float(value))))
        return True
    except Exception as exc:
        logging.warning(f"Camera option failed: {option}={value}: {exc}")
        return False


def apply_exposure_mode() -> None:
    for sensor in exposure_sensors:
        set_option_safe(sensor, rs.option.enable_auto_exposure, 1.0 if AUTO_EXPOSURE else 0.0)
        if not AUTO_EXPOSURE:
            if sensor.supports(rs.option.exposure):
                set_option_safe(sensor, rs.option.exposure, MANUAL_EXPOSURE)
            if sensor.supports(rs.option.gain):
                set_option_safe(sensor, rs.option.gain, MANUAL_GAIN)
    if AUTO_EXPOSURE:
        logging.info("[camera] Auto exposure enabled for all supported D405 sensors.")
    else:
        logging.info(
            f"[camera] Manual exposure enabled: exposure={MANUAL_EXPOSURE:.1f}, "
            f"gain={MANUAL_GAIN:.1f}."
        )


def toggle_auto_exposure() -> None:
    global AUTO_EXPOSURE
    AUTO_EXPOSURE = not AUTO_EXPOSURE
    apply_exposure_mode()


for camera_sensor in profile.get_device().query_sensors():
    if camera_sensor.supports(rs.option.enable_auto_exposure):
        exposure_sensors.append(camera_sensor)
    if camera_sensor.supports(rs.option.enable_auto_white_balance):
        set_option_safe(camera_sensor, rs.option.enable_auto_white_balance, 1.0 if AUTO_WHITE_BALANCE else 0.0)

apply_exposure_mode()


def adjust_manual_exposure(exposure_delta=0.0, gain_delta=0.0) -> None:
    global MANUAL_EXPOSURE, MANUAL_GAIN
    if AUTO_EXPOSURE:
        return
    if exposure_delta:
        ranges = [s.get_option_range(rs.option.exposure) for s in exposure_sensors if s.supports(rs.option.exposure)]
        if ranges:
            MANUAL_EXPOSURE = max(max(r.min for r in ranges), min(min(r.max for r in ranges), MANUAL_EXPOSURE + exposure_delta))
            for sensor in exposure_sensors:
                if sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, MANUAL_EXPOSURE)
    if gain_delta:
        ranges = [s.get_option_range(rs.option.gain) for s in exposure_sensors if s.supports(rs.option.gain)]
        if ranges:
            MANUAL_GAIN = max(max(r.min for r in ranges), min(min(r.max for r in ranges), MANUAL_GAIN + gain_delta))
            for sensor in exposure_sensors:
                if sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, MANUAL_GAIN)
    logging.info(f"Exposure={MANUAL_EXPOSURE:.1f}, gain={MANUAL_GAIN:.1f}")


first_frames = pipeline.wait_for_frames()
ir_left_profile = first_frames.get_infrared_frame(1).get_profile().as_video_stream_profile()
ir_right_profile = first_frames.get_infrared_frame(2).get_profile().as_video_stream_profile()
color_profile = first_frames.get_color_frame().get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
color_intrinsics = color_profile.get_intrinsics()
K_ir = np.array([[ir_intrinsics.fx, 0, ir_intrinsics.ppx], [0, ir_intrinsics.fy, ir_intrinsics.ppy], [0, 0, 1]], dtype=np.float32)
K_color = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx], [0, color_intrinsics.fy, color_intrinsics.ppy], [0, 0, 1]], dtype=np.float32)
ir_to_color = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color.translation).astype(np.float32)
stereo_baseline_m = abs(ir_left_profile.get_extrinsics_to(ir_right_profile).translation[0])
fx_ir, fy_ir, cx_ir, cy_ir = K_ir[0, 0], K_ir[1, 1], K_ir[0, 2], K_ir[1, 2]
u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat, v_flat = u_grid.reshape(-1).astype(np.float32), v_grid.reshape(-1).astype(np.float32)

warm_up_ffs_model(ffs_model)
initialize_initial_position_frame_output()
initialize_fault_snapshot_output()


def estimate_candidate_camera_x(corners, points_3d, u_rgb, v_rgb, in_bounds):
    polygon_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [np.int32(corners)], 1)
    hits = np.zeros(len(points_3d), dtype=bool)
    hits[in_bounds] = polygon_mask[v_rgb[in_bounds], u_rgb[in_bounds]] > 0
    candidate_points = points_3d[hits]
    if len(candidate_points) < 20:
        return None, len(candidate_points)
    # Median is robust against a few background points inside an OBB.
    return float(np.median(candidate_points[:, 0])), len(candidate_points)


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    x_axis = rotation[:, 0] / (np.linalg.norm(rotation[:, 0]) + 1e-12)
    y_axis = rotation[:, 1] - np.dot(rotation[:, 1], x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


sam_initialized = False
current_mask = None
pending_bbox = None
pending_yolo_obbs = None
last_yolo_obbs = None
last_best_idx = -1
last_yolo_time = 0.0
locked_target_corners = None
locked_target_id = -1
locked_target_camera_x = None
locked_target_ready = False
locked_target_height_m = None
locked_cloud_points = None
locked_cloud_u_rgb = None
locked_cloud_v_rgb = None
locked_cloud_in_bounds = None
locked_cloud_colors = None
smooth_grasp = None
smooth_box_center = None
smooth_extent = None
smooth_rotation = None
# 当前相机帧的未滤波抓取点/姿态。后台预抓取跟踪必须使用它：相机随
# 机械臂移动时，smooth_grasp 的滤波滞后会被误认为目标发生了位移。
current_frame_grasp = None
current_frame_rotation = None
height_history = deque(maxlen=20)
published_frames = 0
last_height_m = None
last_side_points = 0
last_side_planes = 0
last_height_evidence_mode = "none"
tracking_frames_without_height = 0
last_height_warning_time = 0.0
last_cloud_fallback_time = 0.0
handoff_clear_streak = 0
handoff_clearance_passed = not HANDOFF_CLEARANCE_ENABLED
handoff_force_live_cloud = False
last_handoff_state = "IDLE"
last_handoff_candidate_points = 0
last_handoff_cluster_points = 0
pregrasp_tracking_deadline = 0.0

if ENABLE_LOCAL_WINDOWS:
    local_window_width = IMG_WIDTH + LOCAL_DIVIDER_WIDTH_PX + int(
        round(CLOUD_VIEW_W * IMG_HEIGHT / CLOUD_VIEW_H)
    )
    cv2.namedWindow(LOCAL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LOCAL_WINDOW_NAME, local_window_width, IMG_HEIGHT)
    cv2.setMouseCallback(LOCAL_WINDOW_NAME, combined_window_mouse_callback)

try:
    while not quit_requested:
        loop_start = time.time()

        if trigger_requested:
            vision_request_active = True
            # A new request supersedes the previous pick indication. The newly
            # selected target will remain visible until the following request.
            # Reset the previous SAM2 session, then hand the new YOLO box to a
            # fresh tracker after the capture below.
            reset_requested = True
            locked_target_corners = None
            locked_target_id = -1
            locked_target_camera_x = None
            locked_target_ready = False
            locked_target_height_m = None
            locked_cloud_points = None
            locked_cloud_u_rgb = None
            locked_cloud_v_rgb = None
            locked_cloud_in_bounds = None
            locked_cloud_colors = None
            handoff_clear_streak = 0
            handoff_clearance_passed = not HANDOFF_CLEARANCE_ENABLED
            handoff_force_live_cloud = False
            last_handoff_state = "WAIT_TARGET"
            last_handoff_candidate_points = 0
            last_handoff_cluster_points = 0
            pregrasp_tracking_deadline = 0.0
            stable_target_frames.clear()
            publish_handoff_clearance("WAIT_TARGET", clear=False)
            logging.info(
                f"[D405] Flushing {CAPTURE_FLUSH_FRAMES} frames before YOLO capture..."
            )
            for _ in range(CAPTURE_FLUSH_FRAMES):
                pipeline.wait_for_frames()
            detection_frames = pipeline.wait_for_frames()
            detection_color = np.asanyarray(detection_frames.get_color_frame().get_data())
            if SAVE_INITIAL_POSITION_FRAMES and not initial_position_frames_saved_for_startup:
                # 记录并立即保存初始位置的两张原始 RGB 帧。RealSense
                # 可能复用底层缓冲区，必须复制后再交给写盘函数。
                initial_position_frames = [detection_color.copy()]
                initial_followup_frames = pipeline.wait_for_frames()
                initial_position_frames.append(
                    np.asanyarray(
                        initial_followup_frames.get_color_frame().get_data()
                    ).copy()
                )
                save_d405_output_frames(
                    initial_position_frames,
                    "initial-position",
                    INITIAL_POSITION_FRAME_COUNT,
                )
                initial_position_frames_saved_for_startup = True
            results = yolo_model(detection_color, conf=0.6, verbose=False)#置信度参数设置目前0.6
            if results and results[0].obb is not None and len(results[0].obb) > 0:
                detected_obbs = results[0].obb.xyxyxyxy.cpu().numpy()
                roi_for_request = get_manual_roi()
                if roi_for_request is not None:
                    left, top, right, bottom = roi_for_request
                    centers = detected_obbs.mean(axis=1)
                    inside_roi = (
                        (centers[:, 0] >= left)
                        & (centers[:, 0] < right)
                        & (centers[:, 1] >= top)
                        & (centers[:, 1] < bottom)
                    )
                    pending_yolo_obbs = detected_obbs[inside_roi]
                    logging.info(
                        f"[ROI] {int(np.count_nonzero(inside_roi))}/{len(detected_obbs)} "
                        f"YOLO candidates have centers inside {roi_for_request}."
                    )
                else:
                    pending_yolo_obbs = detected_obbs

                if len(pending_yolo_obbs) > 0:
                    last_yolo_obbs = pending_yolo_obbs.copy()
                    last_best_idx = -1
                    last_yolo_time = time.time()
                    logging.info(
                        f"[YOLO] Found {len(pending_yolo_obbs)} allowed candidates; "
                        "waiting for 3-D X ranking."
                    )
                else:
                    pending_yolo_obbs = None
                    last_yolo_obbs = None
                    logging.warning(
                        "[ROI] YOLO found objects, but none is inside the locked ROI; "
                        "this request is rejected without fallback outside the ROI."
                    )
                    publish_vision_result(False, "no_yolo_candidate_inside_roi")
            else:
                pending_yolo_obbs = None
                last_yolo_obbs = None
                logging.warning("[YOLO] No cosmetic box detected.")
                publish_vision_result(False, "no_cosmetic_box")
            trigger_requested = False

        frames = pipeline.wait_for_frames()
        # Stamp immediately after frame acquisition.  D405 and the robot
        # driver run on this host, so this is the common wall-clock domain used
        # by the robot feedback history for eye-in-hand compensation.
        frame_stamp = ros_node.get_clock().now().to_msg()
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())

        if reset_requested:
            try:
                sam2_predictor.reset_state()
            except KeyError:
                pass
            sam_initialized = False
            current_mask = None
            smooth_grasp = smooth_box_center = smooth_extent = smooth_rotation = None
            current_frame_grasp = current_frame_rotation = None
            height_history.clear()
            published_frames = 0
            tracking_frames_without_height = 0
            last_height_evidence_mode = "none"
            if clear_target_display_requested:
                locked_target_corners = None
                locked_target_id = -1
                locked_target_camera_x = None
                locked_target_ready = False
                locked_target_height_m = None
                locked_cloud_points = None
                locked_cloud_u_rgb = None
                locked_cloud_v_rgb = None
                locked_cloud_in_bounds = None
                locked_cloud_colors = None
                clear_target_display_requested = False
            reset_requested = False

        if pending_bbox is not None and not sam_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            prompt = np.array([[pending_bbox[0], pending_bbox[1]], [pending_bbox[2], pending_bbox[3]]], dtype=np.float32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=prompt)
            sam_initialized = True
            pending_bbox = None
            logging.info("[SAM2] Tracking selected minimum-camera-X target.")
        elif sam_initialized:
            object_ids, mask_logits = sam2_predictor.track(color_bgr)
            current_mask = (mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze() if len(object_ids) else None
            tracking_frames_without_height += 1

        # YOLO 的相机 X 排序必须使用本次触发刚计算出的 FFS 点云。选中目标后，
        # 相机和场景在机械臂开始抓取前保持静止，因此 SAM2 的后续稳定帧直接复用
        # 这份锁定点云，避免为同一个静止场景重复运行昂贵的 FFS 推理。
        reuse_selection_cloud = (
            locked_cloud_points is not None
            and pending_yolo_obbs is None
            and not locked_target_ready
            and not handoff_force_live_cloud
        )
        if reuse_selection_cloud:
            points_3d = locked_cloud_points
            u_rgb = locked_cloud_u_rgb
            v_rgb = locked_cloud_v_rgb
            in_bounds = locked_cloud_in_bounds
            colors = locked_cloud_colors.copy()
            using_locked_cloud = True
        else:
            ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())
            ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())
            left_rgb, right_rgb = np.stack([ir_left] * 3, axis=-1), np.stack([ir_right] * 3, axis=-1)
            image0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            image1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(image0.shape, divis_by=32, force_square=False)
            image0_p, image1_p = padder.pad(image0, image1)
            with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
                disparity = ffs_model.forward(image0_p, image1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume="pytorch1")
            disparity = padder.unpad(disparity.float()).data.cpu().numpy().reshape(IMG_HEIGHT, IMG_WIDTH).clip(0, None)
            disparity_finite = disparity[np.isfinite(disparity)]
            depth = fx_ir * stereo_baseline_m / (disparity + 1e-6)
            depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0
            bad_edge = (np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)) > 0.5) | (np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)) > 0.5)
            depth[bad_edge] = 0

            z_flat = depth[::PCD_STRIDE, ::PCD_STRIDE].reshape(-1)
            valid_depth = z_flat > 0
            z, u, v = z_flat[valid_depth], u_flat[valid_depth], v_flat[valid_depth]
            points_3d = np.stack(((u - cx_ir) * z / fx_ir, (v - cy_ir) * z / fy_ir, z), axis=-1)
            points_color = (R_ir_to_color @ points_3d.T).T + T_ir_to_color
            u_rgb = (K_color[0, 0] * points_color[:, 0] / points_color[:, 2] + K_color[0, 2]).astype(np.int32)
            v_rgb = (K_color[1, 1] * points_color[:, 1] / points_color[:, 2] + K_color[1, 2]).astype(np.int32)
            in_bounds = (u_rgb >= 0) & (u_rgb < IMG_WIDTH) & (v_rgb >= 0) & (v_rgb < IMG_HEIGHT)
            colors = np.zeros((len(points_3d), 3), dtype=np.float64)
            colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1].astype(np.float64) / 255.0
            using_locked_cloud = False

            # 锁定完成后恢复 LIVE FFS 以维持点云跟随；若某一帧立体网络临时
            # 丢失深度，仍回退到本次目标的锁定快照，避免点云窗口突然变黑。
            if len(points_3d) < 40 and locked_cloud_points is not None:
                points_3d = locked_cloud_points
                u_rgb = locked_cloud_u_rgb
                v_rgb = locked_cloud_v_rgb
                in_bounds = locked_cloud_in_bounds
                colors = locked_cloud_colors.copy()
                using_locked_cloud = True
                now = time.time()
                if now - last_cloud_fallback_time >= 1.0:
                    logging.warning(
                        f"[depth] Live FFS cloud has fewer than 40 points; using locked "
                        f"selection snapshot with {len(points_3d)} points."
                    )
                    last_cloud_fallback_time = now

        # Rank every YOLO box by measured camera-optical X. No ROI, coarse prior,
        # image-center fallback, or image-left heuristic is used.
        if pending_yolo_obbs is not None:
            valid_ratio = len(points_3d) / max(1, len(u_flat))
            median_depth = float(np.median(points_3d[:, 2])) if len(points_3d) else float("nan")
            logging.info(
                f"[depth] valid_points={len(points_3d)}/{len(u_flat)} "
                f"({valid_ratio*100:.1f}%), median_z={median_depth:.3f}m, "
                f"disparity="
                f"{float(np.min(disparity_finite)) if len(disparity_finite) else float('nan'):.2f}/"
                f"{float(np.median(disparity_finite)) if len(disparity_finite) else float('nan'):.2f}/"
                f"{float(np.max(disparity_finite)) if len(disparity_finite) else float('nan'):.2f} "
                f"(min/median/max), exposure={'AUTO' if AUTO_EXPOSURE else 'MANUAL'}"
            )
            ranked = []
            for candidate_index, corners in enumerate(pending_yolo_obbs):
                candidate_x, candidate_point_count = estimate_candidate_camera_x(
                    corners, points_3d, u_rgb, v_rgb, in_bounds
                )
                if candidate_x is not None:
                    ranked.append((candidate_x, candidate_index))
                    logging.info(
                        f"[3-D select] candidate={candidate_index} camera_x={candidate_x:.4f}m "
                        f"valid_3d_points={candidate_point_count}"
                    )
                else:
                    logging.warning(
                        f"[3-D select] candidate={candidate_index} rejected: "
                        f"valid_3d_points={candidate_point_count} < 20"
                    )
            if ranked:
                best_camera_x, best_index = min(ranked, key=lambda item: item[0])
                last_best_idx = best_index
                corners = pending_yolo_obbs[best_index]
                locked_target_corners = corners.copy()
                locked_target_id = int(best_index)
                locked_target_camera_x = float(best_camera_x)
                locked_target_ready = False
                locked_target_height_m = None
                locked_cloud_points = points_3d.copy()
                locked_cloud_u_rgb = u_rgb.copy()
                locked_cloud_v_rgb = v_rgb.copy()
                locked_cloud_in_bounds = in_bounds.copy()
                locked_cloud_colors = colors.copy()
                handoff_clear_streak = 0
                handoff_clearance_passed = not HANDOFF_CLEARANCE_ENABLED
                handoff_force_live_cloud = False
                last_handoff_state = "CHECKING"
                last_handoff_candidate_points = 0
                last_handoff_cluster_points = 0
                publish_handoff_clearance("CHECKING", clear=False)
                x1, y1 = np.min(corners, axis=0)
                x2, y2 = np.max(corners, axis=0)
                pending_bbox = (int(x1), int(y1), int(x2), int(y2))
                reset_requested = True
                logging.info(f"[3-D select] Selected ID={best_index}, the minimum valid camera-X target.")
            else:
                logging.warning("[3-D select] No YOLO candidate had enough valid stereo points; detection rejected.")
                publish_vision_result(False, "no_candidate_with_valid_stereo_points")
            pending_yolo_obbs = None

        display = color_bgr.copy()
        tracked_target_points = None
        handoff_candidate_mask = None
        if current_mask is not None and np.any(current_mask):
            overlay = display.copy()
            overlay[current_mask > 0] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1.0 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
            if contours and locked_target_corners is not None:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) >= 20.0:
                    # The persistent box is derived from the current SAM2 mask,
                    # so it follows the selected target instead of remaining at
                    # the original one-shot YOLO position.
                    locked_target_corners = cv2.boxPoints(
                        cv2.minAreaRect(largest_contour)
                    ).astype(np.float32)

            object_hits = np.zeros(len(points_3d), dtype=bool)
            object_hits[in_bounds] = current_mask[v_rgb[in_bounds], u_rgb[in_bounds]] > 0
            object_point_count = int(np.count_nonzero(object_hits))
            # A live cloud may still contain background points while losing the
            # selected box. Prefer the locked snapshot only when it provides a
            # strictly better SAM-mask overlap.
            if object_point_count < 40 and not using_locked_cloud and locked_cloud_points is not None:
                snapshot_hits = np.zeros(len(locked_cloud_points), dtype=bool)
                snapshot_hits[locked_cloud_in_bounds] = current_mask[
                    locked_cloud_v_rgb[locked_cloud_in_bounds],
                    locked_cloud_u_rgb[locked_cloud_in_bounds],
                ] > 0
                snapshot_point_count = int(np.count_nonzero(snapshot_hits))
                if snapshot_point_count > object_point_count:
                    points_3d = locked_cloud_points
                    u_rgb = locked_cloud_u_rgb
                    v_rgb = locked_cloud_v_rgb
                    in_bounds = locked_cloud_in_bounds
                    colors = locked_cloud_colors.copy()
                    object_hits = snapshot_hits
                    object_point_count = snapshot_point_count
                    using_locked_cloud = True
                    logging.warning(
                        f"[depth] SAM/live overlap={object_point_count} after switching to "
                        "the locked selection snapshot."
                    )
            if object_point_count < 40:
                now = time.time()
                if now - last_height_warning_time >= 1.0:
                    mask_y, mask_x = np.nonzero(current_mask)
                    mask_bbox_text = (
                        f"({int(mask_x.min())},{int(mask_y.min())})-"
                        f"({int(mask_x.max())},{int(mask_y.max())})"
                        if len(mask_x)
                        else "empty"
                    )
                    target_bbox_text = "none"
                    if locked_target_corners is not None:
                        target_bbox_text = (
                            f"({int(np.min(locked_target_corners[:, 0]))},"
                            f"{int(np.min(locked_target_corners[:, 1]))})-"
                            f"({int(np.max(locked_target_corners[:, 0]))},"
                            f"{int(np.max(locked_target_corners[:, 1]))})"
                        )
                    logging.warning(
                        f"[height] SAM mask contains only {object_point_count} valid stereo points; "
                        f"need at least 40 for plane fitting; mask_bbox={mask_bbox_text}, "
                        f"selected_bbox={target_bbox_text}."
                    )
                    last_height_warning_time = now
            if np.any(object_hits):
                colors[object_hits] = colors[object_hits] * 0.2 + MASK_COLOR_RGB * 0.8
                tracked_target_points = points_3d[object_hits].copy()
                object_points = points_3d[object_hits]
                object_uv = np.column_stack((u_rgb[object_hits], v_rgb[object_hits]))
                # Preserve every valid depth point selected by SAM for height
                # evidence. DBSCAN and radial trimming below intentionally make
                # a clean top-plane cloud, but otherwise tend to discard the
                # thin vertical faces that carry the box-height information.
                mask_object_points = object_points.copy()
                mask_object_uv = object_uv.copy()
                if len(object_points) >= 40:
                    object_cloud = o3d.geometry.PointCloud()
                    object_cloud.points = o3d.utility.Vector3dVector(object_points)
                    labels = np.asarray(object_cloud.cluster_dbscan(eps=0.008, min_points=20, print_progress=False))
                    if np.any(labels >= 0):
                        valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                        keep = labels == valid_labels[np.argmax(counts)]
                        object_points, object_uv = object_points[keep], object_uv[keep]
                    centroid = object_points.mean(axis=0)
                    distances = np.linalg.norm(object_points - centroid, axis=1)
                    keep = distances <= np.percentile(distances, 96)
                    filtered, filtered_uv = object_points[keep], object_uv[keep]

                    if len(filtered) >= 40:
                        remaining = o3d.geometry.PointCloud()
                        remaining.points = o3d.utility.Vector3dVector(filtered)
                        plane_candidates = []
                        for _ in range(MAX_PLANES):
                            if len(remaining.points) < MIN_PLANE_POINTS:
                                break
                            model, inliers = remaining.segment_plane(RANSAC_DISTANCE_M, 3, 1000)
                            if len(inliers) < MIN_PLANE_POINTS:
                                break
                            plane_candidates.append((len(inliers), np.asarray(model, dtype=np.float64)))
                            remaining = remaining.select_by_index(inliers, invert=True)

                        if plane_candidates:
                            top_inliers, top_plane = max(plane_candidates, key=lambda item: item[0])
                            raw_normal = top_plane[:3].copy()
                            normal_norm = np.linalg.norm(raw_normal) + 1e-12
                            normal = raw_normal / normal_norm
                            plane_d = float(top_plane[3]) / normal_norm
                            if normal[2] > 0:
                                normal, plane_d = -normal, -plane_d
                            top_coordinate = -plane_d
                            down_from_top = top_coordinate - mask_object_points @ normal
                            raw_side_mask = (
                                (down_from_top > SIDE_START_DEPTH_M)
                                & (down_from_top < MAX_BOX_HEIGHT_M)
                            )
                            raw_side_depths = down_from_top[raw_side_mask]

                            # Preferred evidence is a RANSAC side plane. Thin cosmetic
                            # boxes often expose too few side pixels for RANSAC, so the
                            # fallback only accepts below-top points in a narrow band at
                            # the SAM mask boundary. This retains geometric side evidence
                            # without treating arbitrary top-surface noise as height.
                            last_side_planes = 0
                            for _, candidate_plane in plane_candidates:
                                candidate_normal = candidate_plane[:3]
                                candidate_normal /= np.linalg.norm(candidate_normal) + 1e-12
                                if abs(float(np.dot(candidate_normal, normal))) < 0.55:
                                    last_side_planes += 1

                            mask_u8 = (current_mask > 0).astype(np.uint8)
                            edge_kernel = np.ones((EDGE_EVIDENCE_KERNEL_PX, EDGE_EVIDENCE_KERNEL_PX), np.uint8)
                            mask_interior = cv2.erode(mask_u8, edge_kernel, iterations=1)
                            mask_edge_band = (mask_u8 > mask_interior)
                            edge_flags = mask_edge_band[mask_object_uv[:, 1], mask_object_uv[:, 0]]
                            edge_side_depths = down_from_top[raw_side_mask & edge_flags]

                            # A flat tabletop is a particularly strong third
                            # source of height evidence. Fit it only in a ring
                            # outside the SAM mask and require its normal to be
                            # parallel to the detected box top.
                            table_height = None
                            table_plane_points = 0
                            table_ring_kernel = np.ones((TABLE_RING_KERNEL_PX, TABLE_RING_KERNEL_PX), np.uint8)
                            table_exclusion_kernel = np.ones(
                                (TABLE_EXCLUSION_KERNEL_PX, TABLE_EXCLUSION_KERNEL_PX), np.uint8
                            )
                            table_ring = (
                                cv2.dilate(mask_u8, table_ring_kernel, iterations=1)
                                > cv2.dilate(mask_u8, table_exclusion_kernel, iterations=1)
                            )
                            table_hits = np.zeros(len(points_3d), dtype=bool)
                            table_hits[in_bounds] = table_ring[v_rgb[in_bounds], u_rgb[in_bounds]]
                            table_points = points_3d[table_hits]
                            if len(table_points) >= MIN_TABLE_PLANE_POINTS:
                                table_cloud = o3d.geometry.PointCloud()
                                table_cloud.points = o3d.utility.Vector3dVector(table_points)
                                table_model, table_inliers = table_cloud.segment_plane(
                                    TABLE_RANSAC_DISTANCE_M, 3, 500
                                )
                                table_candidate_normal = np.asarray(table_model[:3], dtype=np.float64)
                                table_candidate_normal /= np.linalg.norm(table_candidate_normal) + 1e-12
                                if (
                                    len(table_inliers) >= MIN_TABLE_PLANE_POINTS
                                    and abs(float(np.dot(table_candidate_normal, normal))) >= 0.90
                                ):
                                    table_inlier_points = table_points[np.asarray(table_inliers, dtype=np.int64)]
                                    table_depths = top_coordinate - table_inlier_points @ normal
                                    table_depths = table_depths[
                                        (table_depths >= MIN_BOX_HEIGHT_M)
                                        & (table_depths <= MAX_BOX_HEIGHT_M)
                                    ]
                                    if len(table_depths) >= MIN_TABLE_PLANE_POINTS:
                                        table_height = float(np.median(table_depths))
                                        table_plane_points = len(table_depths)

                            measured_height = None
                            if table_height is not None:
                                side_depths = np.asarray([table_height], dtype=np.float64)
                                last_height_evidence_mode = "top-table-planes"
                                last_side_points = table_plane_points
                                measured_height = table_height
                            elif last_side_planes > 0 and len(raw_side_depths) >= MIN_SIDE_EVIDENCE_POINTS:
                                side_depths = raw_side_depths
                                last_height_evidence_mode = "ransac-side-plane"
                                measured_height = float(np.percentile(side_depths, HEIGHT_PERCENTILE))
                            elif len(edge_side_depths) >= MIN_EDGE_SIDE_EVIDENCE_POINTS:
                                side_depths = edge_side_depths
                                last_height_evidence_mode = "mask-edge-side-points"
                                measured_height = float(np.percentile(side_depths, HEIGHT_PERCENTILE))
                            else:
                                side_depths = edge_side_depths
                                last_height_evidence_mode = "insufficient"
                            if last_height_evidence_mode != "top-table-planes":
                                last_side_points = len(side_depths)

                            if measured_height is not None:
                                if MIN_BOX_HEIGHT_M <= measured_height <= MAX_BOX_HEIGHT_M:
                                    mask_contours, _ = cv2.findContours(current_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if mask_contours:
                                        box2d = cv2.boxPoints(cv2.minAreaRect(max(mask_contours, key=cv2.contourArea)))
                                        d01 = np.linalg.norm(box2d[0] - box2d[1])
                                        d12 = np.linalg.norm(box2d[1] - box2d[2])
                                        if d01 < d12:
                                            midpoint1, midpoint2 = (box2d[0] + box2d[1]) / 2, (box2d[2] + box2d[3]) / 2
                                        else:
                                            midpoint1, midpoint2 = (box2d[1] + box2d[2]) / 2, (box2d[3] + box2d[0]) / 2
                                        vector2d = midpoint2 - midpoint1
                                        midpoint1 += vector2d * 0.10
                                        midpoint2 -= vector2d * 0.10
                                        point1 = np.mean(filtered[np.argsort(np.linalg.norm(filtered_uv - midpoint1, axis=1))[:5]], axis=0)
                                        point2 = np.mean(filtered[np.argsort(np.linalg.norm(filtered_uv - midpoint2, axis=1))[:5]], axis=0)
                                        x_axis = point2 - point1
                                        x_axis -= np.dot(x_axis, normal) * normal
                                        x_axis /= np.linalg.norm(x_axis) + 1e-12
                                        if x_axis[1] > 0:
                                            x_axis = -x_axis
                                        y_axis = np.cross(normal, x_axis)
                                        y_axis /= np.linalg.norm(y_axis) + 1e-12
                                        axes = np.column_stack((x_axis, y_axis, normal))
                                        x_values, y_values = filtered @ x_axis, filtered @ y_axis
                                        x_mid = 0.5 * (x_values.min() + x_values.max())
                                        y_mid = 0.5 * (y_values.min() + y_values.max())
                                        box_center = x_axis * x_mid + y_axis * y_mid + normal * (top_coordinate - 0.5 * measured_height)
                                        grasp_point = x_axis * x_mid + y_axis * y_mid + normal * (top_coordinate - GRASP_DEPTH_RATIO * measured_height)
                                        raw_extent = np.array([x_values.max() - x_values.min(), y_values.max() - y_values.min(), measured_height])

                                        height_history.append(measured_height)
                                        stable_height = float(np.median(height_history))
                                        height_delta = stable_height - measured_height
                                        box_center += normal * (-0.5 * height_delta)
                                        grasp_point += normal * (-GRASP_DEPTH_RATIO * height_delta)
                                        raw_extent[2] = stable_height
                                        # Keep the current-frame estimate separate
                                        # from the display/stability filter.  The
                                        # filtered estimate is intentionally smooth
                                        # but lags during eye-in-hand motion.
                                        current_frame_grasp = grasp_point.copy()
                                        current_frame_rotation = axes.copy()
                                        if smooth_grasp is None:
                                            smooth_grasp = grasp_point.copy()
                                            smooth_box_center = box_center.copy()
                                            smooth_extent = raw_extent.copy()
                                            smooth_rotation = axes.copy()
                                        else:
                                            smooth_grasp = OBB_SMOOTH * grasp_point + (1 - OBB_SMOOTH) * smooth_grasp
                                            smooth_box_center = OBB_SMOOTH * box_center + (1 - OBB_SMOOTH) * smooth_box_center
                                            smooth_extent = OBB_SMOOTH * raw_extent + (1 - OBB_SMOOTH) * smooth_extent
                                            smooth_rotation = orthonormalize(OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * smooth_rotation)

                                        last_height_m = float(smooth_extent[2])
                                        # A geometrically valid box must not be mistaken
                                        # for a missing-height frame merely because the
                                        # 102 arm is still occupying the space above it.
                                        tracking_frames_without_height = 0

                                        if not handoff_clearance_passed:
                                            clearance = evaluate_camera_right_handoff_clearance(
                                                points_3d,
                                                smooth_box_center,
                                                smooth_extent,
                                                smooth_rotation,
                                                right_extension_m=HANDOFF_CLEARANCE_RIGHT_EXTENSION_M,
                                                side_margin_m=HANDOFF_CLEARANCE_SIDE_MARGIN_M,
                                                vertical_gap_m=HANDOFF_CLEARANCE_VERTICAL_GAP_M,
                                                check_height_m=HANDOFF_CLEARANCE_CHECK_HEIGHT_M,
                                                voxel_size_m=HANDOFF_CLEARANCE_VOXEL_SIZE_M,
                                                min_obstacle_points=HANDOFF_CLEARANCE_MIN_CLUSTER_POINTS,
                                            )
                                            handoff_candidate_mask = clearance.candidate_mask
                                            last_handoff_candidate_points = clearance.candidate_point_count
                                            last_handoff_cluster_points = clearance.largest_cluster_point_count
                                            # After the first snapshot (or any BLOCKED
                                            # result), a CLEAR result only counts when it
                                            # comes from a newly inferred LIVE FFS cloud.
                                            independent_clear_sample = not (
                                                handoff_force_live_cloud and using_locked_cloud
                                            )
                                            if not clearance.clear:
                                                handoff_clear_streak = 0
                                                handoff_force_live_cloud = True
                                                handoff_state = "BLOCKED"
                                            elif not independent_clear_sample:
                                                handoff_force_live_cloud = True
                                                handoff_state = "WAIT_LIVE_CLOUD"
                                            else:
                                                handoff_clear_streak += 1
                                                if handoff_clear_streak >= HANDOFF_CLEARANCE_CLEAR_FRAMES:
                                                    handoff_clearance_passed = True
                                                    handoff_force_live_cloud = False
                                                    handoff_state = "CLEAR"
                                                    # Reuse this verified-clear cloud for
                                                    # the second robot-side pose sample;
                                                    # this avoids a third FFS inference.
                                                    if not using_locked_cloud:
                                                        locked_cloud_points = points_3d.copy()
                                                        locked_cloud_u_rgb = u_rgb.copy()
                                                        locked_cloud_v_rgb = v_rgb.copy()
                                                        locked_cloud_in_bounds = in_bounds.copy()
                                                        locked_cloud_colors = colors.copy()
                                                else:
                                                    handoff_force_live_cloud = True
                                                    handoff_state = "VERIFYING_CLEAR"

                                            if clearance.candidate_point_count:
                                                obstacle_color = (
                                                    np.array([1.0, 0.0, 1.0])
                                                    if not clearance.clear
                                                    else np.array([1.0, 0.65, 0.0])
                                                )
                                                colors[clearance.candidate_mask] = obstacle_color
                                            if handoff_state != last_handoff_state:
                                                logging.info(
                                                    f"[handoff-clearance] {handoff_state}: "
                                                    f"candidate_points={clearance.candidate_point_count}, "
                                                    f"largest_cluster={clearance.largest_cluster_point_count}, "
                                                    f"clear_streak={handoff_clear_streak}/"
                                                    f"{HANDOFF_CLEARANCE_CLEAR_FRAMES}, "
                                                    f"cloud={'LOCKED' if using_locked_cloud else 'LIVE'}"
                                                )
                                            last_handoff_state = handoff_state
                                            publish_handoff_clearance(
                                                handoff_state,
                                                clear=handoff_clearance_passed,
                                                candidate_points=clearance.candidate_point_count,
                                                largest_cluster_points=clearance.largest_cluster_point_count,
                                                clear_streak=handoff_clear_streak,
                                            )

                                        # Publish only after the target core and right-side handoff
                                        # corridor have been independently clear twice. Afterwards SAM2
                                        # remains alive only for the operator display.
                                        if handoff_clearance_passed and not locked_target_ready:
                                            if SAVE_INITIAL_POSITION_FRAMES:
                                                # 这两帧正是形成稳定目标的连续有效帧，
                                                # 保存原始 RGB，不保存带标注的显示图。
                                                stable_target_frames.append(color_bgr.copy())
                                            pose_msg = PoseStamped()
                                            pose_msg.header.stamp = frame_stamp
                                            pose_msg.header.frame_id = "camera_d405_link"
                                            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = map(float, smooth_grasp)
                                            quat = SciPyRot.from_matrix(smooth_rotation).as_quat()
                                            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y = float(quat[0]), float(quat[1])
                                            pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = float(quat[2]), float(quat[3])
                                            pose_pub.publish(pose_msg)
                                            width_msg = Float32()
                                            width_msg.data = float(min(MAX_GRIPPER_OPENING_M, smooth_extent[1] + GRIP_CLEARANCE_M))
                                            width_pub.publish(width_msg)
                                            length_msg = Float32()
                                            # x_axis follows the long side of the segmented
                                            # top surface.  Publish its physical extent
                                            # without gripper clearance so the robot can
                                            # position the near box face relative to the
                                            # barcode scanner.
                                            length_msg.data = float(smooth_extent[0])
                                            length_pub.publish(length_msg)
                                            height_msg = Float32()
                                            height_msg.data = last_height_m
                                            height_pub.publish(height_msg)
                                            published_frames += 1
                                            if published_frames >= PUBLISH_FRAMES_BEFORE_RESET:
                                                locked_target_ready = True
                                                locked_target_height_m = last_height_m
                                                save_d405_output_frames(
                                                    stable_target_frames,
                                                    "stable-target",
                                                    STABLE_TARGET_FRAME_COUNT,
                                                )
                                                stable_target_frames.clear()
                                                # 允许机械臂完成本轮抓取、回到初始位后，
                                                # 下一次视觉触发重新保存初始位两张图。
                                                initial_position_frames_saved_for_startup = False
                                                pregrasp_tracking_deadline = (
                                                    time.monotonic() + PREGRASP_TRACKING_WINDOW_S
                                                )
                                                publish_pregrasp_pose(frame_stamp)
                                                publish_vision_result(True, "stable_target_published")
                                                logging.info(
                                                    f"[D405] Published stable target: length={smooth_extent[0]*1000:.1f}mm, "
                                                    f"height={last_height_m*1000:.1f}mm, "
                                                    f"depth={GRASP_DEPTH_RATIO*100:.0f}%, evidence={last_height_evidence_mode}; "
                                                    "SAM2 display tracking remains active."
                                                )
                                        elif (
                                            handoff_clearance_passed
                                            and locked_target_ready
                                            and time.monotonic() < pregrasp_tracking_deadline
                                        ):
                                            # 稳定目标已经交给机器人后，继续发布后台跟踪结果；
                                            # 机器人只在到达安全抓取上方时消费最新的一份。
                                            publish_pregrasp_pose(frame_stamp)
                                else:
                                    now = time.time()
                                    if now - last_height_warning_time >= 1.0:
                                        logging.warning(f"[height] Rejected {measured_height*1000:.1f}mm outside configured range.")
                                        last_height_warning_time = now
                            else:
                                now = time.time()
                                if now - last_height_warning_time >= 1.0:
                                    logging.warning(
                                        f"[height] Evidence rejected: side_planes={last_side_planes}, "
                                        f"raw_side_points={len(raw_side_depths)}, "
                                        f"edge_side_points={len(edge_side_depths)}, "
                                        f"table_plane_points={table_plane_points}; pose not published."
                                    )
                                    last_height_warning_time = now

        if (
            sam_initialized
            and published_frames == 0
            and tracking_frames_without_height >= MAX_TRACKING_FRAMES_WITHOUT_HEIGHT
        ):
            logging.warning(
                f"[height] No valid height after {tracking_frames_without_height} tracked frames; "
                "resetting SAM2 so the next request starts cleanly."
            )
            publish_vision_result(False, "no_valid_height")
            locked_target_corners = None
            locked_target_id = -1
            locked_target_camera_x = None
            locked_target_ready = False
            locked_target_height_m = None
            locked_cloud_points = None
            locked_cloud_u_rgb = None
            locked_cloud_v_rgb = None
            locked_cloud_in_bounds = None
            locked_cloud_colors = None
            reset_requested = True

        if last_yolo_obbs is not None and time.time() - last_yolo_time <= YOLO_BOX_DISPLAY_TTL_S:
            for index, corners in enumerate(last_yolo_obbs):
                color = (0, 0, 255) if index == last_best_idx else (255, 0, 0)
                cv2.polylines(display, [np.int32(corners)], True, color, 3 if index == last_best_idx else 1)
                cv2.putText(display, f"ID:{index}", tuple(np.int32(corners[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif last_yolo_obbs is not None:
            last_yolo_obbs = None

        # Unlike the short-lived candidate boxes above, keep the chosen target
        # on screen throughout the robot's approach. This overlay is part of
        # both the local OpenCV window and the compressed ROS panel stream.
        if locked_target_corners is not None:
            target_corners_i = np.int32(np.round(locked_target_corners))
            target_color = (0, 0, 255) if locked_target_ready else (0, 165, 255)
            cv2.polylines(display, [target_corners_i], True, target_color, 4, cv2.LINE_AA)
            target_center = np.int32(np.round(np.mean(locked_target_corners, axis=0)))
            cv2.drawMarker(
                display,
                tuple(target_center),
                target_color,
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=3,
            )
            target_label = "NEXT GRASP" if locked_target_ready else "SELECTED / MEASURING"
            target_label += f" | ID:{locked_target_id}"
            if locked_target_camera_x is not None:
                target_label += f" | camX:{locked_target_camera_x:+.3f}m"
            if locked_target_height_m is not None:
                target_label += f" | H:{locked_target_height_m*1000:.1f}mm"
            if smooth_extent is not None:
                target_label += f" | L:{smooth_extent[0]*1000:.1f}mm"
            text_size, font_baseline_px = cv2.getTextSize(
                target_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
            )
            label_x = int(max(2, min(IMG_WIDTH - text_size[0] - 6, np.min(locked_target_corners[:, 0]))))
            label_y = int(max(text_size[1] + 8, np.min(locked_target_corners[:, 1]) - 8))
            cv2.rectangle(
                display,
                (label_x - 2, label_y - text_size[1] - 5),
                (label_x + text_size[0] + 3, label_y + font_baseline_px + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                display,
                target_label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                target_color,
                2,
                cv2.LINE_AA,
            )

        if (
            handoff_candidate_mask is not None
            and len(handoff_candidate_mask) == len(points_3d)
            and np.any(handoff_candidate_mask & in_bounds)
        ):
            projected_indices = np.flatnonzero(handoff_candidate_mask & in_bounds)
            # Keep the panel responsive even when a large robot-link surface
            # contributes thousands of points to the forbidden prism.
            draw_step = max(1, len(projected_indices) // 250)
            point_color = (255, 0, 255) if last_handoff_state == "BLOCKED" else (0, 165, 255)
            for point_index in projected_indices[::draw_step]:
                cv2.circle(
                    display,
                    (int(u_rgb[point_index]), int(v_rgb[point_index])),
                    2,
                    point_color,
                    -1,
                    cv2.LINE_AA,
                )

        draw_manual_roi(display)
        fps = 1.0 / max(1e-6, time.time() - loop_start)
        if last_handoff_state == "BLOCKED":
            status = "HANDOFF BLOCKED - WAITING FOR 102 RETREAT"
        elif last_handoff_state in ("VERIFYING_CLEAR", "WAIT_LIVE_CLOUD"):
            status = (
                f"HANDOFF VERIFYING CLEAR {handoff_clear_streak}/"
                f"{HANDOFF_CLEARANCE_CLEAR_FRAMES}"
            )
        elif locked_target_ready and sam_initialized:
            status = "NEXT GRASP TARGET TRACKING"
        elif sam_initialized:
            status = "TRACKING SELECTED TARGET"
        elif locked_target_ready:
            status = "NEXT GRASP TARGET LOCKED"
        elif locked_target_corners is not None:
            status = "SELECTED TARGET"
        else:
            status = "WAITING FOR TRIGGER"
        exposure_status = "AE" if AUTO_EXPOSURE else f"M Exp:{MANUAL_EXPOSURE:.0f} G:{MANUAL_GAIN:.0f}"
        active_roi = get_manual_roi()
        roi_status = "ROI:FULL" if active_roi is None else f"ROI:{active_roi[0]},{active_roi[1]}-{active_roi[2]},{active_roi[3]}"
        cv2.putText(display, f"{status} | FPS {fps:.1f} | {exposure_status} | {roi_status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0) if sam_initialized else (0, 165, 255), 2)
        cv2.putText(display, "drag mouse=lock ROI | select=min camera X inside ROI | r=clear ROI/reset | a=AE | q=quit", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1)
        if last_handoff_state not in ("IDLE", "WAIT_TARGET"):
            clearance_color = (0, 255, 0) if handoff_clearance_passed else (0, 0, 255)
            cv2.putText(
                display,
                f"handoff={last_handoff_state} points={last_handoff_candidate_points} "
                f"cluster={last_handoff_cluster_points} clear="
                f"{handoff_clear_streak}/{HANDOFF_CLEARANCE_CLEAR_FRAMES}",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                clearance_color,
                2,
            )
        if last_height_m is not None:
            length_text = "" if smooth_extent is None else f" length={smooth_extent[0]*1000:.1f}mm"
            cv2.putText(display, f"height={last_height_m*1000:.1f}mm{length_text} side_planes={last_side_planes} side_pts={last_side_points} depth=75%", (10, IMG_HEIGHT - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(display, f"height evidence: {last_height_evidence_mode}", (10, IMG_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)

        cloud_source = "LOCKED SNAPSHOT" if using_locked_cloud else "LIVE FFS"
        cloud_image = render_cloud(
            points_3d,
            colors,
            smooth_box_center,
            smooth_extent,
            smooth_rotation,
            source_label=cloud_source,
            tracked_target_points=tracked_target_points,
        )
        combined_local_view = compose_local_window(display, cloud_image)
        active_frame_info = build_fault_panel_info(
            display_status=status,
            fps=fps,
            exposure_status=exposure_status,
            active_roi=active_roi,
            cloud_source=cloud_source,
            frame_stamp=frame_stamp,
            point_count=len(points_3d),
            tracked_target_count=0 if tracked_target_points is None else len(tracked_target_points),
        )
        with fault_snapshot_lock:
            latest_panel_image = combined_local_view.copy()
            latest_panel_info = active_frame_info
            latest_panel_captured_at_epoch = float(active_frame_info["captured_at_epoch"])
        # The fault listener only queues an event.  Saving here keeps all image
        # operations in the camera/render thread and uses the completed RGB +
        # point-cloud frame rather than a partially rendered callback state.
        save_pending_fault_snapshots(combined_local_view, active_frame_info)

        publish_jpeg(rgb_pub, display, "d405_local_rgb")
        publish_jpeg(cloud_pub, cloud_image, "d405_local_cloud")
        if ENABLE_LOCAL_WINDOWS:
            cv2.imshow(LOCAL_WINDOW_NAME, combined_local_view)
            local_key = cv2.waitKey(1) & 0xFF
            if local_key != 0xFF:
                handle_key(local_key)

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    if ENABLE_LOCAL_WINDOWS:
        cv2.destroyAllWindows()
    executor.shutdown()
    ros_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join(timeout=1.0)
    logging.info("D405 cosmetic-box node exited.")
