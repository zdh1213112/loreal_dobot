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
import sys
import threading
import time
from collections import deque

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
YOLO_MODEL_PATH = "/home/zdh/yolo_one/yolo_train_web_github/outputs/train/obb_demo-6/weights/best.pt"
VALID_ITERS = 6
MAX_DISP = 192
ZNEAR = 0.16
ZFAR = 5.0
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
IR_PROJECTOR_ON = True

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
PUBLISH_FRAMES_BEFORE_RESET = 3
MAX_TRACKING_FRAMES_WITHOUT_HEIGHT = 18
YOLO_BOX_DISPLAY_TTL_S = 0.8

# Camera/panel parameters
AUTO_EXPOSURE = True
MANUAL_EXPOSURE = 11000.0
MANUAL_GAIN = 8.0
AUTO_WHITE_BALANCE = True
MASK_ALPHA = 0.5
MASK_COLOR_BGR = np.array([75, 70, 203], dtype=np.uint8)
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.float64) / 255.0
PANEL_JPEG_QUALITY = 80
CLOUD_VIEW_W = 720
CLOUD_VIEW_H = 540
ENABLE_LOCAL_WINDOWS = True
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
width_pub = ros_node.create_publisher(Float32, "/gripper_target_width", 10)
height_pub = ros_node.create_publisher(Float32, "/cosmetic_box_height", 10)
rgb_pub = ros_node.create_publisher(CompressedImage, "/vision_panel/d405_local_rgb/image/compressed", 1)
cloud_pub = ros_node.create_publisher(CompressedImage, "/vision_panel/d405_local_cloud/image/compressed", 1)

trigger_requested = False
quit_requested = False
reset_requested = False
clear_target_display_requested = False
manual_roi = None
roi_drawing = False
roi_start = (-1, -1)
roi_end = (-1, -1)
roi_lock = threading.Lock()


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
height_history = deque(maxlen=20)
published_frames = 0
last_height_m = None
last_side_points = 0
last_side_planes = 0
last_height_evidence_mode = "none"
tracking_frames_without_height = 0
last_height_warning_time = 0.0
last_cloud_fallback_time = 0.0

if ENABLE_LOCAL_WINDOWS:
    cv2.namedWindow("D405 Cosmetic Box RGB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("D405 Cosmetic Box RGB", IMG_WIDTH, IMG_HEIGHT)
    cv2.setMouseCallback("D405 Cosmetic Box RGB", mouse_callback)
    cv2.namedWindow("D405 Cosmetic Box Point Cloud", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("D405 Cosmetic Box Point Cloud", CLOUD_VIEW_W, CLOUD_VIEW_H)

try:
    while not quit_requested:
        loop_start = time.time()

        if trigger_requested:
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
            logging.info("[D405] Flushing 15 frames before YOLO capture...")
            for _ in range(15):
                pipeline.wait_for_frames()
            detection_frames = pipeline.wait_for_frames()
            detection_color = np.asanyarray(detection_frames.get_color_frame().get_data())
            results = yolo_model(detection_color, conf=0.1, verbose=False)
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
            else:
                pending_yolo_obbs = None
                last_yolo_obbs = None
                logging.warning("[YOLO] No cosmetic box detected.")
            trigger_requested = False

        frames = pipeline.wait_for_frames()
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())

        if reset_requested:
            try:
                sam2_predictor.reset_state()
            except KeyError:
                pass
            sam_initialized = False
            current_mask = None
            smooth_grasp = smooth_box_center = smooth_extent = smooth_rotation = None
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

        # The scene and camera remain stationary between target selection and
        # pose publication. If FFS temporarily returns no points after SAM2 is
        # initialized, reuse the valid cloud captured for this exact YOLO
        # selection instead of presenting a black cloud or fabricating depth.
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
                    f"[depth] Live FFS cloud has fewer than 40 points; using locked selection "
                    f"snapshot with {len(points_3d)} points."
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
                x1, y1 = np.min(corners, axis=0)
                x2, y2 = np.max(corners, axis=0)
                pending_bbox = (int(x1), int(y1), int(x2), int(y2))
                reset_requested = True
                logging.info(f"[3-D select] Selected ID={best_index}, the minimum valid camera-X target.")
            else:
                logging.warning("[3-D select] No YOLO candidate had enough valid stereo points; detection rejected.")
            pending_yolo_obbs = None

        display = color_bgr.copy()
        tracked_target_points = None
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
                                        # Publish only the initial stable samples.
                                        # Afterwards SAM2 remains alive solely for the
                                        # RGB/point-cloud tracking display until the
                                        # next trigger starts a fresh target session.
                                        if not locked_target_ready:
                                            pose_msg = PoseStamped()
                                            pose_msg.header.stamp = ros_node.get_clock().now().to_msg()
                                            pose_msg.header.frame_id = "camera_d405_link"
                                            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = map(float, smooth_grasp)
                                            quat = SciPyRot.from_matrix(smooth_rotation).as_quat()
                                            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y = float(quat[0]), float(quat[1])
                                            pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = float(quat[2]), float(quat[3])
                                            pose_pub.publish(pose_msg)
                                            width_msg = Float32()
                                            width_msg.data = float(min(MAX_GRIPPER_OPENING_M, smooth_extent[1] + GRIP_CLEARANCE_M))
                                            width_pub.publish(width_msg)
                                            height_msg = Float32()
                                            height_msg.data = last_height_m
                                            height_pub.publish(height_msg)
                                            published_frames += 1
                                            tracking_frames_without_height = 0
                                            if published_frames >= PUBLISH_FRAMES_BEFORE_RESET:
                                                locked_target_ready = True
                                                locked_target_height_m = last_height_m
                                                logging.info(
                                                    f"[D405] Published stable target: height={last_height_m*1000:.1f}mm, "
                                                    f"depth={GRASP_DEPTH_RATIO*100:.0f}%, evidence={last_height_evidence_mode}; "
                                                    "SAM2 display tracking remains active."
                                                )
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

        draw_manual_roi(display)
        fps = 1.0 / max(1e-6, time.time() - loop_start)
        if locked_target_ready and sam_initialized:
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
        if last_height_m is not None:
            cv2.putText(display, f"height={last_height_m*1000:.1f}mm side_planes={last_side_planes} side_pts={last_side_points} depth=75%", (10, IMG_HEIGHT - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(display, f"height evidence: {last_height_evidence_mode}", (10, IMG_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)

        publish_jpeg(rgb_pub, display, "d405_local_rgb")
        cloud_image = render_cloud(
            points_3d,
            colors,
            smooth_box_center,
            smooth_extent,
            smooth_rotation,
            source_label="LOCKED SNAPSHOT" if using_locked_cloud else "LIVE FFS",
            tracked_target_points=tracked_target_points,
        )
        publish_jpeg(cloud_pub, cloud_image, "d405_local_cloud")
        if ENABLE_LOCAL_WINDOWS:
            cv2.imshow("D405 Cosmetic Box RGB", display)
            cv2.imshow("D405 Cosmetic Box Point Cloud", cloud_image)
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
