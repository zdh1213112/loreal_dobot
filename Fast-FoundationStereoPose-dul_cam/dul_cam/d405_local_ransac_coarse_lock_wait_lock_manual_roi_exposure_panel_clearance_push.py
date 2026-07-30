"""
功能说明：D405 末端局部精定位增强版（等待最新粗定位锁定目标）。

主要流程：
1. 监听 ROS 2 话题 `/trigger_d405_vision`，收到 True 后启动一次 D405 精定位流程。
2. 监听 `/coarse_target_obj_for_d405`，接收 Nova5 控制端发布的粗定位锁定目标。
3. 与 `d405_local_ransac_coarse_lock.py` 相比，本版本增加 wait-lock 机制：
   触发后会短暂等待一帧最新的 `/coarse_target_obj_for_d405`，并检查消息新鲜度，
   降低使用旧粗定位目标导致 D405 ROI/目标选择错误的概率。
4. 使用 RealSense D405 双红外 + FFS 生成深度点云，YOLO-OBB 定位目标，SAM2 分割目标区域。
5. 对目标点云做多平面 RANSAC 姿态估计，选择稳定主平面/最宽面生成抓取姿态。
6. 发布精定位抓取结果到 `/target_pose_cam_fine`，frame 为 `camera_d405_link`。
7. 发布估计夹爪开口宽度到 `/gripper_target_width`，供 Nova5/DH 夹爪控制端使用。
8. 排除 SAM 目标点后检查夹爪闭合轴两侧点云；间隙不足时发布相机坐标系侧推向量到
   `/d405/grasp_clearance_push_cam`，向量为零表示可直接下夹爪。

典型配合：D435 粗定位节点 -> Nova5 粗定位移动并发布锁定目标 -> 本节点等待最新锁定目标后精定位。
"""

import os, sys, time, logging, json, threading
import numpy as np
import torch
import yaml
import cv2
import pyrealsense2 as rs
import open3d as o3d
from collections import deque

# ROS 2 和矩阵转换依赖
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Bool, String
from scipy.spatial.transform import Rotation as SciPyRot

# ===== 导入路径设置 =====
SAM2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== GPU config =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Parameters =====
FFS_MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"

VALID_ITERS = 6
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.16          
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]            
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.float64) / 255.0  
IR_PROJECTOR_ON = True  

# ==========================================
# ROS 2 初始化与触发回调机制
# ==========================================
rclpy.init()
ros_node = rclpy.create_node('d405_fine_vision_node')
ros_executor = SingleThreadedExecutor()
ros_executor.add_node(ros_node)
ros_spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
ros_spin_thread.start()
pose_pub = ros_node.create_publisher(PoseStamped, '/target_pose_cam_fine', 10)
width_pub = ros_node.create_publisher(Float32, '/gripper_target_width', 10)
clearance_pub = ros_node.create_publisher(Vector3Stamped, '/d405/grasp_clearance_push_cam', 10)
clearance_contact_pub = ros_node.create_publisher(Vector3Stamped, '/d405/grasp_clearance_contact_cam', 10)
panel_image_pub = ros_node.create_publisher(CompressedImage, '/vision_panel/d405_local_rgb/image/compressed', 1)
panel_cloud_pub = ros_node.create_publisher(CompressedImage, '/vision_panel/d405_local_cloud/image/compressed', 1)
panel_jpeg_quality = 80
CLOUD_VIEW_W = 720
CLOUD_VIEW_H = 540

global_trigger_flag = False
global_tracking_snapshot_flag = False
panel_quit_requested = False
locked_target_pose_msg = None
locked_target_msg_time = 0.0
LOCK_WAIT_TIMEOUT_S = 0.35
LOCK_FRESHNESS_WINDOW_S = 1.5
YOLO_BOX_DISPLAY_TTL_S = 0.5
AUTO_EXPOSURE = False
MANUAL_EXPOSURE = 11000.0
MANUAL_GAIN = 8.0
AUTO_WHITE_BALANCE = True
MIN_ROI_SIZE = 20
AUTO_RESET_AFTER_PUBLISH = False
PUBLISH_FRAMES_BEFORE_RESET = 2

# Clearance is measured from the SAM OBB side face along its local Y axis, which
# is also used below as the gripper closing direction.
CLEARANCE_REQUIRED_M = 0.020
CLEARANCE_SEARCH_EXTRA_M = 0.025
CLEARANCE_TARGET_EXCLUSION_M = 0.004
CLEARANCE_CORRIDOR_X_MARGIN_M = 0.012
CLEARANCE_GRIPPER_SPAN_X_M = 0.060
CLEARANCE_CORRIDOR_Z_MARGIN_M = 0.015
CLEARANCE_CORRIDOR_Z_MIN_FRACTION = -0.25
CLEARANCE_MIN_POINTS = 30
CLEARANCE_NEAR_PERCENTILE = 10.0
CLEARANCE_PUSH_EXTRA_M = 0.006
CLEARANCE_MAX_PUSH_M = 0.030
GRASP_FEEDBACK_DIR = "/home/zdh/ffs_ws/grasp_feedback/d405_clearance_push"
GRASP_FEEDBACK_MAX_RESULT_SAMPLES = 3


def estimate_grasp_clearance_push(points_3d, target_mask, center, extent, axes):
    """Return side clearances and a camera-frame push vector for the tighter side."""
    if points_3d is None or len(points_3d) == 0:
        return None

    scene_points = np.asarray(points_3d, dtype=np.float64)
    scene_points = scene_points[np.isfinite(scene_points).all(axis=1) & ~target_mask]
    if len(scene_points) == 0:
        return None

    local = (scene_points - center) @ axes
    half = 0.5 * np.asarray(extent, dtype=np.float64)
    corridor_half_x = min(
        half[0] + CLEARANCE_CORRIDOR_X_MARGIN_M,
        0.5 * CLEARANCE_GRIPPER_SPAN_X_M,
    )
    common = (
        # Only the target-center finger footprint matters. Obstacles beside a
        # long box end do not intersect the centered gripper descent path.
        (np.abs(local[:, 0]) <= corridor_half_x)
        # Ignore the broad support plane below the box while retaining nearby
        # box tops and upper side faces that a descending finger would strike.
        & (local[:, 2] >= CLEARANCE_CORRIDOR_Z_MIN_FRACTION * half[2])
        & (local[:, 2] <= half[2] + CLEARANCE_CORRIDOR_Z_MARGIN_M)
    )

    clearances = []
    counts = []
    search_limit = CLEARANCE_REQUIRED_M + CLEARANCE_SEARCH_EXTRA_M
    for sign in (-1.0, 1.0):
        outward = sign * local[:, 1] - half[1]
        candidates = outward[
            common
            & (outward >= CLEARANCE_TARGET_EXCLUSION_M)
            & (outward <= search_limit)
        ]
        counts.append(int(len(candidates)))
        if len(candidates) < CLEARANCE_MIN_POINTS:
            clearances.append(float('inf'))
        else:
            clearances.append(float(np.percentile(candidates, CLEARANCE_NEAR_PERCENTILE)))

    closest_index = int(np.argmin(clearances))
    closest_clearance = clearances[closest_index]
    push_cam = np.zeros(3, dtype=np.float64)
    object_push_distance = 0.0
    obstacle_contact_distance = 0.0
    contact_cam = np.zeros(3, dtype=np.float64)
    unsafe_sides = [np.isfinite(value) and value < CLEARANCE_REQUIRED_M for value in clearances]
    if np.isfinite(closest_clearance) and closest_clearance < CLEARANCE_REQUIRED_M:
        obstacle_sign = -1.0 if closest_index == 0 else 1.0
        object_push_distance = min(
            CLEARANCE_MAX_PUSH_M,
            CLEARANCE_REQUIRED_M - closest_clearance + CLEARANCE_PUSH_EXTRA_M,
        )
        obstacle_contact_distance = half[1] + closest_clearance
        contact_cam = obstacle_sign * axes[:, 1] * obstacle_contact_distance
        sweep_distance = obstacle_contact_distance + object_push_distance
        push_cam = obstacle_sign * axes[:, 1] * sweep_distance

    return {
        'status': 'blocked_both' if all(unsafe_sides) else ('blocked_one' if any(unsafe_sides) else 'clear'),
        'negative_clearance_m': clearances[0],
        'positive_clearance_m': clearances[1],
        'negative_points': counts[0],
        'positive_points': counts[1],
        'obstacle_contact_distance_m': obstacle_contact_distance,
        'object_push_distance_m': object_push_distance,
        'contact_cam_m': contact_cam,
        'push_cam_m': push_cam,
    }


def publish_clearance_result(result):
    if result is None:
        return
    push = result['push_cam_m']
    msg = Vector3Stamped()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    msg.header.frame_id = 'camera_d405_link'
    msg.vector.x, msg.vector.y, msg.vector.z = (float(push[0]), float(push[1]), float(push[2]))
    clearance_pub.publish(msg)
    contact = np.asarray(result['contact_cam_m'], dtype=np.float64)
    contact_msg = Vector3Stamped()
    contact_msg.header = msg.header
    contact_msg.vector.x, contact_msg.vector.y, contact_msg.vector.z = (
        float(contact[0]), float(contact[1]), float(contact[2])
    )
    clearance_contact_pub.publish(contact_msg)

    def fmt(value):
        return 'clear' if not np.isfinite(value) else f'{value * 1000.0:.1f}mm'

    logging.info(
        '[Clearance] status=%s, -Y=%s (%d pts), +Y=%s (%d pts), '
        'contact=%.1fmm object_push=%.1fmm sweep_cam=(%.1f, %.1f, %.1f)mm',
        result['status'],
        fmt(result['negative_clearance_m']), result['negative_points'],
        fmt(result['positive_clearance_m']), result['positive_points'],
        result['obstacle_contact_distance_m'] * 1000.0,
        result['object_push_distance_m'] * 1000.0,
        push[0] * 1000.0, push[1] * 1000.0, push[2] * 1000.0,
    )


def clearance_metadata(result):
    if result is None:
        return None

    def finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    return {
        'status': result['status'],
        'negative_clearance_m': finite_or_none(result['negative_clearance_m']),
        'positive_clearance_m': finite_or_none(result['positive_clearance_m']),
        'negative_points': int(result['negative_points']),
        'positive_points': int(result['positive_points']),
        'obstacle_contact_distance_m': float(result['obstacle_contact_distance_m']),
        'object_push_distance_m': float(result['object_push_distance_m']),
        'contact_cam_m': np.asarray(result['contact_cam_m'], dtype=float).tolist(),
        'push_cam_m': np.asarray(result['push_cam_m'], dtype=float).tolist(),
    }


def render_clearance_topdown(points_3d, target_mask, center, extent, axes, result):
    canvas_h, canvas_w = 620, 760
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, 'GRASP CLEARANCE - TARGET LOCAL XY', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    points = np.asarray(points_3d, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    local = (points - center) @ axes
    half = 0.5 * np.asarray(extent, dtype=np.float64)
    corridor_half_x = min(
        half[0] + CLEARANCE_CORRIDOR_X_MARGIN_M,
        0.5 * CLEARANCE_GRIPPER_SPAN_X_M,
    )
    x_limit = max(0.06, half[0] + CLEARANCE_CORRIDOR_X_MARGIN_M + 0.015)
    y_limit = max(0.06, half[1] + CLEARANCE_REQUIRED_M + CLEARANCE_SEARCH_EXTRA_M + 0.010)
    plot_left, plot_top, plot_right, plot_bottom = 55, 60, canvas_w - 35, canvas_h - 95

    def project(local_xy):
        px = plot_left + (local_xy[:, 0] + x_limit) / (2.0 * x_limit) * (plot_right - plot_left)
        py = plot_bottom - (local_xy[:, 1] + y_limit) / (2.0 * y_limit) * (plot_bottom - plot_top)
        return np.column_stack((px, py)).astype(np.int32)

    z_common = (
        (local[:, 2] >= CLEARANCE_CORRIDOR_Z_MIN_FRACTION * half[2])
        & (local[:, 2] <= half[2] + CLEARANCE_CORRIDOR_Z_MARGIN_M)
    )
    visible = z_common & (np.abs(local[:, 0]) <= x_limit) & (np.abs(local[:, 1]) <= y_limit)
    sample_indices = np.flatnonzero(visible)[::max(1, int(np.count_nonzero(visible) / 12000))]
    for point in project(local[sample_indices, :2]):
        cv2.circle(canvas, tuple(point), 1, (95, 95, 95), -1)

    side_colors = ((255, 170, 0), (0, 140, 255))
    for side_index, sign in enumerate((-1.0, 1.0)):
        outward = sign * local[:, 1] - half[1]
        candidate = (
            ~target_mask
            & z_common
            & (np.abs(local[:, 0]) <= corridor_half_x)
            & (outward >= CLEARANCE_TARGET_EXCLUSION_M)
            & (outward <= CLEARANCE_REQUIRED_M + CLEARANCE_SEARCH_EXTRA_M)
        )
        for point in project(local[candidate, :2]):
            cv2.circle(canvas, tuple(point), 2, side_colors[side_index], -1)

    for point in project(local[target_mask & visible, :2]):
        cv2.circle(canvas, tuple(point), 1, (80, 80, 230), -1)

    target_corners = np.array([
        [-half[0], -half[1]], [half[0], -half[1]],
        [half[0], half[1]], [-half[0], half[1]],
    ])
    cv2.polylines(canvas, [project(target_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    required_corners = np.array([
        [-corridor_half_x, -half[1] - CLEARANCE_REQUIRED_M],
        [corridor_half_x, -half[1] - CLEARANCE_REQUIRED_M],
        [corridor_half_x, half[1] + CLEARANCE_REQUIRED_M],
        [-corridor_half_x, half[1] + CLEARANCE_REQUIRED_M],
    ])
    cv2.polylines(canvas, [project(required_corners)], True, (0, 210, 210), 1, cv2.LINE_AA)

    push_local = np.asarray(result['push_cam_m']) @ axes
    if np.linalg.norm(push_local[:2]) > 1e-6:
        arrow = project(np.array([[0.0, 0.0], push_local[:2]]))
        cv2.arrowedLine(canvas, tuple(arrow[0]), tuple(arrow[1]), (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.25)
        contact_sign = 1.0 if push_local[1] >= 0.0 else -1.0
        contact_local = np.array([[0.0, contact_sign * result['obstacle_contact_distance_m']]])
        contact_pixel = project(contact_local)[0]
        cv2.circle(canvas, tuple(contact_pixel), 8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'CONTACT', (contact_pixel[0] + 10, contact_pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"status: {result['status']}", (18, canvas_h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"-Y points={result['negative_points']}  +Y points={result['positive_points']}  "
        f"contact={result['obstacle_contact_distance_m'] * 1000.0:.1f}mm  "
        f"push={result['object_push_distance_m'] * 1000.0:.1f}mm",
        (18, canvas_h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA,
    )
    return canvas


def trigger_callback(msg):
    global global_trigger_flag
    if msg.data:
        global_trigger_flag = True
        logging.info(">>>> [ROS 2] 收到控制端触发信号！即将启动高精度视觉流水线... <<<<")

def tracking_snapshot_callback(msg):
    global global_tracking_snapshot_flag
    if msg.data:
        global_tracking_snapshot_flag = True
        logging.info("[D405 Track] 收到侧推后跟踪快照请求；保持当前 SAM 目标，不重新运行 YOLO/SAM 初始化。")

def coarse_target_lock_callback(msg):
    global locked_target_pose_msg, locked_target_msg_time
    locked_target_pose_msg = msg
    locked_target_msg_time = time.time()
    pos = msg.pose.position
    logging.info(
        f"[D405 Lock] 收到粗定位锁定目标: frame={msg.header.frame_id or 'none'} "
        f"x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}"
    )

def panel_event_callback(msg):
    global need_reset, panel_quit_requested
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
        key = int(event.get("key", -1))
        if key in (ord('r'), ord('R')):
            clear_roi()
            need_reset = True
        elif key == ord('['):
            adjust_manual_exposure(exposure_delta=-50.0)
        elif key == ord(']'):
            adjust_manual_exposure(exposure_delta=50.0)
        elif key == ord('-'):
            adjust_manual_exposure(gain_delta=-1.0)
        elif key in (ord('='), ord('+')):
            adjust_manual_exposure(gain_delta=1.0)
        elif key in (ord('q'), ord('Q')):
            panel_quit_requested = True

def publish_compressed(pub, image, frame_id):
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), panel_jpeg_quality])
    if not ok:
        return
    msg = CompressedImage()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = encoded.tobytes()
    pub.publish(msg)

def publish_panel_frame(image):
    publish_compressed(panel_image_pub, image, "d405_local_rgb")

def render_cloud_panel(points_3d, colors, obb_center=None, obb_extent=None, obb_R=None):
    canvas = np.zeros((CLOUD_VIEW_H, CLOUD_VIEW_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    cv2.putText(canvas, "D405 POINT CLOUD", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

    if points_3d is None or len(points_3d) == 0:
        cv2.putText(canvas, "NO VALID POINTS", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)
        return canvas

    points = np.asarray(points_3d, dtype=np.float64)
    point_colors = np.asarray(colors, dtype=np.float64) if colors is not None and len(colors) == len(points) else np.ones_like(points) * 0.7

    stride = max(1, len(points) // 22000)
    points = points[::stride]
    point_colors = point_colors[::stride]

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    point_colors = point_colors[finite]
    if len(points) == 0:
        return canvas

    center = np.array([0.0, 0.0, 0.65], dtype=np.float64)
    rel = points - center
    yaw = np.deg2rad(-35.0)
    pitch = np.deg2rad(-22.0)
    rot_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rot_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    view = rel @ (rot_yaw @ rot_pitch).T

    scale = 520.0
    px = (CLOUD_VIEW_W * 0.52 + view[:, 0] * scale).astype(np.int32)
    py = (CLOUD_VIEW_H * 0.58 - view[:, 1] * scale).astype(np.int32)
    depth_order = np.argsort(view[:, 2])[::-1]

    bgr = np.clip(point_colors[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
    in_view = (px >= 0) & (px < CLOUD_VIEW_W) & (py >= 0) & (py < CLOUD_VIEW_H)
    for idx in depth_order[in_view[depth_order]]:
        canvas[py[idx], px[idx]] = bgr[idx].tolist()

    if obb_center is not None and obb_extent is not None and obb_R is not None:
        corners_local = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)], dtype=np.float64) * obb_extent
        corners = corners_local @ obb_R.T + obb_center
        rel_c = corners - center
        view_c = rel_c @ (rot_yaw @ rot_pitch).T
        pts2 = np.column_stack((CLOUD_VIEW_W * 0.52 + view_c[:, 0] * scale, CLOUD_VIEW_H * 0.58 - view_c[:, 1] * scale)).astype(np.int32)
        edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
        for a, b in edges:
            cv2.line(canvas, tuple(pts2[a]), tuple(pts2[b]), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(canvas, f"points: {len(points)}", (18, CLOUD_VIEW_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas

def publish_cloud_frame(points_3d, colors, obb_center=None, obb_extent=None, obb_R=None):
    cloud_image = render_cloud_panel(points_3d, colors, obb_center, obb_extent, obb_R)
    publish_compressed(panel_cloud_pub, cloud_image, "d405_local_cloud")
    return cloud_image


def save_grasp_feedback(
    capture_dir,
    raw_bgr,
    annotated_bgr,
    mask,
    cloud_image,
    points_3d,
    colors,
    target_point_mask,
    obb_center,
    obb_extent,
    obb_rotation,
    clearance_result,
    grip_width_m,
):
    os.makedirs(capture_dir, exist_ok=True)
    clearance_topdown = render_clearance_topdown(
        points_3d,
        target_point_mask,
        obb_center,
        obb_extent,
        obb_rotation,
        clearance_result,
    )
    image_outputs = {
        '01_rgb_raw.png': raw_bgr,
        '02_rgb_annotated.png': annotated_bgr,
        '03_sam_mask.png': (mask.astype(np.uint8) * 255),
        '04_point_cloud.png': cloud_image,
        '04b_clearance_topdown.png': clearance_topdown,
    }
    for filename, image in image_outputs.items():
        if image is None or not cv2.imwrite(os.path.join(capture_dir, filename), image):
            raise RuntimeError(f'Failed to save feedback image {filename}')

    np.savez_compressed(
        os.path.join(capture_dir, '05_point_cloud_data.npz'),
        points_camera_m=np.asarray(points_3d, dtype=np.float32),
        colors_rgb=np.asarray(colors, dtype=np.float32),
        sam_target_mask=np.asarray(target_point_mask, dtype=np.uint8),
        obb_center_camera_m=np.asarray(obb_center, dtype=np.float64),
        obb_extent_m=np.asarray(obb_extent, dtype=np.float64),
        obb_rotation_camera=np.asarray(obb_rotation, dtype=np.float64),
    )

    lock_position = None
    if locked_target_pose_msg is not None:
        position = locked_target_pose_msg.pose.position
        lock_position = [float(position.x), float(position.y), float(position.z)]
    metadata = {
        'capture_time_local': time.strftime('%Y-%m-%d %H:%M:%S'),
        'camera_frame_id': 'camera_d405_link',
        'feedback_directory': capture_dir,
        'locked_target_camera_m': lock_position,
        'gripper_width_command_m': float(grip_width_m),
        'obb_center_camera_m': np.asarray(obb_center, dtype=float).tolist(),
        'obb_extent_m': np.asarray(obb_extent, dtype=float).tolist(),
        'obb_rotation_camera': np.asarray(obb_rotation, dtype=float).tolist(),
        'clearance': clearance_metadata(clearance_result),
        'clearance_parameters': {
            'required_m': CLEARANCE_REQUIRED_M,
            'search_extra_m': CLEARANCE_SEARCH_EXTRA_M,
            'target_exclusion_m': CLEARANCE_TARGET_EXCLUSION_M,
            'corridor_x_margin_m': CLEARANCE_CORRIDOR_X_MARGIN_M,
            'gripper_span_x_m': CLEARANCE_GRIPPER_SPAN_X_M,
            'corridor_z_margin_m': CLEARANCE_CORRIDOR_Z_MARGIN_M,
            'corridor_z_min_fraction': CLEARANCE_CORRIDOR_Z_MIN_FRACTION,
            'min_points': CLEARANCE_MIN_POINTS,
            'near_percentile': CLEARANCE_NEAR_PERCENTILE,
            'push_extra_m': CLEARANCE_PUSH_EXTRA_M,
            'max_push_m': CLEARANCE_MAX_PUSH_M,
        },
    }
    with open(os.path.join(capture_dir, '06_metadata.json'), 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    logging.info('[Feedback] 已保存本次抓取关键数据: %s', capture_dir)

trigger_sub = ros_node.create_subscription(Bool, '/trigger_d405_vision', trigger_callback, 10)
tracking_snapshot_sub = ros_node.create_subscription(
    Bool, '/capture_d405_tracking_snapshot', tracking_snapshot_callback, 10
)
coarse_lock_sub = ros_node.create_subscription(
    PoseStamped, '/coarse_target_obj_for_d405', coarse_target_lock_callback, 10
)
panel_event_sub = ros_node.create_subscription(String, '/vision_panel/d405_local_rgb/event', panel_event_callback, 10)
logging.info("ROS 2 节点初始化完成，正在监听触发信号: /trigger_d405_vision")

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)
with open(os.path.join(os.path.dirname(FFS_MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

ffs_model = torch.load(FFS_MODEL_DIR, map_location='cpu', weights_only=False)
ffs_model.args.valid_iters = VALID_ITERS
ffs_model.args.max_disp = MAX_DISP
ffs_model.cuda().eval()

# ===== 2. Load YOLO-OBB model =====
from ultralytics import YOLO
logging.info("Loading YOLO-OBB model...")
# yolo_model = YOLO("/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-6/weights/best.pt")
# yolo_model = YOLO("/home/zdh/ultralytics/runs/obb/train11/weights/best.pt")
# yolo_model = YOLO("/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-12/weights/best.pt")
yolo_model = YOLO("/home/zdh/yolo_one/yolo_train_xense_load_image/outputs/train/obb_demo625-2/weights/best.pt")

# ===== 3. Load SAM2 model =====
logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0

# ===== 4. Initialize RealSense D405 =====
logging.info("Initializing RealSense D405...")
pipeline = rs.pipeline()
config = rs.config()
# config.enable_device('409122274792')
config.enable_device('352122272611')

config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)       

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)

exposure_sensors = []

def set_sensor_option_safe(sensor, option, value):
    try:
        option_range = sensor.get_option_range(option)
        clamped = max(option_range.min, min(option_range.max, float(value)))
        sensor.set_option(option, clamped)
        return True
    except Exception as exc:
        logging.warning(f"设置相机参数失败: {option}={value}, {exc}")
        return False

def apply_camera_exposure_settings(sensor):
    set_sensor_option_safe(sensor, rs.option.enable_auto_exposure, 1.0 if AUTO_EXPOSURE else 0.0)
    if AUTO_EXPOSURE:
        logging.info("相机曝光: auto_exposure=True，使用 RealSense 自动曝光")
        return
    if sensor.supports(rs.option.exposure):
        set_sensor_option_safe(sensor, rs.option.exposure, MANUAL_EXPOSURE)
    if sensor.supports(rs.option.gain):
        set_sensor_option_safe(sensor, rs.option.gain, MANUAL_GAIN)
    logging.info(f"相机曝光: auto_exposure=False exposure={MANUAL_EXPOSURE:.1f} gain={MANUAL_GAIN:.1f}")

for sensor in profile.get_device().query_sensors():
    if sensor.supports(rs.option.enable_auto_exposure):
        exposure_sensors.append(sensor)
        apply_camera_exposure_settings(sensor)
    if sensor.supports(rs.option.enable_auto_white_balance):
        set_sensor_option_safe(sensor, rs.option.enable_auto_white_balance, 1.0 if AUTO_WHITE_BALANCE else 0.0)

def adjust_manual_exposure(exposure_delta=0.0, gain_delta=0.0):
    global MANUAL_EXPOSURE, MANUAL_GAIN
    if not exposure_sensors or AUTO_EXPOSURE:
        return
    try:
        if exposure_delta:
            ranges = [sensor.get_option_range(rs.option.exposure) for sensor in exposure_sensors if sensor.supports(rs.option.exposure)]
            if ranges:
                min_exposure = max(option_range.min for option_range in ranges)
                max_exposure = min(option_range.max for option_range in ranges)
                MANUAL_EXPOSURE = max(min_exposure, min(max_exposure, MANUAL_EXPOSURE + exposure_delta))
                for sensor in exposure_sensors:
                    if sensor.supports(rs.option.exposure):
                        sensor.set_option(rs.option.exposure, MANUAL_EXPOSURE)
        if gain_delta:
            ranges = [sensor.get_option_range(rs.option.gain) for sensor in exposure_sensors if sensor.supports(rs.option.gain)]
            if ranges:
                min_gain = max(option_range.min for option_range in ranges)
                max_gain = min(option_range.max for option_range in ranges)
                MANUAL_GAIN = max(min_gain, min(max_gain, MANUAL_GAIN + gain_delta))
                for sensor in exposure_sensors:
                    if sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, MANUAL_GAIN)
        logging.info(f"当前手动曝光: exposure={MANUAL_EXPOSURE:.1f} gain={MANUAL_GAIN:.1f}")
    except Exception as exc:
        logging.warning(f"手动曝光调节失败: {exc}")

# ===== 5. Get camera intrinsics and extrinsics =====
frames = pipeline.wait_for_frames()
ir_left_profile = frames.get_infrared_frame(1).get_profile().as_video_stream_profile()
color_profile = frames.get_color_frame().get_profile().as_video_stream_profile()

ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([[ir_intrinsics.fx, 0, ir_intrinsics.ppx], [0, ir_intrinsics.fy, ir_intrinsics.ppy], [0, 0, 1]], dtype=np.float32)

color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx], [0, color_intrinsics.fy, color_intrinsics.ppy], [0, 0, 1]], dtype=np.float32)

extrinsics = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(extrinsics.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(extrinsics.translation).astype(np.float32)

ir_right_profile = frames.get_infrared_frame(2).get_profile().as_video_stream_profile()
baseline = abs(ir_left_profile.get_extrinsics_to(ir_right_profile).translation[0])

fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]

u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 6. Warm up FFS =====
dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
d0, d1 = padder.pad(dummy, dummy)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = ffs_model.forward(d0, d1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
torch.cuda.empty_cache()

# ===== 7. Open3D geometry helpers =====
ENABLE_OPEN3D_WINDOW = False
vis = None
if ENABLE_OPEN3D_WINDOW:
    vis = o3d.visualization.Visualizer()
    vis.create_window("D405 Point Cloud", width=720, height=540, left=700, top=50)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
obb_lineset = o3d.geometry.LineSet()
if vis is not None:
    vis.add_geometry(pcd)
    vis.add_geometry(obb_lineset)
    vis.get_render_option().line_width = 5.0

obb_smooth_center = obb_smooth_extent = obb_smooth_R = None
OBB_SMOOTH = 0.65  

extent_history = deque(maxlen=20)
extent_frame_count = 0

def create_camera_frustum(fx_, fy_, cx_, cy_, w, h, scale=0.15):
    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts = [[(u - cx_) / fx_ * scale, -(v - cy_) / fy_ * scale, scale] for u, v in corners_2d]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([[0,0,0]] + pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))
    return ls

if vis is not None:
    vis.add_geometry(create_camera_frustum(fx_ir, fy_ir, cx_ir, cy_ir, IMG_WIDTH, IMG_HEIGHT))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
pca_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
if vis is not None:
    vis.add_geometry(pca_frame)

logging.info("D405 local panel mode: publishing RGB and point cloud panels")

drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = pending_point = current_mask = None
sam2_initialized = need_reset = False
last_yolo_obbs = None
last_best_idx = -1
last_locked_target_uv = None
last_yolo_obbs_time = 0.0
manual_roi = None
published_pose_frames = 0
feedback_trigger_index = 0
pending_feedback_dir = None
feedback_saved_for_trigger = True
feedback_capture_armed = False
feedback_result_sample_count = 0


def begin_feedback_capture():
    global feedback_trigger_index, pending_feedback_dir, feedback_saved_for_trigger
    global feedback_capture_armed, feedback_result_sample_count
    feedback_trigger_index += 1
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    milliseconds = int((time.time() % 1.0) * 1000.0)
    directory_name = f'{timestamp}_{milliseconds:03d}_trigger_{feedback_trigger_index:04d}'
    pending_feedback_dir = os.path.join(GRASP_FEEDBACK_DIR, directory_name)
    feedback_saved_for_trigger = False
    feedback_capture_armed = False
    feedback_result_sample_count = 0
    os.makedirs(pending_feedback_dir, exist_ok=True)
    logging.info('[Feedback] 本次触发关键帧将保存到: %s', pending_feedback_dir)


def save_trigger_feedback(raw_bgr, status, details=None):
    if pending_feedback_dir is None:
        return
    image_path = os.path.join(pending_feedback_dir, '00_trigger_rgb.png')
    if not cv2.imwrite(image_path, raw_bgr):
        logging.error('[Feedback] 无法保存触发原图: %s', image_path)
    payload = {
        'capture_time_local': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'details': details or {},
    }
    with open(os.path.join(pending_feedback_dir, '00_trigger_status.json'), 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def project_locked_target_to_pixel():
    if locked_target_pose_msg is None:
        return None
    frame_id = locked_target_pose_msg.header.frame_id.strip() or "none"
    if frame_id != "camera_d405_link":
        logging.warning(f"[D405 Lock] 锁定目标 frame_id={frame_id} 不是 camera_d405_link，回退到中心选框。")
        return None

    x = float(locked_target_pose_msg.pose.position.x)
    y = float(locked_target_pose_msg.pose.position.y)
    z = float(locked_target_pose_msg.pose.position.z)
    if z <= 1e-6:
        logging.warning("[D405 Lock] 锁定目标 Z 非法，回退到中心选框。")
        return None

    u = float(K_color[0, 0] * x / z + K_color[0, 2])
    v = float(K_color[1, 1] * y / z + K_color[1, 2])
    if not np.isfinite(u) or not np.isfinite(v):
        logging.warning("[D405 Lock] 锁定目标投影无效，回退到中心选框。")
        return None
    return u, v

def wait_for_recent_locked_target(previous_msg_time, timeout_s=LOCK_WAIT_TIMEOUT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if locked_target_pose_msg is not None and locked_target_msg_time > previous_msg_time:
            return True
        time.sleep(0.01)
    return False

def current_locked_target_uv():
    if locked_target_pose_msg is None:
        return None
    if time.time() - locked_target_msg_time > LOCK_FRESHNESS_WINDOW_S:
        return None
    return project_locked_target_to_pixel()

def clamp_point_to_frame(x, y):
    return max(0, min(IMG_WIDTH - 1, int(x))), max(0, min(IMG_HEIGHT - 1, int(y)))

def normalize_roi(start, end):
    x0, y0 = clamp_point_to_frame(*start)
    x1, y1 = clamp_point_to_frame(*end)
    left = max(0, min(x0, x1))
    top = max(0, min(y0, y1))
    right = min(IMG_WIDTH, max(x0, x1) + 1)
    bottom = min(IMG_HEIGHT, max(y0, y1) + 1)
    if right - left <= MIN_ROI_SIZE or bottom - top <= MIN_ROI_SIZE:
        return None
    return left, top, right, bottom

def clear_roi():
    global manual_roi, drawing
    if manual_roi is None:
        return
    manual_roi = None
    drawing = False
    logging.info("ROI Cleared by user.")

def draw_roi_overlay(image):
    roi_color = (0, 200, 255)
    if drawing:
        roi = normalize_roi((ix, iy), (fx_mouse, fy_mouse))
        if roi is not None:
            left, top, right, bottom = roi
            cv2.rectangle(image, (left, top), (right, bottom), roi_color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (ix, iy), (fx_mouse, fy_mouse), roi_color, 1, cv2.LINE_AA)
    elif manual_roi is not None:
        left, top, right, bottom = manual_roi
        cv2.rectangle(image, (left, top), (right, bottom), roi_color, 2, cv2.LINE_AA)
        cv2.putText(image, "ROI LOCKED", (left, max(15, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2, cv2.LINE_AA)

def draw_hud(display, fps, tracking_enabled):
    cv2.putText(display, f"FPS: {fps:.1f}", (IMG_WIDTH - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    status_text = "TRACKING | drag ROI, r=clear/reset, [] exp, -/+ gain, q=quit" if tracking_enabled else "Waiting | drag ROI, [] exp, -/+ gain, q=quit"
    cv2.putText(display, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if tracking_enabled else (0, 165, 255), 2)
    exposure_text = "AE" if AUTO_EXPOSURE else f"Exp:{MANUAL_EXPOSURE:.0f} Gain:{MANUAL_GAIN:.0f}"
    roi_text = "ROI:ON/R" if manual_roi is not None else "ROI:Drag"
    cv2.putText(display, f"{exposure_text} | {roi_text}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point, manual_roi, need_reset
    point = clamp_point_to_frame(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = fx_mouse, fy_mouse = point[0], point[1]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx_mouse, fy_mouse = point[0], point[1]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = point[0], point[1]
        roi = normalize_roi((ix, iy), (fx_mouse, fy_mouse))
        if roi is not None:
            manual_roi = roi
            need_reset = True
            logging.info(f"ROI Locked: {manual_roi}")
        else:
            pending_point = point
            logging.info("ROI too small; using click point prompt for SAM2.")

first_frame = True
frame_count = 0

try:
    while not panel_quit_requested:
        published_this_frame = False
        clearance_result = None
        feedback_target_point_mask = None
        t0 = time.time()

        # =========================================================
        # YOLO 触发逻辑
        # =========================================================
        if global_trigger_flag:
            begin_feedback_capture()
            logging.info("\n" + "="*50)
            logging.info(">>>> 正在清空历史运动残影，准备拍照... <<<<")
            previous_lock_time = locked_target_msg_time
            got_fresh_lock = wait_for_recent_locked_target(previous_lock_time)
            if got_fresh_lock:
                logging.info("🔒 [D405 Lock] 已等到最新粗定位锁定目标，按先验约束选择 YOLO 目标。")
            elif locked_target_pose_msg is not None:
                logging.warning("⚠️ [D405 Lock] 触发后未等到更新锁定消息，继续沿用上一条锁定目标。")
            else:
                logging.warning("⚠️ [D405 Lock] 触发后未收到任何锁定目标，可能回退到中心选框。")
            for _ in range(15):
                pipeline.wait_for_frames()
            
            frames = pipeline.wait_for_frames()
            color_bgr = np.asanyarray(frames.get_color_frame().get_data())
            
            results = yolo_model(color_bgr, conf=0.1, verbose=False)
            
            if len(results) > 0 and results[0].obb is not None and len(results[0].obb) > 0:
                obbs = results[0].obb
                num_objs = len(obbs)
                logging.info(f"🎯 [YOLO] 视野内共检测到 {num_objs} 个目标。")
                
                centers_x = obbs.xyxyxyxy[:, :, 0].mean(dim=1)
                centers_y = obbs.xyxyxyxy[:, :, 1].mean(dim=1)
                last_locked_target_uv = current_locked_target_uv()

                best_idx = None
                roi_rejected = False
                if manual_roi is not None:
                    left, top, right, bottom = manual_roi
                    roi_mask = (centers_x >= left) & (centers_x <= right) & (centers_y >= top) & (centers_y <= bottom)
                    if bool(torch.any(roi_mask)):
                        center_u = 0.5 * (left + right)
                        center_v = 0.5 * (top + bottom)
                        dists = (centers_x - center_u)**2 + (centers_y - center_v)**2
                        dists = torch.where(roi_mask, dists, torch.full_like(dists, float("inf")))
                        best_idx = torch.argmin(dists).item()
                        logging.info(
                            f"💡 [选择依据] 使用手动 ROI {manual_roi}，选择 ROI 内最近中心的 YOLO 目标 ID:{best_idx}。"
                        )
                    else:
                        roi_rejected = True
                        logging.warning(f"⚠️ 手动 ROI {manual_roi} 内没有 YOLO 目标，本次精定位不回退到 ROI 外。")

                if roi_rejected:
                    last_yolo_obbs = None
                    last_best_idx = -1
                    last_yolo_obbs_time = 0.0
                elif best_idx is None and last_locked_target_uv is not None:
                    lock_u, lock_v = last_locked_target_uv
                    dists = (centers_x - lock_u)**2 + (centers_y - lock_v)**2
                    best_idx = torch.argmin(dists).item()
                    logging.info(
                        f"💡 [选择依据] 使用粗定位锁定目标投影点 ({lock_u:.1f}, {lock_v:.1f})，"
                        f"选择最近的 YOLO 目标 ID:{best_idx}。"
                    )
                elif best_idx is None:
                    dists = (centers_x - 320)**2 + (centers_y - 240)**2
                    best_idx = torch.argmin(dists).item()
                    logging.info(f"💡 [选择依据] 未拿到有效锁定目标，回退为画面中心最近的目标 ID:{best_idx}。")
                
                if best_idx is not None:
                    last_yolo_obbs = obbs.xyxyxyxy.cpu().numpy()
                    last_best_idx = best_idx
                    last_yolo_obbs_time = time.time()

                    corners = last_yolo_obbs[best_idx]
                    x1, y1 = np.min(corners, axis=0)
                    x2, y2 = np.max(corners, axis=0)
                    pending_bbox = (int(x1), int(y1), int(x2), int(y2))
                    need_reset = True
                    feedback_capture_armed = True
                    save_trigger_feedback(
                        color_bgr,
                        'yolo_target_selected',
                        {'best_index': int(best_idx), 'pending_bbox_xyxy': list(pending_bbox)},
                    )
                    
                    logging.info("✅ YOLO 锁定目标，即将移交 SAM2 进行实时跟踪...")
                else:
                    save_trigger_feedback(color_bgr, 'no_yolo_target_selected')
                    feedback_saved_for_trigger = True
                    logging.warning("❌ 精定位失败：未在手动 ROI 内选中有效 YOLO 目标。")
                logging.info("="*50 + "\n")
            else:
                save_trigger_feedback(color_bgr, 'yolo_no_detection')
                feedback_saved_for_trigger = True
                logging.warning("❌ 精定位失败：YOLO 未能在画面中检测到目标。")
                last_yolo_obbs = None
                last_best_idx = -1
                last_yolo_obbs_time = 0.0
                
            global_trigger_flag = False
        # =========================================================

        frames = pipeline.wait_for_frames()
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())

        if global_tracking_snapshot_flag:
            begin_feedback_capture()
            feedback_capture_armed = bool(sam2_initialized)
            if feedback_capture_armed:
                save_trigger_feedback(color_bgr, 'sam_tracking_refresh')
            else:
                save_trigger_feedback(color_bgr, 'sam_tracking_unavailable')
                feedback_saved_for_trigger = True
                logging.error('[D405 Track] 当前没有已初始化的 SAM 跟踪目标，无法执行侧推后复检。')
            global_tracking_snapshot_flag = False

        if need_reset:
            try:
                sam2_predictor.reset_state()
            except KeyError:
                pass  
            sam2_initialized = need_reset = False
            current_mask = obb_smooth_center = obb_smooth_extent = obb_smooth_R = None
            published_pose_frames = 0
            extent_history.clear()
            extent_frame_count = 0

        if pending_bbox is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            bbox_arr = np.array([[pending_bbox[0], pending_bbox[1]], [pending_bbox[2], pending_bbox[3]]], dtype=np.float32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox_arr)
            sam2_initialized = True
            pending_bbox = None

        elif pending_point is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(color_bgr)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, points=np.array([[pending_point[0], pending_point[1]]], dtype=np.float32), labels=np.array([1], dtype=np.int32))
            sam2_initialized = True
            pending_point = None

        if sam2_initialized:
            out_obj_ids, out_mask_logits = sam2_predictor.track(color_bgr)
            current_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze() if len(out_obj_ids) > 0 else None

        display = color_bgr.copy()
        
        if current_mask is not None and np.any(current_mask):
            overlay = display.copy()
            overlay[current_mask > 0] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        if last_yolo_obbs is not None and time.time() - last_yolo_obbs_time > YOLO_BOX_DISPLAY_TTL_S:
            last_yolo_obbs = None
            last_best_idx = -1
            last_yolo_obbs_time = 0.0

        if last_yolo_obbs is not None:
            for i, corners in enumerate(last_yolo_obbs):
                corners_int = np.int32(corners)
                color = (0, 0, 255) if i == last_best_idx else (255, 0, 0)
                thickness = 3 if i == last_best_idx else 1
                cv2.polylines(display, [corners_int], isClosed=True, color=color, thickness=thickness)
                cv2.putText(display, f"ID:{i}", tuple(corners_int[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == last_best_idx else 1)

        display_locked_target_uv = current_locked_target_uv()
        if display_locked_target_uv is not None:
            u_lock = int(round(display_locked_target_uv[0]))
            v_lock = int(round(display_locked_target_uv[1]))
            if -50 <= u_lock < IMG_WIDTH + 50 and -50 <= v_lock < IMG_HEIGHT + 50:
                cv2.drawMarker(
                    display,
                    (u_lock, v_lock),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=22,
                    thickness=2,
                )
                cv2.putText(
                    display,
                    "LOCKED_COARSE_TARGET",
                    (max(0, u_lock - 80), max(20, v_lock - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        draw_roi_overlay(display)

        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())
        left_rgb = np.stack([ir_left] * 3, axis=-1)
        right_rgb = np.stack([ir_right] * 3, axis=-1)
        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(IMG_HEIGHT, IMG_WIDTH).clip(0, None)

        depth = fx_ir * baseline / (disp + 1e-6)
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0
        depth[(np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)) > 0.5) | (np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)) > 0.5)] = 0

        z_flat = depth[::PCD_STRIDE, ::PCD_STRIDE].reshape(-1)
        valid_mask = z_flat > 0
        z, u, v = z_flat[valid_mask], u_flat[valid_mask], v_flat[valid_mask]

        points_3d = np.stack([(u - cx_ir) * z / fx_ir, (v - cy_ir) * z / fy_ir, z], axis=-1)
        pts_color = (R_ir_to_color @ points_3d.T).T + T_ir_to_color
        u_rgb = (K_color[0, 0] * pts_color[:, 0] / pts_color[:, 2] + K_color[0, 2]).astype(np.int32)
        v_rgb = (K_color[1, 1] * pts_color[:, 1] / pts_color[:, 2] + K_color[1, 2]).astype(np.int32)
        in_bounds = (u_rgb >= 0) & (u_rgb < IMG_WIDTH) & (v_rgb >= 0) & (v_rgb < IMG_HEIGHT)

        colors = np.zeros((len(z), 3), dtype=np.float64)
        colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1].astype(np.float64) / 255.0

        if current_mask is not None and np.any(current_mask):
            highlight = np.zeros(len(z), dtype=bool)
            highlight[in_bounds] = current_mask[v_rgb[in_bounds], u_rgb[in_bounds]] > 0
            feedback_target_point_mask = highlight.copy()

            if np.any(highlight):
                colors[highlight] = colors[highlight] * 0.2 + MASK_COLOR_RGB * 0.8
                
                obj_pts = points_3d[highlight]
                uv_valid = np.column_stack((u_rgb[highlight], v_rgb[highlight]))

                if len(obj_pts) >= 10:
                    # 1. DBSCAN 去噪
                    obj_pcd_tmp = o3d.geometry.PointCloud()
                    obj_pcd_tmp.points = o3d.utility.Vector3dVector(obj_pts)
                    obj_labels = np.array(obj_pcd_tmp.cluster_dbscan(eps=0.008, min_points=20, print_progress=False))
                    
                    if np.any(obj_labels >= 0):
                        main_label = np.unique(obj_labels[obj_labels >= 0], return_counts=True)[0][np.argmax(np.unique(obj_labels[obj_labels >= 0], return_counts=True)[1])]
                        keep_mask_dbscan = (obj_labels == main_label)
                        obj_pts = obj_pts[keep_mask_dbscan]
                        uv_valid = uv_valid[keep_mask_dbscan] 
                        
                    # 2. 距离百分位过滤
                    centroid = obj_pts.mean(axis=0)
                    dists = np.linalg.norm(obj_pts - centroid, axis=1)
                    keep_mask_dist = dists <= np.percentile(dists, 96)
                    
                    filtered = obj_pts[keep_mask_dist]
                    uv_filtered = uv_valid[keep_mask_dist]

                    if len(filtered) >= 10:
                        center = filtered.mean(axis=0)
                        
                        # ===================================================
                        # 核心升级：多平面 RANSAC 分割找最宽面
                        # ===================================================
                        pcd_obj = o3d.geometry.PointCloud()
                        pcd_obj.points = o3d.utility.Vector3dVector(filtered)
                        
                        max_planes = 3
                        min_plane_points = 20
                        dist_thresh = 0.005 # 5mm 容差
                        
                        best_plane_model = None
                        max_inlier_count = 0
                        remaining_pcd = pcd_obj
                        
                        for i in range(max_planes):
                            if len(remaining_pcd.points) < min_plane_points:
                                break
                            
                            # 执行 RANSAC
                            plane_model, inliers = remaining_pcd.segment_plane(
                                distance_threshold=dist_thresh,
                                ransac_n=3,
                                num_iterations=1000
                            )
                            
                            if len(inliers) < min_plane_points:
                                break
                                
                            # 记录包含内点最多（最宽）的平面模型
                            if len(inliers) > max_inlier_count:
                                max_inlier_count = len(inliers)
                                best_plane_model = plane_model
                                
                            # 剔除已找到的平面点，进入下一轮迭代
                            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                            
                        # 根据 RANSAC 结果获取 Z 轴 (平面法向量)
                        if best_plane_model is not None:
                            Z_axis = np.array(best_plane_model[:3])
                        else:
                            # 极端情况兜底：退回单平面 SVD
                            _, _, Vt = np.linalg.svd(filtered - center, full_matrices=False)
                            Z_axis = Vt[2]
                            
                        Z_axis /= (np.linalg.norm(Z_axis) + 1e-6)
                        if Z_axis[2] > 0: Z_axis = -Z_axis # 强制指向相机
                        # ===================================================

                        # 2D 几何提取物理长边向量 (X_raw)
                        mask_uint8 = (current_mask > 0).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            box = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))
                            d01, d12 = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
                            
                            if d01 < d12:
                                mid1, mid2 = (box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0
                            else:
                                mid1, mid2 = (box[1] + box[2]) / 2.0, (box[3] + box[0]) / 2.0
                            
                            vec_2d = mid2 - mid1
                            # mid1_in, mid2_in = mid1 + vec_2d * 0.15, mid2 - vec_2d * 0.15
                            mid1_in, mid2_in = mid1 + vec_2d * 0.10, mid2 - vec_2d * 0.10
                            
                            dist1 = np.linalg.norm(uv_filtered - mid1_in, axis=1)
                            P1_3D = np.mean(filtered[np.argsort(dist1)[:5]], axis=0)
                            
                            dist2 = np.linalg.norm(uv_filtered - mid2_in, axis=1)
                            P2_3D = np.mean(filtered[np.argsort(dist2)[:5]], axis=0)
                            
                            # 将 X_raw 投影到刚才 RANSAC 找出的最宽平面上，保证正交
                            X_raw = P2_3D - P1_3D
                            X_axis = X_raw - np.dot(X_raw, Z_axis) * Z_axis 
                            X_axis /= (np.linalg.norm(X_axis) + 1e-6)
                            if X_axis[1] > 0: X_axis = -X_axis
                                
                            Y_axis = np.cross(Z_axis, X_axis)
                            Y_axis /= (np.linalg.norm(Y_axis) + 1e-6)
                            
                            axes = np.column_stack([X_axis, Y_axis, Z_axis])

                            local = (filtered - center) @ axes
                            raw_extent = local.max(axis=0) - local.min(axis=0)
                            center = center + axes @ ((local.max(axis=0) + local.min(axis=0)) / 2)

                            extent_frame_count += 1
                            if obb_smooth_center is not None:
                                obb_smooth_center = OBB_SMOOTH * center + (1 - OBB_SMOOTH) * obb_smooth_center
                                obb_smooth_R = OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * obb_smooth_R
                                u0 = obb_smooth_R[:, 0] / np.linalg.norm(obb_smooth_R[:, 0])
                                u1 = obb_smooth_R[:, 1] - np.dot(obb_smooth_R[:, 1], u0) * u0
                                obb_smooth_R = np.column_stack([u0, u1/np.linalg.norm(u1), np.cross(u0, u1/np.linalg.norm(u1))])

                                extent_history.append(raw_extent.copy())
                                ext_alpha = max(0.02, 0.4 * (0.92 ** extent_frame_count))
                                candidate_extent = 0.5 * raw_extent + 0.5 * np.median(np.array(extent_history), axis=0) if len(extent_history) >= 3 else raw_extent
                                max_delta = obb_smooth_extent * 0.05
                                obb_smooth_extent = ext_alpha * (obb_smooth_extent + np.clip(candidate_extent - obb_smooth_extent, -max_delta, max_delta)) + (1 - ext_alpha) * obb_smooth_extent
                            else:
                                obb_smooth_center, obb_smooth_extent, obb_smooth_R = center.copy(), raw_extent.copy(), axes.copy()
                                extent_history.append(raw_extent.copy())

                            # ========= 夹爪控制 =========
                            target_grip_width_m = obb_smooth_extent[1] 
                            final_grip_position = target_grip_width_m + 0.020
                            # print(f"目标真实宽度：{target_grip_width_m:.3f} 米 ({target_grip_width_m*1000:.1f} 毫米)")

                            corners_local = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1]], dtype=np.float64) * (obb_smooth_extent / 2)
                            obb_lineset.points = o3d.utility.Vector3dVector(corners_local @ obb_smooth_R.T + obb_smooth_center)
                            obb_edges = [[0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7]]
                            obb_lineset.lines = o3d.utility.Vector2iVector(obb_edges)
                            obb_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(obb_edges))
                            
                            T_pca = np.eye(4)
                            T_pca[:3, :3], T_pca[:3, 3] = obb_smooth_R, obb_smooth_center
                            pca_frame_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06).transform(T_pca)
                            pca_frame.vertices, pca_frame.vertex_colors = pca_frame_temp.vertices, pca_frame_temp.vertex_colors
                            if vis is not None:
                                vis.update_geometry(pca_frame)
                            
                            # ===== ROS 发布 =====
                            if sam2_initialized:
                                clearance_result = estimate_grasp_clearance_push(
                                    points_3d,
                                    highlight,
                                    obb_smooth_center,
                                    obb_smooth_extent,
                                    obb_smooth_R,
                                )
                                pose_msg = PoseStamped()
                                pose_msg.header.stamp = ros_node.get_clock().now().to_msg()
                                pose_msg.header.frame_id = "camera_d405_link" 
                                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = float(obb_smooth_center[0]), float(obb_smooth_center[1]), float(obb_smooth_center[2])
                                
                                quat = SciPyRot.from_matrix(obb_smooth_R).as_quat()
                                pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

                                pose_pub.publish(pose_msg)
                                width_msg = Float32()
                                width_msg.data = float(final_grip_position)
                                width_pub.publish(width_msg)
                                publish_clearance_result(clearance_result)
                                published_this_frame = True
                        else:
                            obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                    else:
                        obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                else:
                    obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        else:
            obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        # ================= GUI 更新 =================
        fps = 1.0 / max(1e-6, time.time() - t0)
        draw_hud(display, fps, tracking_enabled=sam2_initialized)
        
        if sam2_initialized and obb_smooth_extent is not None:
            cv2.putText(display, f"BBox: {obb_smooth_extent[0]*100:.1f}x{obb_smooth_extent[1]*100:.1f}x{obb_smooth_extent[2]*100:.1f}cm", (10, IMG_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if clearance_result is not None:
            negative = clearance_result['negative_clearance_m']
            positive = clearance_result['positive_clearance_m']
            push = clearance_result['push_cam_m']

            def clearance_text(value):
                return 'clear' if not np.isfinite(value) else f'{value * 1000.0:.1f}mm'

            cv2.putText(
                display,
                f"{clearance_result['status']} -Y:{clearance_text(negative)}({clearance_result['negative_points']}) "
                f"+Y:{clearance_text(positive)}({clearance_result['positive_points']})",
                (10, IMG_HEIGHT - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Sweep cam mm: [{push[0] * 1000.0:.1f}, {push[1] * 1000.0:.1f}, {push[2] * 1000.0:.1f}]",
                (10, IMG_HEIGHT - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 165, 255) if np.linalg.norm(push) > 0.002 else (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if published_this_frame:
            published_pose_frames += 1
            if AUTO_RESET_AFTER_PUBLISH and published_pose_frames >= PUBLISH_FRAMES_BEFORE_RESET:
                need_reset = True
                logging.info("D405 精定位结果已发布，自动回到轻量预览模式。")

        publish_panel_frame(display)
        cloud_image = publish_cloud_frame(points_3d, colors, obb_smooth_center, obb_smooth_extent, obb_smooth_R)
        if (
            published_this_frame
            and not feedback_saved_for_trigger
            and feedback_capture_armed
            and pending_feedback_dir is not None
            and clearance_result is not None
            and feedback_target_point_mask is not None
        ):
            try:
                feedback_result_sample_count += 1
                sample_dir = pending_feedback_dir
                if feedback_result_sample_count > 1:
                    sample_dir = os.path.join(
                        pending_feedback_dir,
                        f'sample_{feedback_result_sample_count:03d}',
                    )
                save_grasp_feedback(
                    sample_dir,
                    color_bgr,
                    display,
                    current_mask,
                    cloud_image,
                    points_3d,
                    colors,
                    feedback_target_point_mask,
                    obb_smooth_center,
                    obb_smooth_extent,
                    obb_smooth_R,
                    clearance_result,
                    final_grip_position,
                )
                feedback_saved_for_trigger = (
                    feedback_result_sample_count >= GRASP_FEEDBACK_MAX_RESULT_SAMPLES
                )
            except Exception as exc:
                logging.exception('[Feedback] 保存本次抓取关键数据失败: %s', exc)

        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if vis is not None:
            if first_frame:
                vis.reset_view_point(True)
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1]); ctr.set_up([0, -1, 0])
                first_frame = False

            vis.update_geometry(pcd)
            vis.update_geometry(obb_lineset)
            vis.poll_events()
            vis.update_renderer()
        else:
            first_frame = False

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    if vis is not None:
        vis.destroy_window()
    ros_executor.shutdown()
    ros_node.destroy_node()
    rclpy.shutdown()
    ros_spin_thread.join(timeout=1.0)
    logging.info("Exited")
