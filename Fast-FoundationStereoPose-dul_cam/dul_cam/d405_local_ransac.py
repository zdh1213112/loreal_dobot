"""
RealSense D405 (末端精定位) + SAM2 + FFS + 触发式启动 + RANSAC多平面提取姿态
新增：多平面 RANSAC 分割逻辑，精准锚定最宽面法向量，解决随机摆放导致的 Z 轴跳变。
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
import pyrealsense2 as rs
import open3d as o3d
from collections import deque

# ROS 2 和矩阵转换依赖
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
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
pose_pub = ros_node.create_publisher(PoseStamped, '/target_pose_cam_fine', 10)
width_pub = ros_node.create_publisher(Float32, '/gripper_target_width', 10)

global_trigger_flag = False

def trigger_callback(msg):
    global global_trigger_flag
    if msg.data:
        global_trigger_flag = True
        logging.info(">>>> [ROS 2] 收到控制端触发信号！即将启动高精度视觉流水线... <<<<")

trigger_sub = ros_node.create_subscription(Bool, '/trigger_d405_vision', trigger_callback, 10)
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
yolo_model = YOLO("/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-10/weights/best.pt")

# ===== 3. Load SAM2 model =====
logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0

# ===== 4. Initialize RealSense D405 =====
logging.info("Initializing RealSense D405...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('409122274792')

config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)       

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)

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

# ===== 7. Open3D visualizer & Helpers =====
vis = o3d.visualization.Visualizer()
vis.create_window("D405 Point Cloud", width=720, height=540, left=700, top=50)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
obb_lineset = o3d.geometry.LineSet()
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

vis.add_geometry(create_camera_frustum(fx_ir, fy_ir, cx_ir, cy_ir, IMG_WIDTH, IMG_HEIGHT))
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))
pca_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
vis.add_geometry(pca_frame)

cv2.namedWindow("D405 RGB + SAM2", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("D405 RGB + SAM2", 30, 50)

drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = pending_point = current_mask = None
sam2_initialized = need_reset = False
last_yolo_obbs = None
last_best_idx = -1

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = x, y
        if abs(fx_mouse - ix) > 8 and abs(fy_mouse - iy) > 8:
            pending_bbox = (min(ix, fx_mouse), min(iy, fy_mouse), max(ix, fx_mouse), max(iy, fy_mouse))
        else:
            pending_point = (x, y)
cv2.setMouseCallback("D405 RGB + SAM2", mouse_callback)

first_frame = True
frame_count = 0

try:
    while True:
        t0 = time.time()
        rclpy.spin_once(ros_node, timeout_sec=0)

        # =========================================================
        # YOLO 触发逻辑
        # =========================================================
        if global_trigger_flag:
            logging.info("\n" + "="*50)
            logging.info(">>>> 正在清空历史运动残影，准备拍照... <<<<")
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
                dists = (centers_x - 320)**2 + (centers_y - 240)**2
                best_idx = torch.argmin(dists).item()
                
                logging.info(f"💡 [选择依据] 目标 ID:{best_idx} 距离画面中心最近，设为最佳目标！")
                
                last_yolo_obbs = obbs.xyxyxyxy.cpu().numpy()
                last_best_idx = best_idx

                corners = last_yolo_obbs[best_idx]
                x1, y1 = np.min(corners, axis=0)
                x2, y2 = np.max(corners, axis=0)
                pending_bbox = (int(x1), int(y1), int(x2), int(y2))
                need_reset = True
                
                logging.info("✅ YOLO 锁定目标，即将移交 SAM2 进行实时跟踪...")
                logging.info("="*50 + "\n")
            else:
                logging.warning("❌ 精定位失败：YOLO 未能在画面中检测到目标。")
                last_yolo_obbs = None
                
            global_trigger_flag = False
        # =========================================================

        frames = pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())   
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())  
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())     

        if need_reset:
            try:
                sam2_predictor.reset_state()
            except KeyError:
                pass  
            sam2_initialized = need_reset = False
            current_mask = obb_smooth_center = obb_smooth_extent = obb_smooth_R = None
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

        if last_yolo_obbs is not None:
            for i, corners in enumerate(last_yolo_obbs):
                corners_int = np.int32(corners)
                color = (0, 0, 255) if i == last_best_idx else (255, 0, 0)
                thickness = 3 if i == last_best_idx else 1
                cv2.polylines(display, [corners_int], isClosed=True, color=color, thickness=thickness)
                cv2.putText(display, f"ID:{i}", tuple(corners_int[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == last_best_idx else 1)

        if drawing and ix >= 0:
            cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

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
                            final_grip_position = target_grip_width_m + 0.010
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
                            vis.update_geometry(pca_frame)
                            
                            # ===== ROS 发布 =====
                            if sam2_initialized:
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
                        else:
                            obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                    else:
                        obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                else:
                    obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        else:
            obb_lineset.points, obb_lineset.lines = o3d.utility.Vector3dVector(np.zeros((0, 3))), o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        # ================= GUI 更新 =================
        fps = 1.0 / (time.time() - t0)
        cv2.putText(display, f"FPS: {fps:.1f}", (IMG_WIDTH - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        status_text = "TRACKING | r=reset q=quit" if sam2_initialized else "Waiting for trigger... | q=quit"
        cv2.putText(display, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if sam2_initialized else (0, 165, 255), 2)
        
        if sam2_initialized and obb_smooth_extent is not None:
            cv2.putText(display, f"BBox: {obb_smooth_extent[0]*100:.1f}x{obb_smooth_extent[1]*100:.1f}x{obb_smooth_extent[2]*100:.1f}cm", (10, IMG_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("D405 RGB + SAM2", display)

        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if first_frame:
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1]); ctr.set_up([0, -1, 0])
            first_frame = False

        vis.update_geometry(pcd)
        vis.update_geometry(obb_lineset)
        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            need_reset = True

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    vis.destroy_window()
    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()
    logging.info("Exited")