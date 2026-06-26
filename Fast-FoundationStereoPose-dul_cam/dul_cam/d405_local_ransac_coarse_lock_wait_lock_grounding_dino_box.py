"""
功能说明：D405 末端局部精定位增强版（等待最新粗定位锁定目标）。

主要流程：
1. 监听 ROS 2 话题 `/trigger_d405_vision`，收到 True 后启动一次 D405 精定位流程。
2. 监听 `/coarse_target_obj_for_d405`，接收 Nova5 控制端发布的粗定位锁定目标。
3. 与 `d405_local_ransac_coarse_lock.py` 相比，本版本增加 wait-lock 机制：
   触发后会短暂等待一帧最新的 `/coarse_target_obj_for_d405`，并检查消息新鲜度，
   降低使用旧粗定位目标导致 D405 ROI/目标选择错误的概率。
4. 使用 RealSense D405 双红外 + FFS 生成深度点云，粗定位 3D 先验动态生成 ROI，SAM2 分割目标区域。
5. 对目标点云做多平面 RANSAC 姿态估计，选择稳定主平面/最宽面生成抓取姿态。
6. 发布精定位抓取结果到 `/target_pose_cam_fine`，frame 为 `camera_d405_link`。
7. 发布估计夹爪开口宽度到 `/gripper_target_width`，供 Nova5/DH 夹爪控制端使用。

典型配合：D435 粗定位节点 -> Nova5 粗定位移动并发布锁定目标 -> 本节点等待最新锁定目标后精定位。
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
from geometry_msgs.msg import PoseStamped, Vector3Stamped, PoseArray
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

# D405 先验 ROI：以 /coarse_target_obj_for_d405 的 3D 位姿和估计物体尺寸投影生成 bbox。
# 欧莱雅化妆品盒尺寸可按现场微调，单位 m。
PRIOR_OBJECT_LENGTH_M = 0.18
PRIOR_OBJECT_WIDTH_M = 0.08
PRIOR_OBJECT_HEIGHT_M = 0.05
PRIOR_ROI_SCALE = 1.10
PRIOR_ROI_MIN_HALF_W_PX = 55
PRIOR_ROI_MIN_HALF_H_PX = 70
PRIOR_ROI_MAX_HALF_W_PX = 130
PRIOR_ROI_MAX_HALF_H_PX = 170
PRIOR_DEPTH_GATE_M = 0.045
USE_PRIOR_DEPTH_GATE = os.environ.get("D405_USE_PRIOR_DEPTH_GATE", "0") == "1"
# D405 视角下盒子通常呈竖向长条；如果现场目标横放，把这里改成 "horizontal"。
PRIOR_ROI_LONG_AXIS = "vertical"
# 粗定位投影中心像素补偿：黄色框中心偏左/偏右/偏上/偏下时调这里。
PRIOR_ROI_OFFSET_U_PX = 25
PRIOR_ROI_OFFSET_V_PX = 70
PRIOR_POLYGON_EXPAND_SCALE = 1.10

# ==========================================
# ROS 2 初始化与触发回调机制
# ==========================================
rclpy.init()
ros_node = rclpy.create_node('d405_fine_vision_node')
pose_pub = ros_node.create_publisher(PoseStamped, '/target_pose_cam_fine', 10)
width_pub = ros_node.create_publisher(Float32, '/gripper_target_width', 10)

global_trigger_flag = False
locked_target_pose_msg = None
locked_target_msg_time = 0.0
locked_target_size_msg = None
locked_target_size_msg_time = 0.0
locked_target_corners_msg = None
locked_target_corners_msg_time = 0.0
LOCK_WAIT_TIMEOUT_S = 0.35
LOCK_PRIOR_WAIT_TIMEOUT_S = 0.45
LOCK_FRESHNESS_WINDOW_S = 1.5

def trigger_callback(msg):
    global global_trigger_flag
    if msg.data:
        global_trigger_flag = True
        logging.info(">>>> [ROS 2] 收到控制端触发信号！即将启动高精度视觉流水线... <<<<")

def coarse_target_lock_callback(msg):
    global locked_target_pose_msg, locked_target_msg_time
    locked_target_pose_msg = msg
    locked_target_msg_time = time.time()
    pos = msg.pose.position
    logging.info(
        f"[D405 Lock] 收到粗定位锁定目标: frame={msg.header.frame_id or 'none'} "
        f"x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}"
    )

def coarse_target_corners_callback(msg):
    global locked_target_corners_msg, locked_target_corners_msg_time
    locked_target_corners_msg = msg
    locked_target_corners_msg_time = time.time()
    logging.info(
        f"[D405 Corners] 收到粗定位角点: frame={msg.header.frame_id or 'none'} count={len(msg.poses)}"
    )

def coarse_target_size_callback(msg):
    global locked_target_size_msg, locked_target_size_msg_time
    locked_target_size_msg = msg
    locked_target_size_msg_time = time.time()
    size = msg.vector
    logging.info(
        f"[D405 Size] 收到粗定位尺寸: frame={msg.header.frame_id or 'none'} "
        f"L={size.x:.3f} W={size.y:.3f} H={size.z:.3f}"
    )

trigger_sub = ros_node.create_subscription(Bool, '/trigger_d405_vision', trigger_callback, 10)
coarse_lock_sub = ros_node.create_subscription(
    PoseStamped, '/coarse_target_obj_for_d405', coarse_target_lock_callback, 10
)
coarse_size_sub = ros_node.create_subscription(
    Vector3Stamped, '/coarse_target_size_for_d405', coarse_target_size_callback, 10
)
coarse_corners_sub = ros_node.create_subscription(
    PoseArray, '/coarse_target_corners_for_d405', coarse_target_corners_callback, 10
)
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

# ===== 2. GroundingDINO box detector =====
# 使用 X-AnyLabeling 的 ONNX GroundingDINO 推理代码；需要本地 onnx 模型路径。
GROUNDING_DINO_PROMPT = os.environ.get("D405_GDINO_PROMPT", "box")
GROUNDING_DINO_MODEL = os.environ.get("D405_GDINO_MODEL", "/home/zdh/ffs_ws//models/groundingdino_swint_ogc_quant.onnx")
GROUNDING_DINO_MODEL_TYPE = os.environ.get("D405_GDINO_MODEL_TYPE", "groundingdino_swint_ogc")
GROUNDING_DINO_INPUT_W = int(os.environ.get("D405_GDINO_INPUT_W", "1200"))
GROUNDING_DINO_INPUT_H = int(os.environ.get("D405_GDINO_INPUT_H", "800"))
GROUNDING_DINO_BOX_THRESHOLD = float(os.environ.get("D405_GDINO_BOX_THRESHOLD", "0.25"))
GROUNDING_DINO_TEXT_THRESHOLD = float(os.environ.get("D405_GDINO_TEXT_THRESHOLD", "0.20"))
GROUNDING_DINO_MAX_CENTER_DIST_PX = float(os.environ.get("D405_GDINO_MAX_CENTER_DIST_PX", "9999"))
GROUNDING_DINO_MIN_AREA_PX = float(os.environ.get("D405_GDINO_MIN_AREA_PX", "150"))
GROUNDING_DINO_MAX_AREA_RATIO = float(os.environ.get("D405_GDINO_MAX_AREA_RATIO", "0.35"))
GROUNDING_DINO_BBOX_PADDING_PX = int(os.environ.get("D405_GDINO_BBOX_PADDING_PX", "12"))

class SimpleOnnxModel:
    def __init__(self, model_path):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
        self.max_text_len = 256
        self.tokenizer = None

    def get_ort_inference(self, blob=None, inputs=None, extract=False):
        return self.ort_session.run(None, inputs)

class SimpleGroundingDINOBase:
    SPECIAL_TOKENS = [101, 102, 1012, 1029]
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGE_STD = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def get_caption(text_prompt):
        caption = str(text_prompt).lower().strip()
        return caption if caption.endswith(".") else caption + "."

    @classmethod
    def preprocess_image(cls, image_rgb, target_size):
        image = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - cls.IMAGE_MEAN) / cls.IMAGE_STD
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, 0).astype(np.float32)

    @staticmethod
    def get_tokenizer():
        from tokenizers import Tokenizer
        candidates = [
            "/home/zdh/X-AnyLabeling/anylabeling/services/auto_labeling/configs/bert/bert_base_uncased_tokenizer.json",
            "/home/zdh/X-AnyLabeling/anylabeling/services/auto_labeling/configs/bert/bert-base-uncased_tokenizer.json",
        ]
        for tokenizer_path in candidates:
            if os.path.isfile(tokenizer_path):
                return Tokenizer.from_file(tokenizer_path)
        raise FileNotFoundError("找不到 GroundingDINO tokenizer json，请检查 X-AnyLabeling/configs/bert。")

    @classmethod
    def generate_masks_with_special_tokens_and_transfer_map(cls, tokenized, special_tokens_list):
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape
        special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token
        idxs = np.argwhere(special_tokens_mask)
        attention_mask = np.eye(num_token, dtype=bool).reshape(1, num_token, num_token)
        attention_mask = np.tile(attention_mask, (bs, 1, 1))
        position_ids = np.zeros((bs, num_token), dtype=int)
        previous_col = 0
        for row, col in idxs:
            if col == 0 or col == num_token - 1:
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
                position_ids[row, previous_col + 1 : col + 1] = np.arange(0, col - previous_col)
            previous_col = col
        return attention_mask, position_ids, None

    @classmethod
    def encode_text(cls, text_prompt, tokenizer, max_text_len):
        caption = cls.get_caption(text_prompt)
        raw = tokenizer.encode(caption)
        tokenized = {
            "input_ids": np.array([raw.ids], dtype=np.int64),
            "token_type_ids": np.array([raw.type_ids], dtype=np.int64),
            "attention_mask": np.array([raw.attention_mask]),
        }
        text_masks, position_ids, _ = cls.generate_masks_with_special_tokens_and_transfer_map(tokenized, cls.SPECIAL_TOKENS)
        if text_masks.shape[1] > max_text_len:
            text_masks = text_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]
        return tokenized, text_masks, position_ids, caption

    @staticmethod
    def get_phrase_token_ranges(input_ids, special_token_ids):
        ranges = []
        start_idx = None
        for i, token_id in enumerate(input_ids):
            if token_id in special_token_ids:
                if start_idx is not None:
                    ranges.append((start_idx, i))
                    start_idx = None
            elif start_idx is None:
                start_idx = i
        return ranges

    @staticmethod
    def get_phrases_from_posmap(posmap, tokenized, tokenizer, left_idx=0, right_idx=255):
        posmap[0 : left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = np.where(posmap)[0]
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)

    @classmethod
    def decode_predictions(cls, logits, boxes, caption, tokenizer, box_threshold, text_threshold, apply_sigmoid=False, with_logits=True):
        logits_filt = np.squeeze(logits, 0)
        if apply_sigmoid:
            logits_filt = cls.sigmoid(logits_filt)
        boxes_filt = np.squeeze(boxes, 0)
        filt_mask = logits_filt.max(axis=1) > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        raw = tokenizer.encode(caption)
        tokenized = {
            "input_ids": np.array(raw.ids, dtype=np.int64),
            "token_type_ids": np.array(raw.type_ids, dtype=np.int64),
            "attention_mask": np.array(raw.attention_mask),
        }
        pred_phrases = []
        phrase_ranges = cls.get_phrase_token_ranges(tokenized["input_ids"], set(cls.SPECIAL_TOKENS))
        num_tokens = len(tokenized["input_ids"])
        for logit in logits_filt:
            posmap = logit > text_threshold
            pred_phrase = cls.get_phrases_from_posmap(posmap, tokenized, tokenizer)
            if not pred_phrase or "##" in pred_phrase:
                best_phrase_score = -np.inf
                best_phrase_range = None
                for start_idx, end_idx in phrase_ranges:
                    phrase_score = logit[start_idx:end_idx].max()
                    if phrase_score > best_phrase_score:
                        best_phrase_score = phrase_score
                        best_phrase_range = (start_idx, end_idx)
                if best_phrase_range:
                    fallback_posmap = np.zeros(num_tokens, dtype=bool)
                    fallback_posmap[best_phrase_range[0] : best_phrase_range[1]] = True
                    pred_phrase = cls.get_phrases_from_posmap(fallback_posmap, tokenized, tokenizer)
            pred_phrases.append([pred_phrase, logit.max() if with_logits else 1.0])
        return boxes_filt, pred_phrases

    @staticmethod
    def rescale_boxes(boxes, img_h, img_w):
        converted = []
        for box in boxes:
            out = box * np.array([img_w, img_h, img_w, img_h])
            out[:2] -= out[2:] / 2
            out[2:] += out[:2]
            converted.append(out)
        return np.array(converted, dtype=int)

def _load_grounding_dino():
    if not os.path.isfile(GROUNDING_DINO_MODEL):
        raise FileNotFoundError(
            f"GroundingDINO ONNX 模型不存在: {GROUNDING_DINO_MODEL}. "
            "请设置 D405_GDINO_MODEL=/path/to/groundingdino.onnx"
        )
    try:
        net = SimpleOnnxModel(GROUNDING_DINO_MODEL)
        net.tokenizer = SimpleGroundingDINOBase.get_tokenizer()
        return net, SimpleGroundingDINOBase
    except Exception as exc:
        raise RuntimeError("GroundingDINO 依赖导入失败：需要 onnxruntime、tokenizers。") from exc

logging.info(f"Loading GroundingDINO model: {GROUNDING_DINO_MODEL}")
gdino_net, GroundingDINOBase = _load_grounding_dino()
logging.info(f"GroundingDINO prompt set to: '{GROUNDING_DINO_PROMPT}'")

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
last_locked_target_uv = None
last_prior_bbox = None
last_prior_polygon = None
last_prior_source = "none"
PRIOR_MASK_PADDING_PX = 10

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

def _locked_target_transform_in_d405():
    if locked_target_pose_msg is None:
        return None
    frame_id = locked_target_pose_msg.header.frame_id.strip() or "none"
    if frame_id != "camera_d405_link":
        logging.warning(f"[D405 Lock] 锁定目标 frame_id={frame_id} 不是 camera_d405_link，无法生成先验 ROI。")
        return None

    pos = locked_target_pose_msg.pose.position
    quat = locked_target_pose_msg.pose.orientation
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = SciPyRot.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    transform[:3, 3] = [float(pos.x), float(pos.y), float(pos.z)]
    if transform[2, 3] <= 1e-6:
        logging.warning("[D405 Lock] 锁定目标 Z 非法，无法生成先验 ROI。")
        return None
    return transform

def _project_points_to_pixels(points_3d):
    z = points_3d[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return None
    pts = points_3d[valid]
    u = K_color[0, 0] * pts[:, 0] / pts[:, 2] + K_color[0, 2]
    v = K_color[1, 1] * pts[:, 1] / pts[:, 2] + K_color[1, 2]
    pixels = np.column_stack([u, v])
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if len(pixels) == 0:
        return None
    return pixels

def current_locked_target_size():
    if locked_target_size_msg is None:
        return np.array([PRIOR_OBJECT_LENGTH_M, PRIOR_OBJECT_WIDTH_M, PRIOR_OBJECT_HEIGHT_M], dtype=np.float64)
    if time.time() - locked_target_size_msg_time > LOCK_FRESHNESS_WINDOW_S:
        logging.warning("[D405 Size] 尺寸消息过旧，回退默认尺寸先验。")
        return np.array([PRIOR_OBJECT_LENGTH_M, PRIOR_OBJECT_WIDTH_M, PRIOR_OBJECT_HEIGHT_M], dtype=np.float64)
    size = locked_target_size_msg.vector
    return np.clip(
        np.array([float(size.x), float(size.y), float(size.z)], dtype=np.float64),
        [0.03, 0.02, 0.01],
        [0.35, 0.20, 0.12],
    )

def expand_polygon_around_center(poly, scale=PRIOR_POLYGON_EXPAND_SCALE):
    poly = np.asarray(poly, dtype=np.float32)
    center = poly.mean(axis=0, keepdims=True)
    expanded = center + (poly - center) * float(scale)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, IMG_WIDTH - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, IMG_HEIGHT - 1)
    return np.round(expanded).astype(np.int32)

def generate_prior_bbox_from_locked_corners():
    global last_prior_polygon
    if locked_target_corners_msg is None:
        logging.warning("[D405 Corners] 没有收到 /coarse_target_corners_for_d405，回退 size ROI。")
        return None
    if time.time() - locked_target_corners_msg_time > LOCK_FRESHNESS_WINDOW_S:
        logging.warning("[D405 Corners] 角点消息过旧，回退 size ROI。")
        return None
    frame_id = locked_target_corners_msg.header.frame_id.strip() or "none"
    if frame_id != "camera_d405_link":
        logging.warning(f"[D405 Corners] frame_id={frame_id} 不是 camera_d405_link，忽略角点 ROI。")
        return None
    if len(locked_target_corners_msg.poses) < 4:
        logging.warning(f"[D405 Corners] 角点数量不足 {len(locked_target_corners_msg.poses)}，回退 size ROI。")
        return None

    points = np.array(
        [[p.position.x, p.position.y, p.position.z] for p in locked_target_corners_msg.poses[:4]],
        dtype=np.float64,
    )
    pixels = _project_points_to_pixels(points)
    if pixels is None or len(pixels) < 4:
        logging.warning("[D405 Corners] 角点投影失败，回退 size ROI。")
        return None
    raw_polygon = np.round(pixels[:4]).astype(np.int32)
    last_prior_polygon = expand_polygon_around_center(raw_polygon)

    center_uv = project_locked_target_to_pixel()
    if center_uv is None:
        center_u, center_v = pixels.mean(axis=0)
    else:
        center_u, center_v = center_uv

    bbox_pixels = last_prior_polygon.astype(np.float64)
    min_u, min_v = np.min(bbox_pixels, axis=0)
    max_u, max_v = np.max(bbox_pixels, axis=0)
    polygon_center_u = 0.5 * (min_u + max_u)
    polygon_center_v = 0.5 * (min_v + max_v)
    center_blend = 0.55
    center_u = center_u * center_blend + polygon_center_u * (1.0 - center_blend)
    center_v = center_v * center_blend + polygon_center_v * (1.0 - center_blend)
    half_w = max(PRIOR_ROI_MIN_HALF_W_PX, 0.5 * (max_u - min_u) * PRIOR_ROI_SCALE)
    half_h = max(PRIOR_ROI_MIN_HALF_H_PX, 0.5 * (max_v - min_v) * PRIOR_ROI_SCALE)
    half_w = min(PRIOR_ROI_MAX_HALF_W_PX, half_w)
    half_h = min(PRIOR_ROI_MAX_HALF_H_PX, half_h)

    x1 = int(np.clip(center_u - half_w, 0, IMG_WIDTH - 1))
    y1 = int(np.clip(center_v - half_h, 0, IMG_HEIGHT - 1))
    x2 = int(np.clip(center_u + half_w, 0, IMG_WIDTH - 1))
    y2 = int(np.clip(center_v + half_h, 0, IMG_HEIGHT - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    logging.info(
        f"💡 [D405 Prior ROI Polygon] center=({center_u:.1f},{center_v:.1f}) bbox=({x1},{y1},{x2},{y2}) "
        f"half=({half_w:.1f},{half_h:.1f})"
    )
    return (x1, y1, x2, y2)

def generate_prior_bbox_from_locked_target():
    transform = _locked_target_transform_in_d405()
    if transform is None:
        return None

    center_uv = project_locked_target_to_pixel()
    if center_uv is None:
        return None
    center_u, center_v = center_uv

    center_u += PRIOR_ROI_OFFSET_U_PX
    center_v += PRIOR_ROI_OFFSET_V_PX

    object_size = current_locked_target_size()
    depth = max(float(transform[2, 3]), 1e-6)

    # 不使用 D435 姿态旋转出来的 3D 盒子投影，避免姿态误差导致 ROI 倾斜/变得很大。
    # 用真实尺寸 + D405 当前深度 + D405 内参生成图像轴对齐 ROI。
    # 由于 D405 抓取视角下盒子常呈竖向长条，这里允许指定最长边投到图像竖向。
    sorted_size = np.sort(object_size)[::-1]
    long_m = sorted_size[0]
    short_m = sorted_size[1] if len(sorted_size) > 1 else sorted_size[0]
    if PRIOR_ROI_LONG_AXIS == "horizontal":
        roi_size_u_m = long_m
        roi_size_v_m = short_m
    else:
        roi_size_u_m = short_m
        roi_size_v_m = long_m

    half_w = 0.5 * roi_size_u_m * K_color[0, 0] / depth * PRIOR_ROI_SCALE
    half_h = 0.5 * roi_size_v_m * K_color[1, 1] / depth * PRIOR_ROI_SCALE

    half_w = min(PRIOR_ROI_MAX_HALF_W_PX, max(PRIOR_ROI_MIN_HALF_W_PX, half_w))
    half_h = min(PRIOR_ROI_MAX_HALF_H_PX, max(PRIOR_ROI_MIN_HALF_H_PX, half_h))

    x1 = int(np.clip(center_u - half_w, 0, IMG_WIDTH - 1))
    y1 = int(np.clip(center_v - half_h, 0, IMG_HEIGHT - 1))
    x2 = int(np.clip(center_u + half_w, 0, IMG_WIDTH - 1))
    y2 = int(np.clip(center_v + half_h, 0, IMG_HEIGHT - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    logging.info(
        f"💡 [D405 Prior ROI Safe] center=({center_u:.1f},{center_v:.1f}) bbox=({x1},{y1},{x2},{y2}) "
        f"half=({half_w:.1f},{half_h:.1f}) axis={PRIOR_ROI_LONG_AXIS} "
        f"offset=({PRIOR_ROI_OFFSET_U_PX},{PRIOR_ROI_OFFSET_V_PX}) z={depth:.3f} "
        f"size=({object_size[0]:.3f},{object_size[1]:.3f},{object_size[2]:.3f})"
    )
    return (x1, y1, x2, y2)

def current_locked_target_depth():
    transform = _locked_target_transform_in_d405()
    if transform is None:
        return None
    return float(transform[2, 3])

def wait_for_recent_locked_target(previous_msg_time, timeout_s=LOCK_WAIT_TIMEOUT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rclpy.spin_once(ros_node, timeout_sec=0.02)
        if locked_target_pose_msg is not None and locked_target_msg_time > previous_msg_time:
            return True
    return False

def wait_for_recent_locked_corners(previous_msg_time, timeout_s=LOCK_PRIOR_WAIT_TIMEOUT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rclpy.spin_once(ros_node, timeout_sec=0.02)
        if locked_target_corners_msg is not None and locked_target_corners_msg_time > previous_msg_time:
            return True
    return False

def wait_for_recent_locked_size(previous_msg_time, timeout_s=LOCK_PRIOR_WAIT_TIMEOUT_S):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rclpy.spin_once(ros_node, timeout_sec=0.02)
        if locked_target_size_msg is not None and locked_target_size_msg_time > previous_msg_time:
            return True
    return False

def current_locked_target_uv():
    if locked_target_pose_msg is None:
        return None
    if time.time() - locked_target_msg_time > LOCK_FRESHNESS_WINDOW_S:
        return None
    return project_locked_target_to_pixel()

def clamp_bbox_with_padding(x1, y1, x2, y2, padding=GROUNDING_DINO_BBOX_PADDING_PX):
    x1 = int(np.clip(np.floor(x1) - padding, 0, IMG_WIDTH - 1))
    y1 = int(np.clip(np.floor(y1) - padding, 0, IMG_HEIGHT - 1))
    x2 = int(np.clip(np.ceil(x2) + padding, 0, IMG_WIDTH - 1))
    y2 = int(np.clip(np.ceil(y2) + padding, 0, IMG_HEIGHT - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def select_grounding_dino_box(color_bgr, locked_uv=None):
    image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    blob = GroundingDINOBase.preprocess_image(image_rgb, (GROUNDING_DINO_INPUT_W, GROUNDING_DINO_INPUT_H))
    tokenized, text_masks, position_ids, caption = GroundingDINOBase.encode_text(
        GROUNDING_DINO_PROMPT, gdino_net.tokenizer, gdino_net.max_text_len
    )
    inputs = {
        "img": blob,
        "input_ids": np.array(tokenized["input_ids"], dtype=np.int64),
        "attention_mask": np.array(tokenized["attention_mask"], dtype=bool),
        "token_type_ids": np.array(tokenized["token_type_ids"], dtype=np.int64),
        "position_ids": np.array(position_ids, dtype=np.int64),
        "text_token_mask": np.array(text_masks, dtype=bool),
    }
    logits, boxes = gdino_net.get_ort_inference(blob, inputs=inputs, extract=False)
    boxes_filt, pred_phrases = GroundingDINOBase.decode_predictions(
        logits,
        boxes,
        caption,
        gdino_net.tokenizer,
        GROUNDING_DINO_BOX_THRESHOLD,
        GROUNDING_DINO_TEXT_THRESHOLD,
        apply_sigmoid=True,
        with_logits=True,
    )
    if len(boxes_filt) == 0:
        logging.warning(
            f"[GroundingDINO] 没有检测到 '{GROUNDING_DINO_PROMPT}', "
            f"box_th={GROUNDING_DINO_BOX_THRESHOLD} text_th={GROUNDING_DINO_TEXT_THRESHOLD}"
        )
        return None, None, None

    boxes_xyxy = GroundingDINOBase.rescale_boxes(boxes_filt, IMG_HEIGHT, IMG_WIDTH).astype(np.float64)
    box_polygons = np.array(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in boxes_xyxy],
        dtype=np.float64,
    )
    lock_uv = locked_uv
    if lock_uv is None:
        lock_uv = current_locked_target_uv()
    if lock_uv is None:
        lock_uv = (IMG_WIDTH * 0.5, IMG_HEIGHT * 0.5)
        logging.warning("[GroundingDINO] 未拿到粗定位投影点，回退选择画面中心最近目标。")

    candidates = []
    rejected = []
    for idx, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        center_u = 0.5 * (x1 + x2)
        center_v = 0.5 * (y1 + y2)
        dist = float(np.hypot(center_u - lock_uv[0], center_v - lock_uv[1]))
        if area < GROUNDING_DINO_MIN_AREA_PX or area > IMG_WIDTH * IMG_HEIGHT * GROUNDING_DINO_MAX_AREA_RATIO:
            rejected.append(f"ID:{idx} area={area:.0f} dist={dist:.1f}")
            continue
        if dist > GROUNDING_DINO_MAX_CENTER_DIST_PX:
            rejected.append(f"ID:{idx} far dist={dist:.1f} area={area:.0f}")
            continue
        candidates.append((dist + area / max(IMG_WIDTH * IMG_HEIGHT, 1) * 60.0, dist, idx, box))

    if not candidates:
        logging.warning(
            f"[GroundingDINO] 检测到 {len(boxes_xyxy)} 个 '{GROUNDING_DINO_PROMPT}'，但候选全被过滤；"
            f"lock_uv=({lock_uv[0]:.1f},{lock_uv[1]:.1f}) rejected={'; '.join(rejected[:6])}"
        )
        return None, box_polygons, None

    candidates.sort(key=lambda item: item[0])
    _, best_dist, best_idx, best_box = candidates[0]
    bbox = clamp_bbox_with_padding(*best_box)
    if bbox is None:
        return None, box_polygons, None
    logging.info(
        f"🎯 [GroundingDINO] 检测到 {len(boxes_xyxy)} 个 '{GROUNDING_DINO_PROMPT}'，"
        f"选择 ID:{best_idx} dist_to_lock={best_dist:.1f}px bbox={bbox}"
    )
    return bbox, box_polygons, best_idx

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
        # 粗定位先验 ROI 触发逻辑
        # =========================================================
        if global_trigger_flag:
            logging.info("\n" + "="*50)
            logging.info(">>>> 正在清空历史运动残影，准备拍照... <<<<")
            previous_lock_time = locked_target_msg_time
            got_fresh_lock = wait_for_recent_locked_target(previous_lock_time)
            locked_uv_for_selection = project_locked_target_to_pixel() if locked_target_pose_msg is not None else None
            if got_fresh_lock:
                logging.info("🔒 [D405 Lock] 已等到最新粗定位锁定目标，将用其投影点选择最近的 GroundingDINO box。")
            elif locked_target_pose_msg is not None:
                logging.warning("⚠️ [D405 Lock] 触发后未等到更新锁定消息，继续沿用上一条锁定目标。")
            else:
                logging.warning("⚠️ [D405 Lock] 触发后未收到任何锁定目标，回退按画面中心选择 GroundingDINO 目标。")
            if locked_uv_for_selection is not None:
                logging.info(f"[GroundingDINO] 本次触发锁定投影点: ({locked_uv_for_selection[0]:.1f}, {locked_uv_for_selection[1]:.1f})")
            for _ in range(15):
                pipeline.wait_for_frames()

            frames = pipeline.wait_for_frames()
            color_bgr = np.asanyarray(frames.get_color_frame().get_data())

            last_prior_bbox = None
            last_prior_polygon = None
            last_prior_source = "grounding_dino"
            current_mask = None

            selected_bbox, detected_boxes, selected_idx = select_grounding_dino_box(color_bgr, locked_uv=locked_uv_for_selection)
            last_yolo_obbs = detected_boxes
            last_best_idx = -1 if selected_idx is None else int(selected_idx)

            if selected_bbox is not None:
                pending_bbox = selected_bbox
                last_prior_bbox = selected_bbox
                need_reset = True
                logging.info("✅ GroundingDINO 锁定 box，即将移交 SAM2 进行实时跟踪...")
                logging.info("="*50 + "\n")
            else:
                logging.warning("❌ 精定位失败：GroundingDINO 未能选择到有效 box。")
                last_best_idx = -1

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
            if pending_bbox is None:
                last_prior_bbox = None
                last_prior_polygon = None
                last_prior_source = "none"
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
            prompt_bbox = None
            if isinstance(pending_point, tuple) and len(pending_point) == 3:
                prompt_points, prompt_labels, prompt_bbox = pending_point
            elif isinstance(pending_point, tuple) and len(pending_point) == 2:
                prompt_points, prompt_labels = pending_point
            else:
                prompt_points = np.array([[pending_point[0], pending_point[1]]], dtype=np.float32)
                prompt_labels = np.array([1], dtype=np.int32)
            if prompt_bbox is not None:
                bbox_arr = np.array([[prompt_bbox[0], prompt_bbox[1]], [prompt_bbox[2], prompt_bbox[3]]], dtype=np.float32)
                sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox_arr)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, points=prompt_points, labels=prompt_labels)
            sam2_initialized = True
            pending_point = None

        if sam2_initialized:
            out_obj_ids, out_mask_logits = sam2_predictor.track(color_bgr)
            current_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze() if len(out_obj_ids) > 0 else None
            if current_mask is not None and last_prior_bbox is not None and last_prior_source not in ("yoloworld", "sam2_point", "grounding_dino"):
                x1, y1, x2, y2 = last_prior_bbox
                x1 = max(0, int(x1) - PRIOR_MASK_PADDING_PX)
                y1 = max(0, int(y1) - PRIOR_MASK_PADDING_PX)
                x2 = min(IMG_WIDTH - 1, int(x2) + PRIOR_MASK_PADDING_PX)
                y2 = min(IMG_HEIGHT - 1, int(y2) + PRIOR_MASK_PADDING_PX)
                roi_mask = np.zeros_like(current_mask, dtype=np.uint8)
                if last_prior_polygon is not None and len(last_prior_polygon) >= 3:
                    poly = last_prior_polygon.copy()
                    poly[:, 0] = np.clip(poly[:, 0], 0, IMG_WIDTH - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, IMG_HEIGHT - 1)
                    cv2.fillPoly(roi_mask, [poly], 1)
                    kernel_size = max(1, PRIOR_MASK_PADDING_PX * 2 + 1)
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
                else:
                    roi_mask[y1:y2 + 1, x1:x2 + 1] = 1
                current_mask = np.logical_and(current_mask > 0, roi_mask > 0).astype(np.uint8)
                current_mask = keep_component_near_locked_target(current_mask, color_bgr)
            elif current_mask is not None and last_prior_source == "sam2_point":
                current_mask = keep_component_near_locked_target(current_mask, color_bgr)

        display = color_bgr.copy()
        
        if last_prior_polygon is not None and len(last_prior_polygon) >= 3:
            poly = last_prior_polygon.copy()
            poly[:, 0] = np.clip(poly[:, 0], 0, IMG_WIDTH - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, IMG_HEIGHT - 1)
            cv2.polylines(display, [poly], True, (0, 255, 255), 2)

        if last_prior_bbox is not None and last_prior_source not in ("yoloworld", "grounding_dino"):
            x1, y1, x2, y2 = [int(v) for v in last_prior_bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 1)
            cv2.putText(
                display,
                f"PRIOR_ROI[{last_prior_source}] {x2 - x1}x{y2 - y1}px",
                (max(0, x1), max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

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
                if USE_PRIOR_DEPTH_GATE:
                    prior_depth = current_locked_target_depth()
                    if prior_depth is not None:
                        depth_keep = np.abs(obj_pts[:, 2] - prior_depth) <= PRIOR_DEPTH_GATE_M
                        if np.count_nonzero(depth_keep) >= 10:
                            obj_pts = obj_pts[depth_keep]
                            uv_valid = uv_valid[depth_keep]

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
