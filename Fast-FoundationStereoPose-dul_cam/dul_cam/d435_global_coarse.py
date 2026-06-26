"""
顶部全局相机 D435：粗定位节点
此版本的相机位置是平行于龙头架安装的；
持续扫描，高频发布目标粗略坐标 (仅使用 YOLO-OBB)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as SciPyRot
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class GlobalCoarseDetector(Node):
    def __init__(self):
        super().__init__('d435_global_detector')
        self.pose_pub = self.create_publisher(PoseStamped, '/target_pose_cam_coarse', 10)
        
        # 1. 绑定 D435 硬件
        logging.info("Initializing RealSense D435 (Top Camera)...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device('254622078230') # D435 序列号
        
        # 粗定位只需 RGB 和 基础 Depth，节省算力带宽
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)


        self.profile = self.pipeline.start(config)
        
        # 获取内参
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.ppx, self.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

        # 2. 加载 YOLO-OBB
        logging.info("Loading YOLO-OBB model...")
        # self.yolo_model = YOLO("/home/zdh/ultralytics/runs/obb/train10/weights/best.pt")
        self.yolo_model = YOLO("/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-4/weights/best.pt")
        
        logging.info("D435 全局检测节点已启动！正在监控传送带...")

    def run(self):
        cv2.namedWindow("D435 Coarse View", cv2.WINDOW_AUTOSIZE)
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())
                
                # 运行 YOLO 检测
                results = self.yolo_model(img, conf=0.5, verbose=False)

                if len(results) > 0 and results[0].obb is not None:
                    # 先让 YOLO 自带的 plot() 画出所有检测到的普通框
                    display = results[0].plot() 
                    
                    obbs = results[0].obb
                    if len(obbs) > 0:
                        # 工业漏斗策略：取传送带最下游（Y坐标最大）的目标
                        centers_x = obbs.xyxyxyxy[:, :, 0].mean(dim=1)
                        centers_y = obbs.xyxyxyxy[:, :, 1].mean(dim=1)
                        best_idx = torch.argmin(centers_x).item()
                        
                        
                        u, v = int(centers_x[best_idx]), int(centers_y[best_idx])
                        
                        # 获取粗略深度
                        z_m = depth_frame.get_distance(u, v)

                        if 0.1 < z_m < 2.0: # 剔除无效深度
                            # 反投影获取物理坐标
                            x_m = (u - self.ppx) * z_m / self.fx
                            y_m = (v - self.ppy) * z_m / self.fy
                            
                            # 获取粗略偏航角 (Yaw)
                            # 从 xywhr [x, y, w, h, r] 中提取最后一位的旋转角(弧度)
                            angle = obbs.xywhr[best_idx, 4].item()
                            q = SciPyRot.from_euler('z', angle).as_quat()

                            # 发布粗定位坐标
                            msg = PoseStamped()
                            msg.header.stamp = self.get_clock().now().to_msg()
                            msg.header.frame_id = "camera_d435_link"
                            msg.pose.position.x = float(x_m)
                            msg.pose.position.y = float(y_m)
                            msg.pose.position.z = float(z_m)
                            
                            msg.pose.orientation.x = float(q[0])
                            msg.pose.orientation.y = float(q[1])
                            msg.pose.orientation.z = float(q[2])
                            msg.pose.orientation.w = float(q[3])
                            
                            self.pose_pub.publish(msg)

                        # ================= 新增：标红突出当前选中目标 =================
                        # 提取最佳目标的 4 个角点并转为整型像素坐标
                        best_corners = obbs.xyxyxyxy[best_idx].cpu().numpy().astype(np.int32)
                        # 在 display 图像上叠加绘制红色粗线框 (B:0, G:0, R:255)
                        cv2.polylines(display, [best_corners], isClosed=True, color=(0, 0, 255), thickness=4)
                        # 加上一个醒目的 TARGET 文本标签
                        cv2.putText(display, "TARGET", tuple(best_corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        # ==============================================================

                else:
                    display = img.copy()

                cv2.putText(display, "D435 Global Coarse Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("D435 Coarse View", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                rclpy.spin_once(self, timeout_sec=0)
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = GlobalCoarseDetector()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()