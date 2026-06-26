"""
顶部全局相机 D435：粗定位节点

安装垂直于龙门架； 坐标系：前X,右Y,下Z
【现代工业极简风】去除冗余科幻元素，保留高级磨砂质感与平滑线条，专为高标准科研/工程展示设计。
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
        
        logging.info("Initializing RealSense D435 (Top Camera)...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device('254622078230')
        
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.profile = self.pipeline.start(config)
        
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.ppx, self.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        logging.info("Loading YOLO-OBB model...")
        self.yolo_model = YOLO("/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-5/weights/best.pt")
        logging.info("D435 全局检测节点已启动！正在监控传送带...")

    def draw_modern_panel(self, img, text_list, is_fallback=False):
        """绘制现代极简风的半透明数据卡片"""
        # 状态指示色：正常为清爽的薄荷绿，降级为警示橙
        accent_color = (0, 165, 255) if is_fallback else (100, 255, 50) 
        
        box_w, box_h = 280, len(text_list) * 30 + 20
        
        # 创建深色半透明遮罩
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (20 + box_w, 20 + box_h), (25, 25, 30), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        # 左侧彩色状态指示条
        cv2.line(img, (20, 20), (20, 20 + box_h), accent_color, 4, cv2.LINE_AA)
        
        # 渲染极简纯白文本
        for i, txt in enumerate(text_list):
            color = accent_color if i == 0 else (240, 240, 240) # 第一行标题用指示色，其余纯白
            font_scale = 0.65 if i == 0 else 0.6
            thickness = 2 if i == 0 else 1
            cv2.putText(img, txt, (35, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

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
                depth_image = np.asanyarray(depth_frame.get_data())
                display = img.copy()

                results = self.yolo_model(img, conf=0.5, verbose=False)

                if len(results) > 0 and results[0].obb is not None and len(results[0].obb) > 0:
                    obbs = results[0].obb
                    
                    centers_x = obbs.xyxyxyxy[:, :, 0].mean(dim=1)
                    centers_y = obbs.xyxyxyxy[:, :, 1].mean(dim=1)
                    best_idx = torch.argmin(centers_x).item()
                    
                    # 1. 渲染背景次要目标 (高级灰)
                    for i in range(len(obbs)):
                        if i != best_idx:
                            corners = obbs.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                            cv2.polylines(display, [corners], True, (120, 120, 120), 1, cv2.LINE_AA)

                    # 2. 提取最佳目标
                    best_corners = obbs.xyxyxyxy[best_idx].cpu().numpy()
                    corners_int = best_corners.astype(np.int32)
                    
                    mask = np.zeros((720, 1280), dtype=np.uint8)
                    cv2.fillPoly(mask, [corners_int], 255)
                    mask = cv2.erode(mask, np.ones((5,5), np.uint8))
                    
                    depth_m = depth_image * self.depth_scale
                    v_idx, u_idx = np.where((mask > 0) & (depth_m > 0.1) & (depth_m < 2.0))
                    
                    if len(v_idx) >= 20: 
                        # ======= 正常 3D 几何结算 =======
                        z_pts = depth_m[v_idx, u_idx]
                        x_pts = (u_idx - self.ppx) * z_pts / self.fx
                        y_pts = (v_idx - self.ppy) * z_pts / self.fy
                        obj_pts = np.column_stack((x_pts, y_pts, z_pts))
                        uv_valid = np.column_stack((u_idx, v_idx))
                        
                        centroid = obj_pts.mean(axis=0)
                        dists = np.linalg.norm(obj_pts - centroid, axis=1)
                        keep_mask = dists <= np.percentile(dists, 90)
                        filtered_pts = obj_pts[keep_mask]
                        uv_filtered = uv_valid[keep_mask]
                        
                        if len(filtered_pts) >= 10:
                            center = filtered_pts.mean(axis=0)
                            
                            d01 = np.linalg.norm(best_corners[0] - best_corners[1])
                            d12 = np.linalg.norm(best_corners[1] - best_corners[2])
                            
                            if d01 < d12:
                                mid1, mid2 = (best_corners[0] + best_corners[1]) / 2.0, (best_corners[2] + best_corners[3]) / 2.0
                            else:
                                mid1, mid2 = (best_corners[1] + best_corners[2]) / 2.0, (best_corners[3] + best_corners[0]) / 2.0
                            
                            vec_2d = mid2 - mid1
                            
                            X_raw = np.array([vec_2d[0] / self.fx, vec_2d[1] / self.fy, 0.0], dtype=np.float64)
                            X_axis = X_raw / (np.linalg.norm(X_raw) + 1e-6)
                            
                            if X_axis[0] < 0: X_axis = -X_axis
                                
                            Z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                            Y_axis = np.cross(Z_axis, X_axis)
                            Y_axis /= (np.linalg.norm(Y_axis) + 1e-6)
                            
                            axes = np.column_stack([X_axis, Y_axis, Z_axis])
                            q = SciPyRot.from_matrix(axes).as_quat()

                            msg = PoseStamped()
                            msg.header.stamp = self.get_clock().now().to_msg()
                            msg.header.frame_id = "camera_d435_link"
                            msg.pose.position.x = float(center[0])
                            msg.pose.position.y = float(center[1])
                            msg.pose.position.z = float(center[2])
                            msg.pose.orientation.x = float(q[0])
                            msg.pose.orientation.y = float(q[1])
                            msg.pose.orientation.z = float(q[2])
                            msg.pose.orientation.w = float(q[3])
                            self.pose_pub.publish(msg)

                            yaw_deg = np.degrees(np.arctan2(X_axis[1], X_axis[0]))
                            
                            # ======= 极简优雅画面渲染 =======
                            # 1. 目标框 (亮青色，粗细适中，抗锯齿)
                            box_color = (255, 220, 50)
                            cv2.polylines(display, [corners_int], True, box_color, 2, cv2.LINE_AA)
                            cv2.putText(display, "TARGET", (corners_int[0][0], corners_int[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

                            # 2. 3D 精致坐标轴
                            def project_to_pixel(p_3d):
                                return (int(p_3d[0] * self.fx / p_3d[2] + self.ppx), int(p_3d[1] * self.fy / p_3d[2] + self.ppy))
                            
                            pt_o = project_to_pixel(center)
                            axis_len = 0.08
                            pt_x = project_to_pixel(center + axes[:, 0] * axis_len)
                            pt_y = project_to_pixel(center + axes[:, 1] * axis_len)
                            pt_z = project_to_pixel(center + axes[:, 2] * axis_len)

                            cv2.arrowedLine(display, pt_o, pt_x, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)
                            cv2.arrowedLine(display, pt_o, pt_y, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)
                            cv2.arrowedLine(display, pt_o, pt_z, (255, 50, 50), 2, cv2.LINE_AA, tipLength=0.15)
                            cv2.circle(display, pt_o, 4, (255, 255, 255), -1, cv2.LINE_AA) 

                            # 3. 现代风数据面板
                            info_text = [
                                "STATUS  |  3D ALIGNED",
                                f"X    :   {center[0]:.3f} m",
                                f"Y    :   {center[1]:.3f} m",
                                f"Z    :   {center[2]:.3f} m",
                                f"YAW  :  {yaw_deg:+.1f} deg"
                            ]
                            self.draw_modern_panel(display, info_text, is_fallback=False)

                    else:
                        # ======= 2D 降级结算 =======
                        u, v = int(centers_x[best_idx]), int(centers_y[best_idx])
                        z_m = depth_frame.get_distance(u, v)

                        if 0.1 < z_m < 2.0:
                            x_m = (u - self.ppx) * z_m / self.fx
                            y_m = (v - self.ppy) * z_m / self.fy
                            
                            theta = obbs.xywhr[best_idx, 4].item()
                            X_axis = np.array([np.cos(theta), np.sin(theta), 0.0])
                            if X_axis[0] < 0: X_axis = -X_axis
                            Z_axis = np.array([0.0, 0.0, -1.0])
                            Y_axis = np.cross(Z_axis, X_axis)
                            
                            axes = np.column_stack([X_axis, Y_axis, Z_axis])
                            q = SciPyRot.from_matrix(axes).as_quat()

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

                            yaw_deg = np.degrees(np.arctan2(X_axis[1], X_axis[0]))
                            
                            # ======= 降级渲染 =======
                            warn_color = (0, 165, 255)
                            cv2.polylines(display, [corners_int], True, warn_color, 2, cv2.LINE_AA)
                            cv2.putText(display, "TARGET 2D", (corners_int[0][0], corners_int[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warn_color, 2, cv2.LINE_AA)
                            
                            pt_o_3d = np.array([x_m, y_m, z_m])
                            def project_to_pixel(p_3d):
                                return (int(p_3d[0] * self.fx / p_3d[2] + self.ppx), int(p_3d[1] * self.fy / p_3d[2] + self.ppy))
                            
                            pt_o = project_to_pixel(pt_o_3d)
                            axis_len = 0.08
                            pt_x = project_to_pixel(pt_o_3d + axes[:, 0] * axis_len)
                            pt_y = project_to_pixel(pt_o_3d + axes[:, 1] * axis_len)
                            pt_z = project_to_pixel(pt_o_3d + axes[:, 2] * axis_len)
                            
                            cv2.arrowedLine(display, pt_o, pt_x, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)
                            cv2.arrowedLine(display, pt_o, pt_y, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)
                            cv2.arrowedLine(display, pt_o, pt_z, (255, 50, 50), 2, cv2.LINE_AA, tipLength=0.15)
                            cv2.circle(display, pt_o, 4, (255, 255, 255), -1, cv2.LINE_AA)

                            info_text = [
                                "STATUS  |  2D FALLBACK",
                                f"X    :   {x_m:.3f} m",
                                f"Y    :   {y_m:.3f} m",
                                f"Z    :   {z_m:.3f} m",
                                f"YAW  :  {yaw_deg:+.1f} deg"
                            ]
                            self.draw_modern_panel(display, info_text, is_fallback=True)

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