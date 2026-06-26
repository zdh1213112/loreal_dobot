"""
顶部全局相机 D435：粗定位节点

安装垂直于龙门架； 坐标系：前X,右Y,下Z
【现代工业极简风】+【鼠标 ROI 框选功能】
- 鼠标左键拖拽：划定检测区域 (ROI)
- 键盘 'r' 键：重置/清除 ROI，恢复全图检测
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
        
        # --- 鼠标框选 ROI 相关状态变量 ---
        self.drawing_roi = False
        self.roi_start = (-1, -1)
        self.roi_end = (-1, -1)
        self.roi = None  # 格式为 (x_min, y_min, x_max, y_max)
        
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
        self.yolo_model = YOLO("/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-15/weights/best.pt")
        logging.info("D435 全局检测节点已启动！正在监控传送带...")

    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件，用于绘制 ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_roi = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.roi = None # 点击时清除旧的 ROI
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_roi:
                self.roi_end = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_roi = False
            self.roi_end = (x, y)
            # 计算边界并保存有效的 ROI
            x1, x2 = min(self.roi_start[0], self.roi_end[0]), max(self.roi_start[0], self.roi_end[0])
            y1, y2 = min(self.roi_start[1], self.roi_end[1]), max(self.roi_start[1], self.roi_end[1])
            if x2 - x1 > 20 and y2 - y1 > 20: # 剔除过小的误触框
                self.roi = (x1, y1, x2, y2)
                logging.info(f"ROI Locked: {self.roi}")

    def draw_modern_panel(self, img, text_list, is_fallback=False):
        """绘制半透明数据卡片"""
        accent_color = (0, 165, 255) if is_fallback else (100, 255, 50) 
        
        # 添加底部操作提示
        text_list.append("-------------")
        text_list.append("L-Click Drag: Set ROI")
        text_list.append("Press 'R': Reset ROI")
        
        box_w, box_h = 280, len(text_list) * 28 + 20
        
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (20 + box_w, 20 + box_h), (25, 25, 30), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        cv2.line(img, (20, 20), (20, 20 + box_h), accent_color, 4, cv2.LINE_AA)
        
        for i, txt in enumerate(text_list):
            color = accent_color if i == 0 else (200, 200, 200) 
            font_scale = 0.65 if i == 0 else 0.55
            thickness = 2 if i == 0 else 1
            if "ROI" in txt: color = (0, 200, 255) # 提示字样加亮为金黄色
            cv2.putText(img, txt, (35, 50 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def run(self):
        cv2.namedWindow("D435 Coarse View", cv2.WINDOW_AUTOSIZE)
        # 绑定鼠标回调函数
        cv2.setMouseCallback("D435 Coarse View", self.mouse_callback)
        
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

                # --- 渲染 ROI 框 ---
                roi_color = (0, 200, 255) # 金黄色
                if self.drawing_roi:
                    cv2.rectangle(display, self.roi_start, self.roi_end, roi_color, 2, cv2.LINE_AA)
                elif self.roi is not None:
                    rx1, ry1, rx2, ry2 = self.roi
                    cv2.rectangle(display, (rx1, ry1), (rx2, ry2), roi_color, 2, cv2.LINE_AA)
                    cv2.putText(display, "ROI LOCKED", (rx1, ry1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2, cv2.LINE_AA)

                results = self.yolo_model(img, conf=0.5, verbose=False)

                if len(results) > 0 and results[0].obb is not None and len(results[0].obb) > 0:
                    obbs = results[0].obb
                    
                    centers_x = obbs.xyxyxyxy[:, :, 0].mean(dim=1)
                    centers_y = obbs.xyxyxyxy[:, :, 1].mean(dim=1)
                    
                    # --- 核心：过滤在 ROI 框外的目标 ---
                    valid_indices = []
                    if self.roi is not None:
                        rx1, ry1, rx2, ry2 = self.roi
                        for i in range(len(centers_x)):
                            cx, cy = centers_x[i].item(), centers_y[i].item()
                            # 仅保留中心点在 ROI 内的目标
                            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                                valid_indices.append(i)
                    else:
                        valid_indices = list(range(len(centers_x)))

                    # 只有当 ROI 内存在目标时，才进行解算
                    if len(valid_indices) > 0:
                        # 在过滤后的结果中，选取 X 轴像素最小的目标 (最左侧)
                        valid_x_vals = [centers_x[i].item() for i in valid_indices]
                        best_local_idx = np.argmin(valid_x_vals)
                        best_idx = valid_indices[best_local_idx]
                        
                        # 1. 渲染背景次要目标 (仅渲染有效的 valid_indices)
                        for i in valid_indices:
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
                                
                                # ======= 画面渲染 =======
                                box_color = (255, 220, 50)
                                cv2.polylines(display, [corners_int], True, box_color, 2, cv2.LINE_AA)
                                cv2.putText(display, "TARGET", (corners_int[0][0], corners_int[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

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
                    else:
                        # 虽然 YOLO 检测到了，但是 ROI 里没有目标
                        self.draw_modern_panel(display, ["STATUS  |  NO TARGET IN ROI"], is_fallback=True)
                else:
                    # 全图无目标
                    self.draw_modern_panel(display, ["STATUS  |  NO TARGETS DETECTED"], is_fallback=True)

                cv2.imshow("D435 Coarse View", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 按下 r 键清空 ROI
                    self.roi = None
                    logging.info("ROI Cleared by user.")
                
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