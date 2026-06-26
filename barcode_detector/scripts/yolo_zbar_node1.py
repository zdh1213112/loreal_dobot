#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from pyzbar.pyzbar import decode, ZBarSymbol
from collections import defaultdict

class YoloBarcodeNode(Node):
    def __init__(self):
        super().__init__('yolo_barcode_node_py')
        self.publisher = self.create_publisher(String, 'detected_barcodes', 10)

        self.get_logger().info("🧠 正在加载 YOLOv8 原生模型...")
        self.model = YOLO('/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.pt')

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        try:
            profile = self.pipeline.start(config)
            sensor = profile.get_device().query_sensors()[1]

            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, 50.0)
            sensor.set_option(rs.option.gain, 16.0)

            if sensor.supports(rs.option.backlight_compensation):
                sensor.set_option(rs.option.backlight_compensation, 1)
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0)

            self.get_logger().info("🚀 D415 YOLO+ZBar (多角度+抗反光级联) 模式启动！")
        except Exception as e:
            self.get_logger().error(f"❌ 相机启动失败: {e}")
            raise e

        self.total_frames = 0
        self.scanned_barcodes = set()
        self.candidate_counts = defaultdict(int)
        self.current_results = []

        self.timer = self.create_timer(0.03, self.process_frame)

    def validate_ean13(self, code):
        if len(code) != 13 or not code.isdigit(): return False
        sum_val = sum(int(d) if i % 2 == 0 else int(d) * 3 for i, d in enumerate(code[:12]))
        return (10 - sum_val % 10) % 10 == int(code[12])

    # ================= 💡 新增：无损图像旋转函数 =================
    def rotate_image_safely(self, mat, angle):
        """旋转图像并自动扩大边界框，防止条码的四个角被裁剪掉，边框用白色填充以形成静区"""
        if angle == 0: return mat
        height, width = mat.shape[:2]
        image_center = (width/2, height/2)
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        abs_cos = abs(rotation_mat[0,0])
        abs_sin = abs(rotation_mat[0,1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]
        # borderValue=255 用纯白填充旋转后的黑边，给 ZBar 制造完美的白色静区
        return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderValue=255)

    def detect_glare(self, gray):
        _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        ratio = cv2.countNonZero(glare_mask) / (gray.shape[0] * gray.shape[1])
        return ratio > 0.01, glare_mask

    def enhance_small_barcode(self, gray, scale):
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), 1.2)
        sharp = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
        return clahe.apply(sharp)

    def suppress_glare_retinex(self, gray):
        gray_f = gray.astype(np.float32) / 255.0
        illum = cv2.GaussianBlur(gray_f, (51, 51), 25)
        reflect = cv2.log(gray_f + 0.01) - cv2.log(illum + 0.01)
        return cv2.normalize(reflect, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def suppress_glare_tophat(self, gray):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, out = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return out

    def suppress_glare_inpaint(self, gray, glare_mask):
        mask_dilated = cv2.dilate(glare_mask, None, iterations=2)
        return cv2.inpaint(gray, mask_dilated, 3, cv2.INPAINT_TELEA)

    def decode_roi(self, scan_gray):
        """只负责解码并返回结果，不涉及坐标计算"""
        if scan_gray is None or scan_gray.size == 0: return None
        barcodes = decode(scan_gray, symbols=[ZBarSymbol.EAN13])
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            if self.validate_ean13(data):
                return data
        return None

    def scan_with_antiglare(self, roi_gray, scale):
        """修改为：成功解码直接返回数据字符串，失败返回 None"""
        has_glare, glare_mask = self.detect_glare(roi_gray)

        data = self.decode_roi(self.enhance_small_barcode(roi_gray, scale))
        if data: return data

        if not has_glare: return None 

        data = self.decode_roi(cv2.resize(self.suppress_glare_retinex(roi_gray), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        if data: return data

        data = self.decode_roi(cv2.resize(self.suppress_glare_tophat(roi_gray), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        if data: return data

        glare_ratio = cv2.countNonZero(glare_mask) / (roi_gray.shape[0] * roi_gray.shape[1])
        if glare_ratio < 0.15:
            fixed = self.suppress_glare_inpaint(roi_gray, glare_mask)
            data = self.decode_roi(self.enhance_small_barcode(fixed, scale))
            if data: return data

        up3 = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        bin_img = cv2.adaptiveThreshold(up3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        data = self.decode_roi(bin_img)
        if data: return data

        return None

    def process_frame(self):
        frames = self.pipeline.poll_for_frames()
        if not frames: return
        color_frame = frames.get_color_frame()
        if not color_frame: return

        image = np.asanyarray(color_frame.get_data())
        self.total_frames += 1

        if self.total_frames % 2 == 0:
            self.current_results.clear()

            results = self.model.predict(image, conf=0.45, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                pad = 20
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

                roi_color = image[y1:y2, x1:x2]
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                
                # ================= 💡 核心升级：多角度轮询攻击 =================
                # 每 30 度转一次，加上 ZBar 自带的容错，构成 360 度绝对无死角天罗地网！
                # for angle in [0, 30, -30, 60, -60, 90]:
                # 分别尝试原图、右转45度、左转45度、垂直转90度
                for angle in [0, 45, -45, 90]:
                    # 1. 旋转截图
                    rotated_roi = self.rotate_image_safely(roi_gray, angle)
                    
                    # 2. 动态放大 (根据旋转后的新宽度计算)
                    scale = max(2.0, min(8.0, 600.0 / rotated_roi.shape[1])) 
                    
                    # 3. 压入抗反光管线解码
                    data = self.scan_with_antiglare(rotated_roi, scale)
                    
                    if data:
                        # ===== 解码成功，进入多帧投票 =====
                        self.candidate_counts[data] += 1
                        if self.candidate_counts[data] >= 2:
                            
                            # 💡 极其清爽的逻辑：不去做恶心的逆向数学映射，直接用 YOLO 给的框画绿框！
                            self.current_results.append(((x1, y1, x2, y2), data))

                            if data not in self.scanned_barcodes:
                                self.scanned_barcodes.add(data)
                                msg = String()
                                msg.data = data
                                self.publisher.publish(msg)
                                self.get_logger().info(f"🟢 完美捕获: {data} (YOLO锁定, {angle}° 矫正成功)")
                        
                        # 这个条码既然已经解出来了，就不需要再试别的角度了，跳出角度循环
                        break 
                # ==============================================================

        if self.total_frames % 100 == 0:
            self.candidate_counts.clear()

        for ((x1, y1, x2, y2), data) in self.current_results:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, data, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 + ZBar Omnidirectional Pipeline", image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloBarcodeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()