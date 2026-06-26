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

        # ================= 1. 初始化 YOLOv8 =================
        self.get_logger().info("🧠 正在加载 YOLOv8 原生模型...")
        self.model = YOLO('/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.pt')

        # ================= 2. 初始化 D415 (复刻 C++ 硬件参数) =================
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

            self.get_logger().info("🚀 D415 YOLO+ZBar (全套抗反光级联) 模式启动！")
        except Exception as e:
            self.get_logger().error(f"❌ 相机启动失败: {e}")
            raise e

        self.total_frames = 0
        self.scanned_barcodes = set()
        self.candidate_counts = defaultdict(int)
        self.current_results = []

        self.timer = self.create_timer(0.03, self.process_frame)

    # ================= 💡 复刻 C++ 核心逻辑 =================
    def validate_ean13(self, code):
        """严格校验 EAN-13，杜绝误码"""
        if len(code) != 13 or not code.isdigit():
            return False
        sum_val = sum(int(d) if i % 2 == 0 else int(d) * 3 for i, d in enumerate(code[:12]))
        check = (10 - sum_val % 10) % 10
        return check == int(code[12])

    def detect_glare(self, gray):
        """检测过曝反光区域"""
        _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_pixels = cv2.countNonZero(glare_mask)
        ratio = glare_pixels / (gray.shape[0] * gray.shape[1])
        return ratio > 0.01, glare_mask

    def enhance_small_barcode(self, gray, scale):
        """策略 0：基础 CLAHE 增强锐化"""
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), 1.2)
        sharp = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
        return clahe.apply(sharp)

    def suppress_glare_retinex(self, gray):
        """策略 1：Retinex 局部亮度归一化去光照"""
        gray_f = gray.astype(np.float32) / 255.0
        illum = cv2.GaussianBlur(gray_f, (51, 51), 25)
        log_gray = cv2.log(gray_f + 0.01)
        log_illum = cv2.log(illum + 0.01)
        reflect = log_gray - log_illum
        out = cv2.normalize(reflect, None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)

    def suppress_glare_tophat(self, gray):
        """策略 2：Top-hat 黑帽形态学提暗线"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, out = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return out

    def suppress_glare_inpaint(self, gray, glare_mask):
        """策略 3：Inpaint 修复破损的高光"""
        mask_dilated = cv2.dilate(glare_mask, None, iterations=2)
        return cv2.inpaint(gray, mask_dilated, 3, cv2.INPAINT_TELEA)

    def scan_and_collect(self, scan_gray, origin_x, origin_y, scale):
        """执行 ZBar 扫描，校验并收集投票"""
        if scan_gray is None or scan_gray.size == 0:
            return False

        barcodes = decode(scan_gray, symbols=[ZBarSymbol.EAN13])
        hit = False
        
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            if not self.validate_ean13(data):
                continue

            self.candidate_counts[data] += 1
            if self.candidate_counts[data] >= 2:
                # 坐标映射：将放大后的 ZBar 多边形坐标还原回 1280x720 原图坐标
                poly = barcode.polygon
                if not poly: continue
                
                pts = np.array([[(p.x / scale) + origin_x, (p.y / scale) + origin_y] for p in poly], np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                
                self.current_results.append(((x, y, x+w, y+h), data))

                if data not in self.scanned_barcodes:
                    self.scanned_barcodes.add(data)
                    msg = String()
                    msg.data = data
                    self.publisher.publish(msg)
                    self.get_logger().info(f"🟢 [YOLO+抗反光解码] 完美捕获: {data}")
            hit = True
        return hit

    def scan_with_antiglare(self, roi_gray, origin_x, origin_y, scale):
        """五级阶梯式抗反光流水线 (复刻 C++)"""
        has_glare, glare_mask = self.detect_glare(roi_gray)

        # 策略 0：标准增强 (命中直接退出，保证极速)
        enhanced = self.enhance_small_barcode(roi_gray, scale)
        if self.scan_and_collect(enhanced, origin_x, origin_y, scale): return True

        if not has_glare: return False # 无反光就不折腾后面的重度算法了

        # 策略 1：Retinex
        retinex = self.suppress_glare_retinex(roi_gray)
        up1 = cv2.resize(retinex, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if self.scan_and_collect(up1, origin_x, origin_y, scale): return True

        # 策略 2：Top-hat
        bh = self.suppress_glare_tophat(roi_gray)
        up2 = cv2.resize(bh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if self.scan_and_collect(up2, origin_x, origin_y, scale): return True

        # 策略 3：Inpaint (仅在中等面积反光使用)
        glare_ratio = cv2.countNonZero(glare_mask) / (roi_gray.shape[0] * roi_gray.shape[1])
        if glare_ratio < 0.15:
            fixed = self.suppress_glare_inpaint(roi_gray, glare_mask)
            enhanced_fixed = self.enhance_small_barcode(fixed, scale)
            if self.scan_and_collect(enhanced_fixed, origin_x, origin_y, scale): return True

        # 策略 4：强对比二值化
        up3 = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        bin_img = cv2.adaptiveThreshold(up3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        if self.scan_and_collect(bin_img, origin_x, origin_y, scale): return True

        return False

    def process_frame(self):
        frames = self.pipeline.poll_for_frames()
        if not frames: return
        color_frame = frames.get_color_frame()
        if not color_frame: return

        image = np.asanyarray(color_frame.get_data())
        self.total_frames += 1

        if self.total_frames % 2 == 0:
            self.current_results.clear()

            # YOLO 找框
            results = self.model.predict(image, conf=0.45, verbose=False)
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # ================= 💡 修改：缩小外扩比例 (固定像素) =================
                # 之前比例扩太大包含太多背景杂波，现在固定扩 15 像素，刚刚好满足静区
                pad = 15
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)

                # 蓝框提示 YOLO 抓取区
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

                roi_color = image[y1:y2, x1:x2]
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                
                # 动态自适应放大倍数 (复刻 C++)
                scale = max(2.0, min(8.0, 600.0 / roi_gray.shape[1])) #ZBar 扫码算法的最佳工作区是图像宽度在 600 像素左右
                
                # 送入全套抗反光流水线
                self.scan_with_antiglare(roi_gray, x1, y1, scale)

        if self.total_frames % 100 == 0:
            self.candidate_counts.clear()

        # 绿框提示成功解出的条码
        for ((x1, y1, x2, y2), data) in self.current_results:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, data, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 + ZBar Advanced Pipeline", image)
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