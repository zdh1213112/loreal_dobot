#!/usr/bin/env python3
"""
D405 YOLO-World 单独测试脚本。

功能：
- 只打开 D405 彩色流。
- 使用 YOLO-World 检测一个文本类别，默认 "cosmetic box"。
- 实时显示检测框、置信度、FPS。
- 按 q/ESC 退出，按 s 保存当前画面。

示例：
D405_YOLOWORLD_MODEL=/home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/models/yolov8m-worldv2.pt \
python /home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/dul_cam/test_d405_yoloworld_cosmetic_box.py
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLO-World cosmetic box detection on RealSense D405 RGB stream.")
    parser.add_argument(
        "--model",
        default=os.environ.get("D405_YOLOWORLD_MODEL", "yolov8s-world.pt"),
        help="YOLO-World .pt path, or model name if available in Ultralytics cache.",
    )
    parser.add_argument(
        "--label",
        default=os.environ.get("D405_YOLOWORLD_LABEL", "box"),
        help="Text prompt/class for YOLO-World.",
    )
    parser.add_argument("--serial", default=os.environ.get("D405_SERIAL", "352122272611"), help="D405 serial number.")
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument("--conf", type=float, default=0.01, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0, cuda:0, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--save-dir", default="/tmp/d405_yoloworld_test", help="Directory for saved screenshots.")
    return parser.parse_args()


def draw_detections(image, boxes_xyxy, confs, label):
    display = image.copy()
    for idx, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
        x1, y1, x2, y2 = np.round(box).astype(int)
        x1 = int(np.clip(x1, 0, image.shape[1] - 1))
        x2 = int(np.clip(x2, 0, image.shape[1] - 1))
        y1 = int(np.clip(y1, 0, image.shape[0] - 1))
        y2 = int(np.clip(y2, 0, image.shape[0] - 1))
        color = (0, 255, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        text = f"{idx}:{label} {conf:.2f}"
        cv2.putText(display, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def main():
    args = parse_args()
    print(f"[YOLO-World] model: {args.model}")
    print(f"[YOLO-World] label: {args.label}")
    print(f"[D405] serial: {args.serial}")

    try:
        model = YOLOWorld(args.model)
    except Exception as exc:
        raise RuntimeError(
            "YOLO-World 模型加载失败。请确认模型路径存在，并且 ffs_ros 环境已安装 clip 依赖。"
        ) from exc
    model.set_classes([args.label])

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    print(f"[D405] color intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    window_name = "D405 YOLO-World cosmetic box test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    last_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_bgr = np.asanyarray(color_frame.get_data())

            infer_kwargs = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
            if args.device is not None:
                infer_kwargs["device"] = args.device
            results = model(color_bgr, **infer_kwargs)

            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
                confs = results[0].boxes.conf.detach().cpu().numpy()

            now = time.time()
            inst_fps = 1.0 / max(now - last_time, 1e-6)
            fps_smooth = inst_fps if fps_smooth <= 0 else fps_smooth * 0.9 + inst_fps * 0.1
            last_time = now

            display = draw_detections(color_bgr, boxes_xyxy, confs, args.label)
            status = f"label='{args.label}' conf={args.conf:.2f} det={len(boxes_xyxy)} FPS={fps_smooth:.1f}"
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, "q/ESC quit | s save", (10, args.height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out_path = save_dir / f"d405_yoloworld_{int(time.time())}_{frame_idx}.jpg"
                cv2.imwrite(str(out_path), display)
                print(f"[SAVE] {out_path}")
            frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
