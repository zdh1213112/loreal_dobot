# 在有 ultralytics 的 Python 环境跑一次
from ultralytics import YOLO
model = YOLO("YOLOV8s_Barcode_Detection.pt")
model.export(
    format="onnx",
    imgsz=960,         # 默认推理尺寸
    dynamic=True,      # 🔥 关键：让 H/W 运行时可变
    simplify=True,     # 简化计算图
    opset=12
)