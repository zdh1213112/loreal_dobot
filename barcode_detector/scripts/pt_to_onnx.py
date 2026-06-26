from ultralytics import YOLO

# 加载模型
model = YOLO("/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.pt")

# 将模型导出为 ONNX 格式，默认包含动态形状或指定固定的 640x640 尺寸
# imgsz=640 是文档中提到的训练尺寸，使用这个尺寸推理效果最好
path = model.export(format="onnx", imgsz=640) 

print(f"模型已成功导出为: {path}")