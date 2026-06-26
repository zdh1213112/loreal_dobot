from ultralytics import YOLO

# 1. 加载你的权重文件
model = YOLO("/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.pt")

# 2. 导出时加入向下兼容的神奇参数：
# opset=12：使用老版本的 ONNX 算子集，完美适配 OpenCV 4.5.4
# simplify=True：使用 onnxslim 强行折叠复杂节点，变成静态的简单网络
path = model.export(
    format="onnx", 
    imgsz=640, 
    opset=11, 
    simplify=True,
    dynamic=False   # 固定 shape，很关键
) 

print(f"✅ 兼容版 ONNX 模型已成功导出至: {path}")