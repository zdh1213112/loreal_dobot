from ultralytics import YOLO

# 1. 加载下载好的微调模型
model = YOLO("/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.pt")

# 2. 对一张图片进行检测
# 请把 'your_test_image.jpg' 替换为你实际的图片路径
results = model("/home/zdh/图片/截图/截图 2026-04-22 11-07-18.png")

# 3. 显示结果（它会自动画出框并标上 barcode 或 qrcode）
results[0].show()