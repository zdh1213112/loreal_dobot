# Barcode ONNX export

The Fast D435 detector expects a fixed `1x3x640x640` input and a postprocessed
detection output shaped `1x300x6`. Use the supplied exporter so these interfaces are
checked before the model is used by ROS.

```bash
/home/zdh/miniconda3/envs/yolo/bin/python \
  /home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/tools/export_barcode_onnx.py \
  /home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/models/best_0929.pt \
  --output /home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/models/best_0929.onnx \
  --imgsz 640 \
  --opset 17 \
  --device cpu
```

Exporting on CPU does not limit inference to CPU. ONNX models do not contain a
fixed execution provider; the Fast D435 node can load the same file with ONNX
Runtime `CUDAExecutionProvider`.

The script requires `ultralytics`, `onnx`, `onnxruntime`, and `numpy`. The
workspace `yolo` Conda environment already contains these packages.
