# Robot Vision & Manipulation Projects

本仓库整理了 `/home/zdh/ffs_ws/src` 中的三个主要项目，用于机器人视觉感知、条码识别、双目位姿估计以及 Dobot Nova5 机械臂控制。

## 项目结构

```text
.
├── barcode_detector/                 # ROS 2 条码/二维码检测节点
├── dobot_nova5_driver/               # Dobot Nova5 ROS 2 控制驱动
└── Fast-FoundationStereoPose-dul_cam/ # 双目深度、目标分割与 3D 位姿估计
```

## 模块简介

- `barcode_detector`：基于 ROS 2、OpenCV、YOLO/ZBar 的条码识别模块，可用于相机图像中的条码检测与结果发布。
- `dobot_nova5_driver`：Dobot Nova5 机械臂 TCP/IP 控制驱动，订阅目标位姿并下发机械臂运动指令。
- `Fast-FoundationStereoPose-dul_cam`：基于 Fast-FoundationStereo、SAM2、YOLO 等工具的双目深度估计、目标跟踪和 3D 位姿估计项目。

## 基本使用

ROS 2 工作空间建议路径：

```bash
cd /home/zdh/ffs_ws
colcon build
source install/setup.bash
```

运行 Dobot Nova5 驱动示例：

```bash
ros2 launch dobot_nova5_driver nova5_driver.launch.py（目前还在开发此命令无效，先使用python直接显式启动）
```

其他视觉脚本请进入对应项目目录，根据各自 README 或脚本说明运行。

## 注意事项

- 本仓库主要保存代码与配置文件。
- `.pt`、`.pth`、`.onnx` 等模型权重文件通常较大，建议单独下载或使用 Git LFS 管理。
- Python 缓存、编译产物和数据文件不建议提交到 Git。
- 使用机械臂前请确认 IP、坐标系、运动范围和急停安全措施。

## 维护者

- zdh
