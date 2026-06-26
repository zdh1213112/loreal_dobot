# D435 + Nova5 RRT 抓取

新增文件：

- `d435_nova5_rrt_grasp_gui.py`
- `d435_nova5_rrt_config.yaml`

这套代码是独立实现，不会修改你之前的 `d435_global_coarse.py`、`d405_local_ransac_coarse_lock_wait_lock.py` 和 Nova5 旧驱动脚本。

## 功能

- 只使用眼在手外的顶部 `D435`
- `YOLO-OBB + depth` 得到目标在 `D435` 相机坐标系中的位置
- 从 `D435` 深度图生成三维点云障碍
- 在相机坐标系中做三维 `RRT`
- 对障碍点云保持 `10 cm` 安全距离
- 生成 `预抓取点 -> 抓取点` 路径
- 通过 `Qt` 界面点击执行 `Nova5`

## Git 上传说明

建议提交 `dul_cam/` 里的源码、`d435_nova5_rrt_config.yaml`，以及根目录下代码依赖的 `Utils.py`、`core/`、`SAM2_streaming/sam2/`、`SAM2_streaming/configs/`、`requirements.txt`。

以下文件体积大或与本机环境相关，已由 `.gitignore` 忽略，需要在运行机器本地准备：

- `weights/23-36-37/model_best_bp2_serialize.pth`
- `SAM2_streaming/checkpoints/sam2.1/sam2.1_hiera_small.pt`
- YOLO 权重，例如配置里的 `/home/zdh/yolo_one/yolo_easy_deploy/outputs/train/obb_demo-4/weights/best.pt`
- 本机硬件/驱动路径，例如 `d435_nova5_rrt_config.yaml` 里的 `driver_root`

`weights/23-36-37/cfg.yaml` 是小配置文件，建议随代码提交。

## 启动

```bash
cd /home/zdh/ffs_ws/src/Fast-FoundationStereoPose/dul_cam
python3 d435_nova5_rrt_grasp_gui.py
```

如果你要用别的配置文件：

```bash
python3 d435_nova5_rrt_grasp_gui.py --config /absolute/path/to/config.yaml
```

## 使用顺序

1. 点 `Connect Robot`
2. 点 `Start D435`
3. 点 `Capture Scene`
4. 点 `Plan RRT`
5. 确认预览图和 `Preview 3D`
6. 点 `Execute`

## 当前实现假设

- 已知 `TCP` 尖端在 `D435` 顶视相机坐标系中的当前起点，默认优先通过机器人反馈和 `base<-d435` 标定矩阵反算。
- 已知 `base <- d435` 外参，默认沿用了你旧代码里的 `handeye_base_to_d435`。
- 目标抓取姿态默认沿用了你旧粗抓逻辑的思路：
  - `Rx = 180 deg`
  - `Ry = 0 deg`
  - `Rz = 目标yaw - 90 deg`

## 当前边界

- 这里只做了单目标、单次抓取、顶部 `D435` 避障路径。
- 没接入你旧流程里的 `D405` 精定位。
- `RRT` 是在笛卡尔空间对 `TCP` 点做规划，不是关节空间 `RRT-Connect`。
- 没做机械臂连杆体积碰撞，只做了 `TCP` 路径点对场景点云的避障。
- 如果相机能看到的点云不完整，遮挡后方的障碍不会被规划器感知到。

## 你后面大概率还要继续补的部分

- 夹爪体积和手腕体积碰撞模型
- 关节可达性和奇异位姿筛查
- 抓取末端姿态模板精调
- 支撑面 / 料框 / 传送带更稳定的分割
- 执行前的路径插值、速度分段和失败回退
