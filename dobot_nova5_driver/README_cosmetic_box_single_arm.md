# D405 + Nova5 101 化妆品盒自动循环

本流程只连接 `192.168.111.101`，不连接或控制 `192.168.111.102`。

## 流程

1. 操作员在 Qt 界面点击执行；101 回到初始关节角 `(14, 14, -115, 25, 83, 10)`，夹爪打开。
2. 触发序列号 `409122274792` 的 D405。
3. YOLO 同时检测到多个盒子时，使用每个检测框内的双目三维点比较相机光学坐标 X，抓取 X 最小的有效目标。
   视觉窗口中的候选框短暂显示，最终选中的目标会持续显示红色粗框、中心十字和 `NEXT GRASP`，直到下一次检测请求或人工按 `r` 清除，便于操作员确认机械臂即将抓取哪个盒子。
4. SAM2 分割后先用净化点云拟合顶面，同时保留未做聚类裁剪的原始分割点作为高度证据。针对相机基本俯视、物体平放桌面的现场，优先使用目标周围桌面平面与顶面间距估高；桌面平面不可用时依次尝试近似垂直的 RANSAC 侧平面和 SAM 轮廓边缘下方点。发布的 TCP 尖端目标在顶面向下 75% 高度处。
5. 抓取并抬起，移动到中转关节角 `(14, -29, -99, 39, 88, 10)`。
6. 移向中转关节前即打开扫码窗口；如果途中已经读到完整条码，直接进入后续 XYZ 点动。尚未读到时，完全复用旧程序的找码动作：读取当前关节角，将 J6 增加 `-90°`，通过 ±355° 关节限位检查后执行关节运动，最多检查四个面。HID 扫码器默认一次完整解码即成功。
7. 保持扫码后的姿态角，先验证完整目标的逆解，再按 User 0 / Tool 1 点到点移动到 `(503, 121, 371) mm`。不再逐轴 MoveJog，避免某个中间 XYZ 状态导致控制器返回 `-1`。
8. 到达 `(503,121,371) mm` 后恢复旧程序的旋转方式：执行 `MoveJog("Ry-", coordtype=1, user=0, tool=1)`，保持 Tool 1 的 TCP XYZ 不主动改变，沿用户坐标系 Ry 负方向旋转约 `-90°` 后停止。旋转继承关节运动速度系数（默认 25%），超时为 60 秒；该动作不再生成分段笛卡尔姿态，也不做中间姿态 IK，结束后检查旋转角度及 TCP XYZ 漂移。
9. 移动到放置位姿 `(531, 328, 85, -179, -1.39, -85.18)`（XYZ 为 mm，角度为度），打开夹爪。
10. 回初始关节位，触发下一轮检测。

## 构建与启动

```bash
cd /home/zdh/ffs_ws
colcon build --packages-select dobot_nova5_driver --symlink-install
source install/setup.bash
ros2 launch dobot_nova5_driver cosmetic_box_single_arm_cycle.launch.py
```

启动文件默认使用 `/home/zdh/miniconda3/envs/ffs_ros/bin/python` 运行视觉节点；该环境已包含当前机器上的 CUDA、FFS、SAM2、RealSense、Open3D、Ultralytics 和 ROS 2 依赖。若环境位置变化，可传入 `vision_python:=...`。

D405 默认启用自动曝光。RGB 视觉窗口或远程面板获得键盘焦点后，按 `a` 可以实时切换自动/手动曝光；左上角显示 `AE` 或 `M Exp:... G:...`。切换到手动模式时使用曝光 `11000`、增益 `8`，并可通过 `[`、`]` 调整曝光、`-`、`+` 调整增益。

扫码器需要读取 `/dev/input/event*`。临时授权脚本已包含在本项目中。构建并 `source install/setup.bash` 后执行：

```bash
sudo "$(ros2 pkg prefix dobot_nova5_driver)/share/dobot_nova5_driver/scripts/grant_cosmetic_barcode_access.sh"
```

该脚本会自动找到稳定的 `/dev/input/by-id/...BTW_Hid_Device-event-kbd` 链接，并将权限授予执行 `sudo` 的实际登录用户；正在运行的扫码节点会在 2 秒内自动连接。

扫码枪拔插后 Linux 会创建新的 `event*` 节点，临时 ACL 会随旧节点消失。建议在部署机器上一次性安装本项目的 udev 规则，然后拔插一次扫码枪：

```bash
"$(ros2 pkg prefix dobot_nova5_driver)/share/dobot_nova5_driver/scripts/install_cosmetic_barcode_udev_rule.sh"
```

扫码节点每 2 秒自动重连；设备不存在与设备存在但无权限会分别打印不同提示。

## 重要参数

- `handeye_flange_to_cam`：当前默认继承原节点矩阵。101 与 D405 `409122274792` 上机前必须确认或重新标定。
- `command_tool_index=1`：放置点、用户坐标点动和视觉抓取都按 Tool 1 下发。
- `user_index=0`：用户坐标为 User 0。
- `auto_start=false`：默认只打开控制界面，不自动运动。可点击“执行完整一轮”或“开始连续循环”。
- `barcode_stable_hits=1`：HID 扫码器每次完整解码只输出一条数据，默认一条即成功。
- `barcode_max_face_rotations=4`：为保持旧程序动作，界面固定检查四个面（当前面加三次 J6 旋转）；仍未扫码时安全停机，不继续放置。
- `grasp_z_offset_m=0.010`：根据当前实机日志默认将视觉抓取 Z 上移 10 mm，用于补偿手眼高度偏差；界面中正值表示抓得更浅。
- `minimum_safe_tcp_z_m=0.010`：抓取命令不会低于 10 mm；低于该值时自动钳位并打印警告。

界面中可以修改初始关节、中转关节、扫码后用户 XYZ、最终放置位姿、运动速度、夹爪力、抓取 Z 修正、TCP 最低安全 Z、抬升高度及扫码稳定参数。下降到位后，程序会在原位等待夹爪完成闭合并检查 `state=2`，确认夹持后才允许抬升；抬升后会再次检查是否掉落。建议先“只采样视觉”，确认结果后再执行完整一轮。

观察状态：

```bash
ros2 topic echo /cosmetic_pick_cycle_status
ros2 topic echo /cosmetic_box_height
ros2 topic echo /detected_barcodes
```

首次实机验证建议将机械臂速度参数降低，并确认：手眼矩阵、Tool 1、User 0、75% 深度方向以及最终放置点均与现场一致。
