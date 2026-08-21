# D405 + Nova5 101 化妆品盒自动循环

本流程只连接 `192.168.111.101`，不连接或控制 `192.168.111.102`。

## 流程

1. 操作员在 Qt 界面点击执行；101 回到初始关节角 `(14, 14, -115, 25, 83, 10)`，夹爪打开。
2. 触发序列号 `409122274792` 的 D405。
3. YOLO 同时检测到多个盒子时，使用每个检测框内的双目三维点比较相机光学坐标 X，抓取 X 最小的有效目标。
   视觉窗口中的候选框短暂显示，最终选中的目标会持续显示红色粗框、中心十字和 `NEXT GRASP`，直到下一次检测请求或人工按 `r` 清除，便于操作员确认机械臂即将抓取哪个盒子。
4. SAM2 分割后先用净化点云拟合顶面，同时保留未做聚类裁剪的原始分割点作为高度证据。针对相机基本俯视、物体平放桌面的现场，优先使用目标周围桌面平面与顶面间距估高；桌面平面不可用时依次尝试近似垂直的 RANSAC 侧平面和 SAM 轮廓边缘下方点。发布的 TCP 尖端目标在顶面向下 75% 高度处。
5. 视觉同时发布点云包围盒长边。抓取并抬起后移动到中转关节角 `(14, -29, -99, 39, 88, 10)`，再沿 User 0 的 X+ 自动靠近扫码器。位移使用 `中转中心距离 - 盒长/2 - 扫码间隙`；默认即 `120 mm - 盒长/2 - 30 mm`。例如盒长 100 mm 时向 X+ 移动 40 mm，使盒子近侧面距扫码器约 30 mm。
6. 抬升确认成功后、开始前往中转关节之前就打开扫码窗口，因此前往 `transfer_joint` 途中读到的条码不会丢失。到达中转点后立即检查：已经扫到稳定条码时跳过 User X+ 靠近和全部 J6 找码，直接进入安全退让；尚未扫到时才自适应靠近扫码器并沿 J6 负方向点动找码。J6 旋转途中约每 20 ms 检查一次条码，成功后立即停止并对齐到最近的标准 90° 面位。
7. 扫码成功并完成 J6 标准面对齐后，盒子仍处于靠近扫码器的位置。程序先保持当前姿态沿 User X− 退回本轮实际 X+ 靠近距离，再额外退让 30 mm；只有核对实际退让量正确后才允许大范围姿态运动。
8. 以退让后的实际姿态为起点，按固定 User 轴顺序合成 `Ry -90°`、再 `Rz +50°`，并与 User XYZ `(557, 200, 320) mm` 组成一个完整目标位姿。程序先验证完整目标逆解，再用一条 User 0 / Tool 1 PTP 让位置和姿态同时变化；结束后同时检查 XYZ 与姿态误差。
9. 移动到放置位姿 `(531, 328, 85, -179.0, -1.39, -85.18)`（XYZ 为 mm，角度为度），打开夹爪。
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
- `scanner_center_distance_m=0.120`：`transfer_joint` 处 TCP 夹持中心到扫码器的 User-X 距离。
- `scanner_face_clearance_m=0.030`：自适应靠近后盒子侧面与扫码器之间保留的间隙。
- `joint_speed=65`、`joint_acc=55`：普通关节运动的速度和加速度；短行程提速重点看加速度。
- `linear_speed=60`、`linear_acc=50`：抓取下降直线运动的速度和加速度。
- `scan_exit_user_xyz=[0.557,0.200,0.320]`：扫码后组合 PTP 的 User XYZ 目标，单位为米。
- `face_up_user_ry_deg=-90`、`post_scan_user_rz_deg=50`：组合目标相对扫码姿态的固定 User 轴旋转；先 Ry，后 Rz。
- `jog_speed_factor=50`：扫码后 XYZ+Ry+Rz 单条组合 PTP 的速度。
- `scanner_approach_speed_factor=50`：只控制中转点后沿 User X+ 靠近扫码器的速度。
- `scanner_retreat_speed_factor=60`：扫码成功后沿 User X− 安全退让的速度。
- `scanner_retreat_extra_m=0.030`：退回实际靠近距离后继续远离扫码器的额外安全余量。
- `barcode_j6_speed_factor=70`：只控制 J6 多面找码，以及扫码成功后的标准面对齐速度；设置过高会增大急停超调。
- `vision_samples=2`：机器人仍使用两帧结果检查 SAM2 目标稳定性；眼在手相机到达初始位后只刷新 2 帧，并复用本次目标选择时的锁定 FFS 点云，避免静止场景重复计算立体深度。
- `vision_result_topic=/d405_vision_result`：视觉端会明确返回本次请求成功或失败；ROI 内无目标、YOLO 无目标、立体点不足时，机械臂不再固定等满 `vision_timeout_s=8.0`。
- `vision_retry_delay_s=0.3`：连续模式一次检测明确失败后，到发起下一次检测之间的等待时间。
- `barcode_face_wait_s=0.1`：每个 J6 标准面等待扫码的最长时间。
- `grasp_close_settle_s=0.05`：夹爪闭合命令完成后、第一次反馈确认前的原位稳定时间；仍保留两次夹持反馈确认。
- `barcode_max_face_rotations=4`：为保持旧程序动作，界面固定检查四个面（当前面加三次 J6 旋转）；仍未扫码时安全停机，不继续放置。
- `grasp_z_offset_m=0.010`：根据当前实机日志默认将视觉抓取 Z 上移 10 mm，用于补偿手眼高度偏差；界面中正值表示抓得更浅。
- `minimum_safe_tcp_z_m=0.010`：抓取命令不会低于 10 mm；低于该值时自动钳位并打印警告。

界面中可以修改初始关节、中转关节、扫码后用户 XYZ、组合 User Ry/Rz 增量、最终放置位姿、普通运动速度、扫码器靠近速度、J6 找码速度、夹爪力、抓取 Z 修正、TCP 最低安全 Z、抬升高度及扫码稳定参数。接受视觉目标后，夹爪预张开会与机械臂移动到目标上方并行，到达上方后仍会确认夹爪已停止才允许下降。下降到位后，程序会在原位等待夹爪完成闭合并检查 `state=2`，确认夹持后才允许抬升；抬升后会再次检查是否掉落。建议先“只采样视觉”，确认结果后再执行完整一轮。

观察状态：

```bash
ros2 topic echo /cosmetic_pick_cycle_status
ros2 topic echo /cosmetic_box_height
ros2 topic echo /detected_barcodes
```

首次实机验证建议将机械臂速度参数降低，并确认：手眼矩阵、Tool 1、User 0、75% 深度方向以及最终放置点均与现场一致。
