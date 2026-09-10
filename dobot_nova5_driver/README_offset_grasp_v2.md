# V2 偏置低位观察与横向插入抓取

启动 `cosmetic_box_single_arm_cycle_v2.launch.py` 默认启用 `offset_grasp_enabled`。
保留原有高位目标复检，然后：

1. 将当前相机 +Y（画面向下）投影到 User 水平面，生成靠近夹爪一侧的偏置点。
2. 高位移到偏置点，再直线下降到本轮抓取 Z。
3. 低位最多观察 `top_surface_barcode_wait_s`（默认 0.5 秒），此前已有命中则立即继续。
4. 保持姿态、抓取 Z 和张开宽度，直线插入中心；到位后走原来的闭爪、确认、抬升及条码分支。

条码检测从接近阶段持续到插入完成。存图阶段新增 `offset_high`、`offset_descent`、`low_observation`、`offset_insert`。
按既定偏置距离执行，不以完整顶面是否落在低位画面内作为运动条件。
低位等待替代原观察位等待；新增平移路径本身会增加耗时。

偏置量为物料半长 + 手指工作段半长 + 20 mm。例如物料长 124 mm 时，默认偏置约 112 mm。
路径不再根据 D405 视野或点云通道结果拦截。各路径点仍会先检查逆解，操作员停止后不会继续下发下一段运动。

## 启动参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `offset_grasp_enabled` | true | 启用新路径；false 使用原中心下降流程 |
| `offset_grasp_speed_percent` | 10 | 新路径 MovL 速度与加速度，代码上限 25 |
| `offset_grasp_clearance_m` | 0.020 | 物料末端与手指工作段之间的偏置余量 |
| `offset_finger_span_m` | 0.060 | 手指沿插入轴的工作段长度 |
偏置路径不做点云通道检查；首次运行应低速观察偏置下降和横移方向。该动作不能保证消除镜面反光。
当前未自动运动或完成实机验证。

恢复旧路径：

```bash
ros2 launch dobot_nova5_driver cosmetic_box_single_arm_cycle_v2.launch.py offset_grasp_enabled:=false
```
