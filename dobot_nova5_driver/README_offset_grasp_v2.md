# V2 偏置低位观察与横向插入抓取

启动 `cosmetic_box_single_arm_cycle_v2.launch.py` 默认启用 `offset_grasp_enabled`。
顶面检测窗口开启后：

1. 根据目标抓取姿态预先推算相机 +Y（画面向下）在 User 水平面的投影，生成靠近夹爪一侧的偏置点。
2. 用一条 PTP 同时完成抓取姿态调整和高位移动到 `offset_high`，不经过物料中心上方。
3. 在 `offset_high` 完成夹爪预张开确认和目标复检，再直线下降到本轮抓取 Z。
4. 低位最多观察 `top_surface_barcode_wait_s`（默认 0.3 秒），此前已有命中则立即继续；检测窗口在下降和后续插入期间仍保持开启。
5. 保持姿态、抓取 Z 和张开宽度，直线插入中心；到位后走原来的闭爪、确认、抬升及条码分支。

条码检测从接近阶段持续到插入完成。存图阶段新增 `offset_high`、`offset_descent`、`low_observation`、`offset_insert`。
按既定偏置距离执行，不以完整顶面是否落在低位画面内作为运动条件。
`offset_descent` 和 `offset_insert` 使用界面统一计算出的 `linear_speed/linear_acc`，不再受独立 10% 默认速度和 25% 上限限制。

偏置量为物料半长 + 手指工作段半长 + 20 mm。例如物料长 124 mm 时，默认偏置约 112 mm。
路径不再根据 D405 视野或点云通道结果拦截。各路径点仍会先检查逆解，操作员停止后不会继续下发下一段运动。

## 启动参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `offset_grasp_enabled` | true | 启用新路径；false 使用原中心下降流程 |
| `offset_grasp_clearance_m` | 0.020 | 物料末端与手指工作段之间的偏置余量 |
| `offset_finger_span_m` | 0.060 | 手指沿插入轴的工作段长度 |

偏置路径不做点云通道检查。该动作不能保证消除镜面反光。

恢复旧路径：

```bash
ros2 launch dobot_nova5_driver cosmetic_box_single_arm_cycle_v2.launch.py offset_grasp_enabled:=false
```
