# V2 偏置低位观察与横向插入抓取

启动 `cosmetic_box_single_arm_cycle_v2.launch.py` 默认启用 `offset_grasp_enabled`。
顶面检测窗口开启后：

1. 根据目标抓取姿态预先推算相机 +Y（画面向下）在 User 水平面的投影，生成靠近夹爪一侧的偏置点。
2. 用一条 PTP 同时完成抓取姿态调整和高位移动到 `offset_high`，不经过物料中心上方。
3. 在距离 `offset_high` 终点默认 30 mm 时完成夹爪预张开确认和目标复检，并提前把直线下降 `MovL` 以 CP=20 排入控制器队列；高位到下降连续衔接，不再等待高位完全停稳。若队列不可用则自动回到分段阻塞路径。
4. 低位最多观察 `top_surface_barcode_wait_s`（默认 0.2 秒），此前已有命中则立即继续；检测窗口在下降和后续插入期间仍保持开启。
5. 保持姿态、抓取 Z 和张开宽度，直线插入中心；到位后走原来的闭爪、确认、抬升及条码分支。侧面条码分支在夹爪确认后会把抬升与前往 `transfer_joint` 的动作做 CP 连续队列，减少抬升到中转之间的停顿；扫码成功后再把沿 User X- 的扫码器退让与扫码后安全高度、固定放置位连续排队，减少退让完成后的停顿；顶面条码分支仍只执行抬升后放置。

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
| `offset_high_clearance_m` | 0.120 | 偏置高位相对抓取深度的净空；横移、姿态调整和部分下降合并为一次 PTP |
| `offset_high_descent_blend_enabled` | true | 高位到垂直下降的 CP 连续队列开关 |
| `offset_high_descent_blend_cp` | 20 | 高位到下降的 Dobot CP 平滑比例 |
| `offset_high_descent_queue_lead_m` | 0.030 | 距离高位终点的提前排队距离（m） |
| `offset_high_descent_command_start_grace_s` | 0.30 | 等待高位首个运动反馈包的宽限时间（s） |
| `offset_finger_span_m` | 0.060 | 手指沿插入轴的工作段长度 |
| `grasp_lift_transfer_blend_enabled` | true | 抬升接近终点时提前排队中转关节 |
| `grasp_lift_transfer_blend_cp` | 20 | 抬升到中转的 CP 平滑比例 |
| `grasp_lift_transfer_queue_lead_m` | 0.010 | 距离抬升终点的提前排队距离 |
| `grasp_lift_transfer_command_start_grace_s` | 0.30 | 提交抬升后等待首个反馈包的宽限时间 |
| `post_scan_place_blend_enabled` | true | 扫码后安全高度到侧面固定放置位的连续队列开关 |
| `post_scan_place_blend_cp` | 20 | 安全高度到固定放置位的 CP 平滑比例 |
| `post_scan_place_queue_lead_m` | 0.020 | 距离安全高度终点的提前排队距离 |
| `post_scan_place_command_start_grace_s` | 0.30 | 提交安全高度动作后等待首个反馈包的宽限时间 |
| `scanner_retreat_post_scan_blend_enabled` | true | 扫码器 User X- 退让与扫码后安全高度/固定放置连续队列开关；仅侧面条码路径使用 |
| `scanner_retreat_post_scan_blend_cp` | 20 | 扫码退让到后续移动的 Dobot CP 平滑比例 |
| `scanner_retreat_post_scan_queue_lead_m` | 0.010 | 扫码退让还剩该距离时排队扫码后安全高度动作（m）；默认仍保留 20 mm 额外退让余量 |
| `scanner_retreat_post_scan_command_start_grace_s` | 0.30 | 提交扫码退让队列后等待首个反馈包的宽限时间（s） |

偏置路径不做点云通道检查。该动作不能保证消除镜面反光。

恢复旧路径：

```bash
ros2 launch dobot_nova5_driver cosmetic_box_single_arm_cycle_v2.launch.py offset_grasp_enabled:=false
```
