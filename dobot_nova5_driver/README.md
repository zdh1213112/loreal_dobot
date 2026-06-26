# Dobot Nova5 ROS 2 Driver

这个包用于把越疆 Nova5 的 TCP/IP Python SDK 接到 `/home/zdh/ffs_ws` 工作空间里。

当前实现：

- 订阅 `PoseStamped` 话题 `/target_pose_tool`
- 将目标位姿转换为 Nova5 TCP 笛卡尔指令
- 默认使用 `MovL`
- 可选切换为 `ServoP`
- 发布当前 TCP 位置到 `/nova5/current_tcp_pose`

建议放置路径：

- `/home/zdh/ffs_ws/src/dobot_nova5_driver`

编译：

```bash
cd /home/zdh/ffs_ws
colcon build --packages-select dobot_nova5_driver
source install/setup.bash
```

运行：

```bash
ros2 launch dobot_nova5_driver nova5_driver.launch.py
```

重要说明：

- 当前节点直接依赖越疆官方 `TCP_IP_Python_V4/dobot_api.py`
- 输入位置单位是米
- 下发到 Dobot SDK 时会自动换算成毫米
- 输入姿态使用四元数，节点内部会转换成 Dobot 需要的欧拉角度数
- 当前 `/nova5/current_tcp_pose` 只发布位置，姿态先固定为单位四元数，后续可以继续补完整姿态反馈
