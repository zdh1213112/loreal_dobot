from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


VISION_SCRIPT = (
    "/home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/dul_cam/"
    "d405_cosmetic_box_leftmost_height75_panel.py"
)


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vision_python",
                default_value="/home/zdh/miniconda3/envs/ffs_ros/bin/python",
                description="Python environment containing CUDA FFS, SAM2, RealSense, Open3D and ROS 2",
            ),
            ExecuteProcess(
                cmd=[LaunchConfiguration("vision_python"), VISION_SCRIPT],
                output="screen",
            ),
            Node(
                package="dobot_nova5_driver",
                executable="barcode_scanner_node",
                name="hid_barcode_scanner_node",
                output="screen",
            ),
            Node(
                package="dobot_nova5_driver",
                executable="nova5_cosmetic_box_cycle",
                name="nova5_cosmetic_box_single_arm_cycle",
                output="screen",
            ),
        ]
    )
