from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
            DeclareLaunchArgument(
                "motion_speed_scale_percent",
                default_value="400",
                description="Unified motion scale; 100 is the legacy effective-speed baseline and 400 is the current faster default",
            ),
            DeclareLaunchArgument(
                "barcode_continuous_rotation",
                default_value="true",
                description=(
                    "Sweep J6 continuously through 270 degrees and snap a detected "
                    "barcode to the nearest 90-degree face; set false for segmented search"
                ),
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
                parameters=[
                    {
                        "motion_speed_scale_percent": ParameterValue(
                            LaunchConfiguration("motion_speed_scale_percent"),
                            value_type=int,
                        ),
                        "barcode_continuous_rotation": ParameterValue(
                            LaunchConfiguration("barcode_continuous_rotation"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
