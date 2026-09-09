from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


VISION_SCRIPT = (
    "/home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/dul_cam/"
    "d405_cosmetic_box_leftmost_height75_panel_v2.py"
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
            DeclareLaunchArgument(
                "handoff_clearance_enabled",
                default_value="true",
                description=(
                    "Enable all D405 handoff/finger obstacle prevention; "
                    "set false only for controlled commissioning"
                ),
            ),
            DeclareLaunchArgument(
                "handoff_overhead_clearance_enabled",
                default_value="true",
                description=(
                    "Enable the D405 overhead/right-corridor check (purple overlay); "
                    "set false to keep finger-path protection while hiding this check"
                ),
            ),
            DeclareLaunchArgument(
                "secondary_collision_check_enabled",
                default_value="true",
                description=(
                    "Enable the 102 read-only TCP Y-clearance interlock for 101; "
                    "set false to run without 102 feedback protection"
                ),
            ),
            DeclareLaunchArgument(
                "grasp_lift_speed_factor",
                default_value="100",
                description="Effective grasp-lift speed percentage",
            ),
            DeclareLaunchArgument(
                "grasp_lift_acc_factor",
                default_value="100",
                description="Effective grasp-lift acceleration percentage",
            ),
            DeclareLaunchArgument(
                "joint_acc",
                default_value="65",
                description="Joint acceleration baseline used by move-above/pregrasp PTP",
            ),
            DeclareLaunchArgument(
                "linear_speed",
                default_value="65",
                description="Linear grasp-descent speed baseline",
            ),
            DeclareLaunchArgument(
                "linear_acc",
                default_value="65",
                description="Linear grasp-descent acceleration baseline",
            ),
            DeclareLaunchArgument(
                "scanner_approach_natural_finish_margin_m",
                default_value="0.015",
                description=(
                    "Allow a safe bounded scanner approach to finish naturally when "
                    "barcode arrives within this remaining distance"
                ),
            ),
            DeclareLaunchArgument(
                "placement_surface_z_m",
                default_value="0.060",
                description="User-frame placement surface height in meters",
            ),
            DeclareLaunchArgument(
                "placement_safety_margin_m",
                default_value="0.010",
                description=(
                    "Additional User-Z clearance above half the measured material "
                    "length during barcode-up placement"
                ),
            ),
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("vision_python"),
                    VISION_SCRIPT,
                    "--ros-args",
                    "-p",
                    [
                        "handoff_clearance_enabled:=",
                        LaunchConfiguration("handoff_clearance_enabled"),
                    ],
                    "-p",
                    [
                        "handoff_overhead_clearance_enabled:=",
                        LaunchConfiguration("handoff_overhead_clearance_enabled"),
                    ],
                ],
                output="screen",
            ),
            Node(
                package="dobot_nova5_driver",
                executable="barcode_scanner_node_v2",
                name="hid_barcode_scanner_node_v2",
                output="screen",
            ),
            Node(
                package="dobot_nova5_driver",
                executable="nova5_cosmetic_box_cycle_v2",
                name="nova5_cosmetic_box_single_arm_cycle_v2",
                output="screen",
                parameters=[
                    {
                        "secondary_collision_check_enabled": ParameterValue(
                            LaunchConfiguration("secondary_collision_check_enabled"),
                            value_type=bool,
                        ),
                        "motion_speed_scale_percent": ParameterValue(
                            LaunchConfiguration("motion_speed_scale_percent"),
                            value_type=int,
                        ),
                        "barcode_continuous_rotation": ParameterValue(
                            LaunchConfiguration("barcode_continuous_rotation"),
                            value_type=bool,
                        ),
                        "grasp_lift_speed_factor": ParameterValue(
                            LaunchConfiguration("grasp_lift_speed_factor"),
                            value_type=int,
                        ),
                        "grasp_lift_acc_factor": ParameterValue(
                            LaunchConfiguration("grasp_lift_acc_factor"),
                            value_type=int,
                        ),
                        "joint_acc": ParameterValue(
                            LaunchConfiguration("joint_acc"),
                            value_type=int,
                        ),
                        "linear_speed": ParameterValue(
                            LaunchConfiguration("linear_speed"),
                            value_type=int,
                        ),
                        "linear_acc": ParameterValue(
                            LaunchConfiguration("linear_acc"),
                            value_type=int,
                        ),
                        "scanner_approach_natural_finish_margin_m": ParameterValue(
                            LaunchConfiguration("scanner_approach_natural_finish_margin_m"),
                            value_type=float,
                        ),
                        "placement_surface_z_m": ParameterValue(
                            LaunchConfiguration("placement_surface_z_m"),
                            value_type=float,
                        ),
                        "placement_safety_margin_m": ParameterValue(
                            LaunchConfiguration("placement_safety_margin_m"),
                            value_type=float,
                        ),
                    }
                ],
            ),
        ]
    )
