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
                "scanner_transfer_barcode_grace_s",
                default_value="0.06",
                description=(
                    "Wait briefly at transfer_joint for a late HID callback before "
                    "starting the monitored scanner approach"
                ),
            ),
            DeclareLaunchArgument("offset_grasp_enabled", default_value="true",
                                  description="Reach offset-high while aligning grasp attitude, then descend and insert"),
            DeclareLaunchArgument("offset_grasp_clearance_m", default_value="0.020"),
            DeclareLaunchArgument("offset_finger_span_m", default_value="0.060"),
            DeclareLaunchArgument(
                "top_surface_barcode_enabled",
                default_value="true",
                description=(
                    "Check the current target region during approach, hover and descent; "
                    "when found, place without scanner/J6/Ry/Rz rotation"
                ),
            ),
            DeclareLaunchArgument(
                "top_surface_barcode_stable_hits",
                default_value="1",
                description="Number of D405 top-surface YOLO barcode detections required",
            ),
            DeclareLaunchArgument(
                "top_surface_barcode_wait_s",
                default_value="0.30",
                description="Maximum hover time to wait for top-surface barcode confirmation",
            ),
            DeclareLaunchArgument(
                "bottom_barcode_recovery_enabled",
                default_value="true",
                description=(
                    "When top and all side faces have no barcode, place on the table, "
                    "flip User Ry- and retry the bottom face"
                ),
            ),
            DeclareLaunchArgument(
                "bottom_flip_user_ry_target_deg",
                default_value="-45.0",
                description="User-Ry target for the bottom-barcode table flip",
            ),
            DeclareLaunchArgument(
                "bottom_flip_j6_pre_return_deg",
                default_value="90.0",
                description="J6 positive return after an unsuccessful -270deg face sweep",
            ),
            DeclareLaunchArgument(
                "bottom_flip_table_retract_m",
                default_value="0.050",
                description="Extra User-X- clearance before placing the box on the table (m)",
            ),
            DeclareLaunchArgument(
                "bottom_flip_table_z_offset_m",
                default_value="0.010",
                description="Clearance retained when reversing the first grasp lift (m)",
            ),
            DeclareLaunchArgument(
                "bottom_flip_lift_m",
                default_value="0.160",
                description="Vertical lift after bottom-face table regrasp (m)",
            ),
            DeclareLaunchArgument(
                "bottom_barcode_place_ry_delta_deg",
                default_value="-45.0",
                description=(
                    "Additional User-Ry rotation at fixed bottom-barcode place pose"
                ),
            ),
            DeclareLaunchArgument(
                "side_barcode_place_rx_delta_deg",
                default_value="-20.0",
                description=(
                    "User-Rx tilt for side-barcode fixed placement"
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
                    "-p",
                    [
                        "top_surface_barcode_stable_hits:=",
                        LaunchConfiguration("top_surface_barcode_stable_hits"),
                    ],
                    "-p",
                    [
                        "top_surface_barcode_enabled:=",
                        LaunchConfiguration("top_surface_barcode_enabled"),
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
                        "scanner_transfer_barcode_grace_s": ParameterValue(
                            LaunchConfiguration("scanner_transfer_barcode_grace_s"),
                            value_type=float,
                        ),
                        "offset_grasp_enabled": ParameterValue(LaunchConfiguration("offset_grasp_enabled"), value_type=bool),
                        "offset_grasp_clearance_m": ParameterValue(LaunchConfiguration("offset_grasp_clearance_m"), value_type=float),
                        "offset_finger_span_m": ParameterValue(LaunchConfiguration("offset_finger_span_m"), value_type=float),
                        "top_surface_barcode_enabled": ParameterValue(
                            LaunchConfiguration("top_surface_barcode_enabled"),
                            value_type=bool,
                        ),
                        "top_surface_barcode_wait_s": ParameterValue(
                            LaunchConfiguration("top_surface_barcode_wait_s"),
                            value_type=float,
                        ),
                        "bottom_barcode_recovery_enabled": ParameterValue(
                            LaunchConfiguration("bottom_barcode_recovery_enabled"),
                            value_type=bool,
                        ),
                        "bottom_flip_user_ry_target_deg": ParameterValue(
                            LaunchConfiguration("bottom_flip_user_ry_target_deg"),
                            value_type=float,
                        ),
                        "bottom_flip_j6_pre_return_deg": ParameterValue(
                            LaunchConfiguration("bottom_flip_j6_pre_return_deg"),
                            value_type=float,
                        ),
                        "bottom_flip_table_retract_m": ParameterValue(
                            LaunchConfiguration("bottom_flip_table_retract_m"),
                            value_type=float,
                        ),
                        "bottom_flip_table_z_offset_m": ParameterValue(
                            LaunchConfiguration("bottom_flip_table_z_offset_m"),
                            value_type=float,
                        ),
                        "bottom_flip_lift_m": ParameterValue(
                            LaunchConfiguration("bottom_flip_lift_m"),
                            value_type=float,
                        ),
                        "bottom_barcode_place_ry_delta_deg": ParameterValue(
                            LaunchConfiguration("bottom_barcode_place_ry_delta_deg"),
                            value_type=float,
                        ),
                        "side_barcode_place_rx_delta_deg": ParameterValue(
                            LaunchConfiguration("side_barcode_place_rx_delta_deg"),
                            value_type=float,
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
