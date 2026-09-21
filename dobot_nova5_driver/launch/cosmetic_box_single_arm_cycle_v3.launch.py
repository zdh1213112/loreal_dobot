"""V3 turntable-integration baseline forked from the complete V2 launch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# The V3 D405 process remains beside its SAM2/FFS resources in the vision
# repository, but has its own entry point so V3 UI behavior cannot alter V2.
VISION_SCRIPT = (
    "/home/zdh/ffs_ws/src/Fast-FoundationStereoPose-dul_cam/dul_cam/"
    "d405_cosmetic_box_leftmost_height75_panel_v3.py"
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
                "turntable_d435_serial",
                default_value="254322071102",
                description="D435 serial used only for turntable side-barcode detection",
            ),
            DeclareLaunchArgument(
                "turntable_d435_confidence",
                default_value="0.60",
                description="Presence-only YOLO confidence threshold for D435 barcode detection",
            ),
            DeclareLaunchArgument(
                "turntable_d435_yolo_stable_hits",
                default_value="2",
                description="Spatially consistent YOLO frames required before D435 confirms a barcode",
            ),
            DeclareLaunchArgument(
                "turntable_d435_model_path",
                default_value=(
                    "/home/zdh/yolo_one/yolo_train_xense_load_image/outputs/"
                    "train/obb_demo111/weights/best.onnx"
                ),
                description="D435 barcode detector weights; fixed-shape ONNX is the default",
            ),
            DeclareLaunchArgument(
                "turntable_d435_inference_provider",
                default_value="cuda",
                description="ONNX provider: cuda (default with CPU fallback), cpu, or auto",
            ),
            DeclareLaunchArgument(
                "turntable_d435_image_size",
                default_value="640",
                description="YOLO inference size; 640 matches training and the fixed ONNX export",
            ),
            DeclareLaunchArgument(
                "turntable_d435_scanner_assist",
                default_value="true",
                description="Use fast camera-scanner pattern detection before the YOLO fallback",
            ),
            DeclareLaunchArgument(
                "turntable_d435_scanner_assist_hits",
                default_value="3",
                description="Spatially stable scanner-pattern frames required for confirmation",
            ),
            DeclareLaunchArgument(
                "turntable_d435_auto_exposure",
                default_value="false",
                description="Use D435 RGB auto exposure; false is recommended for a moving turntable",
            ),
            DeclareLaunchArgument(
                "turntable_d435_exposure",
                default_value="50",
                description="D435 RGB manual exposure for motion-freezing barcode images",
            ),
            DeclareLaunchArgument(
                "turntable_d435_gain",
                default_value="128",
                description="D435 RGB manual gain used to brighten the short-exposure moving image",
            ),
            DeclareLaunchArgument("turntable_d435_roi_x", default_value="0"),
            DeclareLaunchArgument("turntable_d435_roi_y", default_value="0"),
            DeclareLaunchArgument(
                "turntable_d435_roi_width",
                default_value="0",
                description="D435 ROI width; zero uses the remaining image width",
            ),
            DeclareLaunchArgument(
                "turntable_d435_roi_height",
                default_value="0",
                description="D435 ROI height; zero uses the remaining image height",
            ),
            DeclareLaunchArgument(
                "turntable_d435_preview",
                default_value="true",
                description=(
                    "Publish D435 into the combined D405 window; left-drag "
                    "selects ROI and right-click restores full-frame detection"
                ),
            ),
            DeclareLaunchArgument(
                "turntable_d435_continuous_on_start",
                default_value="true",
                description="Keep D435 barcode presence detection active from startup",
            ),
            DeclareLaunchArgument(
                "turntable_do_index",
                default_value="1",
                description="101 controller-cabinet DO wired to the turntable toggle input",
            ),
            DeclareLaunchArgument(
                "turntable_pulse_ms",
                default_value="300",
                description="Low reset and high hold duration of each 0->1->0 toggle pulse",
            ),
            DeclareLaunchArgument(
                "turntable_scan_timeout_s",
                default_value="1.41",
                description="Fixed D435 side-barcode classification window",
            ),
            DeclareLaunchArgument(
                "turntable_stationary_barcode_check_s",
                default_value="1.5",
                description="Check the already-visible stopped face before starting turntable rotation",
            ),
            DeclareLaunchArgument(
                "turntable_settle_s",
                default_value="0.50",
                description="Wait after stop pulse before requesting a fresh D405 target",
            ),
            DeclareLaunchArgument(
                "turntable_assume_stopped_on_start",
                default_value="false",
                description="Set true only after physically confirming the turntable is stopped",
            ),
            DeclareLaunchArgument(
                "turntable_require_place_done",
                default_value="true",
                description="Require either the passive 102 place/retreat trigger or /turntable_place_done",
            ),
            DeclareLaunchArgument(
                "turntable_auto_place_from_secondary_tcp",
                default_value="true",
                description="Passively infer right-arm placement and retreat from read-only 102 TCP feedback",
            ),
            DeclareLaunchArgument(
                "turntable_secondary_place_y_m",
                default_value="0.400",
                description="102 User-0/Tool-1 TCP Y boundary entered during turntable placement",
            ),
            DeclareLaunchArgument(
                "turntable_secondary_safe_z_m",
                default_value="0.200",
                description="Deprecated compatibility argument; 102 TCP Z is not used by the automatic turntable trigger",
            ),
            DeclareLaunchArgument(
                "turntable_secondary_safe_z_stable_s",
                default_value="0.200",
                description="Continuous time with 102 Y below the placement boundary before accepting one event; legacy name retained for compatibility",
            ),
            DeclareLaunchArgument(
                "turntable_surface_z_m",
                default_value="0.126",
                description="Measured User-0 turntable material-support surface Z (126 mm on this cell)",
            ),
            DeclareLaunchArgument(
                "vision_user_z_bias_m",
                default_value="0.0287",
                description="V3-only D405 field correction added to transformed User-0 target Z",
            ),
            DeclareLaunchArgument(
                "turntable_surface_tolerance_m",
                default_value="0.020",
                description="Maximum D405-derived support-plane error relative to turntable top",
            ),
            DeclareLaunchArgument(
                "turntable_tcp_below_target_m",
                default_value="0.0",
                description="Measured gripper/tool extension below the commanded grasp TCP",
            ),
            DeclareLaunchArgument(
                "turntable_surface_clearance_m",
                default_value="0.003",
                description="Required gripper clearance above the turntable surface",
            ),
            DeclareLaunchArgument(
                "grasp_z_offset_m",
                default_value="-0.004",
                description="V3 grasp TCP Z correction; negative descends deeper while the turntable floor remains enforced",
            ),
            DeclareLaunchArgument(
                "motion_speed_scale_percent",
                default_value="400",
                description="Unified motion scale; 100 is the legacy effective-speed baseline and 400 is the current faster default",
            ),
            DeclareLaunchArgument(
                "motion_command_cap_percent",
                default_value="20",
                description="V3 commissioning ceiling for every ordinary robot speed and acceleration command",
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
            DeclareLaunchArgument(
                "scanner_retreat_post_scan_blend_enabled",
                default_value="true",
                description=(
                    "Queue the post-scan safe-height motion before the scanner "
                    "retreat reaches its endpoint"
                ),
            ),
            DeclareLaunchArgument(
                "scanner_retreat_post_scan_blend_cp",
                default_value="20",
                description="Dobot CP blending ratio for scanner retreat to post-scan motion",
            ),
            DeclareLaunchArgument(
                "scanner_retreat_post_scan_queue_lead_m",
                default_value="0.010",
                description=(
                    "Distance before scanner-retreat endpoint at which post-scan "
                    "motion is queued (m)"
                ),
            ),
            DeclareLaunchArgument(
                "scanner_retreat_post_scan_command_start_grace_s",
                default_value="0.30",
                description=(
                    "Grace period for first feedback after scanner-retreat queue "
                    "submission (s)"
                ),
            ),
            DeclareLaunchArgument("offset_grasp_enabled", default_value="true",
                                  description="Reach offset-high while aligning grasp attitude, then descend and insert"),
            DeclareLaunchArgument("offset_grasp_clearance_m", default_value="0.020"),
            DeclareLaunchArgument("offset_finger_span_m", default_value="0.060"),
            DeclareLaunchArgument(
                "offset_high_clearance_m",
                default_value="0.120",
                description=(
                    "Offset-high clearance above grasp depth; lateral motion, "
                    "orientation alignment and partial descent share one PTP"
                ),
            ),
            DeclareLaunchArgument(
                "offset_high_descent_blend_enabled",
                default_value="true",
                description=(
                    "Queue the vertical offset descent before offset-high stops "
                    "so the two approach segments are CP blended"
                ),
            ),
            DeclareLaunchArgument(
                "offset_high_descent_blend_cp",
                default_value="20",
                description="Dobot CP blending ratio for offset-high to descent",
            ),
            DeclareLaunchArgument(
                "offset_high_descent_queue_lead_m",
                default_value="0.030",
                description=(
                    "Distance before offset-high at which the descent is queued (m)"
                ),
            ),
            DeclareLaunchArgument(
                "offset_high_descent_command_start_grace_s",
                default_value="0.30",
                description=(
                    "Grace period for first feedback after offset-high submission (s)"
                ),
            ),
            DeclareLaunchArgument(
                "startup_joint_skip_tolerance_deg",
                default_value="1.0",
                description=(
                    "Skip a redundant startup MovJ when every joint is already "
                    "within this feedback tolerance; zero disables the optimization"
                ),
            ),
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
                default_value="0.20",
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
                "bottom_flip_post_turn_descent_m",
                default_value="0.120",
                description="Descent after the J6 half-turn before second release (m)",
            ),
            DeclareLaunchArgument(
                "bottom_center_tracking_timeout_s",
                default_value="2.0",
                description="Wait for a fresh D405 target before the final regrasp",
            ),
            DeclareLaunchArgument(
                "bottom_center_first_rz_delta_deg",
                default_value="40.0",
                description="Turntable bottom recovery User-Rz rotation before first release",
            ),
            DeclareLaunchArgument(
                "bottom_center_first_tool_rx_delta_deg",
                default_value="70.0",
                description="Tool-Rx rotation after the first turntable release",
            ),
            DeclareLaunchArgument(
                "bottom_center_release_tool_rx_delta_deg",
                default_value="70.0",
                description="Tool-Rx rotation after the second release and J6 half-turn",
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
                "d435_side_face_reference_joint_deg",
                default_value="0.0",
                description="J6 reference for the D435 side-barcode 90-degree face grid",
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
                "grasp_lift_transfer_blend_enabled",
                default_value="false",
                description=(
                    "Legacy non-turntable option; turntable pickup always completes "
                    "a straight safe-height lift before any transfer"
                ),
            ),
            DeclareLaunchArgument(
                "grasp_lift_transfer_blend_cp",
                default_value="20",
                description="Dobot CP blending ratio for the grasp-lift to transfer transition",
            ),
            DeclareLaunchArgument(
                "grasp_lift_transfer_queue_lead_m",
                default_value="0.010",
                description=(
                    "Distance before the lift endpoint at which the transfer "
                    "joint is queued (m)"
                ),
            ),
            DeclareLaunchArgument(
                "grasp_lift_transfer_command_start_grace_s",
                default_value="0.30",
                description=(
                    "Grace period for first feedback after submitting the lift "
                    "before falling back to the blocking transfer path (s)"
                ),
            ),
            DeclareLaunchArgument(
                "post_scan_place_blend_enabled",
                default_value="true",
                description=(
                    "Queue the fixed placement PTP near the end of the post-scan "
                    "safe-height PTP when the side-barcode path is active"
                ),
            ),
            DeclareLaunchArgument(
                "post_scan_place_blend_cp",
                default_value="20",
                description="Dobot CP blending ratio for safe-height to fixed-placement transition",
            ),
            DeclareLaunchArgument(
                "post_scan_place_queue_lead_m",
                default_value="0.020",
                description=(
                    "Distance before the post-scan safe-height endpoint at which "
                    "fixed placement is queued (m)"
                ),
            ),
            DeclareLaunchArgument(
                "post_scan_place_command_start_grace_s",
                default_value="0.30",
                description=(
                    "Grace period for first feedback after submitting the safe-height "
                    "move before falling back to the blocking placement path (s)"
                ),
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
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("vision_python"),
                    "-m",
                    "dobot_nova5_driver.d435_turntable_barcode_node_v3",
                    "--ros-args",
                    "-p",
                    [
                        # ROS CLI parses an all-digit value as an integer.  The
                        # RealSense serial is a string parameter, so retain
                        # literal YAML quotes in the generated argv value.
                        "serial_number:='",
                        LaunchConfiguration("turntable_d435_serial"),
                        "'",
                    ],
                    "-p",
                    [
                        "model_path:=",
                        LaunchConfiguration("turntable_d435_model_path"),
                    ],
                    "-p",
                    [
                        "model_confidence:=",
                        LaunchConfiguration("turntable_d435_confidence"),
                    ],
                    "-p",
                    [
                        "inference_provider:=",
                        LaunchConfiguration("turntable_d435_inference_provider"),
                    ],
                    "-p",
                    [
                        "stable_hits:=",
                        LaunchConfiguration("turntable_d435_yolo_stable_hits"),
                    ],
                    "-p",
                    [
                        "model_image_size:=",
                        LaunchConfiguration("turntable_d435_image_size"),
                    ],
                    "-p",
                    [
                        "scanner_assist_enabled:=",
                        LaunchConfiguration("turntable_d435_scanner_assist"),
                    ],
                    "-p",
                    [
                        "scanner_assist_stable_hits:=",
                        LaunchConfiguration("turntable_d435_scanner_assist_hits"),
                    ],
                    "-p",
                    [
                        "auto_exposure:=",
                        LaunchConfiguration("turntable_d435_auto_exposure"),
                    ],
                    "-p",
                    [
                        "exposure:=",
                        LaunchConfiguration("turntable_d435_exposure"),
                    ],
                    "-p",
                    [
                        "gain:=",
                        LaunchConfiguration("turntable_d435_gain"),
                    ],
                    "-p",
                    ["roi_x:=", LaunchConfiguration("turntable_d435_roi_x")],
                    "-p",
                    ["roi_y:=", LaunchConfiguration("turntable_d435_roi_y")],
                    "-p",
                    [
                        "roi_width:=",
                        LaunchConfiguration("turntable_d435_roi_width"),
                    ],
                    "-p",
                    [
                        "roi_height:=",
                        LaunchConfiguration("turntable_d435_roi_height"),
                    ],
                    "-p",
                    [
                        "preview:=",
                        LaunchConfiguration("turntable_d435_preview"),
                    ],
                ],
                output="screen",
            ),
            Node(
                package="dobot_nova5_driver",
                executable="nova5_cosmetic_box_cycle_v3",
                name="nova5_cosmetic_box_single_arm_cycle_v3",
                output="screen",
                parameters=[
                    {
                        "turntable_do_index": ParameterValue(
                            LaunchConfiguration("turntable_do_index"),
                            value_type=int,
                        ),
                        "turntable_pulse_ms": ParameterValue(
                            LaunchConfiguration("turntable_pulse_ms"),
                            value_type=int,
                        ),
                        "turntable_scan_timeout_s": ParameterValue(
                            LaunchConfiguration("turntable_scan_timeout_s"),
                            value_type=float,
                        ),
                        "turntable_stationary_barcode_check_s": ParameterValue(
                            LaunchConfiguration("turntable_stationary_barcode_check_s"),
                            value_type=float,
                        ),
                        "turntable_settle_s": ParameterValue(
                            LaunchConfiguration("turntable_settle_s"),
                            value_type=float,
                        ),
                        "d435_continuous_on_start": ParameterValue(
                            LaunchConfiguration("turntable_d435_continuous_on_start"),
                            value_type=bool,
                        ),
                        "turntable_assume_stopped_on_start": ParameterValue(
                            LaunchConfiguration("turntable_assume_stopped_on_start"),
                            value_type=bool,
                        ),
                        "turntable_require_place_done": ParameterValue(
                            LaunchConfiguration("turntable_require_place_done"),
                            value_type=bool,
                        ),
                        "turntable_auto_place_from_secondary_tcp": ParameterValue(
                            LaunchConfiguration("turntable_auto_place_from_secondary_tcp"),
                            value_type=bool,
                        ),
                        "turntable_secondary_place_y_m": ParameterValue(
                            LaunchConfiguration("turntable_secondary_place_y_m"),
                            value_type=float,
                        ),
                        "turntable_secondary_safe_z_m": ParameterValue(
                            LaunchConfiguration("turntable_secondary_safe_z_m"),
                            value_type=float,
                        ),
                        "turntable_secondary_safe_z_stable_s": ParameterValue(
                            LaunchConfiguration("turntable_secondary_safe_z_stable_s"),
                            value_type=float,
                        ),
                        "turntable_surface_z_m": ParameterValue(
                            LaunchConfiguration("turntable_surface_z_m"),
                            value_type=float,
                        ),
                        "vision_user_z_bias_m": ParameterValue(
                            LaunchConfiguration("vision_user_z_bias_m"),
                            value_type=float,
                        ),
                        "turntable_surface_tolerance_m": ParameterValue(
                            LaunchConfiguration("turntable_surface_tolerance_m"),
                            value_type=float,
                        ),
                        "turntable_tcp_below_target_m": ParameterValue(
                            LaunchConfiguration("turntable_tcp_below_target_m"),
                            value_type=float,
                        ),
                        "turntable_surface_clearance_m": ParameterValue(
                            LaunchConfiguration("turntable_surface_clearance_m"),
                            value_type=float,
                        ),
                        "grasp_z_offset_m": ParameterValue(
                            LaunchConfiguration("grasp_z_offset_m"),
                            value_type=float,
                        ),
                        "secondary_collision_check_enabled": ParameterValue(
                            LaunchConfiguration("secondary_collision_check_enabled"),
                            value_type=bool,
                        ),
                        "motion_speed_scale_percent": ParameterValue(
                            LaunchConfiguration("motion_speed_scale_percent"),
                            value_type=int,
                        ),
                        "motion_command_cap_percent": ParameterValue(
                            LaunchConfiguration("motion_command_cap_percent"),
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
                        "scanner_retreat_post_scan_blend_enabled": ParameterValue(
                            LaunchConfiguration("scanner_retreat_post_scan_blend_enabled"),
                            value_type=bool,
                        ),
                        "scanner_retreat_post_scan_blend_cp": ParameterValue(
                            LaunchConfiguration("scanner_retreat_post_scan_blend_cp"),
                            value_type=int,
                        ),
                        "scanner_retreat_post_scan_queue_lead_m": ParameterValue(
                            LaunchConfiguration("scanner_retreat_post_scan_queue_lead_m"),
                            value_type=float,
                        ),
                        "scanner_retreat_post_scan_command_start_grace_s": ParameterValue(
                            LaunchConfiguration(
                                "scanner_retreat_post_scan_command_start_grace_s"
                            ),
                            value_type=float,
                        ),
                        "offset_grasp_enabled": ParameterValue(LaunchConfiguration("offset_grasp_enabled"), value_type=bool),
                        "offset_grasp_clearance_m": ParameterValue(LaunchConfiguration("offset_grasp_clearance_m"), value_type=float),
                        "offset_finger_span_m": ParameterValue(LaunchConfiguration("offset_finger_span_m"), value_type=float),
                        "offset_high_clearance_m": ParameterValue(
                            LaunchConfiguration("offset_high_clearance_m"),
                            value_type=float,
                        ),
                        "offset_high_descent_blend_enabled": ParameterValue(
                            LaunchConfiguration("offset_high_descent_blend_enabled"),
                            value_type=bool,
                        ),
                        "offset_high_descent_blend_cp": ParameterValue(
                            LaunchConfiguration("offset_high_descent_blend_cp"),
                            value_type=int,
                        ),
                        "offset_high_descent_queue_lead_m": ParameterValue(
                            LaunchConfiguration("offset_high_descent_queue_lead_m"),
                            value_type=float,
                        ),
                        "offset_high_descent_command_start_grace_s": ParameterValue(
                            LaunchConfiguration(
                                "offset_high_descent_command_start_grace_s"
                            ),
                            value_type=float,
                        ),
                        "startup_joint_skip_tolerance_deg": ParameterValue(
                            LaunchConfiguration("startup_joint_skip_tolerance_deg"),
                            value_type=float,
                        ),
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
                        "bottom_flip_post_turn_descent_m": ParameterValue(
                            LaunchConfiguration("bottom_flip_post_turn_descent_m"),
                            value_type=float,
                        ),
                        "bottom_center_tracking_timeout_s": ParameterValue(
                            LaunchConfiguration("bottom_center_tracking_timeout_s"),
                            value_type=float,
                        ),
                        "bottom_center_first_rz_delta_deg": ParameterValue(
                            LaunchConfiguration("bottom_center_first_rz_delta_deg"),
                            value_type=float,
                        ),
                        "bottom_center_first_tool_rx_delta_deg": ParameterValue(
                            LaunchConfiguration("bottom_center_first_tool_rx_delta_deg"),
                            value_type=float,
                        ),
                        "bottom_center_release_tool_rx_delta_deg": ParameterValue(
                            LaunchConfiguration("bottom_center_release_tool_rx_delta_deg"),
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
                        "d435_side_face_reference_joint_deg": ParameterValue(
                            LaunchConfiguration("d435_side_face_reference_joint_deg"),
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
                        "grasp_lift_transfer_blend_enabled": ParameterValue(
                            LaunchConfiguration("grasp_lift_transfer_blend_enabled"),
                            value_type=bool,
                        ),
                        "grasp_lift_transfer_blend_cp": ParameterValue(
                            LaunchConfiguration("grasp_lift_transfer_blend_cp"),
                            value_type=int,
                        ),
                        "grasp_lift_transfer_queue_lead_m": ParameterValue(
                            LaunchConfiguration("grasp_lift_transfer_queue_lead_m"),
                            value_type=float,
                        ),
                        "grasp_lift_transfer_command_start_grace_s": ParameterValue(
                            LaunchConfiguration("grasp_lift_transfer_command_start_grace_s"),
                            value_type=float,
                        ),
                        "post_scan_place_blend_enabled": ParameterValue(
                            LaunchConfiguration("post_scan_place_blend_enabled"),
                            value_type=bool,
                        ),
                        "post_scan_place_blend_cp": ParameterValue(
                            LaunchConfiguration("post_scan_place_blend_cp"),
                            value_type=int,
                        ),
                        "post_scan_place_queue_lead_m": ParameterValue(
                            LaunchConfiguration("post_scan_place_queue_lead_m"),
                            value_type=float,
                        ),
                        "post_scan_place_command_start_grace_s": ParameterValue(
                            LaunchConfiguration("post_scan_place_command_start_grace_s"),
                            value_type=float,
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
