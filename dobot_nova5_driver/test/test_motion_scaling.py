import threading
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np
import pytest

from dobot_nova5_driver.controller import DobotNova5Controller, TcpPose
from dobot_nova5_driver.dobot_dh_api import DHGripper
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle import (
    CosmeticBoxSingleArmNode,
    CycleTiming,
    PregraspObservation,
    RecoverableGraspError,
    compose_motion_percent,
    grasp_feedback_is_plausible,
    secondary_y_interlock_action,
    secondary_y_retreat_axis,
)
from dobot_nova5_driver.top_surface_geometry import (
    fit_top_surface_axes,
    select_top_plane_candidate,
)


class FakeDashboard:
    def __init__(self):
        self.calls = []

    def _ok(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return "0,{},ok;"

    def SpeedFactor(self, value):
        return self._ok("SpeedFactor", value)

    def AccJ(self, value):
        return self._ok("AccJ", value)

    def VelJ(self, value):
        return self._ok("VelJ", value)

    def AccL(self, value):
        return self._ok("AccL", value)

    def VelL(self, value):
        return self._ok("VelL", value)

    def MovJ(self, *args, **kwargs):
        self.calls.append(("MovJ", args, kwargs))
        return "0,{42},MovJ;"


def test_legacy_default_ratios_are_collapsed_to_baseline_commands():
    assert compose_motion_percent((65, 65, 65), 100) == 27
    assert compose_motion_percent((65, 55), 100) == 36
    assert compose_motion_percent((65, 55, 55), 100) == 20
    assert compose_motion_percent((65, 60, 60), 100) == 23
    assert compose_motion_percent((65, 50, 50), 100) == 16
    assert compose_motion_percent((70,), 100) == 70


def test_speed_scale_increases_the_composed_command_once():
    assert compose_motion_percent((65, 65, 65), 110) == 30
    assert compose_motion_percent((70,), 120) == 84
    assert compose_motion_percent((100,), 200) == 100
    assert compose_motion_percent((65, 65, 65), 400) == 100
    assert compose_motion_percent((65, 60, 60), 400) == 94


def test_grasp_feedback_rejects_gripped_state_without_real_closure():
    assert not grasp_feedback_is_plausible(
        grip_state=2,
        opening_m=0.0891,
        commanded_preshape_m=0.0893,
        minimum_opening_m=0.003,
        minimum_closure_m=0.005,
    )


def test_grasp_feedback_accepts_gripped_state_after_side_closure():
    assert grasp_feedback_is_plausible(
        grip_state=2,
        opening_m=0.0690,
        commanded_preshape_m=0.0893,
        minimum_opening_m=0.003,
        minimum_closure_m=0.005,
    )


@pytest.mark.parametrize("robot_mode", [7, 8, 10])
def test_secondary_interlock_stops_active_left_motion_below_165mm(robot_mode):
    assert secondary_y_interlock_action(0.164, robot_mode, 0.165, 0.145) == "stop"


def test_secondary_interlock_does_not_stop_idle_left_arm_in_warning_band():
    assert secondary_y_interlock_action(0.155, 5, 0.165, 0.145) == "none"


@pytest.mark.parametrize("robot_mode", [5, 7, 8, 10])
def test_secondary_interlock_requests_retreat_below_145mm_in_any_mode(robot_mode):
    assert secondary_y_interlock_action(0.144, robot_mode, 0.165, 0.145) == "retreat"


def test_secondary_interlock_thresholds_are_strictly_less_than():
    assert secondary_y_interlock_action(0.165, 7, 0.165, 0.145) == "none"
    assert secondary_y_interlock_action(0.145, 5, 0.165, 0.145) == "none"


def test_secondary_interlock_rejects_overlapping_thresholds():
    with pytest.raises(ValueError):
        secondary_y_interlock_action(0.140, 7, 0.145, 0.145)


def test_secondary_retreat_axis_moves_101_away_from_102_common_y():
    # Current cell example: 101=-388 mm, 102 common=-548 mm.  Positive User-Y
    # increases the current Y gap.
    assert secondary_y_retreat_axis(-0.388, -0.548) == "Y+"
    assert secondary_y_retreat_axis(-0.600, -0.548) == "Y-"


def _secondary_safety_node(*, cycle_enabled=True, worker_alive=True):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.secondary_safety_lock = threading.Lock()
    node.secondary_protective_stop_latched = threading.Event()
    node.secondary_retreat_active = threading.Event()
    node.secondary_retreat_attempted = threading.Event()
    node.secondary_auto_resume_requested = threading.Event()
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_safety_reason = ""
    node.secondary_resume_thread = None
    node.cycle_enabled = cycle_enabled
    node.worker = SimpleNamespace(is_alive=lambda: worker_alive)
    node.running = True
    node.controller = SimpleNamespace(
        robot_mode=5,
        robot_mode_text=lambda: "enabled and idle",
    )
    node.get_logger = lambda: MagicMock()
    node._publish_status = MagicMock()
    return node


def test_secondary_latch_remembers_interrupted_continuous_cycle():
    node = _secondary_safety_node()

    node._latch_secondary_protective_stop("test trip", robot_mode=5)

    assert not node.cycle_enabled
    assert node.secondary_protective_stop_latched.is_set()
    assert node.secondary_auto_resume_requested.is_set()


def test_secondary_latch_does_not_auto_resume_manual_action():
    node = _secondary_safety_node(worker_alive=False)

    node._latch_secondary_protective_stop("test trip", robot_mode=5)

    assert not node.secondary_auto_resume_requested.is_set()


def test_successful_retreat_restart_starts_fresh_continuous_worker():
    node = _secondary_safety_node(cycle_enabled=False, worker_alive=False)
    node.worker = None
    node.secondary_auto_resume_requested.set()
    node._ensure_worker = MagicMock()

    node._resume_continuous_after_secondary_retreat()

    assert node.cycle_enabled
    assert not node.secondary_auto_resume_requested.is_set()
    node._ensure_worker.assert_called_once_with()


def test_operator_stop_cancels_pending_post_retreat_restart():
    node = _secondary_safety_node(cycle_enabled=False, worker_alive=False)
    node.secondary_auto_resume_requested.set()

    node.stop_continuous_cycle()

    assert not node.cycle_enabled
    assert not node.secondary_auto_resume_requested.is_set()


def test_emergency_retreat_to_200mm_schedules_fresh_continuous_restart():
    node = _secondary_safety_node(cycle_enabled=False, worker_alive=False)
    node.action_lock = threading.RLock()
    node.secondary_protective_stop_latched.set()
    node.secondary_retreat_attempted.set()
    node.secondary_auto_resume_requested.set()
    parameter_values = {
        "secondary_motion_monitor_poll_s": 0.010,
        "secondary_emergency_retreat_timeout_s": 3.0,
        "secondary_emergency_retreat_speed": 100,
        "secondary_emergency_retreat_max_travel_m": 0.200,
        "user_index": 0,
        "command_tool_index": 1,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=parameter_values[name])
    node._secondary_interlock_distances = lambda: (0.165, 0.145, 0.200)
    node._wait_for_action_lock_during_emergency = (
        lambda deadline: node.action_lock.acquire(blocking=False)
    )
    node._read_secondary_y_clearance = MagicMock(
        side_effect=[
            {"gap_y_m": 0.140, "left_y_m": -0.388, "right_common_y_m": -0.528},
            {"gap_y_m": 0.201, "left_y_m": -0.327, "right_common_y_m": -0.528},
            {"gap_y_m": 0.202, "left_y_m": -0.326, "right_common_y_m": -0.528},
        ]
    )
    node.controller = SimpleNamespace(
        robot_mode=5,
        move_jog=MagicMock(),
        wait_until_idle=MagicMock(),
        set_speed_factor=MagicMock(),
    )
    node._schedule_continuous_restart_after_secondary_retreat = MagicMock()

    node._run_secondary_emergency_retreat()

    assert not node.secondary_protective_stop_latched.is_set()
    assert not node.secondary_retreat_attempted.is_set()
    assert node.secondary_auto_resume_requested.is_set()
    assert node.controller.move_jog.call_args_list[0].args == ("Y+",)
    assert node.controller.move_jog.call_args_list[-1].args == ("",)
    node._schedule_continuous_restart_after_secondary_retreat.assert_called_once_with()


def test_gripper_wait_releases_immediately_when_safety_cancels_cycle():
    gripper = DHGripper.__new__(DHGripper)
    gripper.read_grip_state = MagicMock()

    with pytest.raises(RuntimeError, match="cancelled by safety interlock"):
        gripper.wait_until_stopped(cancel_check=lambda: True)

    gripper.read_grip_state.assert_not_called()


def test_top_plane_selection_uses_table_parallel_face_not_largest_side():
    side = np.array([1.0, 0.0, 0.0, -0.1])
    top = np.array([0.0, 0.0, 1.0, -0.4])

    count, selected, source = select_top_plane_candidate(
        [(1200, side), (300, top)],
        np.array([0.0, 0.0, 1.0]),
    )

    assert count == 300
    assert selected is top
    assert source == "table-aligned-object-plane"


def _rectangle_points(length, width, angle_deg=0.0):
    x_values = np.linspace(-length / 2.0, length / 2.0, 18)
    y_values = np.linspace(-width / 2.0, width / 2.0, 14)
    xy = np.array([(x, y) for x in x_values for y in y_values])
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    xy = xy @ rotation.T
    return np.column_stack((xy, np.full(len(xy), 0.4)))


def test_top_surface_axes_follow_metric_top_edges_not_image_silhouette():
    points = _rectangle_points(0.080, 0.060, angle_deg=27.0)

    axes = fit_top_surface_axes(points, np.array([0.0, 0.0, -1.0]), None)

    expected_long = np.array([np.cos(np.deg2rad(27.0)), np.sin(np.deg2rad(27.0)), 0.0])
    assert axes is not None
    assert abs(float(np.dot(axes[:, 0], expected_long))) > 0.99
    assert abs(float(np.dot(axes[:, 2], np.array([0.0, 0.0, -1.0])))) > 0.99


def test_top_surface_first_frame_keeps_legacy_axis_sign_and_avoids_180_turn():
    points = _rectangle_points(0.080, 0.060, angle_deg=27.0)

    axes = fit_top_surface_axes(points, np.array([0.0, 0.0, -1.0]), None)

    expected_legacy_direction = -np.array(
        [np.cos(np.deg2rad(27.0)), np.sin(np.deg2rad(27.0)), 0.0]
    )
    assert axes is not None
    assert axes[1, 0] <= 0.0
    assert float(np.dot(axes[:, 0], expected_legacy_direction)) > 0.99


def test_top_surface_later_frames_keep_first_frame_axis_sign():
    initial = fit_top_surface_axes(
        _rectangle_points(0.080, 0.060, angle_deg=27.0),
        np.array([0.0, 0.0, -1.0]),
        None,
    )

    axes = fit_top_surface_axes(
        _rectangle_points(0.080, 0.060, angle_deg=28.0),
        np.array([0.0, 0.0, -1.0]),
        initial,
    )

    assert initial is not None
    assert axes is not None
    assert float(np.dot(axes[:, 0], initial[:, 0])) > 0.99


def test_near_square_axis_swap_keeps_previous_edge_identity():
    # The new rectangle labels Y as its slightly longer edge.  For a near
    # square target, retain the previous X edge instead of blending a 90-degree
    # label swap into a diagonal orientation.
    points = _rectangle_points(0.068, 0.070)
    reference = np.diag([1.0, -1.0, -1.0])

    axes = fit_top_surface_axes(
        points,
        np.array([0.0, 0.0, -1.0]),
        reference,
    )

    assert axes is not None
    assert abs(float(np.dot(axes[:, 0], reference[:, 0]))) > 0.99


def test_normalized_joint_move_does_not_reapply_global_speed_factor():
    controller = DobotNova5Controller("192.0.2.1")
    dashboard = FakeDashboard()
    controller.dashboard = dashboard
    controller._wait_for_command = lambda *args, **kwargs: None

    controller.enable_single_command_motion_scaling()
    assert [name for name, _, _ in dashboard.calls] == [
        "SpeedFactor",
        "AccJ",
        "VelJ",
        "AccL",
        "VelL",
    ]

    dashboard.calls.clear()
    controller.move_joint([1, 2, 3, 4, 5, 6], speed=27, accel=36)

    assert [name for name, _, _ in dashboard.calls] == ["MovJ"]
    _, _, kwargs = dashboard.calls[0]
    assert kwargs["v"] == 27
    assert kwargs["a"] == 36


def test_feedback_only_connection_does_not_open_dashboard():
    controller = DobotNova5Controller(
        "192.0.2.102",
        dashboard_port=29999,
        feedback_port=30004,
    )
    fake_feedback = MagicMock()

    with (
        patch(
            "dobot_nova5_driver.controller.DobotApiDashboard"
        ) as dashboard_factory,
        patch(
            "dobot_nova5_driver.controller.DobotApiFeedBack",
            return_value=fake_feedback,
        ) as feedback_factory,
        patch.object(controller, "_feedback_loop"),
        patch.object(controller, "_wait_for_feedback"),
        patch.object(controller, "_wait_for_feedback_pose"),
    ):
        controller.connect_feedback_only()

    dashboard_factory.assert_not_called()
    feedback_factory.assert_called_once_with("192.0.2.102", 30004)
    assert controller.dashboard is None
    assert controller.feedback is fake_feedback
    controller.disconnect()


def test_pregrasp_live_position_cannot_replace_reference_orientation():
    reference = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    live = TcpPose(0.41, -0.19, 0.11, 12.0, 78.0, 48.0)

    protected = CosmeticBoxSingleArmNode._keep_reference_orientation(live, reference)

    assert protected.x == live.x
    assert protected.y == live.y
    assert protected.z == live.z
    assert (protected.rx, protected.ry, protected.rz) == (
        reference.rx,
        reference.ry,
        reference.rz,
    )


def test_feedback_packet_timestamp_uses_controller_unix_milliseconds():
    sample_time, source = DobotNova5Controller._feedback_packet_timestamp_s(
        {"TimeStamp": [1788325631000]},
        1788325631.250,
    )

    assert source == "controller_unix_ms"
    assert abs(sample_time - 1788325631.0) < 1e-9


def test_feedback_relative_millisecond_clock_is_mapped_to_host_time():
    controller = DobotNova5Controller("192.0.2.1")
    samples = []
    for index in range(5):
        samples.append(
            controller._feedback_sample_timestamp_s(
                {"TimeStamp": [1000 + index * 8]},
                200.0 + index * 0.008,
            )
        )

    sample_time, source = samples[-1]
    assert source == "controller_relative_ms_mapped"
    assert sample_time == pytest.approx(200.032)


def _pregrasp_estimator(values=None):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    parameter_values = {
        "pregrasp_position_consensus_m": 0.006,
        "pregrasp_position_consensus_samples": 2,
        "pregrasp_unconfirmed_shift_reject_m": 0.020,
        "pregrasp_settled_tcp_speed_mps": 0.015,
        "pregrasp_hover_frame_window_s": 0.25,
        "pregrasp_motion_min_displacement_m": 0.006,
        "pregrasp_motion_max_residual_m": 0.005,
        "pregrasp_prediction_horizon_s": 0.08,
        "pregrasp_position_tolerance_m": 0.007,
        "pregrasp_angle_tolerance_deg": 8.0,
        "pregrasp_max_correction_m": 0.050,
        "pregrasp_max_correction_angle_deg": 30.0,
        "pregrasp_min_hover_clearance_m": 0.030,
        "pregrasp_use_live_pose_for_descent": True,
        "pregrasp_live_orientation_enabled": True,
        "pregrasp_live_orientation_max_delta_deg": 20.0,
        "pregrasp_orientation_consensus_samples": 3,
        "pregrasp_orientation_consensus_spread_deg": 4.0,
        "user_index": 0,
        "command_tool_index": 1,
    }
    if values:
        parameter_values.update(values)
    node.get_parameter = lambda name: SimpleNamespace(value=parameter_values[name])
    return node


def test_pregrasp_static_jitter_keeps_initial_position_and_attitude():
    node = _pregrasp_estimator()
    reference = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    observations = [
        PregraspObservation(1, TcpPose(0.400, -0.200, 0.1, -88.0, 2.0, -42.0), 100.0, 1.0, 0.0),
        PregraspObservation(2, TcpPose(0.409, -0.200, 0.1, 2.0, 92.0, 48.0), 100.2, 1.1, 0.0),
        PregraspObservation(3, TcpPose(0.400, -0.200, 0.1, -88.0, 2.0, -42.0), 100.4, 1.2, 0.0),
        PregraspObservation(4, TcpPose(0.409, -0.200, 0.1, 2.0, 92.0, 48.0), 100.6, 1.3, 0.0),
    ]

    estimate = node._estimate_pregrasp_target(reference, observations, 100.7, 100.7)
    selected = estimate["pose"]

    assert estimate["position_mode"] == "initial-stable-protected"
    assert estimate["orientation_mode"] == "initial-stable-protected"
    assert selected.x == pytest.approx(reference.x)
    assert selected.y == pytest.approx(reference.y)
    assert selected.z == pytest.approx(reference.z)
    assert selected.rx == pytest.approx(reference.rx)
    assert selected.ry == pytest.approx(reference.ry)
    assert selected.rz == pytest.approx(reference.rz)


def test_pregrasp_moving_camera_trend_is_diagnostic_only():
    node = _pregrasp_estimator()
    reference = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    observations = [
        PregraspObservation(1, TcpPose(0.385, -0.200, 0.1, -88.0, 2.0, -42.0), 100.0, 1.0, 0.10),
        PregraspObservation(2, TcpPose(0.391, -0.200, 0.1, -88.0, 2.0, -42.0), 100.2, 1.1, 0.10),
        # The last two frames form a local cluster, but the newest physical
        # measurement is only 4 mm from the correct initial target.  The old
        # velocity extrapolation could turn this into a 9+ mm correction.
        PregraspObservation(3, TcpPose(0.409, -0.200, 0.1, -88.0, 2.0, -42.0), 100.50, 1.2, 0.10),
        PregraspObservation(4, TcpPose(0.404, -0.200, 0.1, -88.0, 2.0, -42.0), 100.65, 1.3, 0.10),
    ]

    estimate = node._estimate_pregrasp_target(reference, observations, 100.7, 100.7)
    selected = estimate["pose"]

    assert estimate["motion_displacement_m"] > 0.0
    assert estimate["near_hover_count"] == 2
    assert estimate["position_mode"] == "initial-stable-protected"
    assert estimate["velocity_xy_mps"].tolist() == [0.0, 0.0]
    assert selected.x == pytest.approx(reference.x)
    assert selected.y == pytest.approx(reference.y)


def test_pregrasp_real_shift_uses_latest_settled_measured_consensus():
    node = _pregrasp_estimator()
    reference = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    observations = [
        PregraspObservation(1, reference, 100.0, 1.0, 0.10),
        PregraspObservation(2, TcpPose(0.416, -0.200, 0.1, -88.0, 2.0, -42.0), 100.50, 1.1, 0.010),
        PregraspObservation(3, TcpPose(0.417, -0.201, 0.1, -88.0, 2.0, -42.0), 100.65, 1.2, 0.005),
    ]

    estimate = node._estimate_pregrasp_target(reference, observations, 100.7, 100.7)
    selected = estimate["pose"]

    assert estimate["position_mode"] == "settled-hover-measured-consensus"
    assert estimate["consensus_count"] == 2
    assert selected.x == pytest.approx(0.4165)
    assert selected.y == pytest.approx(-0.2005)
    assert estimate["velocity_xy_mps"].tolist() == [0.0, 0.0]


def test_pregrasp_translation_does_not_follow_a_ninety_degree_obb_flip():
    node = _pregrasp_estimator()
    reference = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    observations = [
        PregraspObservation(1, TcpPose(0.416, -0.200, 0.1, 2.0, 92.0, 48.0), 100.48, 1.0, 0.010),
        PregraspObservation(2, TcpPose(0.417, -0.200, 0.1, 2.0, 92.0, 48.0), 100.56, 1.1, 0.008),
        PregraspObservation(3, TcpPose(0.418, -0.200, 0.1, 2.0, 92.0, 48.0), 100.64, 1.2, 0.005),
    ]

    estimate = node._estimate_pregrasp_target(reference, observations, 100.7, 100.7)

    assert estimate["position_mode"] == "settled-hover-measured-consensus"
    assert estimate["orientation_mode"] == "initial-stable-protected"


def _mock_pregrasp_revalidation(node, target, selected):
    observation = PregraspObservation(1, selected, 100.6, 1.0, 0.0)
    node._fresh_pregrasp_observations = lambda previous_count: ([observation], 0.01)
    position_delta, angle_delta = node._pose_delta(target, selected)
    node._estimate_pregrasp_target = lambda *args: {
        "pose": selected,
        "position_mode": "settled-hover-measured-consensus",
        "orientation_mode": "initial-stable-protected",
        "position_delta_m": position_delta,
        "angle_delta_deg": angle_delta,
        "velocity_xy_mps": [0.2, 0.0],
        "motion_displacement_m": 0.020,
        "observation_count": 3,
        "near_hover_count": 2,
        "consensus_count": 2,
    }
    node._current_command_pose = lambda: TcpPose(
        target.x, target.y, target.z + 0.10, target.rx, target.ry, target.rz
    )
    node._require_cycle_active = lambda detail: None
    node._publish_status = lambda detail: None
    node.get_logger = lambda: MagicMock()


def test_pregrasp_small_delta_never_becomes_diagonal_descent():
    node = _pregrasp_estimator()
    target = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    selected = TcpPose(0.405, -0.2, 0.1, -88.0, 2.0, -42.0)
    _mock_pregrasp_revalidation(node, target, selected)

    result = node._revalidate_target_at_hover(
        target,
        {"joint_speed": 100, "joint_pose_acc": 100},
        0.0,
        0.01,
        0,
        100.7,
    )

    assert result == target


def test_pregrasp_hover_correction_does_not_extrapolate_after_motion():
    node = _pregrasp_estimator()
    target = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    selected = TcpPose(0.415, -0.2, 0.1, -88.0, 2.0, -42.0)
    _mock_pregrasp_revalidation(node, target, selected)
    node.controller = SimpleNamespace(
        inverse_kinematics=MagicMock(return_value=None),
        current_joint=MagicMock(return_value=[0.0] * 6),
        move_joint_tcp=MagicMock(return_value=None),
    )

    result = node._revalidate_target_at_hover(
        target,
        {"joint_speed": 100, "joint_pose_acc": 100},
        0.0,
        0.01,
        0,
        100.7,
    )

    assert result == selected
    commanded_hover = node.controller.move_joint_tcp.call_args.args[0]
    assert commanded_hover.x == pytest.approx(selected.x)
    assert commanded_hover.y == pytest.approx(selected.y)


def test_pregrasp_large_single_frame_jump_refuses_descent():
    node = _pregrasp_estimator()
    target = TcpPose(0.4, -0.2, 0.1, -88.0, 2.0, -42.0)
    jumped = TcpPose(0.447, -0.2, 0.1, -88.0, 2.0, -42.0)
    observation = PregraspObservation(1, jumped, 100.6, 1.0, 0.0)
    node._fresh_pregrasp_observations = lambda previous_count: ([observation], 0.01)
    node._estimate_pregrasp_target = lambda *args: {
        "pose": target,
        "position_mode": "initial-stable-protected",
        "orientation_mode": "initial-stable-protected",
        "position_delta_m": 0.0,
        "angle_delta_deg": 0.0,
        "velocity_xy_mps": [0.0, 0.0],
        "motion_displacement_m": 0.0,
        "observation_count": 1,
        "near_hover_count": 0,
        "consensus_count": 0,
    }
    node._current_command_pose = lambda: TcpPose(
        target.x, target.y, target.z + 0.10, target.rx, target.ry, target.rz
    )
    node.get_logger = lambda: MagicMock()

    with pytest.raises(RecoverableGraspError, match="refusing descent"):
        node._revalidate_target_at_hover(
            target,
            {"joint_speed": 100, "joint_pose_acc": 100},
            0.0,
            0.01,
            0,
            100.7,
        )


def test_cycle_timing_summary_contains_stage_and_unaccounted_overhead():
    timing = CycleTiming("continuous-1", started_at=10.0)
    timing.add_stage("vision_detection", 0.6, "ok")
    timing.add_stage("move_above", 1.4, "ok")

    with patch(
        "dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle.time.monotonic",
        return_value=12.5,
    ):
        summary = timing.summary("success")

    assert summary["cycle_id"] == "continuous-1"
    assert summary["outcome"] == "success"
    assert summary["total_s"] == 2.5
    assert summary["accounted_s"] == 2.0
    assert summary["overhead_s"] == 0.5
    assert [stage["stage"] for stage in summary["stages"]] == [
        "vision_detection",
        "move_above",
    ]
