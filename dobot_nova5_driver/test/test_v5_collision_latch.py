import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v5 as cycle_v5
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v5 import (
    CosmeticBoxSingleArmNode,
    EmptyTurntableConfirmed,
)
from dobot_nova5_driver.turntable_v3 import PlacementRetreatTrigger


def stopped_node():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.secondary_safety_lock = threading.Lock()
    node.secondary_protective_stop_latched = threading.Event()
    node.secondary_retreat_attempted = threading.Event()
    node.secondary_retreat_active = threading.Event()
    node.secondary_safe_startup_return_active = threading.Event()
    node.secondary_auto_resume_requested = threading.Event()
    node.turntable_scan_cancel = threading.Event()
    node.secondary_recovery_clear_samples = 0
    node.secondary_recovery_clear_since = None
    node.secondary_recovery_last_feedback_at = 0.0
    node.secondary_recovery_last_observed_at = 0.0
    node.secondary_safety_reason = ""
    node.secondary_zone_open_for_102 = False
    node.turntable_waiting_for_place = True
    node.secondary_zone_held_by_102 = False
    node.secondary_pick_guard_active = False
    node.secondary_pick_motion_active = False
    node.secondary_post_place_fetch = False
    node.secondary_fetch_phase_publisher = MagicMock()
    node.secondary_pick_guard_publisher = MagicMock()
    node.secondary_guard_xyz_publisher = MagicMock()
    node.secondary_placement_ready_publisher = MagicMock()
    node.secondary_guard_sequence = 1
    node.secondary_guard_ack_sequence = 1
    node.secondary_material_removed_for_handoff = False
    node.secondary_departure_grant_consumed = False
    node.secondary_last_measurement = {
        "right_y_m": 0.100,
        "feedback_received_at": 1.0,
    }
    node.cycle_enabled = True
    node.worker = SimpleNamespace(is_alive=lambda: True)
    node.controller = SimpleNamespace(
        robot_mode=7,
        robot_mode_text=lambda: "RUNNING",
        stop_motion=MagicMock(),
    )
    node._publish_status = MagicMock()
    node._schedule_continuous_restart_after_secondary_clearance = MagicMock()
    node._secondary_interlock_distances = lambda: (0.160, 0.142, 0.180)
    node._secondary_vla_heartbeat_fresh = lambda: True
    node.get_parameter = lambda name: SimpleNamespace(
        value={
            "secondary_zone_safe_y_m": 0.400,
            "secondary_pick_guard_y_m": 0.325,
            "secondary_collision_x_lt_m": 0.535,
            "secondary_collision_z_gt_m": 0.075,
            "secondary_recovery_stable_s": 1.0,
            "secondary_tcp_max_age_s": 0.05,
        }.get(name, 0.180)
    )
    node.get_logger = lambda: MagicMock()
    return node


def empty_event_node():
    node = stopped_node()
    node.running = True
    node.controller.robot_mode = 5
    node.controller.current_joint = lambda: [25.0, 13.0, -110.0, 15.0, 80.0, 20.0]
    node.turntable_lock = threading.RLock()
    node.turntable_condition = threading.Condition(node.turntable_lock)
    node.turntable_place_done_count = 3
    node.turntable_active_pick_event_number = 3
    node.turntable_material_ready = True
    node.turntable_scan_in_progress = False
    node.turntable_state = "STOPPED"
    node.turntable_ready_barcode = "barcode_detected"
    node.turntable_ready_top_barcode = ""
    node.turntable_scan_error = ""
    node.turntable_secondary_retreat_trigger = MagicMock()
    node.data_lock = threading.RLock()
    node.pose_count = 0
    node.pregrasp_pose_count = 0
    node.pregrasp_observations = deque(maxlen=40)
    node.last_turntable_stop_monotonic = 0.0
    node.last_turntable_stop_epoch = 0.0
    node.last_turntable_stop_pregrasp_count = 0
    node.empty_turntable_confirmation = (3, 0, 0)
    node.empty_turntable_target_invalidated = False
    node.vision_result_count = 0
    node.latest_vision_result = ""
    node.trigger_publisher = MagicMock()
    node._read_secondary_y_clearance = lambda: {
        "right_y_m": 0.344,
        "gap_y_m": 0.300,
    }
    node._set_top_surface_barcode_window = MagicMock()
    node._six_values = lambda _name: [25.0, 13.0, -110.0, 15.0, 80.0, 20.0]
    node.get_parameter = lambda name: SimpleNamespace(
        value={
            "secondary_zone_safe_y_m": 0.400,
            "secondary_pick_guard_y_m": 0.325,
            "secondary_collision_x_lt_m": 0.535,
            "secondary_collision_z_gt_m": 0.075,
            "startup_joint_skip_tolerance_deg": 1.0,
            "empty_turntable_confirm_samples": 2,
            "empty_turntable_min_observation_s": 0.2,
            "empty_turntable_result_timeout_s": 0.2,
            "empty_turntable_guard_probe_interval_s": 0.2,
            "secondary_vla_interlock_required": True,
        }.get(name, 0.180)
    )
    return node


def test_empty_event_requires_two_explicit_full_frame_no_box_results():
    node = empty_event_node()
    results = iter(["failure:no_cosmetic_box", "failure:no_cosmetic_box"])

    def publish(_message):
        node.latest_vision_result = next(results)
        node.vision_result_count += 1

    node.trigger_publisher.publish.side_effect = publish
    started = time.monotonic()
    assert node._confirm_empty_turntable(3)
    assert node.trigger_publisher.publish.call_count == 2
    assert time.monotonic() - started >= 0.19


def test_empty_event_rejects_late_pregrasp_target_after_first_no_box():
    node = empty_event_node()

    def publish(_message):
        node.latest_vision_result = "failure:no_cosmetic_box"
        node.vision_result_count += 1
        node.pregrasp_pose_count += 1

    node.trigger_publisher.publish.side_effect = publish
    assert not node._confirm_empty_turntable(3)
    assert node.trigger_publisher.publish.call_count == 1
    assert node.turntable_material_ready


def test_empty_event_rejects_post_stop_target_already_seen():
    node = empty_event_node()
    node.last_turntable_stop_monotonic = time.monotonic() - 1.0
    node.last_turntable_stop_epoch = time.time() - 1.0
    node.pregrasp_observations.append(SimpleNamespace(
        count=1,
        received_at_monotonic=time.monotonic(),
        frame_time_s=time.time(),
    ))
    assert not node._confirm_empty_turntable(3)
    node.trigger_publisher.publish.assert_not_called()
    node._publish_status.assert_called_with(
        "empty-turntable confirmation blocked for material event #3: "
        "D405 already published a valid post-stop target; retaining the material event"
    )


def test_safety_restart_keeps_gripped_box_and_does_not_open_gripper():
    node = stopped_node()
    node.running = True
    node.cycle_enabled = False
    node.worker = None
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_resume_thread = threading.current_thread()
    node.secondary_auto_resume_requested.set()
    node.controller.robot_mode = 5
    node.gripper = SimpleNamespace(read_grip_state=lambda: cycle_v5.GRIP_GRIPPED)
    node._ensure_worker = MagicMock()

    node._resume_continuous_after_secondary_clearance()

    assert not node.secondary_auto_resume_requested.is_set()
    assert not node.cycle_enabled
    node._ensure_worker.assert_not_called()
    assert "gripper is object gripped" in node._publish_status.call_args.args[0]


def test_safety_restart_requires_verified_gripper_feedback():
    node = stopped_node()
    node.running = True
    node.cycle_enabled = False
    node.worker = None
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_resume_thread = threading.current_thread()
    node.secondary_auto_resume_requested.set()
    node.controller.robot_mode = 5
    node.gripper = SimpleNamespace(read_grip_state=MagicMock(side_effect=RuntimeError("offline")))
    node._ensure_worker = MagicMock()

    node._resume_continuous_after_secondary_clearance()

    assert not node.secondary_auto_resume_requested.is_set()
    assert not node.cycle_enabled
    node._ensure_worker.assert_not_called()
    assert "gripper feedback unavailable" in node._publish_status.call_args.args[0]


def test_safety_restart_continues_when_gripper_is_verified_empty():
    node = stopped_node()
    node.running = True
    node.cycle_enabled = False
    node.worker = None
    node.secondary_safety_shutdown = threading.Event()
    node.secondary_resume_thread = threading.current_thread()
    node.secondary_auto_resume_requested.set()
    node.controller.robot_mode = 5
    node.gripper = SimpleNamespace(read_grip_state=lambda: cycle_v5.GRIP_REACHED)
    node._ensure_worker = MagicMock()

    node._resume_continuous_after_secondary_clearance()

    assert not node.secondary_auto_resume_requested.is_set()
    assert node.cycle_enabled
    node._ensure_worker.assert_called_once_with()


@pytest.mark.parametrize(
    "second_result",
    ["failure:no_yolo_candidate_inside_roi", "failure:no_valid_height", "success:stable_target_published"],
)
def test_empty_event_rejects_ambiguous_or_positive_vision(second_result):
    node = empty_event_node()
    results = iter(["failure:no_cosmetic_box", second_result])

    def publish(_message):
        node.latest_vision_result = next(results)
        node.vision_result_count += 1

    node.trigger_publisher.publish.side_effect = publish
    assert not node._confirm_empty_turntable(3)
    assert node.turntable_material_ready


def test_empty_event_rejects_no_box_when_102_occludes_view():
    node = empty_event_node()
    node.secondary_zone_held_by_102 = True
    assert not node._confirm_empty_turntable(3)
    node.trigger_publisher.publish.assert_not_called()


@pytest.mark.parametrize(
    ("right_y_m", "gap_y_m"),
    [(0.400, 0.300), (0.401, 0.300), (0.344, 0.170)],
)
def test_empty_event_requires_102_outside_place_zone_and_safe_gap(
    right_y_m, gap_y_m
):
    node = empty_event_node()
    node._read_secondary_y_clearance = lambda: {
        "right_y_m": right_y_m,
        "gap_y_m": gap_y_m,
    }
    assert not node._confirm_empty_turntable(3)
    node.trigger_publisher.publish.assert_not_called()


def test_empty_event_reopens_102_only_after_verified_101_startup():
    node = empty_event_node()
    node.secondary_pick_guard_active = True
    assert node._release_confirmed_empty_turntable_event(3)
    assert node.turntable_waiting_for_place
    assert not node.turntable_material_ready
    assert node.turntable_active_pick_event_number is None
    assert not node.secondary_pick_guard_active
    assert node.secondary_zone_open_for_102
    node.turntable_secondary_retreat_trigger.reset.assert_called_once_with(
        require_fresh_entry=True
    )


def test_positive_target_between_confirmation_and_release_keeps_102_closed():
    node = empty_event_node()
    node.pregrasp_pose_count += 1
    assert not node._release_confirmed_empty_turntable_event(3)
    assert node.empty_turntable_target_invalidated
    assert node.turntable_material_ready
    assert not node.secondary_zone_open_for_102


def test_102_at_344mm_gets_lease_after_empty_event_is_confirmed():
    node = empty_event_node()
    node.secondary_pick_guard_active = True

    def publish(_message):
        node.latest_vision_result = "failure:no_cosmetic_box"
        node.vision_result_count += 1

    node.trigger_publisher.publish.side_effect = publish
    assert node._confirm_empty_turntable(3)
    assert node._release_confirmed_empty_turntable_event(3)
    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message="")
    )
    assert response.success
    assert node.secondary_zone_held_by_102


def test_empty_event_keeps_102_closed_if_101_pose_is_not_startup():
    node = empty_event_node()
    node.controller.current_joint = lambda: [45.0, 13.0, -110.0, 15.0, 80.0, 20.0]
    node.secondary_pick_guard_active = True
    assert not node._release_confirmed_empty_turntable_event(3)
    assert node.turntable_material_ready
    assert node.secondary_pick_guard_active
    assert not node.secondary_zone_open_for_102


def test_old_empty_confirmation_cannot_discard_a_new_placement():
    node = empty_event_node()
    node.turntable_place_done_count = 4
    assert not node._release_confirmed_empty_turntable_event(3)
    assert node.turntable_material_ready
    assert not node.secondary_zone_open_for_102


def test_guard_wait_checks_empty_table_before_101_moving():
    node = empty_event_node()
    node.secondary_guard_ack_sequence = -1
    node._read_secondary_y_clearance = lambda: {
        "right_y_m": 0.344,
        "gap_y_m": 0.300,
    }
    node._confirm_empty_turntable = MagicMock(return_value=True)

    with pytest.raises(EmptyTurntableConfirmed):
        node._close_secondary_zone_for_pick(3)

    node._confirm_empty_turntable.assert_called_once_with(3)
    assert node.secondary_pick_guard_active
    assert not node.secondary_zone_open_for_102
    assert node._release_confirmed_empty_turntable_event(3)
    assert node.secondary_zone_open_for_102


def test_102_cannot_acquire_stale_open_lease_after_new_material_event():
    node = empty_event_node()
    node.turntable_waiting_for_place = False
    node.secondary_zone_open_for_102 = True
    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message="")
    )
    assert not response.success
    assert not node.secondary_zone_held_by_102


def test_near_miss_stops_once_and_resumes_only_after_stable_fresh_retreat(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(cycle_v5.time, "monotonic", lambda: now[0])
    node = stopped_node()
    node._latch_secondary_protective_stop("102 approaching", robot_mode=7)
    node._latch_secondary_protective_stop("102 approaching", robot_mode=7)

    assert node.secondary_protective_stop_latched.is_set()
    assert not node.cycle_enabled
    assert node.secondary_auto_resume_requested.is_set()
    node.controller.stop_motion.assert_called_once_with()
    node.secondary_zone_held_by_102 = True
    assert not node._recover_secondary_protective_stop_if_clear(1.0, 5)
    assert node.secondary_protective_stop_latched.is_set()
    node.secondary_zone_held_by_102 = False
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    # Polling the same packet cannot count as an independent fresh sample.
    now[0] = 10.02
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    assert node.secondary_recovery_clear_samples == 1
    recovered = False
    for index in range(2, 28):
        now[0] = 10.0 + (index - 1) * 0.04
        node.secondary_last_measurement["feedback_received_at"] = float(index)
        recovered = node._recover_secondary_protective_stop_if_clear(0.200, 5) or recovered
    assert recovered
    node._schedule_continuous_restart_after_secondary_clearance.assert_called_once()


def test_recovery_timer_restarts_after_an_unsafe_sample(monkeypatch):
    now = [20.0]
    monkeypatch.setattr(cycle_v5.time, "monotonic", lambda: now[0])
    node = stopped_node()
    node._latch_secondary_protective_stop("102 feedback stale", robot_mode=7)
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    now[0] = 20.04
    node.secondary_last_measurement["right_y_m"] = 0.405
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    assert node.secondary_recovery_clear_since is None
    node.secondary_last_measurement["right_y_m"] = 0.100
    recovered = False
    for index in range(2, 28):
        node.secondary_last_measurement["feedback_received_at"] = float(index)
        now[0] = 20.08 + (index - 2) * 0.04
        recovered = node._recover_secondary_protective_stop_if_clear(0.200, 5) or recovered
    assert recovered


def test_recovery_timer_restarts_after_monitor_gap(monkeypatch):
    now = [30.0]
    monkeypatch.setattr(cycle_v5.time, "monotonic", lambda: now[0])
    node = stopped_node()
    node._latch_secondary_protective_stop("102 feedback stale", robot_mode=7)
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    now[0] = 30.20
    node.secondary_last_measurement["feedback_received_at"] = 2.0
    assert not node._recover_secondary_protective_stop_if_clear(0.200, 5)
    assert node.secondary_recovery_clear_samples == 1
    assert node.secondary_recovery_clear_since == 30.20


def test_collision_mode_keeps_latch_even_after_102_retreat():
    node = stopped_node()
    node._latch_secondary_protective_stop("collision", robot_mode=7)
    assert not node._recover_secondary_protective_stop_if_clear(0.300, 9)
    assert node.secondary_protective_stop_latched.is_set()


def test_safe_startup_return_ignores_only_heartbeat_transient():
    node = stopped_node()
    node._secondary_vla_heartbeat_fresh = lambda: False
    node.secondary_zone_held_by_102 = True
    node.secondary_pick_motion_active = True

    assert node._secondary_heartbeat_stop_required(7)
    assert node._secondary_lease_overlap_stop_required()
    with node._safe_startup_return():
        assert node.secondary_safe_startup_return_active.is_set()
        assert not node._secondary_heartbeat_stop_required(7)
        assert not node._secondary_lease_overlap_stop_required()

    assert not node.secondary_safe_startup_return_active.is_set()
    assert node._secondary_heartbeat_stop_required(7)
    assert node._secondary_lease_overlap_stop_required()


def test_safe_startup_return_marker_clears_after_motion_error():
    node = stopped_node()

    with pytest.raises(RuntimeError, match="motion failed"):
        with node._safe_startup_return():
            raise RuntimeError("motion failed")

    assert not node.secondary_safe_startup_return_active.is_set()


def test_pick_guard_violation_stops_motion_but_not_safe_startup_return():
    node = stopped_node()
    node.secondary_pick_guard_active = True

    assert not node._secondary_pick_guard_stop_required(0.325, 7, 0.100, 0.500)
    assert node._secondary_pick_guard_stop_required(0.326, 7, 0.100, 0.500)
    assert not node._secondary_pick_guard_stop_required(0.390, 7, 0.075, 0.500)
    assert not node._secondary_pick_guard_stop_required(0.390, 7, 0.100, 0.535)
    assert not node._secondary_pick_guard_stop_required(0.390, 5, 0.100, 0.500)
    with node._safe_startup_return():
        assert not node._secondary_pick_guard_stop_required(0.390, 7, 0.100, 0.500)


def test_collision_volume_requires_all_three_conditions():
    node = stopped_node()
    node.secondary_pick_guard_active = True
    node.secondary_pick_motion_active = True
    node.secondary_post_place_fetch = True

    assert not node._secondary_pick_corridor_clear(0.390, 0.100, 0.500)
    assert node._secondary_pick_corridor_clear(0.320, 0.100, 0.500)
    assert node._secondary_pick_corridor_clear(0.390, 0.070, 0.500)
    assert node._secondary_pick_corridor_clear(0.390, 0.100, 0.550)
    assert node._secondary_pick_guard_stop_required(0.390, 7, 0.100, 0.500)
    assert node._secondary_pick_guard_stop_required(0.390, 7, float("nan"), 0.500)


def test_y_gap_stop_ends_after_confirmed_turntable_departure():
    node = stopped_node()
    node.secondary_pick_guard_active = True
    node.secondary_pick_motion_active = True
    assert node._secondary_protective_gap_for_phase(0.155, 0.400, 0.100, 0.500) == pytest.approx(0.155)

    node.secondary_pick_guard_active = False
    node.secondary_pick_motion_active = False
    node.secondary_material_removed_for_handoff = True
    assert node._secondary_protective_gap_for_phase(0.155, 0.400, 0.100, 0.500) == float("inf")

    node.secondary_material_removed_for_handoff = False
    assert node._secondary_protective_gap_for_phase(0.155, 0.400, 0.100, 0.500) == pytest.approx(0.155)


def test_101_waits_for_current_guard_ack_even_with_clear_102_pose():
    node = stopped_node()
    node.running = True
    node.secondary_guard_ack_sequence = -1
    node._read_secondary_y_clearance = lambda: {
        "right_y_m": 0.200, "right_z_m": 0.100, "gap_y_m": 0.300
    }
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (node._close_secondary_zone_for_pick(), finished.set()))
    thread.start()

    assert not finished.wait(0.05)
    node.secondary_guard_ack_sequence = node.secondary_guard_sequence
    thread.join(timeout=1.0)
    assert finished.is_set()


def test_102_lease_granted_only_while_101_waits_at_startup():
    node = stopped_node()
    request = SimpleNamespace(data=True)
    response = SimpleNamespace(success=False, message="")
    assert not node._secondary_zone_lease_callback(request, response).success
    node.controller.robot_mode = 5
    node.secondary_zone_open_for_102 = True
    assert node._secondary_zone_lease_callback(request, response).success
    assert node.secondary_zone_held_by_102
    assert not node.secondary_zone_open_for_102
    assert node._secondary_zone_lease_callback(request, response).success

    node._read_secondary_y_clearance = lambda **_kwargs: {
        "right_y_m": 0.401, "gap_y_m": 0.200
    }
    assert not node._secondary_zone_lease_callback(
        SimpleNamespace(data=False), response
    ).success
    assert node.secondary_zone_held_by_102

    node._read_secondary_y_clearance = lambda **_kwargs: {
        "right_y_m": 0.398, "gap_y_m": 0.200
    }
    assert node._secondary_zone_lease_callback(
        SimpleNamespace(data=False), response
    ).success
    assert not node.secondary_zone_held_by_102


def test_102_can_enter_after_box_departure_without_101_y_threshold():
    node = stopped_node()
    node.controller.robot_mode = 7  # 101 is still transferring the old box
    node.secondary_material_removed_for_handoff = True
    node.secondary_pick_guard_active = True
    node._open_secondary_zone_after_departure()
    assert not node.secondary_zone_open_for_102

    node.secondary_pick_guard_active = False
    node.secondary_last_measurement = {"left_y_m": -0.170}
    node._open_secondary_zone_after_departure()
    assert node.secondary_zone_open_for_102
    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message="")
    )
    assert response.success
    assert node.secondary_zone_held_by_102
    assert node.secondary_departure_grant_consumed

    node.secondary_pick_guard_active = True
    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message="")
    )
    assert not response.success


def test_departure_immediately_advertises_the_placement_window():
    node = empty_event_node()
    node.controller.robot_mode = 7
    node.secondary_pick_guard_active = True
    node._mark_turntable_material_removed()
    assert not node.secondary_pick_guard_active
    assert node.secondary_zone_open_for_102
    assert node.turntable_waiting_for_place
    assert node.secondary_placement_ready_publisher.publish.call_args.args[0].data is True

    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message=""),
    )
    assert response.success
    assert node.secondary_placement_ready_publisher.publish.call_args.args[0].data is False


@pytest.mark.parametrize("condition", ["grasping", "occupied", "disabled", "stopped", "consumed"])
def test_placement_window_advertisement_remains_closed_until_safe(condition):
    node = stopped_node()
    node.controller.robot_mode = 5
    node.secondary_zone_open_for_102 = True
    if condition == "grasping":
        node.secondary_pick_guard_active = True
    elif condition == "occupied":
        node.turntable_waiting_for_place = False
    elif condition == "disabled":
        node.cycle_enabled = False
    elif condition == "stopped":
        node.secondary_protective_stop_latched.set()
    else:
        node.secondary_material_removed_for_handoff = True
        node.secondary_departure_grant_consumed = True
    node._publish_secondary_placement_ready()
    assert node.secondary_placement_ready_publisher.publish.call_args.args[0].data is False


def test_102_cannot_enter_during_101_down_grasp_even_if_vla_sees_empty_table():
    node = stopped_node()
    node.secondary_last_measurement = {"left_y_m": -0.170}
    node._open_secondary_zone_after_departure()
    assert not node.secondary_zone_open_for_102

    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message="")
    )
    assert not response.success
    assert not node.secondary_zone_held_by_102


def test_102_cannot_acquire_lease_while_pick_guard_active_even_if_zone_was_open():
    node = stopped_node()
    node.controller.robot_mode = 5
    node.secondary_zone_open_for_102 = True
    node.secondary_pick_guard_active = True

    response = node._secondary_zone_lease_callback(
        SimpleNamespace(data=True), SimpleNamespace(success=False, message="")
    )
    assert not response.success
    assert not node.secondary_zone_held_by_102


def test_102_departure_grant_is_single_use_until_next_pick():
    node = stopped_node()
    node.secondary_material_removed_for_handoff = True
    node._open_secondary_zone_after_departure()
    assert node.secondary_zone_open_for_102
    acquire = SimpleNamespace(data=True)
    release = SimpleNamespace(data=False)
    response = SimpleNamespace(success=False, message="")
    assert node._secondary_zone_lease_callback(acquire, response).success
    assert node.secondary_departure_grant_consumed

    node._read_secondary_y_clearance = lambda **_kwargs: {
        "right_y_m": 0.398, "gap_y_m": 0.200
    }
    assert node._secondary_zone_lease_callback(release, response).success
    node._open_secondary_zone_after_departure()
    assert not node.secondary_zone_open_for_102
    assert not node._secondary_zone_lease_callback(acquire, response).success


def test_101_pick_waits_for_102_lease_release_and_collision_volume_clear():
    node = stopped_node()
    node.running = True
    node.secondary_zone_open_for_102 = True
    node.secondary_zone_held_by_102 = True
    node.secondary_material_removed_for_handoff = True
    measurement = {"right_y_m": 0.398, "right_x_m": 0.500, "right_z_m": 0.100, "gap_y_m": 0.200}
    node._read_secondary_y_clearance = lambda: measurement
    finished = threading.Event()

    def wait_for_release():
        node._close_secondary_zone_for_pick()
        finished.set()

    thread = threading.Thread(target=wait_for_release)
    thread.start()
    assert not finished.wait(0.05)
    node.secondary_guard_ack_sequence = node.secondary_guard_sequence
    assert not node.secondary_zone_open_for_102
    assert not node.secondary_material_removed_for_handoff
    node.secondary_zone_held_by_102 = False
    assert not finished.wait(0.05)
    measurement["right_y_m"] = 0.320
    thread.join(timeout=1.0)
    assert finished.is_set()
    assert node.secondary_pick_guard_active
    published = node.secondary_pick_guard_publisher.publish.call_args.args[0]
    assert published.data == pytest.approx(0.325)


def test_pick_guard_release_restores_400mm_without_resetting_vla():
    node = stopped_node()
    node.secondary_pick_guard_active = True

    node._set_secondary_pick_guard(False)

    assert not node.secondary_pick_guard_active
    published = node.secondary_pick_guard_publisher.publish.call_args.args[0]
    assert published.data == pytest.approx(0.400)
    node._publish_status.assert_called_once()


def test_102_phase_changes_only_after_measured_place_and_retreat():
    node = stopped_node()
    node.turntable_lock = threading.Lock()
    node.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(0.400, 0.0)
    node._accept_turntable_place_done = MagicMock()
    node.get_parameter = lambda _name: SimpleNamespace(value=True)

    node._update_turntable_place_from_secondary_tcp({"right_y_m": 0.399})
    assert not node.secondary_post_place_fetch
    node._update_turntable_place_from_secondary_tcp({"right_y_m": 0.401})
    assert not node.secondary_post_place_fetch
    node._update_turntable_place_from_secondary_tcp({"right_y_m": 0.399})
    assert node.secondary_post_place_fetch
    node._accept_turntable_place_done.assert_called_once()
    node._turntable_place_done_callback(SimpleNamespace(data=True))
    assert not node.secondary_post_place_fetch
    node._update_turntable_place_from_secondary_tcp({"right_y_m": 0.401})
    assert not node.secondary_post_place_fetch


def test_safe_x_fetch_crossing_is_not_a_new_placement():
    node = stopped_node()
    node.turntable_lock = threading.Lock()
    node.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(0.400, 0.0)
    node._accept_turntable_place_done = MagicMock()
    node.secondary_post_place_fetch = True
    values = {"turntable_auto_place_from_secondary_tcp": True,
              "secondary_collision_x_lt_m": 0.535}
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])

    node._update_turntable_place_from_secondary_tcp(
        {"right_x_m": 0.550, "right_y_m": 0.420, "right_z_m": 0.100}
    )
    node._update_turntable_place_from_secondary_tcp(
        {"right_x_m": 0.550, "right_y_m": 0.390, "right_z_m": 0.100}
    )
    node._accept_turntable_place_done.assert_not_called()
    assert node.secondary_post_place_fetch

    node._update_turntable_place_from_secondary_tcp(
        {"right_x_m": 0.500, "right_y_m": 0.420, "right_z_m": 0.200}
    )
    node._update_turntable_place_from_secondary_tcp(
        {"right_x_m": 0.500, "right_y_m": 0.390, "right_z_m": 0.200}
    )
    node._accept_turntable_place_done.assert_called_once()


def test_reset_requires_stopped_worker_and_retired_102():
    node = stopped_node()
    node.cycle_enabled = False
    node.worker = None
    node.controller.robot_mode = 5
    node.secondary_protective_stop_latched.set()
    node._secondary_interlock_distances = lambda: (0.160, 0.142, 0.180)
    node.get_parameter = lambda _name: SimpleNamespace(value=0.400)
    node._read_secondary_y_clearance = lambda: {"gap_y_m": 0.200, "right_y_m": 0.410}

    with pytest.raises(RuntimeError, match="has not cleared"):
        node.reset_secondary_collision_latch()
    assert node.secondary_protective_stop_latched.is_set()

    node._read_secondary_y_clearance = lambda: {"gap_y_m": 0.200, "right_y_m": 0.100}
    node.reset_secondary_collision_latch()
    assert not node.secondary_protective_stop_latched.is_set()
    assert not node.cycle_enabled
