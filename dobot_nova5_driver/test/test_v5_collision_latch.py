import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v5 import (
    CosmeticBoxSingleArmNode,
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
    node.secondary_safety_reason = ""
    node.secondary_zone_open_for_102 = False
    node.secondary_zone_held_by_102 = False
    node.secondary_pick_guard_active = False
    node.secondary_pick_motion_active = False
    node.secondary_post_place_fetch = False
    node.secondary_fetch_phase_publisher = MagicMock()
    node.secondary_pick_guard_publisher = MagicMock()
    node.secondary_guard_xyz_publisher = MagicMock()
    node.secondary_guard_sequence = 1
    node.secondary_guard_ack_sequence = 1
    node.secondary_material_removed_for_handoff = False
    node.secondary_departure_grant_consumed = False
    node.secondary_last_measurement = {"right_y_m": 0.100}
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
        }.get(name, 0.180)
    )
    node.get_logger = lambda: MagicMock()
    return node


def test_near_miss_stops_once_and_resumes_only_after_verified_release():
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
    assert node._recover_secondary_protective_stop_if_clear(0.200, 5)
    node._schedule_continuous_restart_after_secondary_clearance.assert_called_once()


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
