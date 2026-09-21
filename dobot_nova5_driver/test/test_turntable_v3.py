import socket
import threading
from types import SimpleNamespace

import pytest

from dobot_nova5_driver.controller_v3 import (
    BoundedDobotApiDashboard,
    DobotNova5Controller,
    TcpPose,
)
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 import (
    CosmeticBoxSingleArmNode,
)
from dobot_nova5_driver.turntable_v3 import (
    PlacementRetreatTrigger,
    classify_barcode_face,
    estimate_support_surface_z,
    nearest_face_anchor_deg,
    normalize_image_roi,
    turntable_departure_target_z,
    validate_turntable_grasp_height,
)


def test_placement_retreat_requires_place_y_then_stable_retreat_y():
    trigger = PlacementRetreatTrigger(
        place_y_m=0.400,
        stable_s=0.200,
    )

    assert not trigger.update(0.300, 0.0)  # Starting safe must not fire.
    assert not trigger.place_seen
    assert not trigger.update(0.410, 0.1)
    assert trigger.place_seen
    assert not trigger.update(0.390, 0.2)
    assert not trigger.update(0.390, 0.399)
    assert trigger.update(0.390, 0.401)
    assert not trigger.place_seen


def test_placement_retreat_resets_dwell_when_y_becomes_unsafe():
    trigger = PlacementRetreatTrigger(0.400, 0.200)

    assert not trigger.update(0.410, 1.0)
    assert not trigger.update(0.390, 1.1)
    assert not trigger.update(0.405, 1.2)
    assert not trigger.update(0.390, 1.3)
    assert not trigger.update(0.390, 1.49)
    assert trigger.update(0.390, 1.51)


def test_operator_reset_requires_a_fresh_y_entry_after_reaching_retreat_side():
    trigger = PlacementRetreatTrigger(0.400, 0.0)
    assert not trigger.update(0.410, 1.0)
    assert trigger.place_seen

    trigger.reset(require_fresh_entry=True)
    assert not trigger.entry_armed
    assert not trigger.update(0.420, 1.1)
    assert not trigger.update(0.390, 1.2)  # Arms only; old retreat cannot fire.
    assert trigger.entry_armed
    assert not trigger.update(0.410, 1.3)
    assert trigger.update(0.390, 1.4)


def test_short_stale_secondary_feedback_fails_closed_without_idle_reconnect(monkeypatch):
    import dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 as cycle_v3

    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "secondary_collision_check_enabled": True,
        "turntable_auto_place_from_secondary_tcp": False,
        "secondary_user_index": 0,
        "secondary_tool_index": 1,
        "secondary_tcp_max_age_s": 0.05,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.secondary_connection_lock = threading.RLock()
    node._connect_secondary_safety_feedback = lambda: True
    drops = []
    node._drop_secondary_safety_feedback = lambda: drops.append(True)
    received_at = [999.942]
    node.secondary_controller = SimpleNamespace(
        current_feedback_tcp_pose_with_timestamp=lambda: (None, 0, 1, received_at[0]),
        feedback_last_read_error="",
        feedback_raw_timestamp=123,
        feedback_reader_diagnostics=lambda: "reader_active=True",
        feedback_reader_alive=True,
    )
    monkeypatch.setattr(cycle_v3.time, "time", lambda: 1000.0)

    with pytest.raises(RuntimeError, match="stale: age=58ms > 50ms"):
        node._read_secondary_y_clearance()
    assert drops == []

    received_at[0] = 998.9
    with pytest.raises(RuntimeError, match="stale"):
        node._read_secondary_y_clearance()
    assert drops == [True]

    received_at[0] = 999.942
    node.secondary_controller.feedback_reader_alive = False
    with pytest.raises(RuntimeError, match="stale"):
        node._read_secondary_y_clearance()
    assert drops == [True, True]


def test_place_done_resolves_unknown_and_starts_independent_prescan(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, *, target, args, name, daemon):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.target, self.args, self.name, self.daemon))

    import dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 as cycle_v3

    monkeypatch.setattr(cycle_v3.threading, "Thread", FakeThread)
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.turntable_lock = threading.RLock()
    node.turntable_condition = threading.Condition(node.turntable_lock)
    node.turntable_state = "UNKNOWN"
    node.turntable_scan_in_progress = False
    node.turntable_material_ready = False
    node.turntable_place_done_count = 0
    node.turntable_place_done_consumed = 0
    node.turntable_place_done_duplicate_warned = False
    node.turntable_waiting_for_place = True
    node.turntable_scan_error = "old error"
    node.turntable_scan_cancel = threading.Event()
    node.d435_continuous_detection = True
    node.d435_continuous_presence = False
    node.d435_continuous_last_value = ""
    node.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(
        0.400, 0.200
    )
    statuses = []
    node._publish_status = statuses.append

    node._accept_turntable_place_done("test")

    assert node.turntable_state == "STOPPED"
    assert node.turntable_scan_in_progress
    assert not node.turntable_waiting_for_place
    assert node.turntable_place_done_count == 1
    assert node.turntable_place_done_consumed == 1
    assert node.turntable_scan_error == ""
    assert len(started) == 1
    assert started[0][1] == (1,)
    assert "fresh stopped-face D435 confirmation" in statuses[-1]


def test_place_done_reuses_current_continuous_barcode_without_turntable_thread(
    monkeypatch,
):
    import dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 as cycle_v3

    monkeypatch.setattr(
        cycle_v3.threading,
        "Thread",
        lambda **_kwargs: pytest.fail("visible barcode must not start a scan thread"),
    )
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.turntable_lock = threading.RLock()
    node.turntable_condition = threading.Condition(node.turntable_lock)
    node.turntable_state = "UNKNOWN"
    node.turntable_scan_in_progress = False
    node.turntable_material_ready = False
    node.turntable_ready_barcode = ""
    node.turntable_place_done_count = 0
    node.turntable_place_done_consumed = 0
    node.turntable_place_done_duplicate_warned = False
    node.turntable_waiting_for_place = True
    node.turntable_scan_error = "old error"
    node.turntable_scan_cancel = threading.Event()
    node.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(
        0.400, 0.200
    )
    node.d435_continuous_detection = True
    node.d435_continuous_presence = True
    node.d435_continuous_last_value = "barcode_detected"
    statuses = []
    node._publish_status = statuses.append

    node._accept_turntable_place_done("test")

    assert node.turntable_state == "STOPPED"
    assert not node.turntable_scan_in_progress
    assert node.turntable_material_ready
    assert node.turntable_ready_barcode == "barcode_detected"
    assert node.turntable_scan_thread is None
    assert "no turntable pulse is needed" in statuses[-1]


def test_duplicate_place_done_is_ignored_while_material_is_ready(monkeypatch):
    import dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3 as cycle_v3

    monkeypatch.setattr(
        cycle_v3.threading,
        "Thread",
        lambda **_kwargs: pytest.fail("duplicate event must not start a thread"),
    )
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.turntable_lock = threading.RLock()
    node.turntable_condition = threading.Condition(node.turntable_lock)
    node.turntable_state = "STOPPED"
    node.turntable_scan_in_progress = False
    node.turntable_material_ready = True
    node.turntable_place_done_count = 4
    node.turntable_place_done_consumed = 4
    node.turntable_place_done_duplicate_warned = False
    statuses = []
    node._publish_status = statuses.append

    node._accept_turntable_place_done("duplicate")

    assert node.turntable_place_done_count == 4
    assert node.turntable_place_done_duplicate_warned
    assert "ignoring duplicate" in statuses[-1]


def test_secondary_tcp_automatic_place_trigger_uses_y_only():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.get_parameter = lambda name: SimpleNamespace(
        value={"turntable_auto_place_from_secondary_tcp": True}[name]
    )
    node.turntable_lock = threading.RLock()
    node.turntable_waiting_for_place = True
    node.turntable_place_done_count = 0
    node.turntable_place_done_consumed = 0
    node.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(0.400, 0.0)
    statuses = []
    accepted = []
    node._publish_status = statuses.append
    node._accept_turntable_place_done = accepted.append

    # No Z value is supplied: entering and retreating across the Y boundary
    # alone must create the event.
    node._update_turntable_place_from_secondary_tcp({"right_y_m": 0.410})
    node._update_turntable_place_from_secondary_tcp({"right_y_m": 0.390})

    assert len(accepted) == 1
    assert "Z ignored" in accepted[0]
    assert "TCP Z is not used" in statuses[-1]


def test_startup_recovery_reset_forgets_previous_turntable_round():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.turntable_lock = threading.RLock()
    node.turntable_condition = threading.Condition(node.turntable_lock)
    node.turntable_scan_cancel = threading.Event()
    node.turntable_scan_thread = None
    node.turntable_scan_in_progress = True
    node.turntable_material_ready = True
    node.turntable_ready_barcode = "old-barcode"
    node.turntable_scan_error = "old-error"
    node.turntable_waiting_for_place = False
    node.turntable_place_done_count = 7
    node.turntable_place_done_consumed = 7
    node.turntable_place_done_duplicate_warned = True
    node.turntable_barcode_value = "old-window-value"
    node.turntable_barcode_result_count = 3
    node.d435_continuous_last_value = "old-continuous-value"
    node.d435_continuous_presence = True
    node.turntable_secondary_retreat_trigger = PlacementRetreatTrigger(0.400, 0.0)
    node.turntable_secondary_retreat_trigger.update(0.410, 1.0)
    node.last_accepted_target = TcpPose(0.5, -0.1, 0.13, 0.0, 0.0, 0.0)
    node.last_accepted_width_m = 0.05
    node.last_accepted_length_m = 0.12
    node.last_accepted_height_m = 0.04
    stopped = []
    windows = []
    node._stop_turntable_if_running = stopped.append
    node._set_turntable_barcode_window = lambda active: windows.append(
        ("turntable", active)
    )
    node._set_top_surface_barcode_window = lambda active: windows.append(
        ("top", active)
    )

    node._reset_turntable_round_for_operator_recovery()

    assert stopped == ["operator startup recovery"]
    assert windows == [("turntable", False), ("top", False)]
    assert node.turntable_waiting_for_place
    assert not node.turntable_scan_in_progress
    assert not node.turntable_material_ready
    assert node.turntable_ready_barcode == ""
    assert node.turntable_scan_error == ""
    assert node.turntable_place_done_count == 0
    assert node.turntable_place_done_consumed == 0
    assert not node.turntable_scan_cancel.is_set()
    assert not node.turntable_secondary_retreat_trigger.entry_armed
    assert node.last_accepted_target is None
    assert node.last_accepted_width_m is None


def test_move_startup_arms_a_fresh_continuous_wait_after_reset():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.worker = SimpleNamespace(is_alive=lambda: False)
    node.secondary_safety_lock = threading.RLock()
    node.secondary_auto_resume_requested = threading.Event()
    node.secondary_protective_stop_latched = threading.Event()
    node.cycle_enabled = True
    node.get_parameter = lambda name: SimpleNamespace(
        value={"secondary_collision_check_enabled": False}[name]
    )
    node.turntable_scan_cancel = threading.Event()
    node.turntable_condition = threading.Condition(threading.RLock())
    node.controller = SimpleNamespace(robot_mode=5)
    node.action_lock = threading.RLock()
    events = []
    node._publish_status = lambda message: events.append(("status", message))
    node._prepare_robot_for_startup_recovery = lambda: events.append(("prepare",))
    node._move_startup_and_open = lambda **_kwargs: events.append(("startup",))
    node._reset_turntable_round_for_operator_recovery = lambda: events.append(
        ("reset",)
    )

    def ensure_worker():
        assert node.cycle_enabled
        events.append(("worker",))

    node._ensure_worker = ensure_worker

    node.move_startup()

    assert node.cycle_enabled
    assert [(event[0]) for event in events if event[0] != "status"] == [
        "prepare",
        "startup",
        "reset",
        "worker",
    ]
    assert "waiting for a fresh 102" in events[-1][1]


def _make_turntable_scan_node(
    *,
    stopped_face_barcode: str,
    barcode_after_start: str,
    continuous_barcode_after_stop: str = "",
):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "turntable_enabled": True,
        "turntable_stationary_barcode_check_s": 0.05,
        "turntable_scan_timeout_s": 0.10,
        "turntable_settle_s": 0.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.turntable_lock = threading.RLock()
    node.turntable_state = "STOPPED"
    node.turntable_barcode_value = ""
    node.d435_continuous_detection = True
    node.d435_continuous_presence = False
    node.d435_continuous_last_value = ""
    node.turntable_scan_cancel = threading.Event()
    node.running = True
    node.shutting_down = False
    node._wait_for_secondary_y_clearance = lambda *_args, **_kwargs: None
    node._wait_for_turntable_camera_ready = lambda **_kwargs: None
    node._require_cycle_active = lambda *_args: None
    statuses = []
    toggles = []
    node._publish_status = statuses.append

    def set_window(active):
        if active:
            node.turntable_barcode_value = stopped_face_barcode

    def toggle(expected, result, purpose):
        assert node.turntable_state == expected
        toggles.append(purpose)
        node.turntable_state = result
        if purpose == "start":
            node.turntable_barcode_value = barcode_after_start
        elif continuous_barcode_after_stop:
            node.d435_continuous_presence = True
            node.d435_continuous_last_value = continuous_barcode_after_stop

    node._set_turntable_barcode_window = set_window
    node._toggle_turntable = toggle
    return node, statuses, toggles


def test_stopped_face_barcode_skips_turntable_start_pulse():
    node, statuses, toggles = _make_turntable_scan_node(
        stopped_face_barcode="barcode_detected",
        barcode_after_start="",
    )

    result = node._scan_turntable_for_side_barcode(require_cycle_active=False)

    assert result == "barcode_detected"
    assert toggles == []
    assert any("skipping turntable rotation" in status for status in statuses)


def test_turntable_rotates_only_after_stopped_face_has_no_barcode():
    node, statuses, toggles = _make_turntable_scan_node(
        stopped_face_barcode="",
        barcode_after_start="barcode_detected",
    )

    result = node._scan_turntable_for_side_barcode(require_cycle_active=False)

    assert result == "barcode_detected"
    assert toggles == ["start", "stop"]
    assert any("starting turntable search" in status for status in statuses)


def test_barcode_arriving_during_stop_boundary_is_accepted():
    node, statuses, toggles = _make_turntable_scan_node(
        stopped_face_barcode="",
        barcode_after_start="",
        continuous_barcode_after_stop="barcode_detected",
    )

    result = node._scan_turntable_for_side_barcode(require_cycle_active=False)

    assert result == "barcode_detected"
    assert toggles == ["start", "stop"]
    assert any("final stop/settle" in status for status in statuses)


def _make_prescan_worker_node(scan_result="", scan_error=None):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    node.turntable_lock = threading.RLock()
    node.turntable_condition = threading.Condition(node.turntable_lock)
    node.turntable_state = "STOPPED"
    node.turntable_scan_in_progress = True
    node.turntable_material_ready = False
    node.turntable_ready_barcode = ""
    node.turntable_scan_error = ""
    node.turntable_scan_cancel = threading.Event()
    node.running = True
    node.cycle_enabled = True
    node.get_parameter = lambda name: SimpleNamespace(
        value={"turntable_enabled": True, "turntable_place_wait_timeout_s": 0.0}[name]
    )
    statuses = []
    stops = []
    node._publish_status = statuses.append
    node._set_turntable_barcode_window = lambda _active: None
    node._stop_turntable_if_running = lambda reason: stops.append(reason)

    def scan(**_kwargs):
        if scan_error:
            raise RuntimeError(scan_error)
        return scan_result

    node._scan_turntable_for_side_barcode = scan
    return node, statuses, stops


def test_no_side_barcode_after_timeout_releases_stopped_material_for_d405():
    node, statuses, stops = _make_prescan_worker_node()

    node._turntable_prescan_worker(3)

    assert node.turntable_material_ready
    assert node.turntable_ready_barcode == ""
    assert node.turntable_scan_error == ""
    assert not node.turntable_scan_in_progress
    assert node._wait_for_scanned_turntable_material() == ""
    assert stops == []
    assert "ready for 101 to grasp and check its top surface" in statuses[-1]


def test_real_d435_scan_error_still_blocks_left_arm():
    node, _statuses, stops = _make_prescan_worker_node(scan_error="camera unavailable")

    node._turntable_prescan_worker(4)

    assert not node.turntable_material_ready
    assert node.turntable_scan_error == "camera unavailable"
    assert stops
    with pytest.raises(RuntimeError, match="camera unavailable"):
        node._wait_for_scanned_turntable_material()


def test_mouse_roi_is_order_independent_clamped_and_rejects_small_drags():
    assert normalize_image_roi((300, 220), (100, 80), 640, 480) == (
        100,
        80,
        301,
        221,
    )
    assert normalize_image_roi((-20, -10), (700, 500), 640, 480) == (
        0,
        0,
        640,
        480,
    )
    assert normalize_image_roi((10, 10), (20, 20), 640, 480) is None


def test_barcode_face_priority_is_side_then_top_then_bottom():
    assert classify_barcode_face("SIDE", "TOP") == "side"
    assert classify_barcode_face("", "TOP") == "top"
    assert classify_barcode_face("", "") == "bottom"


def test_turntable_departure_uses_absolute_safe_height_when_higher():
    assert turntable_departure_target_z(0.1364, 0.060, 0.320) == pytest.approx(
        0.320
    )


def test_turntable_departure_preserves_minimum_relative_lift_when_higher():
    assert turntable_departure_target_z(0.300, 0.060, 0.320) == pytest.approx(
        0.360
    )


def test_nearest_face_anchor_uses_v2_transfer_j6_grid():
    assert nearest_face_anchor_deg(62.0, 15.0, -90.0, 355.0) == pytest.approx(
        105.0
    )
    assert nearest_face_anchor_deg(42.0, 15.0, -90.0, 355.0) == pytest.approx(
        15.0
    )


def test_zero_degree_d435_reference_snaps_to_zero_degree_grid():
    assert nearest_face_anchor_deg(0.8, 0.0, -90.0, 355.0) == pytest.approx(0.0)
    assert nearest_face_anchor_deg(61.0, 0.0, -90.0, 355.0) == pytest.approx(90.0)


def test_nearest_face_anchor_supports_equivalent_windings_inside_limit():
    assert nearest_face_anchor_deg(-330.0, 15.0, -90.0, 355.0) == pytest.approx(
        -345.0
    )
    assert nearest_face_anchor_deg(320.0, 15.0, -90.0, 355.0) == pytest.approx(
        285.0
    )


def _make_lift_face_snap_node(*, reject_fk=False):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    params = {
        "grasp_lift_m": 0.060,
        "scan_exit_user_xyz": [0.560, 0.375, 0.320],
        "user_index": 0,
        "command_tool_index": 1,
        "jog_tolerance_m": 0.002,
        "barcode_flip_watch_joint_index": 5,
        "d435_side_face_reference_joint_deg": 0.0,
        "barcode_flip_step_deg": -90.0,
        "barcode_flip_safe_joint_limit_deg": 355.0,
        "barcode_flip_jog_tolerance_deg": 1.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=params[name])
    node._require_cycle_active = lambda _where: None
    node._is_barcode_flip_joint_safe = lambda _joints, _delta: True
    statuses = []
    node._publish_status = statuses.append
    node.get_logger = lambda: SimpleNamespace(warning=statuses.append)

    class FakeController:
        def __init__(self):
            self.pose = TcpPose(0.550, -0.150, 0.134, 0.0, 0.0, 20.0)
            self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 20.0]
            self.moves = []

        def move_linear_tcp(self, target, **_kwargs):
            self.moves.append(target)
            self.pose = target
            if target.z == pytest.approx(0.320):
                self.joints[5] = target.rz

        def current_joint(self):
            return list(self.joints)

        def forward_kinematics(self, joints, **_kwargs):
            if reject_fk:
                raise RuntimeError("FK unavailable")
            return TcpPose(self.pose.x, self.pose.y, self.pose.z, 0.0, 0.0, joints[5])

        def inverse_kinematics(self, pose, **_kwargs):
            return [0.0, 0.0, 0.0, 0.0, 0.0, pose.rz]

    controller = FakeController()
    node.controller = controller
    node._current_command_pose = lambda: controller.pose
    motion = {
        "grasp_lift_speed": 20,
        "grasp_lift_acc": 20,
        "barcode_alignment_speed": 20,
        "barcode_alignment_acc": 20,
    }
    return node, controller, statuses, motion


def test_turntable_side_lift_and_face_snap_use_one_ascending_movl():
    node, controller, statuses, motion = _make_lift_face_snap_node()

    assert node._execute_turntable_lift_face_snap(motion)
    assert [(move.z, move.rz) for move in controller.moves] == pytest.approx(
        [(0.320, 0.0)]
    )
    assert "combined turntable departure" in statuses[-1]


def test_turntable_side_lift_uses_original_straight_path_if_preflight_fails():
    node, controller, statuses, motion = _make_lift_face_snap_node(reject_fk=True)

    assert not node._execute_turntable_lift_face_snap(motion)
    assert [(move.z, move.rz) for move in controller.moves] == pytest.approx(
        [(0.320, 20.0)]
    )
    assert any("preflight rejected" in status for status in statuses)


def test_turntable_combined_motion_failure_does_not_retry_or_issue_snap():
    node, controller, _statuses, motion = _make_lift_face_snap_node()

    def fail_motion(target, **_kwargs):
        controller.moves.append(target)
        raise RuntimeError("MovL rejected")

    controller.move_linear_tcp = fail_motion
    with pytest.raises(RuntimeError, match="MovL rejected"):
        node._execute_turntable_lift_face_snap(motion)
    assert len(controller.moves) == 1


def test_forward_kinematics_converts_controller_mm_to_user_tool_meters():
    controller = DobotNova5Controller.__new__(DobotNova5Controller)
    called = []

    class Dashboard:
        def PositiveKin(self, *joints, user, tool):
            called.append((joints, user, tool))
            return "0,{550,-150,200,0,0,20},PositiveKin();"

    controller.dashboard = Dashboard()
    controller._command_lock = threading.RLock()
    pose = controller.forward_kinematics(
        [1, 2, 3, 4, 5, 20], user_index=0, tool_index=1
    )

    assert pose == TcpPose(0.550, -0.150, 0.200, 0.0, 0.0, 20.0)
    assert called == [((1.0, 2.0, 3.0, 4.0, 5.0, 20.0), 0, 1)]


class FakeDashboard:
    def __init__(self):
        self.commands = []
        self.output = 0

    def DOInstant(self, index, value):
        self.commands.append((int(index), int(value)))
        self.output = int(value)
        return f"0,{{}},DOInstant({index},{value});"

    def GetDO(self, index):
        return f"0,{{{self.output}}},GetDO({index});"


class TimeoutSocket:
    def settimeout(self, _timeout):
        pass

    def sendall(self, _payload):
        pass

    def recv(self, _size):
        raise socket.timeout("simulated missing Dashboard reply")

    def shutdown(self, _how):
        pass

    def close(self):
        pass


def test_v3_dashboard_missing_reply_releases_caller_and_reconnects():
    dashboard = BoundedDobotApiDashboard.__new__(BoundedDobotApiDashboard)
    dashboard.ip = "192.0.2.1"
    dashboard.port = 29999
    dashboard._v3_io_lock = threading.Lock()
    dashboard.socket_dobot = TimeoutSocket()
    reconnected = []
    dashboard._replace_socket = lambda: reconnected.append(True)

    with pytest.raises(TimeoutError, match="GetDO\\(1\\).+did not return"):
        dashboard.sendRecvMsg("GetDO(1)")

    assert reconnected == [True]


def test_support_surface_uses_remaining_quarter_height():
    assert estimate_support_surface_z(0.140, 0.080) == pytest.approx(0.120)


def test_turntable_height_check_accepts_consistent_target():
    result = validate_turntable_grasp_height(
        vision_target_z_m=0.140,
        command_target_z_m=0.150,
        box_height_m=0.080,
        configured_surface_z_m=0.120,
        surface_tolerance_m=0.005,
        tcp_below_target_m=0.010,
        surface_clearance_m=0.005,
    )
    assert result.estimated_surface_z_m == pytest.approx(0.120)
    assert result.minimum_command_tcp_z_m == pytest.approx(0.135)


def test_turntable_height_check_rejects_wrong_support_plane():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_turntable_grasp_height(
            vision_target_z_m=0.100,
            command_target_z_m=0.110,
            box_height_m=0.080,
            configured_surface_z_m=0.120,
            surface_tolerance_m=0.005,
            tcp_below_target_m=0.0,
            surface_clearance_m=0.003,
        )


def test_turntable_height_check_rejects_unconfigured_surface():
    with pytest.raises(ValueError, match="not configured"):
        validate_turntable_grasp_height(
            vision_target_z_m=0.100,
            command_target_z_m=0.110,
            box_height_m=0.040,
            configured_surface_z_m=-1.0,
            surface_tolerance_m=0.020,
            tcp_below_target_m=0.0,
            surface_clearance_m=0.003,
        )


def test_controller_turntable_pulse_is_zero_one_zero(monkeypatch):
    controller = DobotNova5Controller("192.0.2.1")
    dashboard = FakeDashboard()
    controller.dashboard = dashboard
    monkeypatch.setattr("dobot_nova5_driver.controller_v3.time.sleep", lambda _: None)

    controller.pulse_digital_output(3, 300)

    assert dashboard.commands == [(3, 0), (3, 1), (3, 0)]
    assert controller.read_digital_output(3) == 0
