"""Pure policy and fake-IO regression checks; never open real serial/robots."""

import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dobot_nova5_driver.alarm_light_v5 import (
    HANDOFF_STATE_TOPIC, Indication, PROCESS_TOPICS, SoundPolicy, StationPolicy, coil_command,
    light_pattern, pattern_commands,
)
from dobot_nova5_driver.alarm_light_node_v5 import SerialLightWorker
from dobot_nova5_driver import alarm_light_node_v5
from dobot_nova5_driver.alarm_light_display_v5 import AlarmLightDisplayMixin


def test_defaults_off_creates_no_ros_entities_or_timer():
    parameters = {}

    class Display(AlarmLightDisplayMixin):
        declare_parameter = staticmethod(lambda key, default: parameters.setdefault(key, default))
        get_parameter = staticmethod(lambda key: SimpleNamespace(value=parameters[key]))

    node = Display()
    node.create_publisher = Mock(side_effect=AssertionError("must not publish"))
    node.create_subscription = Mock(side_effect=AssertionError("must not subscribe"))
    node.create_timer = Mock(side_effect=AssertionError("must not add timers"))
    node._configure_alarm_light_display()
    assert parameters["alarm_light_enabled"] is False
    node.create_publisher.assert_not_called()
    node.create_subscription.assert_not_called()
    node.create_timer.assert_not_called()


def test_fault_priority_survives_normal_progress_and_camera_readiness():
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-1"))
    policy.receive_timing(dict(event="cycle_summary", outcome="fault"))
    policy.receive_cycle_status("GUI parameters applied")
    policy.receive_d435_ready(True)
    policy.receive_timing(dict(event="stage", stage="vision_detection", outcome="ok"))
    assert policy.decide().fault
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-2"))
    assert policy.decide().state == "initializing"
    policy.receive_cycle_status("moving to startup joint")
    assert policy.decide().state == "running"


def test_long_silent_inference_scan_and_material_wait_never_become_faults():
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-1"))
    policy.receive_cycle_status("moving to startup joint")
    policy.observe_publishers({key: 1 for key in PROCESS_TOPICS}, 0)
    for now in (3, 20, 53, 90, 600):
        assert policy.decide(now).state == "running"
        assert not SoundPolicy().update(policy.decide(now), now)
    for text in ("waiting for stopped, D435-scanned material", "top-surface barcode detection armed",
                 "D405 handoff zone looks clear; confirming with a fresh FFS cloud"):
        policy.receive_cycle_status(text)
        assert policy.decide().state == "running"


@pytest.mark.parametrize("outcome", ["no_target", "post_guard_vision_retry"])
def test_normal_automatic_retry_outcomes_remain_green(outcome):
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_summary", cycle_id="continuous-1", outcome=outcome))
    assert policy.decide().state == "running"


@pytest.mark.parametrize("outcome", ["grasp_retry", "bottom_regrasp_auto_rescan", "empty_turntable_reopened_for_102"])
def test_confirmed_recovery_outcomes_are_yellow_until_validated_grasp_target(outcome):
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_summary", cycle_id="continuous-1", outcome=outcome))
    assert policy.decide().warning
    assert light_pattern(policy.decide()) == (0x07, False)
    policy.receive_cycle_status("moving to startup joint")
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-2"))
    policy.receive_timing(dict(event="stage", stage="turntable_ready_wait", outcome="ok"))
    assert policy.decide().warning
    policy.receive_timing(dict(event="stage", stage="grasp_prepare", outcome="ok"))
    assert policy.decide().state == "running"


@pytest.mark.parametrize("status", [
    "recoverable grasp failure at pregrasp revalidation: target lost; returning to startup",
    "cycle 4: grasp retry recovery completed; requesting a fresh D405 target",
    "final bottom regrasp lost its D405 target; returning 101 to startup before rescanning",
    "automatic bottom-regrasp rescan 1/2: new material event #3",
])
def test_actual_controller_recovery_messages_create_warning_without_waiting_for_summary(status):
    policy = StationPolicy()
    policy.receive_cycle_status(status)
    assert policy.decide().warning
    policy.receive_cycle_status("GUI parameters applied")
    policy.receive_cycle_status("continuous cycle started")
    assert policy.decide().warning
    policy.receive_cycle_status("101 turntable grasp/flip phase armed; the 102 TCP collision-volume guard is active")
    assert policy.decide().state == "running"


def test_repeated_no_target_counts_distinct_attempts_not_duplicate_messages():
    policy = StationPolicy(no_target_warning_count=3)
    for index in (1, 2):
        event = dict(event="cycle_summary", cycle_id=f"continuous-{index}", outcome="no_target")
        policy.receive_timing(event)
        policy.receive_timing(event)
        policy.receive_timing(dict(event="cycle_start", cycle_id=f"continuous-{index + 1}"))
        assert not policy.decide().warning
    policy.receive_timing(dict(event="cycle_summary", cycle_id="continuous-3", outcome="no_target"))
    assert policy.decide().code == "target_missing"
    policy.receive_timing(dict(event="stage", stage="vision_detection", outcome="ok"))
    assert policy.decide().warning  # A None target also has an 'ok' stage.
    policy.receive_timing(dict(event="stage", stage="grasp_prepare", outcome="ok"))
    assert not policy.decide().warning
    assert policy.no_target_streak == 0
    with pytest.raises(ValueError):
        StationPolicy(no_target_warning_count=0)


def test_confirmed_empty_turntable_warning_clears_only_for_real_replacement_not_synthetic_rescan():
    policy = StationPolicy()
    policy.receive_cycle_status("moving to startup joint")
    policy.receive_cycle_status("material event #1 was empty in repeated D405 checks; 101 remains at startup and 102 may place a fresh box")
    assert policy.decide().code == "target_missing"
    policy.receive_cycle_status("accepted material event #2 from automatic bottom-regrasp D405-loss retry; place_done confirms the turntable is stopped")
    assert policy.decide().warning
    policy.receive_cycle_status("accepted material event #3 from automatic 102 TCP place/retreat trigger; place_done confirms the turntable is stopped")
    assert not policy.decide().warning


def test_point_cloud_warning_is_scoped_to_an_active_grasp_request_and_fresh_confirmed_block():
    policy = StationPolicy(clock_ns=lambda: 100)
    blocked = dict(state="BLOCKED", clear=False, stamp_ns=101, candidate_points=800)
    assert not policy.receive_handoff(blocked)  # Standby/normal 102 placement.
    policy.receive_cycle_status("cycle 1: using post-stop D405 target")
    for data in ([], {}, {**blocked, "stamp_ns": 99}, {**blocked, "stamp_ns": True},
                 {**blocked, "state": "VERIFYING_BLOCKED"}, {**blocked, "clear": "false"}):
        assert not policy.receive_handoff(data)
        assert not policy.decide().warning
    assert policy.receive_handoff(blocked)
    assert policy.decide().code == "grasp_obstacle"
    for state in ("CHECKING", "VERIFYING_CLEAR", "REACQUIRE_TARGET", "WAIT_LIVE_CLOUD"):
        policy.receive_handoff(dict(state=state, clear=False, stamp_ns=102))
        assert policy.decide().warning
    assert not policy.receive_handoff(dict(state="CLEAR", clear=True, stamp_ns=100))
    assert policy.receive_handoff(dict(state="CLEAR", clear=True, stamp_ns=103))
    assert not policy.decide().warning
    assert not policy.receive_handoff({**blocked, "stamp_ns": 102})
    policy.receive_cycle_status("101 turntable grasp/flip phase armed; the 102 TCP collision-volume guard is active")
    assert not policy.receive_handoff({**blocked, "stamp_ns": 104})  # Held-box clouds.


def test_camera_clear_only_clears_obstacle_not_an_independent_grasp_recovery():
    policy = StationPolicy(clock_ns=lambda: 100)
    policy.receive_cycle_status("recoverable grasp failure at before lift: empty grasp; returning to startup")
    policy.receive_cycle_status("requesting fresh D405 target; robot remains at its current safe pose until the pose is verified")
    policy.receive_handoff(dict(state="BLOCKED", clear=False, stamp_ns=101))
    assert policy.decide().code == "grasp_obstacle"
    policy.receive_handoff(dict(state="CLEAR", clear=True, stamp_ns=102))
    assert policy.decide().code == "grasp_recovery"


@pytest.mark.parametrize("status", ["D405 finger corridor BLOCKED; waiting for obstacle clearance (points=10, cluster=10)",
                                  "D405 handoff zone BLOCKED; waiting for 102 to retreat (points=800, cluster=800)"])
def test_controller_confirmed_obstacle_status_also_indicates_prohibition(status):
    policy = StationPolicy()
    policy.receive_cycle_status(status)
    assert policy.decide().code == "grasp_obstacle"
    policy.receive_cycle_status("D405 handoff zone looks clear; confirming with a fresh FFS cloud")
    assert policy.decide().warning
    policy.receive_cycle_status("D405 handoff zone CLEAR; accepting stable grasp samples")
    assert not policy.decide().warning


def test_yellow_warning_never_clears_a_fault_and_is_suppressed_when_operator_stops():
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_summary", outcome="safety_pause"))
    policy.receive_cycle_status("D405 finger corridor BLOCKED; waiting for obstacle clearance")
    assert policy.decide().fault
    policy.receive_timing(dict(event="cycle_start"))
    policy.receive_cycle_status("cycle stopped")
    policy.receive_cycle_status("D405 finger corridor BLOCKED; waiting for obstacle clearance")
    assert policy.decide().code == "operator_pause"
    assert not policy.grasp_observation_active


def test_warning_tone_is_short_once_per_continuous_episode_and_optional():
    sound = SoundPolicy()
    obstacle = Indication("warning", "grasp_obstacle", "障碍物")
    recovery = Indication("warning", "grasp_recovery", "恢复")
    assert sound.update(obstacle, 0)
    assert sound.update(obstacle, 0.49)
    for now in (0.5, 1, 30, 60):
        assert not sound.update(obstacle, now)
    assert not sound.update(recovery, 61)  # Another reason, same recovery run.
    assert not sound.update(Indication("interlock_wait", "waiting_102_heartbeat", "等待"), 62)
    assert not sound.update(recovery, 63)
    sound.update(Indication("running", "run", "运行"), 64)
    assert sound.update(obstacle, 65)  # Resolved, then a new incident.
    sound.acknowledge(True)
    assert not sound.update(obstacle, 65.1)
    assert not SoundPolicy(False).update(obstacle, 0)
    quiet_warning = SoundPolicy(warning_beep_enabled=False)
    assert not quiet_warning.update(obstacle, 0)
    assert quiet_warning.update(Indication("fault", "fault", "故障"), 1)


def test_only_confirmed_publisher_disappearance_indicates_disconnection():
    policy = StationPolicy()
    counts = {key: 1 for key in PROCESS_TOPICS}
    policy.observe_publishers(counts, 0)
    counts["d405"] = 0
    policy.observe_publishers(counts, 10)
    policy.observe_publishers(counts, 14.9)
    assert not policy.decide().fault
    policy.observe_publishers(counts, 15)
    assert policy.decide().code == "node:d405"
    counts["d405"] = 1
    policy.observe_publishers(counts, 16)
    assert not policy.decide().fault


def test_explicit_safety_pause_is_fault_and_camera_fault_cannot_be_cleared_by_cycle_start():
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_summary", outcome="safety_pause"))
    assert policy.decide().fault
    policy.receive_timing(dict(event="cycle_start"))
    policy.receive_d435_status("FAULT: D435 turntable barcode node failed: disconnected")
    policy.receive_timing(dict(event="cycle_start"))
    assert policy.decide().code == "d435_fault"
    policy.receive_d435_ready(True)
    assert not policy.decide().fault


def test_operator_pause_is_not_overwritten_by_a_late_normal_stage():
    policy = StationPolicy()
    policy.receive_cycle_status("continuous cycle started")
    policy.receive_cycle_status("continuous cycle will stop after the current blocking motion")
    policy.receive_timing(dict(event="stage", stage="return_startup", outcome="ok"))
    assert policy.decide().state == "waiting"
    assert not SoundPolicy().update(policy.decide(), 0)


def test_malformed_events_and_unknown_progress_do_not_create_alarms():
    policy = StationPolicy()
    for data in (None, [], {}, dict(event="unknown"), dict(event="cycle_summary", outcome=None)):
        assert not policy.receive_timing(data)
    policy.observe_publishers({key: 0 for key in PROCESS_TOPICS}, 10000)
    assert policy.decide().state == "initializing"


def test_initialization_and_start_request_stay_yellow_flashing_until_verified_progress():
    policy = StationPolicy()
    assert light_pattern(policy.decide()) == (0x07, False)
    policy.receive_d435_ready(True)
    assert policy.decide().state == "initializing"
    for text in ("cycle enabled", "continuous cycle started"):
        policy.receive_cycle_status(text)
        assert light_pattern(policy.decide()) == (0x07, False)
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-1"))
    assert policy.decide().state == "initializing"
    policy.receive_cycle_status("moving to startup joint")
    assert light_pattern(policy.decide()) == (0x03, False)
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-2"))
    assert light_pattern(policy.decide()) == (0x03, False)


@pytest.mark.parametrize("stage", ["startup motion", "single-cycle startup motion", "vision capture",
                                  "offset-high", "move-above"])
def test_heartbeat_wait_is_silent_yellow_flash_until_matching_clearance(stage):
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-1"))
    for now in range(30):
        policy.receive_cycle_status(f"waiting for 102 VLA interlock heartbeat before {stage}")
        policy.receive_d435_ready(True)
        policy.receive_cycle_status("GUI parameters applied")
        policy.receive_cycle_status("continuous cycle started")
        policy.receive_timing(dict(event="cycle_start", cycle_id="continuous-1"))
        policy.receive_timing(dict(event="stage", stage="turntable_ready_wait", outcome="ok"))
        policy.receive_timing(dict(event="cycle_summary", cycle_id="continuous-0", outcome="no_target"))
        assert policy.decide().code == "waiting_102_heartbeat"
        assert light_pattern(policy.decide()) == (0x07, False)
        assert not SoundPolicy().update(policy.decide(), now)
    policy.receive_cycle_status("102 TCP Y clearance CLEAR: gap=200.0mm; continuing another stage")
    assert policy.decide().state == "interlock_wait"
    policy.receive_cycle_status(f"102 TCP Y clearance CLEAR: gap=200.0mm; continuing {stage}")
    assert policy.decide().state == "running"


@pytest.mark.parametrize("progress", ["moving to startup joint", "startup joint already reached (max error=0.00deg)"])
def test_startup_heartbeat_recovery_can_be_observed_without_a_repeated_clearance_message(progress):
    policy = StationPolicy()
    policy.receive_cycle_status("waiting for 102 VLA interlock heartbeat before startup motion")
    policy.receive_cycle_status(progress)
    assert policy.decide().state == "running"


def test_wait_for_vision_heartbeat_is_not_cleared_by_unrelated_motion_or_failed_clearance():
    policy = StationPolicy()
    policy.receive_cycle_status("waiting for 102 VLA interlock heartbeat before vision capture")
    policy.receive_cycle_status("moving to startup joint")
    policy.receive_timing(dict(event="stage", stage="prevision_secondary_clearance", outcome="error"))
    assert policy.decide().state == "interlock_wait"
    policy.receive_timing(dict(event="stage", stage="prevision_secondary_clearance", outcome="ok"))
    assert policy.decide().state == "running"


def test_heartbeat_wait_cannot_override_fault_or_a_deliberate_operator_pause():
    policy = StationPolicy()
    policy.receive_timing(dict(event="cycle_summary", outcome="safety_pause"))
    policy.receive_cycle_status("waiting for 102 VLA interlock heartbeat before startup motion")
    assert policy.decide().fault
    assert light_pattern(policy.decide())[0] == 0x16
    policy.receive_timing(dict(event="cycle_start"))
    assert not policy.decide().fault
    policy.receive_cycle_status("cycle stopped")
    policy.receive_cycle_status("waiting for 102 VLA interlock heartbeat before startup motion")
    assert policy.decide().code == "operator_pause"
    assert light_pattern(policy.decide()) == (0x02, False)


def test_lamp_node_uses_one_hz_and_does_not_log_or_write_again_for_send_completion(monkeypatch):
    import rclpy
    import rclpy.node

    writer = SimpleNamespace(error="", applied=None, submit=Mock(), close=Mock())
    logger = Mock()
    now = [0.0]

    class FakeNode:
        def __init__(self, _name):
            self.parameters = {}
            self.periods = []

        def declare_parameter(self, name, default):
            self.parameters[name] = default

        def get_parameter(self, name):
            return SimpleNamespace(value=self.parameters[name])

        def create_timer(self, period, _callback):
            self.periods.append(period)
            return SimpleNamespace(cancel=Mock())

        create_publisher = Mock(return_value=Mock())
        create_subscription = Mock()
        create_service = Mock()
        count_publishers = Mock(return_value=1)
        get_logger = Mock(return_value=logger)
        destroy_node = Mock()
        destroy_timer = Mock()

    def exercise(node):
        from rclpy.qos import ReliabilityPolicy

        assert node.periods == [1.0]
        handoff_subscription = next(call for call in node.create_subscription.call_args_list
                                    if call.args[1] == HANDOFF_STATE_TOPIC)
        assert handoff_subscription.args[3].depth == 1
        assert handoff_subscription.args[3].reliability == ReliabilityPolicy.BEST_EFFORT
        assert writer.submit.call_args.args == ((0x07, False),)
        node.receive_timing(SimpleNamespace(data='{"event":"cycle_start","cycle_id":"continuous-1"}'))
        node.receive_cycle(SimpleNamespace(data="waiting for 102 VLA interlock heartbeat before startup motion"))
        assert writer.submit.call_args.args == ((0x07, False),)
        submits = writer.submit.call_count
        messages = logger.info.call_count
        for _ in range(5):
            node.receive_cycle(SimpleNamespace(data="waiting for 102 VLA interlock heartbeat before startup motion"))
            node.update()
        assert writer.submit.call_count == submits
        assert logger.info.call_count == messages
        assert "102 VLA 心跳" in logger.info.call_args.args[0]
        node.receive_cycle(SimpleNamespace(data="moving to startup joint"))
        assert writer.submit.call_args.args == ((3, False),)
        submits = writer.submit.call_count
        messages = logger.info.call_count
        writer.applied = (3, False)
        # A local send completion and repeated normal workflow/graph checks
        # must not duplicate a station event in the log or resend green.
        for _ in range(5):
            node.receive_cycle(SimpleNamespace(data="waiting for stopped, D435-scanned material"))
            node.update()
        assert writer.submit.call_count == submits
        assert logger.info.call_count == messages
        now[0] = 10.0
        node.receive_cycle(SimpleNamespace(data="recoverable grasp failure at before lift: empty grasp; returning to startup"))
        assert writer.submit.call_args.args == ((0x07, True),)
        assert node.periods == [1.0, 0.5]
        tone_timer = node.warning_tone_timer
        now[0] = 10.5
        messages = logger.info.call_count
        node.finish_warning_tone()
        tone_timer.cancel.assert_called_once()
        assert node.warning_tone_timer is None
        assert writer.submit.call_args.args == ((0x07, False),)
        assert logger.info.call_count == messages
        now[0] = 40.0
        node.update()
        assert writer.submit.call_args.args == ((0x07, False),)
        assert node.periods == [1.0, 0.5]
        node.receive_timing(SimpleNamespace(data='{"event":"stage","stage":"grasp_prepare","outcome":"ok"}'))
        assert writer.submit.call_args.args == ((3, False),)
        now[0] = 42.0
        node.policy.clock_ns = lambda: 100
        node.receive_cycle(SimpleNamespace(data="cycle 2: using post-stop D405 target"))
        node.receive_handoff(SimpleNamespace(data="{"))
        assert writer.submit.call_args.args == ((3, False),)
        node.receive_handoff(SimpleNamespace(data='{"state":"BLOCKED","clear":false,"stamp_ns":101}'))
        assert writer.submit.call_args.args == ((0x07, True),)
        assert node.warning_tone_timer is not None
        messages = logger.info.call_count
        node.receive_timing(SimpleNamespace(data='{"event":"cycle_summary","outcome":"safety_pause"}'))
        assert writer.submit.call_args.args == ((0x16, True),)
        assert node.warning_tone_timer is None
        assert logger.info.call_count == messages + 1

    monkeypatch.setattr(rclpy.node, "Node", FakeNode)
    monkeypatch.setattr(rclpy, "init", Mock())
    monkeypatch.setattr(rclpy, "spin", exercise)
    monkeypatch.setattr(rclpy, "ok", lambda: False)
    monkeypatch.setattr(alarm_light_node_v5, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(alarm_light_node_v5, "SerialLightWorker", Mock(return_value=writer))
    alarm_light_node_v5.main()
    writer.close.assert_called_once_with((0x01, False))


def test_silence_keeps_red_and_new_fault_rearms_sound():
    sound = SoundPolicy()
    fault = Indication("fault", "feedback102", "反馈失效")
    assert sound.update(fault, 0)
    assert not sound.update(fault, 4)
    assert sound.update(fault, 30)
    sound.acknowledge(True)
    assert not sound.update(fault, 31)
    assert light_pattern(fault, acknowledged=True) == (0x01, False)
    assert sound.update(Indication("fault", "camera", "相机故障"), 32)
    assert not sound.update(Indication("running", "run", "运行"), 33)
    assert not SoundPolicy(False).update(fault, 0)


def test_vendor_crc_addresses_and_complete_patterns():
    assert coil_command(1, 1).hex() == "01050001ff00ddfa"
    assert coil_command(1, 0, False).hex() == "010500000000cdca"
    red = light_pattern(Indication("fault", "fault", "故障"), buzzer=True)
    commands = pattern_commands(red, 1)
    assert [int.from_bytes(cmd[2:4], "big") for cmd in commands] == [0, 0x16, 0x09]
    assert light_pattern(Indication("running", "run", "运行")) == (3, False)
    yellow = pattern_commands(light_pattern(StationPolicy().decide()), 1)
    assert yellow[-1].hex() == "01050007ff003dfb"
    yellow_tone = pattern_commands(light_pattern(Indication("warning", "grasp_obstacle", "障碍物"), buzzer=True), 1)
    assert [int.from_bytes(command[2:4], "big") for command in yellow_tone] == [0, 0x07, 0x04]
    for device_id in (0, 248, -1):
        with pytest.raises(ValueError):
            pattern_commands(red, device_id)


class FakeSerial:
    def __init__(self):
        self.commands = []
        self.written = threading.Event()
        self.closed = False

    def reset_input_buffer(self):
        pass

    def write(self, command):
        self.commands.append((time.monotonic(), command))
        self.written.set()
        return len(command)

    def close(self):
        self.closed = True


def wait_applied(worker, pattern):
    with worker._condition:
        assert worker._condition.wait_for(lambda: worker.applied == pattern, timeout=2)


def test_worker_serial_parameters_spacing_dedup_and_shutdown():
    channel = FakeSerial()
    factory = Mock(return_value=channel)
    worker = SerialLightWorker("fake-only", serial_factory=factory)
    try:
        worker.submit((3, False))
        wait_applied(worker, (3, False))
        worker.submit((3, False))
        assert len(channel.commands) == 2
        assert channel.commands[1][0] - channel.commands[0][0] >= 0.045
        kwargs = factory.call_args.kwargs
        assert kwargs["port"] == "fake-only" and kwargs["parity"] == "E"
        assert kwargs["baudrate"] == 9600 and kwargs["exclusive"] is True
        assert kwargs["write_timeout"] == 0.2
    finally:
        worker.close((None, False))
    assert channel.closed
    assert channel.commands[-1][1] == coil_command(1, 0, False)


def test_new_fault_replaces_pending_old_green_without_replaying_history():
    channel = FakeSerial()
    entered, release = threading.Event(), threading.Event()
    original_write = channel.write

    def gated_write(command):
        if not entered.is_set():
            entered.set()
            assert release.wait(1)
        return original_write(command)

    channel.write = gated_write
    worker = SerialLightWorker("fake-only", serial_factory=lambda **_: channel)
    try:
        worker.submit((3, False))
        assert entered.wait(1)
        worker.submit((0x16, False))
        release.set()
        wait_applied(worker, (0x16, False))
        assert coil_command(1, 3) not in [command for _, command in channel.commands]
        assert channel.commands[-1][1] == coil_command(1, 0x16)
    finally:
        release.set()
        worker.close((None, False))


def test_unavailable_serial_failure_is_contained_and_submit_stays_nonblocking():
    failed = threading.Event()

    def unavailable(**_kwargs):
        failed.set()
        raise OSError("fake unplugged")

    worker = SerialLightWorker("fake-only", serial_factory=unavailable)
    try:
        worker.submit((3, False))
        assert failed.wait(1)
        worker.submit((0x16, False))
        assert worker.desired == (0x16, False)
    finally:
        worker.close((0x01, False), timeout=0.05)
    assert worker._thread.join(timeout=0.5) is None
    assert not worker._thread.is_alive()


def test_camera_and_motion_paths_have_no_light_reporting_or_sampling_hooks():
    package = Path(__file__).parents[1]
    camera = package.parent / "Fast-FoundationStereoPose-dul_cam/dul_cam/d405_cosmetic_box_leftmost_height75_panel_v5.py"
    d435 = package / "dobot_nova5_driver/d435_turntable_barcode_node_fast.py"
    driver = package / "dobot_nova5_driver/nova5_cosmetic_box_single_arm_cycle_v5.py"
    assert "alarm_status" not in camera.read_text()
    assert "alarm_status" not in d435.read_text()
    text = driver.read_text()
    for hook in ("_alarm_stage_enter", "_alarm_cycle_outcome", "_alarm_status_timer", "_publish_alarm_light_snapshot"):
        assert hook not in text


def test_launch_default_false_really_disables_process_and_true_enables_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros-log"))
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node
    path = Path(__file__).parents[1] / "launch/cosmetic_box_single_arm_cycle_v5_d435.launch.py"
    spec = importlib.util.spec_from_file_location("v5_alarm_launch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = module.generate_launch_description().entities
    flag = next(a for a in actions if isinstance(a, DeclareLaunchArgument) and a.name == "alarm_light_enabled")
    context = LaunchContext()
    flag.execute(context)
    assert context.launch_configurations["alarm_light_enabled"] == "false"
    lamp = next(a for a in actions if isinstance(a, Node) and a.condition is not None)
    assert not lamp.condition.evaluate(context)
    context.launch_configurations["alarm_light_enabled"] = "true"
    assert lamp.condition.evaluate(context)
