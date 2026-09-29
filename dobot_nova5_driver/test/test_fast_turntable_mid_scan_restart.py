import threading
import time
from types import SimpleNamespace

from dobot_nova5_driver.controller_v3 import DobotNova5Controller
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_v3_fast import (
    CosmeticBoxSingleArmNode,
)


def _make_node(
    *,
    barcode_during_mid_pair: bool = False,
    midpoint_pair_duration_s: float = 0.0,
):
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "turntable_enabled": True,
        "turntable_stationary_barcode_check_s": 0.0,
        "turntable_scan_timeout_s": 0.10,
        "turntable_mid_scan_restart_s": 0.03,
        "turntable_mid_scan_low_hold_ms": 100,
        "turntable_settle_s": 0.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.turntable_lock = threading.RLock()
    node.turntable_state = "STOPPED"
    node.turntable_barcode_value = ""
    node.turntable_scan_cancel = threading.Event()
    node.running = True
    node.shutting_down = False
    node._wait_for_secondary_y_clearance = lambda *_args, **_kwargs: None
    node._wait_for_turntable_camera_ready = lambda **_kwargs: None
    node._require_cycle_active = lambda *_args: None
    node._set_turntable_barcode_window = lambda _active: None
    statuses = []
    toggles = []
    events = []
    node._publish_status = statuses.append

    def toggle(expected, result, purpose):
        assert node.turntable_state == expected
        toggles.append(purpose)
        events.append(("toggle", purpose))
        node.turntable_state = result
        return time.monotonic()

    node._toggle_turntable = toggle

    def midpoint_restart():
        assert node.turntable_state == "RUNNING"
        toggles.append("mid-scan stop/start")
        events.append(("toggle", "mid-scan stop/start"))
        if midpoint_pair_duration_s > 0.0:
            time.sleep(midpoint_pair_duration_s)
        if barcode_during_mid_pair:
            node.turntable_barcode_value = "barcode_detected"
        return time.monotonic()

    node._restart_turntable_back_to_back = midpoint_restart
    return node, statuses, toggles, events


def test_fast_scan_restarts_immediately_at_midpoint_then_uses_original_deadline():
    node, statuses, toggles, events = _make_node(midpoint_pair_duration_s=0.02)

    started_at = time.monotonic()
    result = node._scan_turntable_for_side_barcode(require_cycle_active=False)
    elapsed_s = time.monotonic() - started_at

    assert result == ""
    assert toggles == ["start", "mid-scan stop/start", "stop"]
    assert any("minimum 100ms low hold" in text for text in statuses)
    assert any("remaining 0.1s rotation interval" in text for text in statuses)
    assert elapsed_s >= 0.115


def test_fast_scan_stops_after_barcode_arrives_during_midpoint_pair():
    node, _statuses, toggles, _events = _make_node(barcode_during_mid_pair=True)

    result = node._scan_turntable_for_side_barcode(require_cycle_active=False)

    assert result == "barcode_detected"
    assert toggles == ["start", "mid-scan stop/start", "stop"]


class _Dashboard:
    def __init__(self, events):
        self.events = events

    def DOInstant(self, index, value):
        self.events.append(("do", int(value)))
        return f"0,{{}},DOInstant({index},{value});"


def test_controller_midpoint_pulse_pair_has_only_configured_low_hold(monkeypatch):
    events = []
    controller = DobotNova5Controller("192.0.2.1")
    controller.dashboard = _Dashboard(events)
    monkeypatch.setattr(
        "dobot_nova5_driver.controller_v3.time.sleep",
        lambda duration: events.append(("sleep", duration)),
    )

    controller.pulse_digital_output_twice_back_to_back(
        1, 100, 100, first_pulse_ms=100
    )

    assert events == [
        ("do", 0),
        ("do", 1),
        ("sleep", 0.1),
        ("do", 0),
        ("sleep", 0.1),
        ("do", 1),
        ("sleep", 0.1),
        ("do", 0),
    ]


def test_final_stop_keeps_low_hold_after_midpoint_restart():
    node = CosmeticBoxSingleArmNode.__new__(CosmeticBoxSingleArmNode)
    values = {
        "turntable_do_index": 1,
        "turntable_pulse_ms": 100,
        "turntable_stop_pulse_ms": 300,
        "turntable_mid_scan_low_hold_ms": 100,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.turntable_lock = threading.RLock()
    node.turntable_state = "RUNNING"
    calls = []

    class _Controller:
        def pulse_digital_output(self, index, pulse_ms, *, pre_low_hold_ms):
            calls.append((index, pulse_ms, pre_low_hold_ms))
            return time.monotonic()

    node.controller = _Controller()
    node._publish_status = lambda _text: None

    node._toggle_turntable("RUNNING", "STOPPED", "stop")

    assert calls == [(1, 300, 100)]
    assert node.turntable_state == "STOPPED"
