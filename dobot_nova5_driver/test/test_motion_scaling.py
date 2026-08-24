from unittest.mock import patch

from dobot_nova5_driver.controller import DobotNova5Controller
from dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle import (
    CycleTiming,
    compose_motion_percent,
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
