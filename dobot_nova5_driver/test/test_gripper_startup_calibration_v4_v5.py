"""Exercise V4/V5 startup with the real DH driver and simulated registers only."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dobot_nova5_driver import dobot_dh_api_v3 as dh


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Dashboard:
    def __init__(self, clock):
        self.clock = clock
        self.initialized = dh.INIT_DONE
        self.calibration_started = None
        self.calibration_duration = 0.8
        self.calibration_start_delay = 0.0
        self.init_feedback_sticky = False
        self.position_moving_until = 0.0
        self.ready_blips = ()
        self.reject_calibration = False
        self.reject_open_count = 0
        self.reject_force_count = 0
        self.command_error_id = -1
        self.open_stuck = False
        self.open_blocked = False
        self.writes = []
        self.write_times = []
        self.registers = {dh.REG_CUR_POS: 0, dh.REG_FORCE: 20}

    def SetToolMode(self, *_args):
        return "0,{},SetToolMode();"

    def SetTool485(self, *_args):
        return "0,{},SetTool485();"

    def ModbusClose(self, *_args):
        return "0,{},ModbusClose();"

    def ModbusCreate(self, *_args):
        return "0,{0},ModbusCreate();"

    def SetHoldRegs(self, _index, register, _count, encoded_value):
        value = int(encoded_value.strip("{}"))
        self.writes.append((register, value))
        self.write_times.append((self.clock.now, register, value))
        if register == dh.REG_INITIALIZE:
            if self.reject_calibration:
                return "-1,{},SetHoldRegs();"
            self.calibration_started = self.clock.now
        if register == dh.REG_FORCE and self.reject_force_count:
            self.reject_force_count -= 1
            return f"{self.command_error_id},{{}},SetHoldRegs();"
        if register == dh.REG_POSITION:
            elapsed = self.clock.now - self.calibration_started
            if elapsed < self.calibration_start_delay + self.calibration_duration:
                return "-1,{},SetHoldRegs();"
            if self.reject_open_count:
                self.reject_open_count -= 1
                return f"{self.command_error_id},{{}},SetHoldRegs();"
        if register == dh.REG_POSITION and not self.open_stuck:
            self.registers[dh.REG_CUR_POS] = 800 if self.open_blocked else value
        self.registers[register] = value
        return "0,{},SetHoldRegs();"

    def GetHoldRegs(self, _index, register, _count):
        if register == dh.REG_INIT_STATE:
            if self.calibration_started is None:
                value = self.initialized
            else:
                elapsed = self.clock.now - self.calibration_started
                done = elapsed >= self.calibration_start_delay + self.calibration_duration
                old = elapsed < self.calibration_start_delay
                blip = any(low <= elapsed < high for low, high in self.ready_blips)
                value = int(self.init_feedback_sticky or old or blip or done)
        elif register == dh.REG_GRIP_STATE:
            if self.open_stuck and dh.REG_POSITION in self.registers:
                value = dh.GRIP_IN_MOTION
            elif self.open_blocked and dh.REG_POSITION in self.registers:
                value = dh.GRIP_GRIPPED
            else:
                value = dh.GRIP_REACHED
        elif register == dh.REG_CUR_POS and self.clock.now < self.position_moving_until:
            value = int(self.clock.now * 100)
        else:
            value = self.registers[register]
        return f"0,{{{value}}},GetHoldRegs();"


@pytest.fixture(params=["v4", "v5"])
def startup(request, monkeypatch):
    module = importlib.import_module(
        "dobot_nova5_driver.nova5_cosmetic_box_single_arm_cycle_" + request.param
    )
    clock = Clock()
    monkeypatch.setattr(dh, "time", clock)
    dashboard = Dashboard(clock)
    parameters = dict(robot_ip="192.0.2.101", dashboard_port=29999,
                      dh_tool_identify=1, dh_slave_id=1, dh_force=55,
                      dh_timeout_s=10.0)
    node = SimpleNamespace(
        controller=SimpleNamespace(dashboard=dashboard),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_logger=lambda: logger,
        _publish_status=Mock(),
    )
    logger = Mock()
    initialize = lambda: module.CosmeticBoxSingleArmNode._initialize_gripper(node)
    return SimpleNamespace(node=node, module=module, dashboard=dashboard,
                           parameters=parameters, clock=clock, logger=logger,
                           initialize=initialize)


@pytest.mark.parametrize("initial_state", [0, dh.INIT_DONE])
def test_startup_calibrates_even_when_already_initialized(startup, initial_state):
    startup.dashboard.initialized = initial_state
    gripper = startup.initialize()

    assert isinstance(gripper, dh.DHGripper)
    assert startup.dashboard.writes == [
        (dh.REG_INITIALIZE, dh.INIT_TRIGGER),
        (dh.REG_FORCE, 55),
        (dh.REG_POSITION, dh.DH_POS_OPEN),
    ]
    assert gripper.read_init_state() == dh.INIT_DONE
    assert gripper.read_position_raw() == dh.DH_POS_OPEN


def test_every_new_startup_repeats_calibration(startup):
    startup.initialize()
    startup.initialize()
    assert startup.dashboard.writes.count((dh.REG_INITIALIZE, dh.INIT_TRIGGER)) == 2


@pytest.mark.parametrize("timeout", [3.0, 20.0])
def test_calibration_timeout_aborts_startup_before_ready_or_open(startup, timeout):
    startup.parameters["dh_timeout_s"] = timeout
    startup.dashboard.calibration_duration = float("inf")

    with pytest.raises(TimeoutError, match="recalibration timed out"):
        startup.initialize()

    assert startup.clock.now >= max(15.0, timeout)
    assert startup.dashboard.writes == [(dh.REG_INITIALIZE, dh.INIT_TRIGGER)]
    startup.logger.info.assert_not_called()


def test_rejected_calibration_does_not_allow_startup_to_continue(startup):
    startup.dashboard.reject_calibration = True
    with pytest.raises(RuntimeError, match="SetHoldRegs.*failed"):
        startup.initialize()
    assert startup.dashboard.writes == [(dh.REG_INITIALIZE, dh.INIT_TRIGGER)]
    startup.logger.info.assert_not_called()


def test_startup_is_not_ready_until_opening_is_confirmed(startup):
    startup.dashboard.open_stuck = True
    with pytest.raises(TimeoutError, match="did not stop"):
        startup.initialize()
    assert (dh.REG_INITIALIZE, dh.INIT_TRIGGER) in startup.dashboard.writes
    startup.logger.info.assert_not_called()


def test_manual_recalibration_remains_available_after_startup(startup):
    startup.node.gripper = startup.initialize()
    startup.module.CosmeticBoxSingleArmNode.recalibrate_gripper(startup.node)
    assert startup.dashboard.writes.count((dh.REG_INITIALIZE, dh.INIT_TRIGGER)) == 2
    assert startup.dashboard.registers[dh.REG_CUR_POS] == dh.DH_POS_OPEN
    assert "completed" in startup.node._publish_status.call_args.args[0]


@pytest.mark.parametrize("sticky", [False, True])
def test_old_initialized_feedback_does_not_release_open_during_delayed_calibration(startup, sticky):
    startup.dashboard.calibration_start_delay = 0.9
    startup.dashboard.calibration_duration = 1.9
    startup.dashboard.init_feedback_sticky = sticky
    startup.initialize()
    open_times = [stamp for stamp, register, _ in startup.dashboard.write_times
                  if register == dh.REG_POSITION]
    assert open_times and min(open_times) >= 3.0


def test_brief_ready_feedback_is_not_enough_to_finish_calibration(startup):
    startup.dashboard.calibration_duration = 4.0
    startup.dashboard.ready_blips = ((0.2, 0.4), (3.0, 3.15))
    startup.initialize()
    command_times = [stamp for stamp, register, _ in startup.dashboard.write_times
                     if register in (dh.REG_FORCE, dh.REG_POSITION)]
    assert min(command_times) >= 4.3


def test_position_must_stop_even_if_initialized_and_reached_feedback_is_stale(startup):
    startup.dashboard.position_moving_until = 4.0
    startup.initialize()
    command_times = [stamp for stamp, register, _ in startup.dashboard.write_times
                     if register in (dh.REG_FORCE, dh.REG_POSITION)]
    assert min(command_times) >= 4.3


@pytest.mark.parametrize("register", ["open", "force"])
def test_calibration_completion_tolerates_two_transient_write_refusals_without_recalibrating(startup, register):
    setattr(startup.dashboard, f"reject_{register}_count", 2)
    gripper = startup.initialize()
    assert gripper.read_position_raw() == dh.DH_POS_OPEN
    assert startup.dashboard.writes.count((dh.REG_INITIALIZE, dh.INIT_TRIGGER)) == 1
    rejected_register = dh.REG_POSITION if register == "open" else dh.REG_FORCE
    assert sum(reg == rejected_register for reg, _ in startup.dashboard.writes) == 3


def test_persistent_open_refusal_still_aborts_startup_after_bounded_retries(startup):
    startup.dashboard.reject_open_count = 100
    with pytest.raises(RuntimeError, match="SetHoldRegs.*failed"):
        startup.initialize()
    assert startup.dashboard.writes.count((dh.REG_POSITION, dh.DH_POS_OPEN)) == 3
    assert startup.dashboard.writes.count((dh.REG_INITIALIZE, dh.INIT_TRIGGER)) == 1
    startup.logger.info.assert_not_called()


def test_other_write_errors_are_not_retried(startup):
    startup.dashboard.reject_open_count = 100
    startup.dashboard.command_error_id = -2
    with pytest.raises(RuntimeError, match="ErrorID -2"):
        startup.initialize()
    assert startup.dashboard.writes.count((dh.REG_POSITION, dh.DH_POS_OPEN)) == 1


def test_blocked_open_cannot_be_reported_as_startup_ready(startup):
    startup.dashboard.open_blocked = True
    with pytest.raises(RuntimeError, match="full-open.*not confirmed"):
        startup.initialize()
    startup.logger.info.assert_not_called()


def test_calibration_can_finish_without_commanding_open_when_not_requested(startup):
    gripper = dh.DHGripper(startup.dashboard, dh.DobotDHConfig(force=55))
    gripper.recalibrate(open_after=False)
    assert startup.dashboard.writes == [
        (dh.REG_INITIALIZE, dh.INIT_TRIGGER), (dh.REG_FORCE, 55),
    ]


@pytest.mark.parametrize("cancel_at", ["before_trigger", "during_calibration", "before_open"])
def test_cancellation_does_not_send_the_next_motion(startup, cancel_at):
    gripper = dh.DHGripper(startup.dashboard, dh.DobotDHConfig(force=55))

    def cancelled():
        if cancel_at == "before_trigger":
            return True
        if cancel_at == "during_calibration":
            return startup.clock.now >= 0.8
        return (dh.REG_FORCE, 55) in startup.dashboard.writes

    with pytest.raises(RuntimeError, match="recalibration cancelled"):
        gripper.recalibrate(cancel_check=cancelled)
    assert not any(register == dh.REG_POSITION for register, _ in startup.dashboard.writes)
    if cancel_at == "before_trigger":
        assert startup.dashboard.writes == []
    elif cancel_at == "during_calibration":
        assert startup.dashboard.writes == [(dh.REG_INITIALIZE, dh.INIT_TRIGGER)]


def test_feedback_read_refusal_is_not_hidden_as_a_retryable_open_error(startup):
    startup.dashboard.GetHoldRegs = lambda *_args: "-1,{},GetHoldRegs();"
    with pytest.raises(dh.DobotCommandError, match="GetHoldRegs.*ErrorID -1"):
        startup.initialize()
    assert startup.dashboard.writes == [(dh.REG_INITIALIZE, dh.INIT_TRIGGER)]
    startup.logger.info.assert_not_called()
