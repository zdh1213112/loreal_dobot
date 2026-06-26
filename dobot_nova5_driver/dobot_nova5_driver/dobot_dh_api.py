#!/usr/bin/env python3

"""Standalone Dobot Nova5 + DH AG-95 control helper.

This file is intentionally independent from the LeRobot Robot/RobotConfig
framework. It only uses Dobot's TCP Python SDK in ``TCP_IP_Python_V4/dobot_api.py``.

Typical usage from the repository root:

    # Connect right arm dashboard and run a simple gripper open/close test.
    python3 src/lerobot/robots/bi_dobot_nova5_dh/dobot_dh_api.py \
        --robot-ip 192.168.5.102 --cycles 3

    # Also move the arm to a joint target, in degrees.
    python3 src/lerobot/robots/bi_dobot_nova5_dh/dobot_dh_api.py \
        --robot-ip 192.168.5.102 --joint 270 0 90 0 -90 0

Notes:
    DH AG-95 register convention:
        0.0 = fully closed, 1.0 = fully open
    Dobot end-effector RS485 proxy flow:
        SetToolPower(1)
        SetToolMode(1, 1, identify)
        SetTool485(115200, "N", 1, identify)
        ModbusCreate("192.168.201.1", 60000, slave_id, isRTU=True)
"""

from __future__ import annotations

import argparse
import re
import time
from contextlib import suppress
from dataclasses import dataclass

from TCP_IP_Python_V4.dobot_api import DobotApiDashboard  # noqa: E402

REG_INITIALIZE = 0x0100
REG_FORCE = 0x0101
REG_POSITION = 0x0103
REG_INIT_STATE = 0x0200
REG_GRIP_STATE = 0x0201
REG_CUR_POS = 0x0202

INIT_TRIGGER = 0xA5
INIT_DONE = 1
DH_POS_OPEN = 1000

GRIP_IN_MOTION = 0
GRIP_REACHED = 1
GRIP_GRIPPED = 2
GRIP_DROPPED = 3


def parse_dobot_response(response: str) -> tuple[int, list[str]]:
    """Parse Dobot responses such as ``0,{1},RobotMode();``."""
    if not isinstance(response, str):
        raise RuntimeError(f"Invalid Dobot response type: {type(response).__name__}")
    match = re.match(r"\s*(-?\d+)\s*,\s*\{([^}]*)\}", response)
    if match is None:
        raise RuntimeError(f"Could not parse Dobot response: {response!r}")
    error_id = int(match.group(1))
    values = [value.strip() for value in match.group(2).split(",") if value.strip()]
    return error_id, values


def raise_if_error(response: str, command_name: str) -> list[str]:
    error_id, values = parse_dobot_response(response)
    if error_id == 0:
        return values
    raise RuntimeError(
        f"{command_name} failed with ErrorID {error_id}: {response.strip()}"
    )


@dataclass
class DobotDHConfig:
    # robot_ip: str = "192.168.5.102"
    robot_ip: str = "192.168.142.102"
    dashboard_port: int = 29999

    master_ip: str = "192.168.201.1"
    master_port: int = 60000
    tool_identify: int = 1

    slave_id: int = 1
    baudrate: int = 115200
    parity: str = "N"
    stop_bit: int = 1
    force: int = 30

    enable_robot: bool = True
    power_cycle_tool: bool = False
    tool_power_wait_s: float = 2.0


class DHGripper:
    """DH AG-95 helper using Dobot's end-effector RS485 Modbus proxy."""

    def __init__(self, dashboard: DobotApiDashboard, config: DobotDHConfig):
        if not 20 <= config.force <= 100:
            raise ValueError(f"force must be in [20, 100], got {config.force}.")
        self.dashboard = dashboard
        self.force = config.force
        self.modbus_index: int | None = None

        response = dashboard.ModbusCreate(
            config.master_ip,
            config.master_port,
            config.slave_id,
            True,
        )
        values = raise_if_error(response, "ModbusCreate")
        if not values:
            raise RuntimeError(
                f"ModbusCreate returned no master index: {response.strip()}"
            )
        self.modbus_index = int(values[0])
        print(f"ModbusCreate master_index={self.modbus_index}")

    def read_register(self, reg: int) -> int:
        if self.modbus_index is None:
            raise RuntimeError("DH gripper Modbus connection is closed.")
        response = self.dashboard.GetHoldRegs(self.modbus_index, reg, 1)
        values = raise_if_error(response, f"GetHoldRegs(0x{reg:04X})")
        if not values:
            raise RuntimeError(f"GetHoldRegs(0x{reg:04X}) returned no value.")
        return int(values[0])

    def write_register(self, reg: int, value: int) -> None:
        if self.modbus_index is None:
            raise RuntimeError("DH gripper Modbus connection is closed.")
        response = self.dashboard.SetHoldRegs(
            self.modbus_index,
            reg,
            1,
            "{" + str(value) + "}",
        )
        raise_if_error(response, f"SetHoldRegs(0x{reg:04X})")

    def initialize(self, timeout_s: float = 10.0, init_open: bool = True) -> None:
        state = self.read_init_state()
        if state != INIT_DONE:
            print("Triggering DH gripper initialization...")
            self.write_register(REG_INITIALIZE, INIT_TRIGGER)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self.read_init_state() == INIT_DONE:
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError(
                    f"DH gripper initialization timed out after {timeout_s:.1f}s."
                )

        self.set_force(self.force)
        if init_open:
            self.open(wait=True)
        print("DH gripper ready.")

    def read_init_state(self) -> int:
        return self.read_register(REG_INIT_STATE)

    def read_position(self) -> float:
        raw = self.read_register(REG_CUR_POS)
        raw = max(0, min(DH_POS_OPEN, raw))
        return raw / DH_POS_OPEN

    def read_grip_state(self) -> int:
        return self.read_register(REG_GRIP_STATE)

    def set_force(self, force: int) -> None:
        if not 20 <= force <= 100:
            raise ValueError(f"force must be in [20, 100], got {force}.")
        self.force = force
        self.write_register(REG_FORCE, force)

    def set_position(
        self, position: float, wait: bool = False, timeout_s: float = 10.0
    ) -> None:
        if not 0.0 <= position <= 1.0:
            raise ValueError(f"position must be in [0, 1], got {position}.")
        self.write_register(REG_POSITION, int(round(position * DH_POS_OPEN)))
        if wait:
            self.wait_until_stopped(timeout_s=timeout_s)

    def open(self, wait: bool = False) -> None:
        self.set_position(1.0, wait=wait)

    def close(self, wait: bool = False) -> None:
        self.set_position(0.0, wait=wait)

    def wait_until_stopped(self, timeout_s: float = 10.0) -> int:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.read_grip_state()
            if state in (GRIP_REACHED, GRIP_GRIPPED, GRIP_DROPPED):
                return state
            time.sleep(0.05)
        raise TimeoutError(f"DH gripper did not stop within {timeout_s:.1f}s.")

    def disconnect(self) -> None:
        if self.modbus_index is not None:
            with suppress(Exception):
                self.open(wait=False)
            with suppress(Exception):
                self.dashboard.ModbusClose(self.modbus_index)
            self.modbus_index = None


class DobotDHController:
    """Standalone controller for one Dobot Nova5 arm and one DH AG-95 gripper."""

    def __init__(self, config: DobotDHConfig):
        self.config = config
        self.dashboard: DobotApiDashboard | None = None
        self.gripper: DHGripper | None = None

    def connect(self) -> None:
        print(
            f"Connecting Dobot dashboard {self.config.robot_ip}:{self.config.dashboard_port}..."
        )
        self.dashboard = DobotApiDashboard(
            self.config.robot_ip, self.config.dashboard_port
        )

        if self.config.enable_robot:
            self.clear_error_if_needed()
            raise_if_error(self.dashboard.EnableRobot(), "EnableRobot")
            self.wait_until_ready(timeout_s=30.0)

        self.configure_tool_rs485()
        self.gripper = DHGripper(self.dashboard, self.config)
        self.gripper.initialize(init_open=True)

    def configure_tool_rs485(self) -> None:
        if self.dashboard is None:
            raise RuntimeError("Dashboard is not connected.")

        # if self.config.power_cycle_tool:
        #     print("Power-cycling end tool...")
        #     raise_if_error(
        #         self.dashboard.SetToolPower(0, self.config.tool_identify),
        #         "SetToolPower(0)",
        #     )
        #     time.sleep(0.5)

        # raise_if_error(
        #     self.dashboard.SetToolPower(1, self.config.tool_identify),
        #     "SetToolPower(1)",
        # )
        # time.sleep(self.config.tool_power_wait_s)

        raise_if_error(
            self.dashboard.SetToolMode(1, 1, self.config.tool_identify),
            "SetToolMode",
        )
        raise_if_error(
            self.dashboard.SetTool485(
                self.config.baudrate,
                self.config.parity,
                self.config.stop_bit,
                self.config.tool_identify,
            ),
            "SetTool485",
        )

    def robot_mode(self) -> int:
        if self.dashboard is None:
            raise RuntimeError("Dashboard is not connected.")
        values = raise_if_error(self.dashboard.RobotMode(), "RobotMode")
        if not values:
            raise RuntimeError("RobotMode returned no value.")
        return int(float(values[0]))

    def clear_error_if_needed(self) -> None:
        if self.dashboard is None:
            raise RuntimeError("Dashboard is not connected.")
        if self.robot_mode() == 9:
            print("Robot is in error mode, sending ClearError...")
            raise_if_error(self.dashboard.ClearError(), "ClearError")
            time.sleep(0.5)

    def wait_until_ready(self, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            mode = self.robot_mode()
            if mode == 5:
                print("Robot ready.")
                return
            if mode == 9:
                raise RuntimeError("Robot entered error mode. Check controller alarms.")
            time.sleep(0.1)
        raise TimeoutError(f"Robot did not become ready within {timeout_s:.1f}s.")

    def move_joint(
        self, joint_deg: list[float], vel: int = 30, wait: bool = True
    ) -> None:
        if self.dashboard is None:
            raise RuntimeError("Dashboard is not connected.")
        if len(joint_deg) != 6:
            raise ValueError(f"joint_deg must contain 6 values, got {len(joint_deg)}.")
        if not 1 <= vel <= 100:
            raise ValueError(f"vel must be in [1, 100], got {vel}.")

        print(f"MovJ joint={joint_deg}, vel={vel}")
        response = self.dashboard.MovJ(*[float(v) for v in joint_deg], 1, v=vel)
        raise_if_error(response, "MovJ(joint)")
        if wait:
            self.wait_until_ready(timeout_s=90.0)

    def open_gripper(self, wait: bool = True) -> None:
        if self.gripper is None:
            raise RuntimeError("Gripper is not connected.")
        self.gripper.open(wait=wait)

    def close_gripper(self, wait: bool = True) -> None:
        if self.gripper is None:
            raise RuntimeError("Gripper is not connected.")
        self.gripper.close(wait=wait)

    def disconnect(self) -> None:
        if self.gripper is not None:
            self.gripper.disconnect()
            self.gripper = None
        if self.dashboard is not None:
            with suppress(Exception):
                self.dashboard.Stop()
            self.dashboard.close()
            self.dashboard.socket_dobot = 0
            self.dashboard = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Dobot Nova5 + DH AG-95 simple controller."
    )
    parser.add_argument("--robot-ip", default="192.168.5.102")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--master-ip", default="192.168.201.1")
    parser.add_argument("--master-port", type=int, default=60000)
    parser.add_argument("--identify", type=int, choices=(1, 2), default=1)
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--parity", choices=("N", "E", "O"), default="N")
    parser.add_argument("--stop-bit", type=int, choices=(1, 2), default=1)
    parser.add_argument("--force", type=int, default=30)
    parser.add_argument("--power-cycle-tool", action="store_true")
    parser.add_argument("--no-enable", action="store_true")
    parser.add_argument(
        "--cycles", type=int, default=1, help="Gripper open/close cycles."
    )
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument(
        "--joint",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Optional joint target in degrees. If omitted, the arm will not move.",
    )
    parser.add_argument("--joint-vel", type=int, default=30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = DobotDHConfig(
        robot_ip=args.robot_ip,
        dashboard_port=args.dashboard_port,
        master_ip=args.master_ip,
        master_port=args.master_port,
        tool_identify=args.identify,
        slave_id=args.slave_id,
        baudrate=args.baudrate,
        parity=args.parity,
        stop_bit=args.stop_bit,
        force=args.force,
        enable_robot=not args.no_enable,
        power_cycle_tool=args.power_cycle_tool,
    )

    controller = DobotDHController(config)
    try:
        controller.connect()
        if args.joint is not None:
            controller.move_joint(list(args.joint), vel=args.joint_vel)

        for index in range(1, args.cycles + 1):
            print(f"Gripper cycle {index}/{args.cycles}: close")
            controller.close_gripper(wait=True)
            time.sleep(args.hold_s)
            print(f"Gripper cycle {index}/{args.cycles}: open")
            controller.open_gripper(wait=True)
            time.sleep(args.hold_s)

        if controller.gripper is not None:
            print(f"Current gripper position: {controller.gripper.read_position():.3f}")
    finally:
        controller.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
