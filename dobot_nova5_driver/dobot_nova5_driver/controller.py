import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .TCP_IP_Python_V4.dobot_api import DobotApiDashboard, DobotApiFeedBack


MM_PER_METER = 1000.0
ROBOT_MODE_TEXT = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWEROFF",
    4: "DISABLED",
    5: "ENABLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "SINGLE_MOVE",
    9: "ERROR",
    10: "PAUSE",
    11: "COLLISION",
}


@dataclass
class TcpPose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


class DobotNova5Controller:
    def __init__(
        self,
        robot_ip: str,
        dashboard_port: int = 29999,
        feedback_port: int = 30004,
        startup_joint: Optional[list[float]] = None,
        startup_speed: int = 20,
    ) -> None:
        self.robot_ip = robot_ip
        self.dashboard_port = dashboard_port
        self.feedback_port = feedback_port
        self.startup_joint = startup_joint or [270.0, 0.0, 90.0, 0.0, -90.0, 0.0]
        self.startup_speed = startup_speed

        self.dashboard: DobotApiDashboard | None = None
        self.feedback: DobotApiFeedBack | None = None
        self.feedback_data = None
        self._feedback_thread: threading.Thread | None = None
        self._stop_feedback = threading.Event()
        self._command_lock = threading.Lock()

    def connect(self, go_to_start: bool = False, auto_enable: bool = False) -> None:
        self.dashboard = DobotApiDashboard(self.robot_ip, self.dashboard_port)
        self.feedback = DobotApiFeedBack(self.robot_ip, self.feedback_port)
        self._stop_feedback.clear()
        self._feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self._feedback_thread.start()
        self._wait_for_feedback()

        if auto_enable:
            if self.robot_mode == 9:
                self._raise_if_error(self.dashboard.ClearError(), "ClearError")
                self._wait_until(lambda: self.robot_mode != 9, timeout_s=10.0, detail="clear robot error")

            self._raise_if_enable_issue(self.dashboard.EnableRobot())
            self._wait_until(lambda: self.robot_mode == 5, timeout_s=30.0, detail="robot enable ready")

            if go_to_start:
                self.move_joint(self.startup_joint, speed=self.startup_speed)

    def disconnect(self) -> None:
        self._stop_feedback.set()
        if self.dashboard is not None:
            self.dashboard.close()
            self.dashboard = None
        if self.feedback is not None:
            self.feedback.close()
            self.feedback = None

    @property
    def is_connected(self) -> bool:
        return self.dashboard is not None and self.feedback is not None

    @property
    def robot_mode(self) -> int:
        if self.feedback_data is None:
            return -1
        return int(self.feedback_data["RobotMode"][0])

    def _tcp_pose_from_values(self, values) -> TcpPose:
        if len(values) < 6:
            raise RuntimeError("Pose data did not contain 6 values")
        return TcpPose(
            x=float(values[0]) / MM_PER_METER,
            y=float(values[1]) / MM_PER_METER,
            z=float(values[2]) / MM_PER_METER,
            rx=float(values[3]),
            ry=float(values[4]),
            rz=float(values[5]),
        )

    def feedback_tool_index(self) -> int:
        if self.feedback_data is None:
            raise RuntimeError("Feedback not ready")
        return int(self.feedback_data["Tool"][0])

    def feedback_user_index(self) -> int:
        if self.feedback_data is None:
            raise RuntimeError("Feedback not ready")
        return int(self.feedback_data["User"][0])

    def current_tcp_pose(self, user_index: Optional[int] = None, tool_index: Optional[int] = None) -> TcpPose:
        if self.feedback_data is None:
            raise RuntimeError("Feedback not ready")
        if (user_index is None) != (tool_index is None):
            raise ValueError("user_index and tool_index must be provided together")
        if user_index is not None and tool_index is not None:
            if self.feedback_user_index() == int(user_index) and self.feedback_tool_index() == int(tool_index):
                tcp = self.feedback_data["ToolVectorActual"][0]
                return self._tcp_pose_from_values(tcp)
            return self.read_pose(user_index=user_index, tool_index=tool_index)
        tcp = self.feedback_data["ToolVectorActual"][0]
        return self._tcp_pose_from_values(tcp)

    def current_joint(self) -> list[float]:
        if self.feedback_data is None:
            raise RuntimeError("Feedback not ready")
        joints = self.feedback_data["QActual"][0]
        return [float(v) for v in joints]

    def current_command_id(self) -> int:
        if self.feedback_data is None:
            return -1
        return int(self.feedback_data["CurrentCommandId"][0])

    def robot_mode_text(self) -> str:
        return ROBOT_MODE_TEXT.get(self.robot_mode, str(self.robot_mode))

    def power_on(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.PowerOn(), "PowerOn")

    def enable_robot(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_enable_issue(self.dashboard.EnableRobot())
        self._wait_until(lambda: self.robot_mode == 5, timeout_s=30.0, detail="robot enable ready")

    def disable_robot(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.DisableRobot(), "DisableRobot")

    def clear_error(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.ClearError(), "ClearError")

    def reset_robot(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.ResetRobot(), "ResetRobot")

    def stop_motion(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.Stop(), "Stop")
        self._wait_until(lambda: self.robot_mode != 10, timeout_s=5.0, detail="robot exit pause after stop")

    def wait_until_idle(self, timeout_s: float = 5.0) -> None:
        self._wait_until(lambda: self.robot_mode == 5, timeout_s=timeout_s, detail="robot idle")

    def pause_motion(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.Pause(), "Pause")

    def continue_motion(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.Continue(), "Continue")

    def set_speed_factor(self, speed: int) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.SpeedFactor(int(speed)), "SpeedFactor")

    def set_joint_profile(self, speed: Optional[int] = None, accel: Optional[int] = None) -> bool:
        self._ensure_dashboard()
        supported = True
        with self._command_lock:
            if accel is not None:
                if not self._raise_if_supported(self.dashboard.AccJ(int(accel)), "AccJ"):
                    supported = False
            if speed is not None:
                if not self._raise_if_supported(self.dashboard.VelJ(int(speed)), "VelJ"):
                    supported = False
        return supported

    def set_linear_profile(self, speed: Optional[int] = None, accel: Optional[int] = None) -> bool:
        self._ensure_dashboard()
        supported = True
        with self._command_lock:
            if accel is not None:
                if not self._raise_if_supported(self.dashboard.AccL(int(accel)), "AccL"):
                    supported = False
            if speed is not None:
                if not self._raise_if_supported(self.dashboard.VelL(int(speed)), "VelL"):
                    supported = False
        return supported

    def set_user_index(self, index: int) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.User(int(index)), "User")

    def set_tool_index(self, index: int) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.Tool(int(index)), "Tool")

    def start_drag(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.StartDrag(), "StartDrag")

    def stop_drag(self) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.StopDrag(), "StopDrag")

    def move_jog(self, axis_id: str, coord_type: int = 1, user: int = 0, tool: int = 0) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            if axis_id == "":
                response = self.dashboard.MoveJog("")
            else:
                response = self.dashboard.MoveJog(axis_id=axis_id, coordtype=coord_type, user=user, tool=tool)
            self._raise_if_error(
                response,
                f"MoveJog({axis_id})",
            )

    def read_pose(self, user_index: Optional[int] = None, tool_index: Optional[int] = None) -> TcpPose:
        self._ensure_dashboard()
        if (user_index is None) != (tool_index is None):
            raise ValueError("user_index and tool_index must be provided together")
        with self._command_lock:
            if user_index is None and tool_index is None:
                values = self._raise_if_error(self.dashboard.GetPose(), "GetPose")
            else:
                values = self._raise_if_error(
                    self.dashboard.GetPose(user=int(user_index), tool=int(tool_index)),
                    f"GetPose(user={int(user_index)}, tool={int(tool_index)})",
                )
        return self._tcp_pose_from_values(values)

    def move_to_startup(self) -> None:
        if self.robot_mode == 10:
            self.stop_motion()
        self.move_joint(self.startup_joint, speed=self.startup_speed)

    def move_joint(self, joints_deg: list[float], speed: int = 20) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(self.dashboard.SpeedFactor(int(speed)), "SpeedFactor")
            response = self.dashboard.MovJ(
                float(joints_deg[0]),
                float(joints_deg[1]),
                float(joints_deg[2]),
                float(joints_deg[3]),
                float(joints_deg[4]),
                float(joints_deg[5]),
                1,
                v=int(speed),
            )
        command_id = self._require_command_id(response, "MovJ")
        self._wait_for_command(command_id, timeout_s=60.0)

    def move_joint_tcp(
        self,
        pose_m_deg: TcpPose,
        speed: int = 20,
        accel: Optional[int] = None,
        cp: Optional[int] = None,
        user_index: Optional[int] = None,
        tool_index: Optional[int] = None,
    ) -> None:
        self._ensure_dashboard()
        if (user_index is None) != (tool_index is None):
            raise ValueError("user_index and tool_index must be provided together")
        with self._command_lock:
            response = self.dashboard.MovJ(
                pose_m_deg.x * MM_PER_METER,
                pose_m_deg.y * MM_PER_METER,
                pose_m_deg.z * MM_PER_METER,
                pose_m_deg.rx,
                pose_m_deg.ry,
                pose_m_deg.rz,
                0,
                user=-1 if user_index is None else int(user_index),
                tool=-1 if tool_index is None else int(tool_index),
                a=-1 if accel is None else int(accel),
                v=int(speed),
                cp=-1 if cp is None else int(cp),
            )
        command_id = self._require_command_id(response, "MovJ")
        self._wait_for_command(command_id, timeout_s=60.0)

    def move_linear_tcp(
        self,
        pose_m_deg: TcpPose,
        speed: int = 10,
        accel: Optional[int] = None,
        user_index: Optional[int] = None,
        tool_index: Optional[int] = None,
    ) -> None:
        self._ensure_dashboard()
        if (user_index is None) != (tool_index is None):
            raise ValueError("user_index and tool_index must be provided together")
        with self._command_lock:
            response = self.dashboard.MovL(
                pose_m_deg.x * MM_PER_METER,
                pose_m_deg.y * MM_PER_METER,
                pose_m_deg.z * MM_PER_METER,
                pose_m_deg.rx,
                pose_m_deg.ry,
                pose_m_deg.rz,
                0,
                user=-1 if user_index is None else int(user_index),
                tool=-1 if tool_index is None else int(tool_index),
                a=-1 if accel is None else int(accel),
                v=int(speed),
            )
        command_id = self._require_command_id(response, "MovL")
        self._wait_for_command(command_id, timeout_s=60.0)

    def inverse_kinematics(
        self,
        pose_m_deg: TcpPose,
        user_index: Optional[int] = None,
        tool_index: Optional[int] = None,
        joint_near: Optional[list[float]] = None,
    ) -> list[float]:
        self._ensure_dashboard()
        joint_near_arg = ""
        use_joint_near = -1
        if joint_near is not None:
            if len(joint_near) != 6:
                raise ValueError(f"joint_near must contain 6 values, got {len(joint_near)}")
            joint_near_arg = "{" + ",".join(f"{float(value):.6f}" for value in joint_near) + "}"
            use_joint_near = 1
        with self._command_lock:
            response = self.dashboard.InverseKin(
                pose_m_deg.x * MM_PER_METER,
                pose_m_deg.y * MM_PER_METER,
                pose_m_deg.z * MM_PER_METER,
                pose_m_deg.rx,
                pose_m_deg.ry,
                pose_m_deg.rz,
                user=-1 if user_index is None else int(user_index),
                tool=-1 if tool_index is None else int(tool_index),
                useJointNear=use_joint_near,
                JointNear=joint_near_arg,
            )
        values = self._raise_if_error(response, "InverseKin")
        if len(values) < 6:
            raise RuntimeError(f"InverseKin did not return 6 joint values: {response.strip()}")
        return [float(value) for value in values[:6]]

    def rel_move_tool_joint(
        self,
        offset_pose_m_deg: TcpPose,
        speed: int = 20,
        accel: Optional[int] = None,
        cp: Optional[int] = None,
        user_index: Optional[int] = None,
        tool_index: Optional[int] = None,
    ) -> None:
        self._ensure_dashboard()
        if (user_index is None) != (tool_index is None):
            raise ValueError("user_index and tool_index must be provided together")
        with self._command_lock:
            response = self.dashboard.RelMovJTool(
                offset_pose_m_deg.x * MM_PER_METER,
                offset_pose_m_deg.y * MM_PER_METER,
                offset_pose_m_deg.z * MM_PER_METER,
                offset_pose_m_deg.rx,
                offset_pose_m_deg.ry,
                offset_pose_m_deg.rz,
                user=-1 if user_index is None else int(user_index),
                tool=-1 if tool_index is None else int(tool_index),
                a=-1 if accel is None else int(accel),
                v=int(speed),
                cp=-1 if cp is None else int(cp),
            )
        command_id = self._require_command_id(response, "RelMovJTool")
        self._wait_for_command(command_id, timeout_s=60.0)

    def rel_move_user_joint(
        self,
        offset_pose_m_deg: TcpPose,
        speed: int = 20,
        accel: Optional[int] = None,
        cp: Optional[int] = None,
        user_index: Optional[int] = None,
        tool_index: Optional[int] = None,
    ) -> None:
        self._ensure_dashboard()
        if (user_index is None) != (tool_index is None):
            raise ValueError("user_index and tool_index must be provided together")
        with self._command_lock:
            response = self.dashboard.RelMovJUser(
                offset_pose_m_deg.x * MM_PER_METER,
                offset_pose_m_deg.y * MM_PER_METER,
                offset_pose_m_deg.z * MM_PER_METER,
                offset_pose_m_deg.rx,
                offset_pose_m_deg.ry,
                offset_pose_m_deg.rz,
                user=-1 if user_index is None else int(user_index),
                tool=-1 if tool_index is None else int(tool_index),
                a=-1 if accel is None else int(accel),
                v=int(speed),
                cp=-1 if cp is None else int(cp),
            )
        command_id = self._require_command_id(response, "RelMovJUser")
        self._wait_for_command(command_id, timeout_s=60.0)

    def relative_point_user(
        self,
        base_pose_m_deg: TcpPose,
        offset_pose_m_deg: TcpPose,
    ) -> TcpPose:
        self._ensure_dashboard()
        with self._command_lock:
            response = self.dashboard.RelPointUser(
                0,
                base_pose_m_deg.x * MM_PER_METER,
                base_pose_m_deg.y * MM_PER_METER,
                base_pose_m_deg.z * MM_PER_METER,
                base_pose_m_deg.rx,
                base_pose_m_deg.ry,
                base_pose_m_deg.rz,
                offset_pose_m_deg.x * MM_PER_METER,
                offset_pose_m_deg.y * MM_PER_METER,
                offset_pose_m_deg.z * MM_PER_METER,
                offset_pose_m_deg.rx,
                offset_pose_m_deg.ry,
                offset_pose_m_deg.rz,
            )
        values = self._raise_if_error(response, "RelPointUser")
        if len(values) < 6:
            raise RuntimeError(f"RelPointUser did not return 6 pose values: {response.strip()}")
        return self._tcp_pose_from_values(values[:6])

    def servo_tcp(self, pose_m_deg: TcpPose, duration_s: float = 0.1, aheadtime: float = 50.0, gain: float = 300.0) -> None:
        self._ensure_dashboard()
        with self._command_lock:
            self._raise_if_error(
                self.dashboard.ServoP(
                    pose_m_deg.x * MM_PER_METER,
                    pose_m_deg.y * MM_PER_METER,
                    pose_m_deg.z * MM_PER_METER,
                    pose_m_deg.rx,
                    pose_m_deg.ry,
                    pose_m_deg.rz,
                    t=duration_s,
                    aheadtime=aheadtime,
                    gain=gain,
                ),
                "ServoP",
            )

    def _feedback_loop(self) -> None:
        while not self._stop_feedback.is_set() and self.feedback is not None:
            try:
                packet = self.feedback.feedBackData()
                if packet is not None:
                    self.feedback_data = packet
            except Exception:
                time.sleep(0.05)

    def _wait_for_feedback(self, timeout_s: float = 3.0) -> None:
        self._wait_until(lambda: self.feedback_data is not None, timeout_s=timeout_s, detail="first feedback packet")

    def _wait_for_command(self, command_id: int, timeout_s: float) -> None:
        def done() -> bool:
            if self.feedback_data is None:
                return False
            current_command_id = int(self.feedback_data["CurrentCommandId"][0])
            mode = self.robot_mode
            if mode == 9:
                raise RuntimeError("Robot entered error mode while waiting for motion completion")
            if mode == 10:
                raise RuntimeError("Motion paused before command completion")
            return mode == 5 and current_command_id == command_id

        self._wait_until(done, timeout_s=timeout_s, detail=f"command {command_id} completion")

    def _raise_if_enable_issue(self, response: str) -> None:
        error_id, values = self._parse_response(response)
        if error_id == 0:
            return
        if values:
            _ = values
        mode = self.robot_mode
        if mode not in (5, 6, 7, 8):
            raise RuntimeError(f"EnableRobot failed: {response.strip()} robot_mode={mode}")

    def _require_command_id(self, response: str, command_name: str) -> int:
        values = self._raise_if_error(response, command_name)
        if not values:
            raise RuntimeError(f"{command_name} did not return command id: {response.strip()}")
        return int(float(values[0]))

    def _raise_if_error(self, response: str, command_name: str) -> list[str]:
        error_id, values = self._parse_response(response)
        if error_id != 0:
            raise RuntimeError(f"{command_name} failed: {response.strip()}")
        return values

    def _raise_if_supported(self, response: str, command_name: str) -> bool:
        error_id, values = self._parse_response(response)
        if error_id == 0:
            return True
        if error_id == -7:
            return False
        raise RuntimeError(f"{command_name} failed: {response.strip()}")

    def _parse_response(self, raw: str) -> tuple[int, list[str]]:
        if not isinstance(raw, str):
            raise RuntimeError(f"Invalid response type: {type(raw).__name__}")
        match = re.match(r"\s*(-?\d+)\s*,\s*\{([^}]*)\}", raw)
        if match is None:
            raise RuntimeError(f"Could not parse Dobot response: {raw!r}")
        error_id = int(match.group(1))
        values = [value.strip() for value in match.group(2).split(",") if value.strip()]
        return error_id, values

    def _wait_until(self, predicate, timeout_s: float, detail: str) -> None:
        start = time.time()
        while True:
            result = predicate()
            if result:
                return
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out while waiting for {detail}")
            time.sleep(0.05)

    def _ensure_dashboard(self) -> None:
        if self.dashboard is None:
            raise RuntimeError("Robot dashboard not connected")
