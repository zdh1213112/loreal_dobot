"""V5 station indication policy and the existing KELONWEI wire protocol.

No ROS, Qt, camera, GPU or serial imports. Indication never authorizes motion.
The 0x05 addresses intentionally match alarm_light_controller/app.py; the
vendor's separate 0x01 channel numbering is different and is not used here.
"""

import math
import time
from dataclasses import dataclass


CYCLE_STATUS_TOPIC = "/cosmetic_pick_cycle_status"
CYCLE_TIMING_TOPIC = "/cosmetic_pick_cycle_timing"
D435_STATUS_TOPIC = "/turntable_barcode_camera_status"
D435_READY_TOPIC = "/turntable_barcode_camera_ready"
HANDOFF_STATE_TOPIC = "/d405_handoff_zone_state"
PROCESS_TOPICS = {"cycle": CYCLE_STATUS_TOPIC, "d405": "/d405_vision_result",
                  "d435": D435_STATUS_TOPIC}


@dataclass(frozen=True)
class Indication:
    state: str
    code: str
    reason: str

    @property
    def fault(self):
        return self.state == "fault"

    @property
    def warning(self):
        return self.state == "warning"


class StationPolicy:
    """Consume existing workflow events; elapsed inference time is not a fault."""
    def __init__(self, *, no_target_warning_count=3, clock_ns=None):
        if int(no_target_warning_count) < 1:
            raise ValueError("no_target_warning_count must be positive")
        self.base = Indication("initializing", "initializing", "等待工作站初始化")
        self.faults = {}
        self.warnings = {}
        self.no_target_warning_count = int(no_target_warning_count)
        self.no_target_streak = 0
        self.last_no_target_cycle = None
        self.clock_ns = clock_ns or time.time_ns
        self.grasp_observation_active = False
        self.handoff_barrier_ns = 0
        self.last_handoff_stamp_ns = 0
        self.operator_paused = False
        self.interlock_wait_stage = None
        self.seen_publishers = set()
        self.missing_since = {}

    def _run(self):
        if not self.operator_paused and self.interlock_wait_stage is None:
            self.base = Indication("running", "automatic_run", "自动运行（识别、扫码、上料等待、重试及双臂交接）")

    def _warn(self, code, reason):
        if not self.operator_paused:
            self.warnings[code] = Indication("warning", code, reason)

    def _reset_target_warnings(self):
        self.warnings.clear()
        self.no_target_streak = 0
        self.last_no_target_cycle = None
        self.grasp_observation_active = False

    def _begin_observation(self):
        if not self.operator_paused:
            self.grasp_observation_active = True
            self.handoff_barrier_ns = self.clock_ns()

    def _valid_grasp_target(self):
        self._reset_target_warnings()
        self._run()

    def _start(self):
        had_cycle_fault = self.faults.pop("cycle", None) is not None
        restarting = self.operator_paused or had_cycle_fault
        self.operator_paused = False
        if restarting:
            self.interlock_wait_stage = None
            self._reset_target_warnings()
        # A start request is published before the startup interlock is checked.
        # It is not evidence that motion/initialization has been permitted.
        if restarting or (self.base.state != "running" and self.interlock_wait_stage is None):
            self.base = Indication("initializing", "starting", "流程启动中，等待启动条件与 102 联锁放行")

    def _continue_after_interlock(self):
        self.interlock_wait_stage = None
        self._run()

    def receive_timing(self, data):
        if not isinstance(data, dict):
            return False
        event = data.get("event")
        if event == "cycle_start":
            self._start()
        elif event == "cycle_summary":
            outcome = data.get("outcome")
            if not isinstance(outcome, str):
                return False
            if outcome in ("fault", "safety_pause"):
                reason = "101 流程已因故障停止" if outcome == "fault" else "双臂安全保护已暂停流程"
                self.faults["cycle"] = Indication("fault", outcome, reason)
            elif outcome == "cancelled":
                self.operator_paused = True
                self.interlock_wait_stage = None
                self._reset_target_warnings()
                self.base = Indication("waiting", "operator_pause", "流程已暂停")
            elif outcome in ("grasp_retry", "bottom_regrasp_auto_rescan"):
                self._warn("grasp_recovery" if outcome == "grasp_retry" else "bottom_regrasp_recovery",
                    "抓取失败，正在返回初始位并重新识别/重抓" if outcome == "grasp_retry" else
                    "翻底重抓丢失视觉目标，正在返回初始位重新扫码")
            elif outcome == "no_target":
                cycle_id = data.get("cycle_id")
                if cycle_id and cycle_id != self.last_no_target_cycle:
                    self.last_no_target_cycle = cycle_id
                    self.no_target_streak += 1
                    if self.no_target_streak >= self.no_target_warning_count:
                        self._warn("target_missing", "已上料但连续多次未获取有效抓取目标，请检查摆放或遮挡；系统继续重试")
                self._run()
            elif outcome == "empty_turntable_reopened_for_102":
                self.grasp_observation_active = False
                self.warnings.pop("grasp_obstacle", None)
                self._warn("target_missing", "上料事件后多次检查确认空盘，已开放 102 重新上料")
            elif outcome == "success":
                self._reset_target_warnings()
                if str(data.get("cycle_id", "")).startswith("continuous-"):
                    self._run()
                elif not self.operator_paused and self.interlock_wait_stage is None:
                    self.base = Indication("ready", "ready", "本轮结束，等待操作")
            elif str(data.get("cycle_id", "")).startswith("continuous-"):
                self._run()
            elif not self.operator_paused and self.interlock_wait_stage is None:
                self.base = Indication("ready", "ready", "本轮结束，等待操作")
        elif event == "stage":
            # A recoverable failed stage is not a station fault. The final
            # cycle outcome/control interlock decides whether work stopped.
            clearance_stage = {
                "vision capture": "prevision_secondary_clearance",
                "offset-high": "secondary_y_clearance",
                "move-above": "secondary_y_clearance",
            }.get(self.interlock_wait_stage)
            if clearance_stage and data.get("stage") == clearance_stage and data.get("outcome") == "ok":
                self.interlock_wait_stage = None
            if data.get("stage") == "grasp_prepare" and data.get("outcome") == "ok":
                self._valid_grasp_target()
            self._run()
        else:
            return False
        return True

    def receive_cycle_status(self, text):
        text = str(text).strip()
        if text.startswith("FAULT:"):
            self.faults["cycle"] = Indication("fault", "fault", text[:512])
        elif text.startswith(("101 motion paused by 102", "102 TCP safety feedback failed while 101 was moving; stopping 101:")):
            self.faults["cycle"] = Indication("fault", "safety_pause", text[:512])
        elif text in ("cycle enabled", "continuous cycle started") or text.startswith((
                "single cycle request accepted;",
                "101/102 turntable zone recovered after verified 102 retreat",
                "101/102 Y clearance recovered; starting a fresh continuous cycle")):
            self._start()
        elif text in ("cycle stopped", "robot stopped; continuous cycle disabled") or text.startswith(
                "continuous cycle will stop after the current blocking motion"):
            self.operator_paused = True
            self.interlock_wait_stage = None
            self._reset_target_warnings()
            self.base = Indication("waiting", "operator_pause", "操作员已暂停流程")
        elif text.startswith("recoverable grasp failure at ") or ": grasp retry recovery completed;" in text:
            self.grasp_observation_active = False
            self._warn("grasp_recovery", "抓取失败，正在返回初始位并重新识别/重抓")
        elif text.startswith(("final bottom regrasp lost its D405 target;", "automatic bottom-regrasp rescan ")):
            self.grasp_observation_active = False
            self._warn("bottom_regrasp_recovery", "翻底重抓丢失视觉目标，正在返回初始位重新扫码")
        elif text.startswith(("accepted material event #", "new material event #")):
            # The bottom-regrasp retry uses a new logical event for the SAME
            # box. Only a real replacement placement starts a new warning run.
            if "automatic bottom-regrasp D405-loss retry" not in text:
                self._reset_target_warnings()
        elif text.startswith("material event #") and "was empty in repeated D405 checks;" in text:
            self.grasp_observation_active = False
            self.warnings.pop("grasp_obstacle", None)
            self._warn("target_missing", "上料事件后多次检查确认空盘，已开放 102 重新上料")
        elif text.startswith(("requesting fresh D405 target;", "turntable stopped; consuming fresh post-stop D405 tracking frames")) or (
                text.startswith("cycle ") and (": using post-stop D405 target" in text or ": detecting box after turntable stop" in text)):
            self._begin_observation()
        elif text.startswith(("D405 finger corridor BLOCKED;", "D405 handoff zone BLOCKED;")):
            self._warn("grasp_obstacle", "抓取目标周围或夹指通道存在障碍物点云，禁止抓取；等待通道清空")
        elif text.startswith("D405 handoff zone CLEAR; accepting stable grasp samples"):
            self.warnings.pop("grasp_obstacle", None)
        elif text.startswith("101 turntable grasp/flip phase armed;"):
            self._valid_grasp_target()
        elif text.startswith("waiting for 102 VLA interlock heartbeat before "):
            if not self.operator_paused:
                self.interlock_wait_stage = text.split(" before ", 1)[1]
                stage_name = {
                    "startup motion": "启动运动",
                    "single-cycle startup motion": "单轮启动运动",
                    "vision capture": "视觉采集",
                    "offset-high": "偏移高位接近",
                    "move-above": "抓取上方接近",
                }.get(self.interlock_wait_stage, self.interlock_wait_stage[:120])
                self.base = Indication("interlock_wait", "waiting_102_heartbeat",
                    f"{stage_name}前等待 102 VLA 心跳；101 保持静止，请启动或检查 102 VLA")
        elif text.startswith("102 TCP Y clearance CLEAR:"):
            if self.interlock_wait_stage and text.endswith(f"; continuing {self.interlock_wait_stage}"):
                self._continue_after_interlock()
        elif text == "moving to startup joint" or text.startswith("startup joint already reached "):
            if self.interlock_wait_stage in (None, "startup motion", "single-cycle startup motion"):
                self._continue_after_interlock()
        # Normal progress, missing targets, a GUI parameter update and scan
        # messages must neither create faults nor clear an existing one.

    def receive_handoff(self, data):
        """Use only the camera's existing, fresh, confirmed grasp prohibition.

        Ignore standby/rotation clouds and VERIFYING_BLOCKED single-frame hits.
        This only indicates an existing decision; it never authorizes a grasp.
        """
        if self.operator_paused or not self.grasp_observation_active or not isinstance(data, dict):
            return False
        stamp = data.get("stamp_ns")
        if type(stamp) is not int or stamp < self.handoff_barrier_ns or stamp <= self.last_handoff_stamp_ns:
            return False
        state = data.get("state")
        if state == "BLOCKED" and data.get("clear") is False:
            self.last_handoff_stamp_ns = stamp
            self._warn("grasp_obstacle", "抓取目标周围或夹指通道存在障碍物点云，禁止抓取；等待通道清空")
        elif state == "CLEAR" and data.get("clear") is True:
            self.last_handoff_stamp_ns = stamp
            self.warnings.pop("grasp_obstacle", None)
        else:
            return False
        return True

    def receive_d435_status(self, text):
        text = str(text).strip()
        if text.startswith("FAULT:"):
            self.faults["d435"] = Indication("fault", "d435_fault", text[:512])

    def receive_d435_ready(self, ready):
        if ready:
            self.faults.pop("d435", None)
            # One camera being ready does not imply station startup/interlock
            # completion. Keep initialization/wait indication until progress.

    def observe_publishers(self, counts, now):
        """ROS graph cache only: no image/robot queries or loop heartbeats.

        Silence during work never enters here. Only loss of a previously seen
        publisher, confirmed across five seconds, indicates a disconnected node.
        """
        for source in PROCESS_TOPICS:
            key = f"node:{source}"
            if counts.get(source, 0) > 0:
                self.seen_publishers.add(source)
                self.missing_since.pop(source, None)
                if self.faults.pop(key, None) is not None:
                    self.base = Indication("waiting", "component_reconnected", "节点已重新连接，等待流程恢复")
            elif source in self.seen_publishers:
                since = self.missing_since.setdefault(source, now)
                if now - since >= 5.0:
                    self.faults[key] = Indication("fault", key, f"{source} 节点发布端已退出或断开")

    def decide(self, _now=None):
        if self.faults:
            return next(iter(self.faults.values()))
        if self.operator_paused or self.interlock_wait_stage is not None:
            return self.base
        for code in ("grasp_obstacle", "bottom_regrasp_recovery", "grasp_recovery", "target_missing"):
            if code in self.warnings:
                return self.warnings[code]
        return self.base


class SoundPolicy:
    """Acknowledgement only silences indication; it never clears a fault."""
    WARNING_BEEP_S = 0.5

    def __init__(self, enabled=True, *, warning_beep_enabled=True):
        self.enabled = enabled
        self.warning_beep_enabled = warning_beep_enabled
        self.warning_active = False
        self.warning_started_at = 0.0
        self.fault_code = None
        self.started_at = 0.0
        self.acknowledged = False

    def update(self, indication, now):
        code = indication.code if indication.fault else None
        if code != self.fault_code:
            self.fault_code, self.started_at, self.acknowledged = code, now, False
        if code:
            self.warning_active = False
            return self.enabled and not self.acknowledged and (now - self.started_at) % 30.0 < 3.0
        if indication.warning:
            if not self.warning_active:
                self.warning_active, self.warning_started_at, self.acknowledged = True, now, False
            return (self.enabled and self.warning_beep_enabled and not self.acknowledged
                    and now - self.warning_started_at < self.WARNING_BEEP_S)
        if indication.state in ("running", "ready", "waiting", "off"):
            self.warning_active = False
        return False

    def acknowledge(self, muted):
        self.acknowledged = bool(muted)


def append_crc(payload):
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return payload + bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def coil_command(slave_id, address, enabled=True):
    if not 1 <= slave_id <= 247:
        raise ValueError("alarm light device ID must be 1..247 (no broadcast)")
    return append_crc(bytes((slave_id, 5, address >> 8, address & 0xFF, 0xFF if enabled else 0, 0)))


def light_pattern(indication, *, acknowledged=False, buzzer=False):
    # Device-native blinking; no Python blink loop or per-frame writes.
    address = {"initializing": 0x07, "ready": 0x12, "running": 0x03,
               "interlock_wait": 0x07, "warning": 0x07, "waiting": 0x02, "recovering": 0x07,
               "fault": 0x01 if acknowledged else 0x16, "off": None}[indication.state]
    return address, bool(buzzer)


def pattern_commands(pattern, slave_id):
    address, buzzer = pattern
    commands = [coil_command(slave_id, 0, False)]
    if address is not None:
        commands.append(coil_command(slave_id, address))
    if buzzer:
        # Yellow's one-shot tone needs an immediate ON, not a periodic
        # buzzer's possibly silent first half-cycle. The lamp node turns it
        # off after the short window; red retains the original pulsed sound.
        commands.append(coil_command(slave_id, 0x04 if address == 0x07 else 0x09))
    return commands


def finite_positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value
