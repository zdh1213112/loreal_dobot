"""Independent opt-in V5 indication process. Never connects to robot ports."""

import json
import threading
import time

from .alarm_light_v5 import (
    CYCLE_STATUS_TOPIC, CYCLE_TIMING_TOPIC, D435_READY_TOPIC, D435_STATUS_TOPIC,
    HANDOFF_STATE_TOPIC, VLA102_STATE_TOPIC, PROCESS_TOPICS, Indication, SoundPolicy, StationPolicy, finite_positive,
    light_pattern, pattern_commands,
)


class SerialLightWorker:
    """One serial owner; replace pending patterns instead of queuing history."""
    def __init__(self, port, slave_id=1, *, command_gap_s=0.05, serial_factory=None):
        pattern_commands((None, False), slave_id)  # validate before starting
        self.port, self.slave_id = port, slave_id
        self.gap = max(0.05, finite_positive(command_gap_s, "command_gap_s"))
        self.factory = serial_factory
        self.desired = self.applied = None
        self.error = ""
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._serial = None
        self._thread = threading.Thread(target=self._run, name="v5-alarm-serial", daemon=True)
        self._thread.start()

    def submit(self, pattern):
        with self._condition:
            self.desired = pattern
            self._condition.notify_all()

    def _open(self):
        factory = self.factory
        if factory is None:
            from serial import Serial
            factory = Serial
        self._serial = factory(port=self.port, baudrate=9600, bytesize=8, parity="E",
                               stopbits=1, timeout=0.1, write_timeout=0.2, exclusive=True)
        self.applied = None

    def _close(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None

    def _run(self):
        try:
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop.is_set() or self.desired != self.applied)
                    if self._stop.is_set():
                        break
                    pattern = self.desired
                if pattern is None or pattern == self.applied:
                    continue
                try:
                    if self._serial is None:
                        self._open()
                    self._serial.reset_input_buffer()
                    self.applied = None
                    complete = True
                    for command in pattern_commands(pattern, self.slave_id):
                        if self._stop.is_set() or self.desired != pattern:
                            complete = False
                            break
                        if self._serial.write(command) != len(command):
                            raise OSError("short alarm-light serial write")
                        # No tcdrain/flush and no reply wait in a ROS callback.
                        self._stop.wait(self.gap)
                    if complete:
                        with self._condition:
                            self.applied = pattern
                            self.error = ""
                            self._condition.notify_all()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.applied = None
                    self._close()
                    self._stop.wait(2.0)
        finally:
            self._close()

    def close(self, final_pattern, timeout=1.0):
        deadline = time.monotonic() + timeout
        self.submit(final_pattern)
        with self._condition:
            self._condition.wait_for(lambda: self.applied == final_pattern,
                                     timeout=max(0.0, deadline - time.monotonic()))
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))


def main(args=None):
    # Keeping ROS imports here lets protocol/policy/transport tests use fake IO.
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, String
    from std_srvs.srv import SetBool

    class AlarmLightNode(Node):
        def __init__(self):
            super().__init__("station_alarm_light_v5")
            for name, default in (("port", "/dev/ttyUSB0"), ("device_id", 1),
                                  ("buzzer_enabled", True), ("command_gap_s", 0.05),
                                  ("warning_beep_enabled", True), ("no_target_warning_count", 3)):
                self.declare_parameter(name, default)
            value = lambda name: self.get_parameter(name).value
            self.policy = StationPolicy(no_target_warning_count=int(value("no_target_warning_count")))
            self.sound = SoundPolicy(bool(value("buzzer_enabled")),
                                     warning_beep_enabled=bool(value("warning_beep_enabled")))
            self.output = self.create_publisher(String, "/station_v5/alarm_status",
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.create_subscription(String, CYCLE_STATUS_TOPIC, self.receive_cycle, 10)
            self.create_subscription(String, CYCLE_TIMING_TOPIC, self.receive_timing, 20)
            # Existing tiny JSON result only, not a point cloud/image stream.
            # Best effort/depth one adds no reliable backpressure to D405.
            self.create_subscription(String, HANDOFF_STATE_TOPIC, self.receive_handoff,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.create_subscription(String, VLA102_STATE_TOPIC, self.receive_vla102,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.create_subscription(String, D435_STATUS_TOPIC,
                lambda msg: self.policy.receive_d435_status(msg.data), 10)
            self.create_subscription(Bool, D435_READY_TOPIC,
                lambda msg: self.policy.receive_d435_ready(msg.data),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.create_service(SetBool, "/station_v5/alarm_acknowledge", self.acknowledge)
            self.writer = SerialLightWorker(str(value("port")), int(value("device_id")),
                                            command_gap_s=float(value("command_gap_s")))
            self.last = None
            self.last_log_key = None
            self.last_pattern = None
            self.last_indication = Indication("initializing", "initializing", "工作站正在初始化")
            self.last_status_at = float("-inf")
            self.warning_tone_timer = None
            self.create_timer(1.0, self.update)
            self.update()

        def receive_cycle(self, message):
            previous = self.policy.decide()
            self.policy.receive_cycle_status(message.data)
            if self.policy.decide() != previous:
                self.update(check_graph=False)

        def receive_timing(self, message):
            if len(message.data) > 16384:
                return
            try:
                previous = self.policy.decide()
                self.policy.receive_timing(json.loads(message.data))
                if self.policy.decide() != previous:
                    self.update(check_graph=False)
            except (ValueError, TypeError):
                pass

        def receive_handoff(self, message):
            if len(message.data) > 8192:
                return
            try:
                previous = self.policy.decide()
                self.policy.receive_handoff(json.loads(message.data))
                if self.policy.decide() != previous:
                    self.update(check_graph=False)
            except (ValueError, TypeError):
                pass

        def stop_warning_tone_timer(self):
            if self.warning_tone_timer is not None:
                self.warning_tone_timer.cancel()
                self.destroy_timer(self.warning_tone_timer)
                self.warning_tone_timer = None

        def receive_vla102(self, message):
            if len(message.data) > 8192:
                return
            try:
                previous = self.policy.decide()
                self.policy.receive_vla102(json.loads(message.data))
                if self.policy.decide() != previous:
                    self.update(check_graph=False)
            except (ValueError, TypeError):
                pass

        def finish_warning_tone(self):
            self.stop_warning_tone_timer()
            self.update(check_graph=False)

        def acknowledge(self, request, response):
            self.sound.acknowledge(request.data)
            self.update()
            response.success = True
            response.message = "仅改变蜂鸣器静音；不清除故障、不启动机械臂"
            return response

        def update(self, *, check_graph=True):
            now = time.monotonic()
            if check_graph:
                self.policy.observe_publishers(
                    {source: self.count_publishers(topic) for source, topic in PROCESS_TOPICS.items()}, now)
            indication = self.policy.decide(now)
            buzzer = self.sound.update(indication, now)
            if indication.warning and buzzer:
                if self.warning_tone_timer is None:
                    remaining = self.sound.WARNING_BEEP_S - (now - self.sound.warning_started_at)
                    self.warning_tone_timer = self.create_timer(max(0.01, remaining), self.finish_warning_tone)
            else:
                self.stop_warning_tone_timer()
            pattern = light_pattern(indication, acknowledged=self.sound.acknowledged, buzzer=buzzer)
            if pattern != self.last_pattern:
                self.writer.submit(pattern)
                self.last_pattern = pattern
            payload = dict(state=indication.state, code=indication.code, reason=indication.reason,
                           acknowledged=self.sound.acknowledged, buzzer=buzzer,
                           serial_error=self.writer.error,
                           command_applied=self.writer.applied == pattern)
            key = tuple(payload.items())
            if key != self.last or now - self.last_status_at >= 2.0:
                message = String()
                message.data = json.dumps(payload, ensure_ascii=False)
                self.output.publish(message)
                self.last_status_at = now
                self.last = key
            # Local send completion, periodic sound and ACK changes are not
            # new station events. Never print the same alarm every 3 seconds.
            log_key = (indication.state, indication.code, indication.reason, self.writer.error)
            if log_key != self.last_log_key:
                self.get_logger().info(f"[状态灯] {indication.reason}; serial_error={self.writer.error or 'none'}")
                self.last_log_key = log_key
            self.last_indication = indication

        def destroy_node(self):
            # Do not erase a known red fault on shutdown. Normal exit turns
            # outputs off; a killed process/device power loss cannot be fixed
            # by software without a device communication watchdog.
            final = (0x01, False) if self.last_indication.fault else (None, False)
            self.stop_warning_tone_timer()
            self.writer.close(final)
            return super().destroy_node()

    rclpy.init(args=args)
    node = None
    try:
        node = AlarmLightNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
