"""Optional GUI display only. No sampling, publishers, timers or robot calls."""

import json
import time


class AlarmLightDisplayMixin:
    def _configure_alarm_light_display(self):
        self.declare_parameter("alarm_light_enabled", False)
        self._alarm_light_enabled = bool(self.get_parameter("alarm_light_enabled").value)
        if not self._alarm_light_enabled:
            return
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        from std_srvs.srv import SetBool
        self._alarm_output_text = "状态灯正在连接"
        self._alarm_output_at = 0.0
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/station_v5/alarm_status", self._alarm_output_callback, qos)
        self._alarm_ack_client = self.create_client(SetBool, "/station_v5/alarm_acknowledge")

    def _alarm_output_callback(self, message):
        try:
            data = json.loads(message.data)
            reason = str(data.get("reason", "状态未知"))[:512]
            if data.get("serial_error"):
                reason = "状态灯通信失败：" + str(data["serial_error"])[:256] + "；" + reason
            elif not data.get("command_applied"):
                reason = "正在发送灯控命令；" + reason
            self._alarm_output_text = reason
            self._alarm_output_at = time.monotonic()
        except (ValueError, TypeError, AttributeError):
            pass

    def acknowledge_alarm_light(self):
        from std_srvs.srv import SetBool
        if self._alarm_ack_client.service_is_ready():
            request = SetBool.Request()
            request.data = True
            self._alarm_ack_client.call_async(request)
