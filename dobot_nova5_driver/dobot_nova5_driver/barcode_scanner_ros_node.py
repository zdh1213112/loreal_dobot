"""Publish USB-keyboard scanner values to ROS 2 without controlling a status light."""

import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .scanner_input import ScannerInputReader, find_scanner_event_device


class BarcodeScannerNode(Node):
    def __init__(self) -> None:
        super().__init__("hid_barcode_scanner_node")
        self.declare_parameter("device_path", "")
        self.declare_parameter("grab_device", True)
        self.declare_parameter("output_topic", "/detected_barcodes")
        self.publisher = self.create_publisher(String, str(self.get_parameter("output_topic").value), 10)
        self.reader = None
        self.reader_lock = threading.Lock()
        self.last_missing_warning_time = 0.0
        self.last_permission_warning_time = 0.0
        self.last_logged_barcode = ""
        self.last_barcode_log_time = 0.0
        self.retry_timer = self.create_timer(2.0, self._ensure_reader)
        self._ensure_reader()

    def _ensure_reader(self) -> None:
        with self.reader_lock:
            if self.reader is not None and self.reader.running:
                return
            configured = str(self.get_parameter("device_path").value).strip()
            device_path = configured or find_scanner_event_device()
            if not device_path:
                now = time.monotonic()
                if now - self.last_missing_warning_time >= 10.0:
                    self.get_logger().warning(
                        "Barcode scanner is disconnected or not present; "
                        "reconnect it and the node will retry automatically"
                    )
                    self.last_missing_warning_time = now
                return
            real_device_path = os.path.realpath(device_path)
            if not os.path.exists(device_path) or not os.path.exists(real_device_path):
                now = time.monotonic()
                if now - self.last_missing_warning_time >= 10.0:
                    self.get_logger().warning(
                        f"Barcode scanner device link is stale: {device_path}; "
                        "reconnect it and the node will retry automatically"
                    )
                    self.last_missing_warning_time = now
                return
            if not os.access(real_device_path, os.R_OK):
                now = time.monotonic()
                if now - self.last_permission_warning_time >= 30.0:
                    self.get_logger().error(
                        f"Barcode scanner is present at {real_device_path} but is not readable. "
                        "Run the project script: sudo \"$(ros2 pkg prefix dobot_nova5_driver)/share/"
                        "dobot_nova5_driver/scripts/grant_cosmetic_barcode_access.sh\""
                    )
                    self.last_permission_warning_time = now
                return
            self.reader = ScannerInputReader(
                device_path,
                self._publish_scan,
                grab_device=bool(self.get_parameter("grab_device").value),
            )
            self.reader.start()
            self.last_missing_warning_time = 0.0
            self.last_permission_warning_time = 0.0
            self.get_logger().info(f"Barcode scanner started: {device_path}")

    def _publish_scan(self, value: str) -> None:
        msg = String()
        msg.data = value
        self.publisher.publish(msg)
        now = time.monotonic()
        if value != self.last_logged_barcode or now - self.last_barcode_log_time >= 2.0:
            self.get_logger().info(f"Barcode: {value}")
            self.last_logged_barcode = value
            self.last_barcode_log_time = now

    def destroy_node(self):
        if self.reader is not None:
            self.reader.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BarcodeScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
