#!/usr/bin/env python3
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


@dataclass
class PanelFrame:
    name: str
    image: np.ndarray
    stamp: float
    rect: Tuple[int, int, int, int]
    source_size: Tuple[int, int]


class VisionPanelViewer(Node):
    window_name = "Vision Panel"

    def __init__(self):
        super().__init__("vision_panel_viewer")
        self.declare_parameter("window_width", 1600)
        self.declare_parameter("tile_height", 480)
        self.declare_parameter("stale_timeout", 2.0)
        self.declare_parameter("jpeg_topics", [
            "barcode:/vision_panel/barcode/image/compressed:/vision_panel/barcode/event",
            "d405_local_rgb:/vision_panel/d405_local_rgb/image/compressed:/vision_panel/d405_local_rgb/event",
            "d405_local_cloud:/vision_panel/d405_local_cloud/image/compressed:/vision_panel/d405_local_cloud/event",
            "d435_global:/vision_panel/d435_global/image/compressed:/vision_panel/d435_global/event",
        ])

        self.window_width = max(640, int(self.get_parameter("window_width").value))
        self.tile_height = max(120, int(self.get_parameter("tile_height").value))
        self.stale_timeout = max(0.2, float(self.get_parameter("stale_timeout").value))
        topic_specs = [str(item) for item in self.get_parameter("jpeg_topics").value]

        self.frames: Dict[str, PanelFrame] = {}
        self.event_publishers: Dict[str, object] = {}
        self.active_panel: Optional[str] = None
        self.mouse_capture_panel: Optional[str] = None
        self.last_canvas: Optional[np.ndarray] = None
        self.quit_requested = False

        for spec in topic_specs:
            parts = spec.split(":")
            if len(parts) != 3:
                self.get_logger().warn(f"忽略无效 panel topic 配置: {spec}")
                continue
            name, image_topic, event_topic = parts
            self.create_subscription(CompressedImage, image_topic, lambda msg, panel=name: self.image_callback(panel, msg), 1)
            self.event_publishers[name] = self.create_publisher(String, event_topic, 10)
            self.get_logger().info(f"Panel 输入: {name} <- {image_topic}, event -> {event_topic}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.tile_height * 2)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.timer = self.create_timer(0.03, self.render_once)

    def image_callback(self, panel: str, msg: CompressedImage) -> None:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return
        now = time.monotonic()
        source_h, source_w = image.shape[:2]
        old_rect = self.frames[panel].rect if panel in self.frames else (0, 0, 0, 0)
        self.frames[panel] = PanelFrame(
            name=panel,
            image=image,
            stamp=now,
            rect=old_rect,
            source_size=(source_w, source_h),
        )

    def live_panels(self) -> List[PanelFrame]:
        now = time.monotonic()
        panels = [frame for frame in self.frames.values() if now - frame.stamp <= self.stale_timeout]
        order = ["d435_global", "d405_local_rgb", "d405_local_cloud", "barcode"]
        panels.sort(key=lambda frame: order.index(frame.name) if frame.name in order else 99)
        return panels

    def layout(self, panels: List[PanelFrame]) -> Tuple[np.ndarray, Dict[str, PanelFrame]]:
        if not panels:
            canvas = np.zeros((self.tile_height, self.window_width, 3), dtype=np.uint8)
            cv2.putText(canvas, "Waiting for vision panel streams...", (40, self.tile_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
            return canvas, {}

        count = len(panels)
        cols = 1 if count == 1 else 2
        rows = int(math.ceil(count / cols))
        cell_w = self.window_width // cols
        cell_h = self.tile_height
        canvas = np.zeros((rows * cell_h, self.window_width, 3), dtype=np.uint8)
        laid_out: Dict[str, PanelFrame] = {}

        for idx, frame in enumerate(panels):
            row = idx // cols
            col = idx % cols
            x0 = col * cell_w
            y0 = row * cell_h
            image_h, image_w = frame.image.shape[:2]
            scale = min(cell_w / image_w, cell_h / image_h)
            draw_w = max(1, int(image_w * scale))
            draw_h = max(1, int(image_h * scale))
            resized = cv2.resize(frame.image, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
            pad_x = x0 + (cell_w - draw_w) // 2
            pad_y = y0 + (cell_h - draw_h) // 2
            canvas[pad_y:pad_y + draw_h, pad_x:pad_x + draw_w] = resized

            rect = (pad_x, pad_y, pad_x + draw_w, pad_y + draw_h)
            frame.rect = rect
            laid_out[frame.name] = frame
            border_color = (0, 220, 255) if frame.name == self.active_panel else (80, 80, 80)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), border_color, 2)
            cv2.putText(canvas, frame.name, (x0 + 16, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2, cv2.LINE_AA)

        return canvas, laid_out

    def render_once(self) -> None:
        panels = self.live_panels()
        canvas, laid_out = self.layout(panels)
        self.frames.update(laid_out)
        self.last_canvas = canvas
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        if key == 27:
            self.quit_requested = True
            rclpy.shutdown()
            return
        if self.active_panel is not None:
            self.publish_key(self.active_panel, key)

    def panel_at(self, x: int, y: int) -> Optional[Tuple[PanelFrame, int, int]]:
        for frame in self.live_panels():
            x0, y0, x1, y1 = frame.rect
            if x0 <= x < x1 and y0 <= y < y1:
                src_w, src_h = frame.source_size
                src_x = int((x - x0) * src_w / max(1, x1 - x0))
                src_y = int((y - y0) * src_h / max(1, y1 - y0))
                return frame, max(0, min(src_w - 1, src_x)), max(0, min(src_h - 1, src_y))
        return None

    def mouse_callback(self, event, x, y, flags, param) -> None:
        hit = self.panel_at(x, y)
        if hit is None and self.mouse_capture_panel in self.frames:
            frame = self.frames[self.mouse_capture_panel]
            x0, y0, x1, y1 = frame.rect
            if x1 > x0 and y1 > y0:
                src_w, src_h = frame.source_size
                src_x = int((max(x0, min(x1 - 1, x)) - x0) * src_w / max(1, x1 - x0))
                src_y = int((max(y0, min(y1 - 1, y)) - y0) * src_h / max(1, y1 - y0))
                hit = frame, max(0, min(src_w - 1, src_x)), max(0, min(src_h - 1, src_y))
        if hit is None:
            return
        frame, src_x, src_y = hit
        if frame.name == "d405_local_cloud":
            self.active_panel = "d405_local_rgb"
            return

        self.active_panel = frame.name
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_capture_panel = frame.name
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_capture_panel = None
        publisher = self.event_publishers.get(frame.name)
        if publisher is None:
            return
        msg = String()
        msg.data = json.dumps({
            "type": "mouse",
            "event": int(event),
            "x": int(src_x),
            "y": int(src_y),
            "flags": int(flags),
        })
        publisher.publish(msg)

    def publish_key(self, panel: str, key: int) -> None:
        publisher = self.event_publishers.get(panel)
        if publisher is None:
            return
        msg = String()
        msg.data = json.dumps({"type": "key", "key": int(key)})
        publisher.publish(msg)

    def destroy_node(self) -> bool:
        cv2.destroyWindow(self.window_name)
        return super().destroy_node()


def main():
    rclpy.init()
    node = VisionPanelViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
