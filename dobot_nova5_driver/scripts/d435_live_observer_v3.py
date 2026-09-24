#!/usr/bin/env python3
"""Record the running V3 D435 preview and barcode events without camera access.

This is a read-only ROS subscriber. It does not start/stop the turntable or
issue any robot command. Start the normal V3 workflow separately.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class D435LiveObserver(Node):
    def __init__(self, output: Path, save_interval_s: float):
        super().__init__("d435_live_observer_v3")
        self.output = output
        self.save_interval_s = save_interval_s
        self.started_at = time.monotonic()
        self.last_saved_at = float("-inf")
        self.frame_count = 0
        self.saved_count = 0
        self.event_count = 0
        self.last_frame_at = None
        self.intervals = []
        self.last_image = None
        self.frame_rows = []
        self.event_rows = []
        self.create_subscription(
            CompressedImage,
            "/vision_panel/d435_turntable/image/compressed",
            self.on_image,
            10,
        )
        for topic in (
            "/turntable_barcode_result",
            "/d435_continuous_barcode_result",
            "/turntable_barcode_camera_status",
        ):
            self.create_subscription(
                String,
                topic,
                lambda message, source=topic: self.on_event(source, message),
                10,
            )

    def elapsed(self):
        return round(time.monotonic() - self.started_at, 3)

    def on_image(self, message):
        now = time.monotonic()
        self.frame_count += 1
        if self.last_frame_at is not None:
            self.intervals.append(now - self.last_frame_at)
        self.last_frame_at = now
        self.last_image = bytes(message.data)
        if now - self.last_saved_at < self.save_interval_s:
            return
        self.last_saved_at = now
        filename = f"preview_{self.saved_count:04d}.jpg"
        (self.output / filename).write_bytes(self.last_image)
        image = cv2.imdecode(
            np.frombuffer(self.last_image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            return
        height, width = image.shape
        roi = image[int(height * 0.40) : int(height * 0.98),
                    int(width * 0.12) : int(width * 0.92)]
        row = {
            "elapsed_s": self.elapsed(),
            "image": filename,
            "width": width,
            "height": height,
            "roi_mean_gray": round(float(roi.mean()), 1),
            "roi_laplacian_var": round(
                float(cv2.Laplacian(roi, cv2.CV_64F).var()), 1
            ),
        }
        self.frame_rows.append(row)
        self.saved_count += 1
        print(json.dumps({"frame": row}, ensure_ascii=False), flush=True)

    def on_event(self, topic, message):
        row = {"elapsed_s": self.elapsed(), "topic": topic, "data": message.data}
        self.event_rows.append(row)
        self.event_count += 1
        if "success:" in message.data or "confirmed" in message.data:
            if self.last_image is not None:
                (self.output / f"event_{self.event_count:04d}.jpg").write_bytes(
                    self.last_image
                )
        print(json.dumps({"event": row}, ensure_ascii=False), flush=True)

    def save_results(self, duration_s):
        for name, rows, columns in (
            ("frames.csv", self.frame_rows,
             ["elapsed_s", "image", "width", "height", "roi_mean_gray",
              "roi_laplacian_var"]),
            ("events.csv", self.event_rows, ["elapsed_s", "topic", "data"]),
        ):
            with (self.output / name).open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        summary = {
            "duration_s": duration_s,
            "preview_frames": self.frame_count,
            "preview_fps": round(self.frame_count / max(duration_s, 0.001), 2),
            "median_preview_interval_ms": (
                round(float(np.median(self.intervals)) * 1000, 2)
                if self.intervals else None
            ),
            "saved_frames": self.saved_count,
            "barcode_results": sum(
                "success:" in row["data"] for row in self.event_rows
                if row["topic"] in (
                    "/turntable_barcode_result",
                    "/d435_continuous_barcode_result",
                )
            ),
        }
        (self.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--save-interval", type=float, default=0.10)
    parser.add_argument("--wait-for-preview", type=float, default=60.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    observer = D435LiveObserver(args.output, args.save_interval)
    try:
        print(
            "WAITING for V3 D435 preview: "
            f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')} "
            f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', 'default')}",
            flush=True,
        )
        wait_started = time.monotonic()
        next_diagnostic_at = wait_started + 3.0
        while (
            observer.frame_count == 0
            and time.monotonic() - wait_started < args.wait_for_preview
        ):
            rclpy.spin_once(observer, timeout_sec=0.1)
            if time.monotonic() >= next_diagnostic_at:
                next_diagnostic_at = time.monotonic() + 3.0
                preview_publishers = observer.count_publishers(
                    "/vision_panel/d435_turntable/image/compressed"
                )
                status_publishers = observer.count_publishers(
                    "/turntable_barcode_camera_status"
                )
                print(
                    f"WAITING: preview_publishers={preview_publishers}, "
                    f"status_publishers={status_publishers}, "
                    f"visible_d435_nodes="
                    f"{[name for name in observer.get_node_names() if 'd435' in name.lower()]}",
                    flush=True,
                )
        if observer.frame_count == 0:
            observer.save_results(time.monotonic() - wait_started)
            print(
                "NO PREVIEW: start the normal V3 launch, then check that its "
                "D435 node is ready and both terminals use the same ROS_DOMAIN_ID",
                flush=True,
            )
            return
        print(
            "READY: receiving D435 frames; run the normal V3 workflow "
            "and rotate the turntable now",
            flush=True,
        )
        started = time.monotonic()
        while time.monotonic() - started < args.duration:
            rclpy.spin_once(observer, timeout_sec=0.1)
        observer.save_results(time.monotonic() - started)
    finally:
        observer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
