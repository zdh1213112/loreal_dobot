#!/usr/bin/env python3
"""Standalone, read-only D435 color/turntable-barcode comparison tool.

This program opens only the D435 color sensor. It never publishes a ROS topic,
toggles DO1, or commands either robot. Stop the V3 D435 node before running it:
librealsense normally permits only one owner of a camera stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dobot_nova5_driver.turntable_barcode_detector_v3 import (  # noqa: E402
    DEFAULT_BARCODE_MODEL,
    TurntableBarcodeDetector,
)


DEFAULT_SERIAL = "254322071102"
CSV_FIELDS = (
    "elapsed_s", "mode", "profile", "frame_number", "camera_timestamp_ms",
    "capture_interval_ms", "detect_ms", "candidate", "source", "confidence",
    "candidate_hits", "required_hits", "confirmed", "rect", "roi_mean",
    "roi_laplacian_var", "candidate_laplacian_var", "sensor_exposure",
    "sensor_gain", "frame_actual_exposure_us", "image",
)


@dataclass(frozen=True, order=True)
class ColorProfile:
    width: int
    height: int
    fps: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"


def parse_profile(value: str) -> ColorProfile:
    try:
        size, fps = value.lower().split("@", 1)
        width, height = size.split("x", 1)
        profile = ColorProfile(int(width), int(height), int(fps))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("profile must be WIDTHxHEIGHT@FPS") from exc
    if min(profile.width, profile.height, profile.fps) <= 0:
        raise argparse.ArgumentTypeError("profile dimensions and FPS must be positive")
    return profile


def parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if min(x, y) < 0 or min(width, height) <= 0:
        raise argparse.ArgumentTypeError("ROI needs nonnegative x/y and positive size")
    return x, y, width, height


def rect_iou(first, second) -> float:
    if first is None or second is None:
        return 0.0
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1, aw * ah + bw * bh - intersection)


def moving_stripe_match(first, second) -> bool:
    if first is None or second is None:
        return False
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    area_ratio = (bw * bh) / max(1, aw * ah)
    distance = math.hypot((bx + bw / 2) - (ax + aw / 2),
                          (by + bh / 2) - (ay + ah / 2))
    return 0.30 <= area_ratio <= 3.30 and distance <= max(
        90.0, 1.75 * max(aw, ah, bw, bh)
    )


class StableBarcode:
    """Mirror the V3 2-hit YOLO / 3-hit assist confirmation thresholds."""

    def __init__(self) -> None:
        self.source = ""
        self.value = ""
        self.last_rect = None
        self.first_rect = None
        self.last_at = 0.0
        self.hits = 0
        self.latched = False
        self.last_seen_at = 0.0

    def observe(self, hit, now: float) -> tuple[int, int, bool]:
        if hit is None:
            if self.latched and now - self.last_seen_at >= 1.0:
                self.latched = False
            if now - self.last_at > 0.70:
                self.hits = 0
                self.last_rect = None
            return self.hits, 0, False
        source = str(hit.source)
        family = "barcode_visual" if source == "scanner_pattern" or source.startswith(
            "yolo"
        ) else source
        if source == "scanner_pattern":
            required, max_gap = 3, 0.25
            spatial = rect_iou(self.last_rect, hit.rect) >= 0.20
        elif source == "moving_stripes":
            required, max_gap = 3, 0.70
            spatial = moving_stripe_match(self.last_rect, hit.rect)
        else:
            required, max_gap = 2, 0.70
            spatial = rect_iou(self.last_rect, hit.rect) >= 0.15
        same = (family == self.source and hit.value == self.value
                and now - self.last_at <= max_gap and spatial)
        self.hits = self.hits + 1 if same else 1
        if not same:
            self.first_rect = tuple(hit.rect)
        self.source, self.value = family, str(hit.value)
        self.last_rect, self.last_at = tuple(hit.rect), now
        self.last_seen_at = now
        movement_ok = True
        if source == "moving_stripes" and self.first_rect is not None:
            fx, _, fw, _ = self.first_rect
            x, _, width, _ = hit.rect
            movement_ok = abs((x + width / 2) - (fx + fw / 2)) >= max(
                10.0, width * 0.10
            )
        confirmed = self.hits >= required and movement_ok and not self.latched
        if confirmed:
            self.latched = True
            self.hits = 0
            self.last_rect = None
            self.first_rect = None
        return (required if confirmed else self.hits), required, confirmed


def option_range(sensor, option):
    if not sensor.supports(option):
        return None
    value = sensor.get_option_range(option)
    return {"min": value.min, "max": value.max,
            "step": value.step, "default": value.default}


def set_option(sensor, option, requested: float, label: str) -> float:
    limits = option_range(sensor, option)
    if limits is None:
        raise RuntimeError(f"camera does not support {label}")
    value = max(limits["min"], min(limits["max"], float(requested)))
    if limits["step"] > 0:
        value = limits["min"] + round(
            (value - limits["min"]) / limits["step"]
        ) * limits["step"]
    sensor.set_option(option, value)
    actual = float(sensor.get_option(option))
    if abs(actual - requested) > 0.01:
        print(f"OPTION {label}: requested={requested}, actual={actual} "
              f"(range={limits})", flush=True)
    return actual


def find_device(serial: str):
    devices = list(rs.context().query_devices())
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    found = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"D435 serial {serial} not found; connected serials={found}")


def color_profiles(device) -> list[ColorProfile]:
    profiles = set()
    for sensor in device.query_sensors():
        for stream in sensor.get_stream_profiles():
            if stream.stream_type() != rs.stream.color or stream.format() != rs.format.bgr8:
                continue
            video = stream.as_video_stream_profile()
            profiles.add(ColorProfile(video.width(), video.height(), stream.fps()))
    return sorted(profiles, key=lambda item: (item.width * item.height,
                                               item.width, item.height, item.fps))


def color_sensor_from_profile(pipeline_profile):
    device = pipeline_profile.get_device()
    try:
        return device.first_color_sensor()
    except (AttributeError, RuntimeError):
        for sensor in device.query_sensors():
            if any(profile.stream_type() == rs.stream.color
                   for profile in sensor.get_stream_profiles()):
                return sensor
    raise RuntimeError("D435 color sensor not found")


def read_option(sensor, option):
    try:
        return float(sensor.get_option(option)) if sensor.supports(option) else None
    except RuntimeError:
        return None


def frame_exposure_us(frame):
    try:
        key = rs.frame_metadata_value.actual_exposure
        return int(frame.get_frame_metadata(key)) if frame.supports_frame_metadata(key) else None
    except (AttributeError, RuntimeError):
        return None


def color_frame_as_bgr(frame, pixel_format):
    """Keep native mode format-independent without setting a camera format."""

    data = np.asanyarray(frame.get_data())
    if pixel_format == rs.format.bgr8:
        return data
    if pixel_format == rs.format.rgb8:
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if pixel_format == rs.format.yuyv:
        return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_YUY2)
    if pixel_format == rs.format.uyvy:
        return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_UYVY)
    if pixel_format == rs.format.mjpeg:
        decoded = cv2.imdecode(np.frombuffer(frame.get_data(), np.uint8),
                               cv2.IMREAD_COLOR)
        if decoded is not None:
            return decoded
    raise RuntimeError(f"native color format {pixel_format} cannot be converted to BGR; "
                       "rerun with --mode auto or manual and an explicit BGR8 profile")


def detail_roi_for_profile(width: int, height: int) -> tuple[int, int, int, int]:
    """Keep the 1280x720 V3 crop size near the same physical image center."""

    crop_width = min(760, width)
    crop_height = min(440, height)
    center_x = width * (280 + 760 / 2) / 1280
    center_y = height * (220 + 440 / 2) / 720
    left = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, round(center_y - crop_height / 2)))
    return left, top, crop_width, crop_height


class QualityLab:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.profiles = color_profiles(find_device(args.serial))
        if not self.profiles:
            raise RuntimeError("D435 reports no BGR8 color profiles")
        if args.mode != "native" and args.profile not in self.profiles:
            raise RuntimeError(f"unsupported BGR8 profile {args.profile}; use --list-profiles")
        self.mode = args.mode
        self.profile = args.profile
        self.exposure = float(args.exposure)
        self.gain = float(args.gain)
        self.pipeline = None
        self.sensor = None
        self.color_format = None
        self.original_options = {}
        self.frame_count = 0
        self.candidate_count = 0
        self.confirmed_count = 0
        self.saved_count = 0
        self.detect_ms = []
        self.by_setting = {}
        self.started_at = time.monotonic()
        self.last_frame_at = None
        self.last_save_at = 0.0
        self.allowed_roi = args.roi
        self.drag_origin = None
        self.drag_cursor = None
        self.verifier = StableBarcode()
        self.detector = TurntableBarcodeDetector(
            model_path=args.model,
            confidence=args.confidence,
            image_size=640,
            scanner_assist=True,
            inference_provider=args.provider,
            detail_roi=(280, 220, 760, 440),
            full_frame_interval=6,
            yolo_min_candidate_area_ratio=0.01,
            moving_stripes_enabled=True,
        )
        print("Loading the same barcode detector as V3...", flush=True)
        self.detector.load_model()
        print(f"Detector provider={self.detector.active_provider}; "
              f"confidence={self.detector.confidence:.2f}", flush=True)

    def start_camera(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.args.serial)
        if self.mode == "native":
            # No profile, exposure, gain, image-quality or AE options are set.
            config.enable_stream(rs.stream.color)
        else:
            config.enable_stream(rs.stream.color, self.profile.width,
                                 self.profile.height, rs.format.bgr8, self.profile.fps)
        started = False
        sensor = None
        try:
            active = pipeline.start(config)
            started = True
            sensor = color_sensor_from_profile(active)
            self.original_options = {
                option: read_option(sensor, option)
                for option in (
                    rs.option.enable_auto_exposure,
                    rs.option.auto_exposure_priority,
                    rs.option.exposure,
                    rs.option.gain,
                    rs.option.sharpness,
                )
            }
            if self.mode == "auto":
                set_option(sensor, rs.option.enable_auto_exposure, 1, "auto_exposure")
                if sensor.supports(rs.option.auto_exposure_priority):
                    set_option(sensor, rs.option.auto_exposure_priority, 0,
                               "auto_exposure_priority")
            elif self.mode == "manual":
                set_option(sensor, rs.option.enable_auto_exposure, 0, "auto_exposure")
                self.exposure = set_option(sensor, rs.option.exposure,
                                           self.exposure, "exposure")
                self.gain = set_option(sensor, rs.option.gain, self.gain, "gain")
            if self.mode != "native" and self.args.sharpness is not None:
                set_option(sensor, rs.option.sharpness, self.args.sharpness,
                           "sharpness")
            actual = active.get_stream(rs.stream.color).as_video_stream_profile()
            self.profile = ColorProfile(actual.width(), actual.height(), actual.fps())
            self.color_format = actual.format()
            self.detector.detail_roi = detail_roi_for_profile(
                self.profile.width, self.profile.height
            )
            self.detector._yolo_frame_index = 0
            self.pipeline, self.sensor = pipeline, sensor
            self.last_frame_at = None
            self.verifier = StableBarcode()
            self.detector._previous_stripe_gray = None
            for _ in range(5):
                pipeline.wait_for_frames(1000)
            print(f"CAMERA mode={self.mode}, actual={self.profile}, "
                  f"format={self.color_format}, detail_roi={self.detector.detail_roi}, "
                  f"AE={read_option(sensor, rs.option.enable_auto_exposure)}, "
                  f"exposure={read_option(sensor, rs.option.exposure)}, "
                  f"gain={read_option(sensor, rs.option.gain)}", flush=True)
        except Exception:
            if started:
                if self.mode != "native":
                    self.restore_camera_options(sensor)
                pipeline.stop()
            raise

    def restore_camera_options(self, sensor):
        """Leave production camera settings as they were before this test."""

        if sensor is None or not self.original_options:
            return
        auto_option = rs.option.enable_auto_exposure
        restore_order = (
            rs.option.auto_exposure_priority, rs.option.exposure,
            rs.option.gain, rs.option.sharpness,
        )
        try:
            if self.original_options.get(auto_option) is not None:
                set_option(sensor, auto_option, 0, "restore_auto_exposure_off")
            for option in restore_order:
                value = self.original_options.get(option)
                if value is not None:
                    set_option(sensor, option, value, f"restore_{option}")
            original_auto = self.original_options.get(auto_option)
            if original_auto is not None:
                set_option(sensor, auto_option, original_auto,
                           "restore_auto_exposure")
        except RuntimeError as exc:
            print(f"WARNING: camera option restoration failed: {exc}",
                  file=sys.stderr, flush=True)

    def stop_camera(self):
        if self.pipeline is not None:
            try:
                if self.mode != "native":
                    self.restore_camera_options(self.sensor)
            finally:
                self.pipeline.stop()
                self.pipeline = None
                self.sensor = None
                self.color_format = None
                self.original_options = {}

    def restart_profile(self, new_profile: ColorProfile):
        old_profile = self.profile
        self.allowed_roi = None
        print("ROI cleared on profile change; draw a new ROI for the new resolution",
              flush=True)
        self.stop_camera()
        self.profile = new_profile
        try:
            self.start_camera()
        except Exception as exc:
            print(f"PROFILE {new_profile} failed: {exc}; restoring {old_profile}",
                  flush=True)
            self.profile = old_profile
            self.start_camera()

    def next_resolution(self):
        if self.mode == "native":
            print("Native has no explicit profile; restart with --mode auto/manual "
                  "to select a resolution", flush=True)
            return
        sizes = sorted({(p.width, p.height) for p in self.profiles},
                       key=lambda item: (item[0] * item[1], item))
        index = sizes.index((self.profile.width, self.profile.height))
        width, height = sizes[(index + 1) % len(sizes)]
        choices = [p for p in self.profiles if (p.width, p.height) == (width, height)]
        selected = min(choices, key=lambda p: (abs(p.fps - self.profile.fps), -p.fps))
        self.restart_profile(selected)

    def next_fps(self):
        if self.mode == "native":
            print("Native has no explicit profile; restart with --mode auto/manual "
                  "to select FPS", flush=True)
            return
        choices = [p for p in self.profiles if (p.width, p.height)
                   == (self.profile.width, self.profile.height)]
        index = choices.index(self.profile)
        self.restart_profile(choices[(index + 1) % len(choices)])

    def on_mouse(self, event, x, y, flags, userdata):
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_origin = (x, y)
            self.drag_cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_origin is not None:
            self.drag_cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_origin is not None:
            left, right = sorted((self.drag_origin[0], x))
            top, bottom = sorted((self.drag_origin[1], y))
            if right - left >= 20 and bottom - top >= 20:
                self.allowed_roi = (left, top, right - left, bottom - top)
                self.verifier = StableBarcode()
                print(f"ROI={self.allowed_roi}; outside hits will be ignored", flush=True)
            self.drag_origin = self.drag_cursor = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.allowed_roi = None
            self.verifier = StableBarcode()
            print("ROI cleared", flush=True)

    def handle_key(self, key: int) -> bool:
        if key in (27, ord("q")):
            return False
        if key == ord("r"):
            self.next_resolution()
        elif key == ord("f"):
            self.next_fps()
        elif key == ord("a") and self.mode != "native":
            self.mode = "manual" if self.mode == "auto" else "auto"
            if self.mode == "auto":
                set_option(self.sensor, rs.option.enable_auto_exposure, 1,
                           "auto_exposure")
                if self.sensor.supports(rs.option.auto_exposure_priority):
                    set_option(self.sensor, rs.option.auto_exposure_priority, 0,
                               "auto_exposure_priority")
            else:
                set_option(self.sensor, rs.option.enable_auto_exposure, 0,
                           "auto_exposure")
                self.exposure = set_option(self.sensor, rs.option.exposure,
                                           self.exposure, "exposure")
                self.gain = set_option(self.sensor, rs.option.gain, self.gain, "gain")
            print(f"MODE {self.mode}", flush=True)
        elif key in (ord("["), ord("]"), ord(","), ord(".")):
            if self.mode != "manual":
                print("Exposure/gain keys require manual mode", flush=True)
            elif key in (ord("["), ord("]")):
                delta = -10 if key == ord("[") else 10
                self.exposure = set_option(self.sensor, rs.option.exposure,
                                           self.exposure + delta, "exposure")
            else:
                delta = -8 if key == ord(",") else 8
                self.gain = set_option(self.sensor, rs.option.gain,
                                       self.gain + delta, "gain")
        elif key == ord("s"):
            self.last_save_at = 0.0
        return True

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        with (self.output / "frames.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            self.start_camera()
            try:
                if not self.args.headless:
                    cv2.namedWindow("D435 barcode quality lab", cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback("D435 barcode quality lab", self.on_mouse)
                    print("Keys: q/Esc exit; r resolution; f FPS; a auto/manual; "
                          "[ ] exposure; , . gain; s save; drag ROI; right-click clear",
                          flush=True)
                self.started_at = time.monotonic()
                while self.args.duration <= 0 or time.monotonic() - self.started_at < self.args.duration:
                    frames = self.pipeline.wait_for_frames(1000)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    image = color_frame_as_bgr(color, self.color_format)
                    now = time.monotonic()
                    interval_ms = (now - self.last_frame_at) * 1000 if self.last_frame_at else None
                    self.last_frame_at = now
                    roi = self.allowed_roi
                    allowed = None if roi is None else (
                        roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]
                    )
                    detect_started = time.monotonic()
                    hit = self.detector.detect(image, allowed)
                    detect_ms = (time.monotonic() - detect_started) * 1000
                    self.detect_ms.append(detect_ms)
                    hits, required, confirmed = self.verifier.observe(hit, now)
                    self.frame_count += 1
                    self.candidate_count += int(hit is not None)
                    self.confirmed_count += int(confirmed)
                    setting = f"{self.mode} {self.profile}"
                    setting_stats = self.by_setting.setdefault(
                        setting, {"frames": 0, "candidate_frames": 0,
                                  "confirmed_events": 0, "detect_ms_total": 0.0}
                    )
                    setting_stats["frames"] += 1
                    setting_stats["candidate_frames"] += int(hit is not None)
                    setting_stats["confirmed_events"] += int(confirmed)
                    setting_stats["detect_ms_total"] += detect_ms
                    if confirmed:
                        print(f"CONFIRMED t={now-self.started_at:.3f}s "
                              f"source={hit.source} rect={hit.rect}", flush=True)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    x, y, width, height = roi or (0, 0, image.shape[1], image.shape[0])
                    crop = gray[y:min(y + height, gray.shape[0]),
                                x:min(x + width, gray.shape[1])]
                    roi_mean = float(np.mean(crop)) if crop.size else None
                    roi_sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else None
                    hit_sharp = None
                    if hit is not None:
                        hx, hy, hw, hh = hit.rect
                        hit_crop = gray[max(0, hy):min(hy + hh, gray.shape[0]),
                                        max(0, hx):min(hx + hw, gray.shape[1])]
                        if hit_crop.size:
                            hit_sharp = float(cv2.Laplacian(hit_crop, cv2.CV_64F).var())
                    save = (self.args.save_every > 0
                            and now - self.last_save_at >= self.args.save_every) or confirmed
                    image_name = ""
                    if save:
                        image_name = f"raw_{self.saved_count:05d}.png"
                        if not cv2.imwrite(str(self.output / image_name), image):
                            raise RuntimeError(f"failed to save {image_name}")
                        self.saved_count += 1
                        self.last_save_at = now
                    writer.writerow({
                        "elapsed_s": round(now - self.started_at, 4),
                        "mode": self.mode,
                        "profile": str(self.profile),
                        "frame_number": color.get_frame_number(),
                        "camera_timestamp_ms": round(color.get_timestamp(), 3),
                        "capture_interval_ms": round(interval_ms, 2) if interval_ms else "",
                        "detect_ms": round(detect_ms, 2),
                        "candidate": int(hit is not None),
                        "source": hit.source if hit else "",
                        "confidence": round(hit.confidence, 3) if hit else "",
                        "candidate_hits": hits,
                        "required_hits": required,
                        "confirmed": int(confirmed),
                        "rect": str(hit.rect) if hit else "",
                        "roi_mean": round(roi_mean, 2) if roi_mean is not None else "",
                        "roi_laplacian_var": round(roi_sharp, 2) if roi_sharp is not None else "",
                        "candidate_laplacian_var": round(hit_sharp, 2) if hit_sharp is not None else "",
                        "sensor_exposure": read_option(self.sensor, rs.option.exposure),
                        "sensor_gain": read_option(self.sensor, rs.option.gain),
                        "frame_actual_exposure_us": frame_exposure_us(color) or "",
                        "image": image_name,
                    })
                    if self.frame_count % 30 == 0:
                        file.flush()
                    if not self.args.headless:
                        display = image.copy()
                        if roi is not None:
                            cv2.rectangle(display, (x, y), (x + width, y + height),
                                          (0, 200, 255), 2)
                        if self.drag_origin is not None and self.drag_cursor is not None:
                            cv2.rectangle(display, self.drag_origin, self.drag_cursor,
                                          (255, 200, 0), 2)
                        if hit is not None:
                            hx, hy, hw, hh = hit.rect
                            cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh),
                                          (0, 255, 0) if confirmed else (0, 255, 255), 3)
                        fps_text = (f"delivered={1000 / interval_ms:.1f}fps"
                                    if interval_ms and interval_ms > 0 else
                                    "delivered=--fps")
                        evidence_text = (
                            "CONFIRMED" if confirmed else
                            f"CANDIDATE {hits}/{required}" if hit else "NO HIT"
                        )
                        sharp_text = (
                            f"ROIsharp={roi_sharp:.0f}" if roi_sharp is not None
                            else "ROIsharp=--"
                        )
                        if hit_sharp is not None:
                            sharp_text += f" BARsharp={hit_sharp:.0f}"
                        lines = [
                            f"{self.mode.upper()} {self.profile}  AE={read_option(self.sensor, rs.option.enable_auto_exposure)} "
                            f"exp={read_option(self.sensor, rs.option.exposure)} gain={read_option(self.sensor, rs.option.gain)}",
                            f"{fps_text} detect={detect_ms:.0f}ms "
                            f"candidates={self.candidate_count} confirmed={self.confirmed_count}",
                            f"{evidence_text} {hit.source if hit else ''} {sharp_text}",
                        ]
                        for index, line in enumerate(lines):
                            cv2.putText(display, line, (12, 30 + index * 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 4)
                            cv2.putText(display, line, (12, 30 + index * 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                                        (0, 240, 0) if confirmed and index == 2 else (255, 255, 255), 2)
                        cv2.imshow("D435 barcode quality lab", display)
                        if not self.handle_key(cv2.waitKey(1) & 0xFF):
                            break
            except KeyboardInterrupt:
                print("Interrupted; writing summary for frames collected so far",
                      flush=True)
            finally:
                self.stop_camera()
                if not self.args.headless:
                    cv2.destroyAllWindows()
        elapsed = time.monotonic() - self.started_at
        summary = {
            "duration_s": round(elapsed, 3),
            "last_mode": self.mode,
            "last_profile": str(self.profile),
            "delivered_frames": self.frame_count,
            "delivered_fps": round(self.frame_count / max(0.001, elapsed), 2),
            "candidate_frames": self.candidate_count,
            "confirmed_events": self.confirmed_count,
            "saved_raw_png": self.saved_count,
            "median_detect_ms": round(float(np.median(self.detect_ms)), 2)
            if self.detect_ms else None,
            "model": self.detector.model_path,
            "provider": self.detector.active_provider,
            "roi": self.allowed_roi,
            "by_setting": {
                setting: {
                    "frames": stats["frames"],
                    "candidate_frames": stats["candidate_frames"],
                    "confirmed_events": stats["confirmed_events"],
                    "mean_detect_ms": round(
                        stats["detect_ms_total"] / max(1, stats["frames"]), 2
                    ),
                }
                for setting, stats in self.by_setting.items()
            },
        }
        (self.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--list-profiles", action="store_true",
                        help="list actual BGR8 color profiles and option ranges; no stream started")
    parser.add_argument("--mode", choices=("native", "auto", "manual"), default="manual",
                        help="native sets no profile/AE/exposure/gain/sharpness options")
    parser.add_argument("--profile", type=parse_profile, default=ColorProfile(1280, 720, 30))
    parser.add_argument("--exposure", type=float, default=70,
                        help="RealSense color exposure option value; manual mode only")
    parser.add_argument("--gain", type=float, default=128,
                        help="RealSense color gain option value; manual mode only")
    parser.add_argument("--sharpness", type=float,
                        help="optional sensor sharpness; not applied in native mode")
    parser.add_argument("--model", default=DEFAULT_BARCODE_MODEL)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--provider", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--roi", type=parse_roi,
                        help="acceptance ROI x,y,width,height; inference remains on original frame")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="seconds; 0 means until q/Esc or Ctrl-C")
    parser.add_argument("--save-every", type=float, default=1.0,
                        help="seconds between lossless raw PNG saves; 0 disables periodic saves")
    parser.add_argument("--output", type=Path,
                        help="new output directory for PNG, frames.csv and summary.json")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.list_profiles:
        try:
            device = find_device(args.serial)
            print(f"DEVICE {device.get_info(rs.camera_info.name)} serial={args.serial}")
            print("BGR8 color profiles:")
            for profile in color_profiles(device):
                print(" ", profile)
            for sensor in device.query_sensors():
                if any(profile.stream_type() == rs.stream.color
                       for profile in sensor.get_stream_profiles()):
                    for label, option in (
                        ("AE", rs.option.enable_auto_exposure),
                        ("exposure", rs.option.exposure),
                        ("gain", rs.option.gain),
                        ("sharpness", rs.option.sharpness),
                        ("AE priority", rs.option.auto_exposure_priority),
                    ):
                        print(f"{label}: {option_range(sensor, option)}")
        except RuntimeError as exc:
            parser.exit(1, f"Camera query failed: {exc}\n"
                        "Stop V3 and run on the host with D435 device access.\n")
        return
    if args.output is None:
        parser.error("--output is required except with --list-profiles")
    if args.output.exists():
        parser.error(f"output already exists; choose a new directory: {args.output}")
    if args.duration < 0 or args.save_every < 0:
        parser.error("duration and save-every must be nonnegative")
    try:
        QualityLab(args).run()
    except KeyboardInterrupt:
        print("Interrupted; use a finite --duration for an automatic summary", flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}\nStop V3 first if it owns the D435 camera.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
