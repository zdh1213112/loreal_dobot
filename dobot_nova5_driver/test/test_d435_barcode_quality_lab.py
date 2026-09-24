"""No-camera tests for the standalone D435 image/barcode comparison tool."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyrealsense2 as rs


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "d435_barcode_quality_lab.py"
SPEC = importlib.util.spec_from_file_location("d435_barcode_quality_lab", SCRIPT)
lab = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)


class QualityLabPureTests(unittest.TestCase):
    def test_profile_and_roi_parsing(self):
        self.assertEqual(lab.parse_profile("1920x1080@30"),
                         lab.ColorProfile(1920, 1080, 30))
        self.assertEqual(lab.parse_roi("387,291,631,299"),
                         (387, 291, 631, 299))
        with self.assertRaises(Exception):
            lab.parse_profile("1280x720")

    def test_yolo_needs_two_spatially_matching_hits(self):
        verify = lab.StableBarcode()
        hit = SimpleNamespace(source="yolo_detail", value="barcode_detected",
                              rect=(100, 100, 80, 40))
        self.assertEqual(verify.observe(hit, 1.0), (1, 2, False))
        self.assertEqual(verify.observe(hit, 1.2), (2, 2, True))
        self.assertEqual(verify.observe(hit, 1.3), (1, 2, False))

    def test_different_region_does_not_confirm(self):
        verify = lab.StableBarcode()
        first = SimpleNamespace(source="yolo", value="barcode_detected",
                                rect=(100, 100, 80, 40))
        other = SimpleNamespace(source="yolo", value="barcode_detected",
                                rect=(500, 100, 80, 40))
        verify.observe(first, 1.0)
        self.assertEqual(verify.observe(other, 1.2), (1, 2, False))

    def test_native_rgb_frame_is_converted_to_bgr(self):
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        frame = SimpleNamespace(get_data=lambda: rgb)
        bgr = lab.color_frame_as_bgr(frame, rs.format.rgb8)
        self.assertEqual(bgr[0, 0].tolist(), [0, 0, 255])

    def test_detail_crop_tracks_turntable_at_higher_resolution(self):
        self.assertEqual(lab.detail_roi_for_profile(1280, 720),
                         (280, 220, 760, 440))
        self.assertEqual(lab.detail_roi_for_profile(1920, 1080),
                         (610, 440, 760, 440))


if __name__ == "__main__":
    unittest.main()
