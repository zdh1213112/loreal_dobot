"""Synthetic checks for the D405 top-face barcode depth interlock."""

import numpy as np

from top_surface_barcode_detector_v3 import TopSurfaceBarcode, verify_top_plane_barcode


HIT = TopSurfaceBarcode("barcode_detected", (10, 20, 100, 60))
PLANE = np.array([0.0, 0.0, 1.0, -0.30])
PIXELS = np.array([(x, y) for x in range(25, 100, 12) for y in range(32, 75, 8)])


def _points(z):
    return np.column_stack((PIXELS[:, 0] * 0.0002, PIXELS[:, 1] * 0.0002,
                            np.full(len(PIXELS), z)))


def test_live_barcode_pixels_across_top_plane_are_confirmed():
    evidence = verify_top_plane_barcode(HIT, _points(0.302), PIXELS, PLANE)
    assert evidence["verified"] is True
    assert evidence["sample_pixels"] >= 12
    assert min(evidence["inlier_coverage_xy"]) >= 0.3


def test_side_label_depth_does_not_confirm_top_barcode():
    evidence = verify_top_plane_barcode(HIT, _points(0.34), PIXELS, PLANE)
    assert evidence["verified"] is False
    assert evidence["reason"] == "barcode depth is not on the top plane"


def test_top_edge_strip_is_not_enough_to_classify_side_label_as_top():
    strip_pixels = np.array([(x, y) for x in range(22, 100, 4) for y in (32, 34)])
    strip_points = np.column_stack((strip_pixels[:, 0] * 0.0002,
                                    strip_pixels[:, 1] * 0.0002,
                                    np.full(len(strip_pixels), 0.302)))
    evidence = verify_top_plane_barcode(HIT, strip_points, strip_pixels, PLANE)
    assert evidence["verified"] is False
    assert evidence["reason"] == "top-plane points cover only a narrow barcode strip"


def test_missing_geometry_and_duplicate_pixels_fail_closed():
    assert not verify_top_plane_barcode(HIT, None, PIXELS, PLANE)["verified"]
    duplicated_pixels = np.repeat(PIXELS[:2], 10, axis=0)
    duplicated_points = np.repeat(_points(0.30)[:2], 10, axis=0)
    evidence = verify_top_plane_barcode(HIT, duplicated_points, duplicated_pixels, PLANE)
    assert evidence["verified"] is False
    assert evidence["sample_pixels"] == 2
