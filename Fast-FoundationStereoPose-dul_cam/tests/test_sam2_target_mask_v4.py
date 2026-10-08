import cv2
import numpy as np

from sam2_target_mask_v4 import prompt_from_obb, refine_sam_mask


def test_rotated_yolo_box_adds_background_points_outside_the_product():
    corners = np.array(
        [[443.4, 273.9], [602.7, 79.6], [544.7, 32.1], [385.5, 226.5]],
        dtype=np.float32,
    )

    bbox, points, labels = prompt_from_obb(corners, (480, 640, 3))

    assert bbox.shape == (2, 2)
    assert np.array_equal(labels[:3], [1, 1, 1])
    assert np.count_nonzero(labels == 0) >= 2
    for point, label in zip(points, labels):
        distance = cv2.pointPolygonTest(corners, tuple(map(float, point)), True)
        if label == 1:
            assert distance >= 0.0
        else:
            assert distance < -5.0


def test_refinement_keeps_rotated_box_and_drops_small_artifacts():
    raw = np.zeros((240, 320), dtype=np.uint8)
    cv2.fillConvexPoly(
        raw,
        np.array([[60, 180], [180, 40], [210, 70], [90, 210]], dtype=np.int32),
        1,
    )
    raw[22:24, 25:27] = 1
    raw[200:202, 290:292] = 1

    cleaned = refine_sam_mask(raw)

    assert cleaned is not None
    assert np.count_nonzero(cleaned) >= 0.95 * np.count_nonzero(raw)
    assert not np.any(cleaned & (raw == 0))
    assert not cleaned[22, 25]
    assert not cleaned[200, 290]


def test_refinement_rejects_mask_that_tracks_table_instead_of_selected_box():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[40:100, 40:100] = 1  # selected box
    mask[130:230, 180:300] = 1  # larger, disconnected table region
    selected_obb = np.array(
        [[40, 40], [100, 40], [100, 100], [40, 100]], dtype=np.float32
    )

    assert refine_sam_mask(mask, selected_obb=selected_obb) is None
