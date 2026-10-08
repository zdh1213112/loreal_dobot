"""Conservative SAM2 prompts and mask cleanup for a YOLO-selected box."""

import cv2
import numpy as np


def prompt_from_obb(corners: np.ndarray, image_shape: tuple[int, ...]):
    """Pair the required axis-aligned SAM box with object/background points."""

    polygon = np.asarray(corners, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        raise ValueError("YOLO OBB must contain four finite image points")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    lower = np.clip(polygon.min(axis=0), [0, 0], [width - 1, height - 1])
    upper = np.clip(polygon.max(axis=0), [0, 0], [width - 1, height - 1])
    if np.any(upper - lower < 2):
        raise ValueError("YOLO OBB is too small for a SAM2 prompt")
    bbox = np.stack((lower, upper)).astype(np.float32)

    center = polygon.mean(axis=0)
    edges = np.roll(polygon, -1, axis=0) - polygon
    major_axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    positive = [center, center - 0.18 * major_axis, center + 0.18 * major_axis]
    points = [np.clip(point, [0, 0], [width - 1, height - 1]) for point in positive]
    labels = [1] * len(points)

    # The four corners of an AABB around a diagonal box are background. Only
    # use a candidate when it is clearly outside the rotated YOLO polygon.
    x1, y1 = lower
    x2, y2 = upper
    inset = min(4.0, 0.08 * float(min(x2 - x1, y2 - y1)))
    for candidate in (
        (x1 + inset, y1 + inset),
        (x2 - inset, y1 + inset),
        (x2 - inset, y2 - inset),
        (x1 + inset, y2 - inset),
    ):
        if cv2.pointPolygonTest(polygon, tuple(map(float, candidate)), True) < -5.0:
            points.append(np.asarray(candidate, dtype=np.float32))
            labels.append(0)
    return bbox, np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def refine_sam_mask(
    raw_mask: np.ndarray,
    *,
    selected_obb: np.ndarray | None = None,
) -> np.ndarray | None:
    """Drop narrow protrusions/islands; reject an initial mask on the table."""

    mask = np.asarray(raw_mask)
    if mask.ndim != 2:
        raise ValueError("SAM2 mask must be two-dimensional")
    binary = (mask > 0).astype(np.uint8)
    raw_area = int(binary.sum())
    if raw_area < 40:
        return None
    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if component_count <= 1:
        return None
    selected_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = labels == selected_label
    clean_area = int(np.count_nonzero(clean))
    if clean_area < 40 or clean_area < 0.70 * raw_area:
        return None

    if selected_obb is not None:
        polygon = np.asarray(selected_obb, dtype=np.float32)
        if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
            raise ValueError("selected OBB must contain four finite image points")
        support = np.zeros(clean.shape, dtype=np.uint8)
        cv2.fillConvexPoly(support, np.rint(polygon).astype(np.int32), 1)
        support = cv2.dilate(
            support,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        )
        overlap_fraction = np.count_nonzero(clean & (support > 0)) / clean_area
        if overlap_fraction < 0.70:
            return None
    return clean
