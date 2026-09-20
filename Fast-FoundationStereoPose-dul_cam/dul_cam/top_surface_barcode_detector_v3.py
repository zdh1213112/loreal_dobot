"""Barcode decoding on the visible top surface of a tracked D405 target.

The cosmetic-box vision process already owns the D405 camera.  This helper
therefore consumes its current RGB frame and SAM2 mask instead of opening a
second RealSense pipeline.  The mask is eroded before decoding so labels on a
vertical side or in the background are rejected as far as the image geometry
allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:  # pragma: no cover - depends on the host's libzbar install
    zbar_decode = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional when OpenCV/ZBar is sufficient
    YOLO = None


@dataclass(frozen=True)
class TopSurfaceBarcode:
    value: str
    rect: tuple[int, int, int, int] | None = None


class TopSurfaceBarcodeDetector:
    """Decode a barcode whose centre lies in the interior of a SAM2 mask."""

    def __init__(
        self,
        *,
        minimum_interior_pixels: int = 4,
        model_path: str = "/home/zdh/yolo_one/yolo_train_xense_load_image/outputs/train/obb_demo111/weights/best.pt",
        model_confidence: float = 0.20,
    ) -> None:
        self.minimum_interior_pixels = max(1, int(minimum_interior_pixels))
        self.model_path = str(model_path)
        self.model_confidence = max(0.05, min(0.95, float(model_confidence)))
        self._yolo_model = None
        self._yolo_load_attempted = False
        self._cv_detector = None
        # Main D405 process reads this after each attempt to render a compact
        # explanation of the barcode ROI and decoder path on its live panel.
        self.last_debug: dict = {
            "interior_bbox": None,
            "interior_pixels": 0,
            "candidate_boxes": [],
            "accepted_candidate": None,
            "decoder": "none",
            "message": "not attempted",
        }
        try:
            if hasattr(cv2, "barcode_BarcodeDetector"):
                self._cv_detector = cv2.barcode_BarcodeDetector()
        except Exception:
            self._cv_detector = None

    @staticmethod
    def _interior_mask(
        mask: np.ndarray,
        target_corners: np.ndarray | None = None,
    ) -> np.ndarray | None:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.ndim != 2 or not np.any(binary):
            return None
        # The SAM mask belongs to the target originally selected by YOLO.  Its
        # current tracked quadrilateral is also applied so decoding never
        # expands into a neighbouring package or the background.
        if target_corners is not None:
            corners = np.asarray(target_corners, dtype=np.float32).reshape(-1, 2)
            if len(corners) >= 3:
                target_region = np.zeros_like(binary)
                cv2.fillPoly(target_region, [np.int32(np.round(corners))], 1)
                binary &= target_region
                if not np.any(binary):
                    return None
        # Preserve the complete visible target, including edge labels.
        return binary.astype(bool)

    @staticmethod
    def _crop_target_roi(
        image: np.ndarray, interior: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        ys, xs = np.nonzero(interior)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = np.asarray(image[y0:y1, x0:x1]).copy()
        return crop, (x0, y0)

    @staticmethod
    def _valid_zbar_result(
        result,
        interior: np.ndarray,
        origin: tuple[int, int],
        scale: float,
    ) -> bool:
        rect = getattr(result, "rect", None)
        if rect is None:
            return True
        cx = int(round((float(rect.left) + float(rect.width) * 0.5) / scale)) + origin[0]
        cy = int(round((float(rect.top) + float(rect.height) * 0.5) / scale)) + origin[1]
        h, w = interior.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return False
        return bool(interior[cy, cx])

    def _decode_zbar(self, image: np.ndarray, interior: np.ndarray, origin: tuple[int, int]):
        if zbar_decode is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # The top label is often only 40--70 pixels wide in the D405 RGB
        # stream.  ZBar is much more reliable after enlarging, rotating the
        # portrait label into landscape, and trying a few contrast variants.
        variants = [(gray, False)]
        for scale in (2.0, 3.0, 4.0):
            up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            variants.extend((
                (up, False),
                (cv2.detailEnhance(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR), sigma_s=10, sigma_r=0.15)[:, :, 0], False),
                (cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX), False),
            ))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        variants.append((clahe.apply(gray), False))
        # Try both orientations.  The barcode on a box top is frequently
        # aligned along the long axis, which appears portrait in the camera.
        expanded_variants = []
        for variant, transformed in variants:
            expanded_variants.extend((
                (variant, transformed),
                (cv2.rotate(variant, cv2.ROTATE_90_CLOCKWISE), True),
                (cv2.rotate(variant, cv2.ROTATE_90_COUNTERCLOCKWISE), True),
            ))
        variants = expanded_variants
        for variant, transformed in variants:
            if variant.size == 0:
                continue
            try:
                hits = zbar_decode(variant)
            except Exception:
                continue
            scale = variant.shape[1] / max(1, gray.shape[1])
            for hit in hits:
                if "QRCODE" in str(getattr(hit, "type", "")).upper():
                    continue
                value = bytes(hit.data).decode("utf-8", errors="ignore").strip()
                if not value:
                    continue
                if not transformed and not self._valid_zbar_result(hit, interior, origin, scale):
                    continue
                rect = getattr(hit, "rect", None)
                mapped_rect = None
                if rect is not None and not transformed:
                    mapped_rect = (
                        int(origin[0] + rect.left / scale),
                        int(origin[1] + rect.top / scale),
                        int(rect.width / scale),
                        int(rect.height / scale),
                    )
                return TopSurfaceBarcode(value=value, rect=mapped_rect)
        return None

    def _decode_opencv(self, image: np.ndarray, interior: np.ndarray, origin: tuple[int, int]):
        if self._cv_detector is None:
            return None
        try:
            ok, decoded, points, _ = self._cv_detector.detectAndDecode(image)
        except Exception:
            return None
        if not ok:
            return None
        values = decoded if isinstance(decoded, (tuple, list)) else [decoded]
        for index, raw in enumerate(values):
            value = str(raw).strip()
            if not value:
                continue
            if points is not None:
                try:
                    pts = np.asarray(points[index], dtype=np.float32).reshape(-1, 2)
                    center = np.mean(pts, axis=0).astype(int)
                    x, y = int(center[0]) + origin[0], int(center[1]) + origin[1]
                    if not (0 <= y < interior.shape[0] and 0 <= x < interior.shape[1]) or not interior[y, x]:
                        continue
                except Exception:
                    pass
            return TopSurfaceBarcode(value=value)
        return None

    def _decode_yolo(self, image: np.ndarray, interior: np.ndarray) -> TopSurfaceBarcode | None:
        """Locate a barcode with YOLO inside the selected target region.

        The top-surface branch only needs face classification.  It deliberately
        returns as soon as a class-0 barcode box is inside the SAM interior;
        ZBar/OpenCV string decoding is not required for this decision.
        """

        if YOLO is None:
            return None
        if not self.load_model():
            return None
        try:
            predictions = self._yolo_model.predict(
                image, conf=self.model_confidence, verbose=False
            )
        except Exception:
            return None
        if not predictions:
            return None
        boxes = getattr(predictions[0], "boxes", None)
        if boxes is None:
            return None
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            return None
        try:
            candidates = np.asarray(xyxy.cpu().numpy(), dtype=np.float32).reshape(-1, 4)
        except Exception:
            return None
        classes = getattr(boxes, "cls", None)
        try:
            class_ids = np.asarray(classes.cpu().numpy()).reshape(-1) if classes is not None else None
        except Exception:
            class_ids = None
        h, w = interior.shape[:2]
        confidences = getattr(boxes, "conf", None)
        try:
            confidence_values = (
                np.asarray(confidences.cpu().numpy(), dtype=np.float32).reshape(-1)
                if confidences is not None
                else None
            )
        except Exception:
            confidence_values = None
        self.last_debug["candidate_boxes"] = []
        for index, (x1, y1, x2, y2) in enumerate(candidates):
            # The shipped model labels class 0 as barcode and class 1 as
            # qrcode.  QR codes are intentionally excluded from this branch.
            if class_ids is not None and index < len(class_ids) and int(class_ids[index]) != 0:
                continue
            left = max(0, min(w - 1, int(np.floor(x1))))
            top = max(0, min(h - 1, int(np.floor(y1))))
            right = max(left + 1, min(w, int(np.ceil(x2))))
            bottom = max(top + 1, min(h, int(np.ceil(y2))))
            cx, cy = (left + right) // 2, (top + bottom) // 2
            class_id = (
                int(class_ids[index])
                if class_ids is not None and index < len(class_ids)
                else 0
            )
            confidence = (
                float(confidence_values[index])
                if confidence_values is not None and index < len(confidence_values)
                else None
            )
            inside = bool(interior[cy, cx])
            self.last_debug["candidate_boxes"].append(
                {
                    "rect": (left, top, right, bottom),
                    "class_id": class_id,
                    "confidence": confidence,
                    "center_inside": inside,
                }
            )
            if not inside:
                continue
            self.last_debug["accepted_candidate"] = (left, top, right - left, bottom - top)
            self.last_debug["decoder"] = "YOLO"
            self.last_debug["message"] = (
                f"YOLO barcode detected (conf={confidence:.2f})"
                if confidence is not None
                else "YOLO barcode detected"
            )
            return TopSurfaceBarcode(
                value="barcode_detected",
                rect=(left, top, right - left, bottom - top),
            )
        return None

    @staticmethod
    def _map_debug_to_global(debug: dict, origin: tuple[int, int]) -> dict:
        """Map ROI-local debug rectangles back to full-camera coordinates."""

        ox, oy = [int(v) for v in origin]
        mapped = dict(debug)
        bbox = mapped.get("interior_bbox")
        if bbox is not None and len(bbox) == 4:
            mapped["interior_bbox"] = (
                int(bbox[0]) + ox,
                int(bbox[1]) + oy,
                int(bbox[2]) + ox,
                int(bbox[3]) + oy,
            )
        candidates = []
        for candidate in mapped.get("candidate_boxes", []):
            item = dict(candidate)
            rect = item.get("rect")
            if rect is not None and len(rect) == 4:
                item["rect"] = (
                    int(rect[0]) + ox,
                    int(rect[1]) + oy,
                    int(rect[2]) + ox,
                    int(rect[3]) + oy,
                )
            candidates.append(item)
        mapped["candidate_boxes"] = candidates
        accepted = mapped.get("accepted_candidate")
        if accepted is not None and len(accepted) == 4:
            # Accepted candidates are stored as x,y,w,h, unlike YOLO debug
            # boxes which use x0,y0,x1,y1.
            mapped["accepted_candidate"] = (
                int(accepted[0]) + ox,
                int(accepted[1]) + oy,
                int(accepted[2]),
                int(accepted[3]),
            )
        return mapped

    def load_model(self) -> bool:
        """Load the optional barcode YOLO model before the first motion cycle."""

        if YOLO is None:
            self._yolo_load_attempted = True
            return False
        if self._yolo_model is not None:
            return True
        if not self._yolo_load_attempted:
            self._yolo_load_attempted = True
            try:
                self._yolo_model = YOLO(self.model_path)
            except Exception:
                self._yolo_model = None
        return self._yolo_model is not None

    def detect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        target_corners: np.ndarray | None = None,
    ) -> TopSurfaceBarcode | None:
        self.last_debug = {
            "interior_bbox": None,
            "interior_pixels": 0,
            "candidate_boxes": [],
            "accepted_candidate": None,
            "decoder": "none",
            "message": "invalid image",
        }
        if image is None or np.asarray(image).size == 0:
            return None
        interior = self._interior_mask(mask, target_corners)
        if interior is None or int(np.count_nonzero(interior)) < self.minimum_interior_pixels:
            self.last_debug["message"] = "empty or too-small SAM interior"
            return None
        ys, xs = np.nonzero(interior)
        self.last_debug["interior_bbox"] = (
            int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        )
        self.last_debug["interior_pixels"] = int(np.count_nonzero(interior))
        cropped = self._crop_target_roi(image, interior)
        if cropped is None:
            self.last_debug["message"] = "empty crop"
            return None
        crop, origin = cropped
        # Run the barcode YOLO model on the selected material ROI itself. This
        # prevents unrelated labels elsewhere in the D405 frame from affecting
        # the top-surface decision and gives YOLO more pixels for the target.
        local_interior = interior[origin[1] : origin[1] + crop.shape[0], origin[0] : origin[0] + crop.shape[1]]
        local_ys, local_xs = np.nonzero(local_interior)
        self.last_debug["interior_bbox"] = (
            int(local_xs.min()),
            int(local_ys.min()),
            int(local_xs.max()),
            int(local_ys.max()),
        ) if len(local_xs) else None
        result = self._decode_yolo(crop, local_interior)
        self.last_debug = self._map_debug_to_global(self.last_debug, origin)
        if result is not None:
            if result.rect is not None:
                result = TopSurfaceBarcode(
                    value=result.value,
                    rect=(
                        int(result.rect[0]) + origin[0],
                        int(result.rect[1]) + origin[1],
                        int(result.rect[2]),
                        int(result.rect[3]),
                    ),
                )
            if self.last_debug.get("decoder") == "none":
                self.last_debug["decoder"] = "YOLO+ZBar"
            self.last_debug["accepted_candidate"] = result.rect
            if self.last_debug.get("message") in (None, "no barcode decoded"):
                self.last_debug["message"] = f"decoded {result.value}"
            return result
        # If YOLO did not detect a box on the full ROI crop, still try a
        # replicated border around the target crop to give edge labels more
        # input margin. This does not recover physically occluded pixels.
        roi_h, roi_w = crop.shape[:2]
        pad = max(4, int(round(0.08 * min(roi_w, roi_h))))
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        padded_mask = cv2.copyMakeBorder(
            local_interior.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        ).astype(bool)
        self.last_debug["interior_bbox"] = (
            int(local_xs.min()) + pad, int(local_ys.min()) + pad,
            int(local_xs.max()) + pad, int(local_ys.max()) + pad,
        )
        self.last_debug["candidate_boxes"] = []
        padded_result = self._decode_yolo(padded, padded_mask)
        self.last_debug = self._map_debug_to_global(
            self.last_debug, (origin[0] - pad, origin[1] - pad)
        )
        if padded_result is not None:
            if padded_result.rect is not None:
                padded_result = TopSurfaceBarcode(
                    value=padded_result.value,
                    rect=(
                        int(padded_result.rect[0]) + origin[0] - pad,
                        int(padded_result.rect[1]) + origin[1] - pad,
                        int(padded_result.rect[2]),
                        int(padded_result.rect[3]),
                    ),
                )
            return padded_result
        result = self._decode_zbar(crop, local_interior, (0, 0))
        if result is not None:
            if result.rect is not None:
                result = TopSurfaceBarcode(
                    value=result.value,
                    rect=(
                        int(result.rect[0]) + origin[0],
                        int(result.rect[1]) + origin[1],
                        int(result.rect[2]),
                        int(result.rect[3]),
                    ),
                )
            self.last_debug["decoder"] = "ZBar"
            self.last_debug["accepted_candidate"] = result.rect
            self.last_debug["message"] = f"decoded {result.value}"
            return result
        result = self._decode_opencv(crop, interior, origin)
        if result is not None:
            self.last_debug["decoder"] = "OpenCV"
            self.last_debug["accepted_candidate"] = result.rect
            self.last_debug["message"] = f"decoded {result.value}"
            return result
        self.last_debug["message"] = "no barcode decoded"
        return None
