"""CUDA/detail-ROI D435 side-barcode detector for the fast V3 fork.

This module is intentionally separate from ``turntable_barcode_detector_v3``.
The production V3 node keeps its original model and inference path; this copy
is used only by the opt-in fast-D435 launch file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - reported clearly by load_model
    ort = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - reported clearly by load_model
    YOLO = None


DEFAULT_BARCODE_MODEL = "/home/zdh/ffs_ws/models/merge_0928_fast.onnx"


@dataclass(frozen=True)
class TurntableBarcodeHit:
    value: str
    rect: tuple[int, int, int, int]
    confidence: float
    source: str = "yolo"


def barcode_region_has_visual_detail(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
) -> bool:
    """Reject only near-uniform YOLO boxes, regardless of their ROI position.

    A dark robot pedestal produced a stable barcode prediction in the cell
    recording despite containing no visible bars.  Inspect the central part
    of each candidate so a crop boundary, package edge, or camera overlay
    cannot provide the entire contrast.  Low contrast *and* few local edges
    are required for rejection; a partially visible edge-of-ROI barcode is
    still eligible when its visible bars contain image detail.
    """

    if image is None or np.asarray(image).size == 0:
        return False
    image_height, image_width = image.shape[:2]
    x, y, width, height = [int(value) for value in rect]
    if width <= 0 or height <= 0:
        return False
    left = max(0, x + int(round(width * 0.15)))
    top = max(0, y + int(round(height * 0.15)))
    right = min(image_width, x + max(1, int(round(width * 0.85))))
    bottom = min(image_height, y + max(1, int(round(height * 0.85))))
    if right - left < 3 or bottom - top < 3:
        return False
    interior = image[top:bottom, left:right]
    gray = (
        cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
        if interior.ndim == 3
        else interior
    )
    gray = np.asarray(gray, dtype=np.float32)
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    horizontal_edges = float(np.mean(np.abs(np.diff(gray, axis=1)) >= 8.0))
    vertical_edges = float(np.mean(np.abs(np.diff(gray, axis=0)) >= 8.0))
    return contrast >= 14.0 or max(horizontal_edges, vertical_edges) >= 0.02


def barcode_region_has_parallel_bar_structure(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    minimum_runs: int = 6,
    minimum_line_coverage: float = 0.22,
    minimum_direction_dominance: float = 1.15,
) -> bool:
    """Require repeated, substantially parallel edges in a YOLO proposal.

    The one-class model supplies a useful candidate region, but shiny
    shrink-wrap can make an entire package score highly. Contrast alone cannot
    distinguish those broad wrinkles from a barcode. A linear barcode instead
    produces several edge runs with one dominant direction, and each run
    continues through a meaningful fraction of the candidate crop. Evaluate
    both vertical- and horizontal-bar layouts so 90-degree rotations remain
    valid.
    """

    if image is None or np.asarray(image).size == 0:
        return False
    image_height, image_width = image.shape[:2]
    x, y, width, height = [int(value) for value in rect]
    if width <= 0 or height <= 0:
        return False
    # A package boundary and the proposal border are not barcode evidence.
    inset_x = max(1, int(round(width * 0.06)))
    inset_y = max(1, int(round(height * 0.06)))
    left = max(0, x + inset_x)
    top = max(0, y + inset_y)
    right = min(image_width, x + width - inset_x)
    bottom = min(image_height, y + height - inset_y)
    if right - left < 12 or bottom - top < 12:
        return False
    interior = image[top:bottom, left:right]
    gray = (
        cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
        if interior.ndim == 3
        else np.asarray(interior)
    )
    gray = cv2.GaussianBlur(np.asarray(gray, dtype=np.uint8), (3, 3), 0.6)
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    required_runs = max(4, int(minimum_runs))
    required_coverage = max(0.10, min(0.80, float(minimum_line_coverage)))
    required_dominance = max(1.0, float(minimum_direction_dominance))

    def axis_has_bars(
        primary: np.ndarray,
        secondary: np.ndarray,
        collapse_axis: int,
    ) -> bool:
        # The percentile adapts to pale/blurred labels; the absolute floor
        # prevents sensor noise on a smooth package from becoming an edge.
        edge_threshold = max(24.0, float(np.percentile(primary, 80.0)))
        strong = primary >= edge_threshold
        line_coverage = np.mean(strong, axis=collapse_axis)
        active = line_coverage >= required_coverage
        if active.size == 0:
            return False
        run_starts = active & np.concatenate(
            (np.array([True], dtype=bool), np.logical_not(active[:-1]))
        )
        run_count = int(np.count_nonzero(run_starts))
        active_fraction = float(np.mean(active))
        direction_dominance = float(np.mean(primary)) / max(
            1e-6, float(np.mean(secondary))
        )
        return (
            run_count >= required_runs
            and active_fraction >= 0.10
            and direction_dominance >= required_dominance
        )

    # Vertical bars have X gradients continuing through rows; horizontal bars
    # have Y gradients continuing through columns.
    return axis_has_bars(gradient_x, gradient_y, collapse_axis=0) or axis_has_bars(
        gradient_y, gradient_x, collapse_axis=1
    )


class TurntableBarcodeDetector:
    """Detect whether the single-class barcode model sees a barcode region."""

    @staticmethod
    def _rect_center_in_roi(
        rect: tuple[int, int, int, int],
        allowed_roi: tuple[int, int, int, int] | None,
    ) -> bool:
        if allowed_roi is None:
            return True
        x, y, width, height = rect
        left, top, right, bottom = allowed_roi
        center_x = x + width * 0.5
        center_y = y + height * 0.5
        return left <= center_x < right and top <= center_y < bottom

    def __init__(
        self,
        model_path: str = DEFAULT_BARCODE_MODEL,
        confidence: float = 0.75,
        image_size: int = 640,
        scanner_assist: bool = True,
        inference_provider: str = "cuda",
        require_cuda: bool = True,
        wide_roi_fallback: bool = True,
        wide_roi_aspect_ratio: float = 2.0,
        wide_roi_tile_fraction: float = 0.70,
        wide_roi_unsharp_sigma: float = 1.2,
        wide_roi_unsharp_amount: float = 1.0,
        detail_roi: tuple[int, int, int, int] | None = (280, 220, 760, 440),
        full_frame_interval: int = 6,
        yolo_min_candidate_area_ratio: float = 0.01,
        moving_stripes_enabled: bool = True,
    ):
        self.model_path = str(model_path)
        self.confidence = max(0.05, min(0.95, float(confidence)))
        # The exported ONNX graph and the training run both use 640x640.  A
        # larger inference size was slower and less accurate on blurred
        # validation images, so keep the trained size explicit.
        self.image_size = max(320, int(image_size))
        self.scanner_assist = bool(scanner_assist)
        self.wide_roi_fallback = bool(wide_roi_fallback)
        self.wide_roi_aspect_ratio = max(1.0, float(wide_roi_aspect_ratio))
        self.wide_roi_tile_fraction = max(
            0.51, min(1.0, float(wide_roi_tile_fraction))
        )
        self.wide_roi_unsharp_sigma = max(
            0.1, float(wide_roi_unsharp_sigma)
        )
        self.wide_roi_unsharp_amount = max(
            0.0, float(wide_roi_unsharp_amount)
        )
        self.detail_roi = (
            tuple(int(value) for value in detail_roi)
            if detail_roi is not None and len(detail_roi) == 4
            else None
        )
        # The fixed detail crop is the fast path for the known turntable
        # region.  Periodic full-frame inference keeps coverage for a box
        # outside that crop without paying for two model passes on every
        # camera frame.  A value of one restores full-frame inference on
        # every frame.
        self.full_frame_interval = max(1, int(full_frame_interval))
        self._yolo_frame_index = 0
        self.yolo_min_candidate_area_ratio = max(
            0.0, min(0.25, float(yolo_min_candidate_area_ratio))
        )
        self.moving_stripes_enabled = bool(moving_stripes_enabled)
        self._previous_stripe_gray: np.ndarray | None = None
        requested_provider = str(inference_provider).strip().lower()
        if requested_provider not in {"cuda", "cpu", "auto"}:
            raise ValueError(
                "inference_provider must be one of: cuda, cpu, auto"
            )
        self.requested_provider = requested_provider
        self.require_cuda = bool(require_cuda)
        if self.require_cuda and self.requested_provider != "cuda":
            raise ValueError(
                "fast D435 detector requires inference_provider='cuda'"
            )
        self.active_provider = "not_loaded"
        self.provider_fallback_reason = ""
        self.model = None
        self.onnx_session = None
        self.onnx_input_name = ""
        self.onnx_session_options = None
        barcode_namespace = getattr(cv2, "barcode", None)
        barcode_detector_type = getattr(
            barcode_namespace, "BarcodeDetector", None
        ) or getattr(cv2, "barcode_BarcodeDetector", None)
        self.scanner_detector = (
            barcode_detector_type()
            if self.scanner_assist and barcode_detector_type is not None
            else None
        )

    @staticmethod
    def _preload_cuda_dependencies() -> None:
        """Load pip-installed cuDNN before ONNX Runtime creates CUDA EP."""

        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            return
        site_packages = Path(ort.__file__).resolve().parent.parent
        cudnn_directory = site_packages / "nvidia" / "cudnn" / "lib"
        if cudnn_directory.is_dir():
            # CUDA 12.8 runtime libraries are installed system-wide. cuDNN 9
            # lives in the Python environment and must be loaded explicitly
            # because it is not on the launch process' LD_LIBRARY_PATH.
            preload(
                cuda=False,
                cudnn=True,
                msvc=False,
                directory=str(cudnn_directory),
            )

    def _create_onnx_session(self, use_cuda: bool) -> None:
        if self.onnx_session_options is None:
            raise RuntimeError("ONNX session options are not initialized")
        providers = None
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": 1,
                    },
                ),
            ]
            if not self.require_cuda:
                providers.append("CPUExecutionProvider")
        elif "CPUExecutionProvider" in set(ort.get_available_providers()):
            providers = ["CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(
            self.model_path,
            sess_options=self.onnx_session_options,
            providers=providers,
        )
        session_providers = self.onnx_session.get_providers()
        self.active_provider = (
            str(session_providers[0]) if session_providers else "unknown"
        )

    def _fallback_to_cpu(self, reason: str) -> None:
        if self.require_cuda:
            raise RuntimeError(
                "fast D435 CUDA inference is required; " + str(reason)
            )
        self.provider_fallback_reason = str(reason)
        self.onnx_session = None
        self._create_onnx_session(use_cuda=False)

    def _warmup_onnx_session(self) -> None:
        """Pay the provider's one-time graph/kernel cost before scanning."""

        tensor = np.zeros(
            (1, 3, self.image_size, self.image_size), dtype=np.float32
        )
        try:
            self.onnx_session.run(None, {self.onnx_input_name: tensor})
        except Exception as exc:
            if self.active_provider != "CUDAExecutionProvider":
                raise
            self._fallback_to_cpu(f"CUDA warm-up inference failed: {exc}")
            self.onnx_session.run(None, {self.onnx_input_name: tensor})

    def load_model(self) -> None:
        if self.require_cuda and Path(self.model_path).suffix.lower() != ".onnx":
            raise RuntimeError(
                "fast D435 CUDA mode requires an ONNX model; "
                f"received {self.model_path!r}"
            )
        if Path(self.model_path).suffix.lower() == ".onnx":
            if ort is None:
                raise RuntimeError(
                    "onnxruntime is unavailable in the D435 vision Python environment"
                )
            options = ort.SessionOptions()
            # Keep routine session diagnostics quiet. Real model/session
            # errors remain visible.
            options.log_severity_level = 3
            self.onnx_session_options = options
            available = set(ort.get_available_providers())
            wants_cuda = self.requested_provider in {"cuda", "auto"}
            use_cuda = wants_cuda and "CUDAExecutionProvider" in available
            if wants_cuda and not use_cuda:
                if self.require_cuda:
                    raise RuntimeError(
                        "fast D435 requires CUDAExecutionProvider, but ONNX "
                        f"Runtime reports available providers={sorted(available)}"
                    )
                self.provider_fallback_reason = (
                    "CUDAExecutionProvider is not available; using CPU"
                )
            if use_cuda:
                try:
                    self._preload_cuda_dependencies()
                    self._create_onnx_session(use_cuda=True)
                    if self.active_provider != "CUDAExecutionProvider":
                        self._fallback_to_cpu(
                            "CUDA provider initialization selected "
                            f"{self.active_provider}"
                        )
                except Exception as exc:
                    if self.require_cuda:
                        raise RuntimeError(
                            "fast D435 CUDA provider initialization failed: "
                            f"{exc}"
                        ) from exc
                    self._fallback_to_cpu(
                        f"CUDA provider initialization failed: {exc}"
                    )
            else:
                self._create_onnx_session(use_cuda=False)
            model_input = self.onnx_session.get_inputs()[0]
            self.onnx_input_name = str(model_input.name)
            shape = list(model_input.shape)
            if len(shape) != 4 or shape[2:] != [self.image_size, self.image_size]:
                raise RuntimeError(
                    "D435 barcode ONNX input shape does not match model_image_size: "
                    f"model={shape}, requested={self.image_size}"
                )
            self._warmup_onnx_session()
            return
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is unavailable in the D435 vision Python environment"
            )
        # The selected barcode-label checkpoint is an ordinary detection
        # model; declare its task explicitly instead of inferring it by path.
        self.model = YOLO(self.model_path, task="detect")
        self.active_provider = "Ultralytics/PyTorch"

    def _predict_onnx(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the exported end-to-end ONNX graph and map boxes to the ROI."""

        if self.onnx_session is None:
            raise RuntimeError("turntable barcode ONNX session is not loaded")
        height, width = image.shape[:2]
        scale = min(self.image_size / height, self.image_size / width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        left = int(round(pad_width / 2.0 - 0.1))
        right = int(round(pad_width / 2.0 + 0.1))
        top = int(round(pad_height / 2.0 - 0.1))
        bottom = int(round(pad_height / 2.0 + 0.1))
        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(
            letterboxed[:, :, ::-1].transpose(2, 0, 1)
        )[None].astype(np.float32)
        tensor /= 255.0
        try:
            raw_output = self.onnx_session.run(
                None, {self.onnx_input_name: tensor}
            )[0]
        except Exception as exc:
            if self.active_provider != "CUDAExecutionProvider":
                raise
            self._fallback_to_cpu(f"CUDA inference failed: {exc}")
            raw_output = self.onnx_session.run(
                None, {self.onnx_input_name: tensor}
            )[0]
        output = np.asarray(raw_output, dtype=np.float32)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] < 6:
            raise RuntimeError(
                f"unexpected D435 barcode ONNX output shape: {output.shape}"
            )
        rows = output[0]
        keep = (rows[:, 4] >= self.confidence) & (rows[:, 5].astype(int) == 0)
        rows = rows[keep]
        if rows.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        candidates = rows[:, :4].copy()
        candidates[:, (0, 2)] = (candidates[:, (0, 2)] - left) / scale
        candidates[:, (1, 3)] = (candidates[:, (1, 3)] - top) / scale
        candidates[:, (0, 2)] = np.clip(candidates[:, (0, 2)], 0, width)
        candidates[:, (1, 3)] = np.clip(candidates[:, (1, 3)], 0, height)
        return (
            candidates,
            rows[:, 5].astype(np.int32),
            rows[:, 4].astype(np.float32),
        )

    def detect_scanner_pattern(
        self,
        image: np.ndarray,
        allowed_roi: tuple[int, int, int, int] | None = None,
    ) -> TurntableBarcodeHit | None:
        """Locate a barcode-like quadrilateral without decoding its payload."""

        if self.scanner_detector is None:
            return None
        if image is None or np.asarray(image).size == 0:
            return None
        try:
            detected, points = self.scanner_detector.detect(image)
        except (cv2.error, TypeError, ValueError):
            return None
        if not detected or points is None:
            return None
        quadrilaterals = np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)
        if quadrilaterals.size == 0:
            return None
        image_height, image_width = image.shape[:2]
        eligible = []
        for polygon in quadrilaterals:
            x, y, width, height = cv2.boundingRect(polygon)
            left = max(0, min(image_width - 1, int(x)))
            top = max(0, min(image_height - 1, int(y)))
            right = max(left + 1, min(image_width, int(x + width)))
            bottom = max(top + 1, min(image_height, int(y + height)))
            rect = (left, top, right - left, bottom - top)
            if self._rect_center_in_roi(rect, allowed_roi):
                eligible.append((abs(cv2.contourArea(polygon)), rect))
        if not eligible:
            return None
        _, rect = max(eligible, key=lambda item: item[0])
        return TurntableBarcodeHit(
            value="barcode_detected",
            rect=rect,
            # OpenCV's presence-only detector does not expose a confidence.
            confidence=1.0,
            source="scanner_pattern",
        )

    def _detect_yolo_once(
        self,
        image: np.ndarray,
        allowed_roi: tuple[int, int, int, int] | None = None,
    ) -> TurntableBarcodeHit | None:
        """Run one model view and keep its strongest detailed candidate."""

        if self.onnx_session is not None:
            candidates, class_ids, confidence_values = self._predict_onnx(image)
        else:
            predictions = self.model.predict(
                image,
                conf=self.confidence,
                imgsz=self.image_size,
                classes=[0],
                max_det=5,
                verbose=False,
            )
            if not predictions:
                return None
            boxes = getattr(predictions[0], "boxes", None)
            if boxes is None or getattr(boxes, "xyxy", None) is None:
                return None
            candidates = np.asarray(
                boxes.xyxy.cpu().numpy(), dtype=np.float32
            ).reshape(-1, 4)
            classes = getattr(boxes, "cls", None)
            confidences = getattr(boxes, "conf", None)
            class_ids = (
                np.asarray(classes.cpu().numpy(), dtype=np.int32).reshape(-1)
                if classes is not None
                else np.zeros(len(candidates), dtype=np.int32)
            )
            confidence_values = (
                np.asarray(confidences.cpu().numpy(), dtype=np.float32).reshape(-1)
                if confidences is not None
                else np.ones(len(candidates), dtype=np.float32)
            )
        height, width = image.shape[:2]
        hits = []
        for index, candidate in enumerate(candidates):
            # This model has exactly one class.  Its metadata calls class 0
            # "box", but the Roboflow training project and annotations are
            # barcode regions.
            if index < len(class_ids) and int(class_ids[index]) != 0:
                continue
            x1, y1, x2, y2 = [float(value) for value in candidate]
            left = max(0, min(width - 1, int(np.floor(x1))))
            top = max(0, min(height - 1, int(np.floor(y1))))
            right = max(left + 1, min(width, int(np.ceil(x2))))
            bottom = max(top + 1, min(height, int(np.ceil(y2))))
            rect = (left, top, right - left, bottom - top)
            if not self._rect_center_in_roi(rect, allowed_roi):
                continue
            candidate_area_ratio = (
                float((right - left) * (bottom - top))
                / float(max(1, width * height))
            )
            if candidate_area_ratio < self.yolo_min_candidate_area_ratio:
                continue
            confidence = (
                float(confidence_values[index])
                if index < len(confidence_values)
                else 1.0
            )
            if not barcode_region_has_visual_detail(
                image, (left, top, right - left, bottom - top)
            ):
                continue
            if not barcode_region_has_parallel_bar_structure(
                image, (left, top, right - left, bottom - top)
            ):
                continue
            hits.append((confidence, left, top, right, bottom))
        if not hits:
            return None
        confidence, left, top, right, bottom = max(hits, key=lambda item: item[0])
        return TurntableBarcodeHit(
            # The V3 workflow only classifies barcode presence.  Decoding a
            # moving crop adds a large slow path and its digits are discarded.
            value="barcode_detected",
            rect=(left, top, right - left, bottom - top),
            confidence=confidence,
            source="yolo",
        )

    @staticmethod
    def _translate_hit_x(
        hit: TurntableBarcodeHit, offset_x: int
    ) -> TurntableBarcodeHit:
        x, y, width, height = hit.rect
        return TurntableBarcodeHit(
            value=hit.value,
            rect=(int(x) + int(offset_x), int(y), int(width), int(height)),
            confidence=hit.confidence,
            source="yolo_tiled",
        )

    @staticmethod
    def _translate_hit_xy(
        hit: TurntableBarcodeHit,
        offset_x: int,
        offset_y: int,
        source: str,
    ) -> TurntableBarcodeHit:
        x, y, width, height = hit.rect
        return TurntableBarcodeHit(
            value=hit.value,
            rect=(int(x) + int(offset_x), int(y) + int(offset_y),
                  int(width), int(height)),
            confidence=hit.confidence,
            source=source,
        )

    def _detail_roi_view(
        self, image: np.ndarray
    ) -> tuple[int, int, np.ndarray] | None:
        """Return the fixed turntable detail view in native image coordinates."""

        if self.detail_roi is None:
            return None
        height, width = image.shape[:2]
        x, y, roi_width, roi_height = self.detail_roi
        if roi_width <= 0 or roi_height <= 0:
            return None
        left = max(0, min(width - 1, int(x)))
        top = max(0, min(height - 1, int(y)))
        right = max(left + 1, min(width, left + int(roi_width)))
        bottom = max(top + 1, min(height, top + int(roi_height)))
        if right - left < 160 or bottom - top < 120:
            return None
        return left, top, image[top:bottom, left:right]

    @staticmethod
    def _shift_roi(
        allowed_roi: tuple[int, int, int, int] | None,
        offset_x: int,
        offset_y: int,
    ) -> tuple[int, int, int, int] | None:
        if allowed_roi is None:
            return None
        left, top, right, bottom = allowed_roi
        return (
            int(left) - int(offset_x),
            int(top) - int(offset_y),
            int(right) - int(offset_x),
            int(bottom) - int(offset_y),
        )

    def _wide_roi_tiles(
        self, image: np.ndarray
    ) -> list[tuple[int, np.ndarray]]:
        """Return two overlapping views that retain detail in a very wide ROI."""

        height, width = image.shape[:2]
        if (
            not self.wide_roi_fallback
            or height <= 0
            or width / float(height) < self.wide_roi_aspect_ratio
        ):
            return []
        tile_width = max(
            1,
            min(width, int(round(width * self.wide_roi_tile_fraction))),
        )
        offsets = (0, max(0, width - tile_width))
        return [
            (offset, image[:, offset : offset + tile_width])
            for offset in offsets
        ]

    def _unsharp_for_wide_roi(self, image: np.ndarray) -> np.ndarray:
        """Restore low-contrast bar edges lost to motion and ROI letterboxing."""

        amount = self.wide_roi_unsharp_amount
        if amount <= 0.0:
            return image
        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=self.wide_roi_unsharp_sigma,
            sigmaY=self.wide_roi_unsharp_sigma,
        )
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0.0)

    def detect_yolo(
        self,
        image: np.ndarray,
        allowed_roi: tuple[int, int, int, int] | None = None,
    ) -> TurntableBarcodeHit | None:
        """Detect a barcode with a detail-first, periodic full-frame schedule.

        The turntable ROI is typically about 2.5:1. Letterboxing that entire
        view into the model's square input reduces its useful height to about
        255 pixels, which made small, pale moving barcodes disappear. The
        fixed detail crop is therefore evaluated on every frame. Full-frame
        inference is retained periodically so a manually selected barcode
        outside the fixed crop is still found, while avoiding the previous
        full-frame-plus-detail two-pass cost on every frame.
        """

        if self.model is None and self.onnx_session is None:
            raise RuntimeError("turntable barcode model is not loaded")
        if image is None or np.asarray(image).size == 0:
            return None

        self._yolo_frame_index += 1
        full_frame_due = (
            self.detail_roi is None
            or self.full_frame_interval <= 1
            or self._yolo_frame_index == 1
            or self._yolo_frame_index % self.full_frame_interval == 0
        )

        # The native 1280x720 frame makes a small side barcode occupy too few
        # pixels after letterboxing to 640x640.  A fixed turntable crop gives
        # the same extra detail on every frame, independent of a hand-drawn
        # acceptance ROI.  This is the normal one-pass path.
        detail_view = self._detail_roi_view(image)
        if detail_view is not None:
            offset_x, offset_y, detail = detail_view
            enhanced = self._unsharp_for_wide_roi(detail)
            detail_roi = self._shift_roi(allowed_roi, offset_x, offset_y)
            hit = self._detect_yolo_once(enhanced, detail_roi)
            if hit is not None:
                return self._translate_hit_xy(
                    hit, offset_x, offset_y, source="yolo_detail"
                )

        if not full_frame_due:
            return None

        # Periodic full-frame coverage catches targets outside the fixed
        # detail crop and preserves the original acceptance-ROI semantics.
        full_hit = self._detect_yolo_once(image, allowed_roi)
        if full_hit is not None:
            return full_hit

        # A very wide manually selected ROI can still benefit from the two
        # overlapping views, but only on the periodic full-frame frames.
        tiles = self._wide_roi_tiles(image)
        if tiles:
            tiled_hits = []
            for offset_x, tile in tiles:
                enhanced = self._unsharp_for_wide_roi(tile)
                tile_roi = self._shift_roi(allowed_roi, offset_x, 0)
                hit = self._detect_yolo_once(enhanced, tile_roi)
                if hit is not None:
                    tiled_hits.append(self._translate_hit_x(hit, offset_x))
            if tiled_hits:
                return max(tiled_hits, key=lambda item: item.confidence)
        return None

    def detect_moving_stripes(
        self,
        image: np.ndarray,
        allowed_roi: tuple[int, int, int, int] | None = None,
    ) -> TurntableBarcodeHit | None:
        """Find moving, coherent 1-D bars when the trained model misses.

        The image is always the native full D435 frame.  An optional ROI only
        filters candidate locations; it never changes image scale or texture.
        """

        if not self.moving_stripes_enabled or image is None:
            return None
        height, width = image.shape[:2]
        if width < 1100 or height < 600:
            self._previous_stripe_gray = None
            return None
        gray_full = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else np.asarray(image, dtype=np.uint8)
        )
        previous = self._previous_stripe_gray
        self._previous_stripe_gray = gray_full.copy()
        if previous is None or previous.shape != gray_full.shape:
            return None

        left, top = int(width * 0.28), int(height * 0.30)
        right, bottom = int(width * 0.82), int(height * 0.81)
        # The upper white package in the cell has a pale, reflective label.
        # CLAHE restores local bar contrast without changing the raw frame
        # used for temporal motion, so this fallback can see the label even
        # when YOLO has no proposal for it.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_full = clahe.apply(gray_full)
        search_gray = enhanced_full[top:bottom, left:right]
        hits = []
        for horizontal_bars in (False, True):
            # Transposing the search view lets the existing vertical-bar
            # geometry and thresholds evaluate 90-degree rotated barcodes
            # symmetrically.  Candidate coordinates are mapped back to the
            # unchanged full-frame image before motion and ROI checks.
            view = (
                np.ascontiguousarray(search_gray.T)
                if horizontal_bars else search_gray
            )
            gray = cv2.GaussianBlur(view, (3, 3), 0.5)
            edge_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            edge_mask = np.asarray(edge_x > 18.0, dtype=np.uint8) * 255
            elongated = cv2.morphologyEx(
                edge_mask, cv2.MORPH_OPEN, np.ones((9, 1), np.uint8)
            )
            grouped = cv2.morphologyEx(
                elongated, cv2.MORPH_CLOSE, np.ones((5, 18), np.uint8)
            )
            contours, _ = cv2.findContours(
                grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, box_width, box_height = cv2.boundingRect(contour)
                if not (
                    50 <= box_width <= 300
                    and 24 <= box_height <= 160
                    and box_width / float(box_height) >= 1.10
                ):
                    continue
                patch = gray[y:y + box_height, x:x + box_width]
                gradient_x = np.abs(
                    cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
                )
                gradient_y = np.abs(
                    cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
                )
                line_fraction = np.mean(gradient_x > 18.0, axis=0)
                continuous_columns = int(np.count_nonzero(line_fraction > 0.30))
                dominance = float(np.mean(gradient_x)) / max(
                    float(np.mean(gradient_y)), 1.0
                )
                if continuous_columns < 35 or dominance < 1.20:
                    continue
                rect = (
                    (left + y, top + x, box_height, box_width)
                    if horizontal_bars else
                    (left + x, top + y, box_width, box_height)
                )
                full_x, full_y, full_width, full_height = rect
                before = previous[
                    full_y:full_y + full_height,
                    full_x:full_x + full_width,
                ]
                current = gray_full[
                    full_y:full_y + full_height,
                    full_x:full_x + full_width,
                ]
                moving_fraction = float(
                    np.mean(cv2.absdiff(before, current) > 10)
                )
                if moving_fraction < 0.08:
                    continue
                if not self._rect_center_in_roi(rect, allowed_roi):
                    continue
                if not barcode_region_has_parallel_bar_structure(
                    enhanced_full,
                    rect,
                    minimum_runs=5,
                    minimum_line_coverage=0.16,
                    minimum_direction_dominance=1.08,
                ):
                    continue
                hits.append(
                    (continuous_columns * dominance * moving_fraction, rect)
                )
        if not hits:
            return None
        _, rect = max(hits, key=lambda item: item[0])
        return TurntableBarcodeHit(
            value="barcode_detected",
            rect=rect,
            confidence=0.0,  # geometric evidence, not a YOLO probability
            source="moving_stripes",
        )

    def detect(
        self,
        image: np.ndarray,
        allowed_roi: tuple[int, int, int, int] | None = None,
    ) -> TurntableBarcodeHit | None:
        """Prefer scanner/YOLO evidence; use moving stripes only on misses."""

        scanner_hit = self.detect_scanner_pattern(image, allowed_roi)
        if scanner_hit is not None:
            return scanner_hit
        yolo_hit = self.detect_yolo(image, allowed_roi)
        if yolo_hit is not None:
            return yolo_hit
        return self.detect_moving_stripes(image, allowed_roi)
