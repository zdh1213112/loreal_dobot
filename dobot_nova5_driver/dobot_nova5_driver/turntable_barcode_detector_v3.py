"""Presence-only D435 side-barcode detector for the turntable workflow."""

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


DEFAULT_BARCODE_MODEL = (
    "/home/zdh/yolo_one/yolo_train_xense_load_image/outputs/"
    "train/obb_demo111/weights/best.onnx"
)


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


class TurntableBarcodeDetector:
    """Detect whether the single-class barcode model sees a barcode region."""

    def __init__(
        self,
        model_path: str = DEFAULT_BARCODE_MODEL,
        confidence: float = 0.45,
        image_size: int = 640,
        scanner_assist: bool = True,
        inference_provider: str = "cuda",
    ):
        self.model_path = str(model_path)
        self.confidence = max(0.05, min(0.95, float(confidence)))
        # The exported ONNX graph and the training run both use 640x640.  A
        # larger inference size was slower and less accurate on blurred
        # validation images, so keep the trained size explicit.
        self.image_size = max(320, int(image_size))
        self.scanner_assist = bool(scanner_assist)
        requested_provider = str(inference_provider).strip().lower()
        if requested_provider not in {"cuda", "cpu", "auto"}:
            raise ValueError(
                "inference_provider must be one of: cuda, cpu, auto"
            )
        self.requested_provider = requested_provider
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
                "CPUExecutionProvider",
            ]
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
                            f"{self.active_provider}; using CPU"
                        )
                except Exception as exc:
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
        # task="detect" is required for the fixed-shape ONNX export and also
        # avoids guessing from the misleading historical "obb_demo" folder.
        self.model = YOLO(self.model_path, task="detect")

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
        self, image: np.ndarray
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
        polygon = max(quadrilaterals, key=lambda item: abs(cv2.contourArea(item)))
        x, y, width, height = cv2.boundingRect(polygon)
        image_height, image_width = image.shape[:2]
        left = max(0, min(image_width - 1, int(x)))
        top = max(0, min(image_height - 1, int(y)))
        right = max(left + 1, min(image_width, int(x + width)))
        bottom = max(top + 1, min(image_height, int(y + height)))
        return TurntableBarcodeHit(
            value="barcode_detected",
            rect=(left, top, right - left, bottom - top),
            # OpenCV's presence-only detector does not expose a confidence.
            confidence=1.0,
            source="scanner_pattern",
        )

    def detect_yolo(self, image: np.ndarray) -> TurntableBarcodeHit | None:
        """Select the strongest YOLO hit with visible detail in its box."""

        if self.model is None and self.onnx_session is None:
            raise RuntimeError("turntable barcode model is not loaded")
        if image is None or np.asarray(image).size == 0:
            return None
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
            confidence = (
                float(confidence_values[index])
                if index < len(confidence_values)
                else 1.0
            )
            if not barcode_region_has_visual_detail(
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

    def detect(self, image: np.ndarray) -> TurntableBarcodeHit | None:
        """Use the fast scanner pattern path first, then fall back to YOLO."""

        scanner_hit = self.detect_scanner_pattern(image)
        if scanner_hit is not None:
            return scanner_hit
        return self.detect_yolo(image)
