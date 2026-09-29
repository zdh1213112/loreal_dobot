#!/usr/bin/env python3
"""Export a one-class Ultralytics barcode detector to runtime-ready ONNX.

The fast D435 detector in this workspace expects a static NCHW input and an
end-to-end detection output shaped ``[batch, max_det, 6]``. Each output row is
``x1, y1, x2, y2, confidence, class_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO barcode .pt checkpoint to D435-compatible ONNX."
    )
    parser.add_argument("weights", type=Path, help="Input Ultralytics .pt checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (default: beside the checkpoint)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--batch", type=int, default=1, help="Static batch size")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, for example cpu or 0. Runtime CUDA is independent of this.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold embedded in the NMS graph",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold embedded in the NMS graph",
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="Maximum NMS detections per image"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX graph simplification",
    )
    return parser.parse_args()


def require_export_dependencies():
    missing = []
    modules = {}
    for name in ("onnx", "onnxruntime", "ultralytics"):
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing Python packages: {joined}. "
            "Run this script with the workspace 'yolo' Conda environment, or install "
            "ultralytics, onnx and onnxruntime in the active environment."
        )
    return modules["onnx"], modules["onnxruntime"], modules["ultralytics"]


def dimensions(value_info) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return result


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else weights.with_suffix(".onnx")
    )
    if not weights.is_file():
        raise SystemExit(f"Checkpoint does not exist: {weights}")
    if weights.suffix.lower() != ".pt":
        raise SystemExit(f"Expected a .pt checkpoint, received: {weights}")
    if output.suffix.lower() != ".onnx":
        raise SystemExit(f"Output path must end in .onnx: {output}")
    if args.imgsz <= 0 or args.imgsz % 32:
        raise SystemExit("--imgsz must be a positive multiple of 32")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")

    # Keep Ultralytics' optional settings file away from read-only home mounts.
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics-config")
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    onnx, ort, ultralytics = require_export_dependencies()
    model = ultralytics.YOLO(str(weights), task="detect")
    names = dict(model.names)
    if names != {0: "barcode"}:
        raise SystemExit(
            "Expected exactly one class named 'barcode' for the D435 detector; "
            f"checkpoint contains {names!r}"
        )

    detection_head = model.model.model[-1]
    model_is_end_to_end = bool(getattr(detection_head, "end2end", False))
    export_nms = not model_is_end_to_end
    postprocess = "model end-to-end head" if model_is_end_to_end else "exported NMS"

    print(f"Loading: {weights}")
    print(
        f"Export: imgsz={args.imgsz}, batch={args.batch}, opset={args.opset}, "
        f"device={args.device}, postprocess={postprocess}, max_det={args.max_det}"
    )
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=args.batch,
            opset=args.opset,
            device=args.device,
            dynamic=False,
            half=False,
            simplify=not args.no_simplify,
            nms=export_nms,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )
    ).resolve()

    output.parent.mkdir(parents=True, exist_ok=True)
    if exported != output:
        shutil.move(str(exported), str(output))

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    input_shapes = [(item.name, dimensions(item)) for item in graph.graph.input]
    output_shapes = [(item.name, dimensions(item)) for item in graph.graph.output]
    expected_input = [args.batch, 3, args.imgsz, args.imgsz]
    if len(input_shapes) != 1 or input_shapes[0][1] != expected_input:
        raise RuntimeError(
            f"Unexpected ONNX input interface: {input_shapes}; expected {expected_input}"
        )
    if (
        len(output_shapes) != 1
        or len(output_shapes[0][1]) != 3
        or output_shapes[0][1][0] != args.batch
        or output_shapes[0][1][2] != 6
    ):
        raise RuntimeError(
            "Unexpected ONNX output interface: "
            f"{output_shapes}; expected [batch, max_det, 6]"
        )

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    dummy = np.zeros(expected_input, dtype=np.float32)
    runtime_outputs = session.run(None, {input_info.name: dummy})
    runtime_shapes = [list(np.asarray(item).shape) for item in runtime_outputs]
    if len(runtime_shapes) != 1 or runtime_shapes[0][-1] != 6:
        raise RuntimeError(f"ONNX Runtime returned incompatible shapes: {runtime_shapes}")

    print(f"Saved: {output}")
    print(f"Size: {output.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"SHA256: {sha256sum(output)}")
    print(f"ONNX inputs: {input_shapes}")
    print(f"ONNX outputs: {output_shapes}")
    print(f"CPU smoke-test outputs: {runtime_shapes}")
    print("Validation: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Export interrupted", file=sys.stderr)
        raise SystemExit(130)
