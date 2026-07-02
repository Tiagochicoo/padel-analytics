"""
export_trt.py
=============
Export a trained YOLOv26 checkpoint to ONNX and/or NVIDIA TensorRT (.engine).

    python src/export_trt.py --weights data/models/player_best.pt --format onnx
    python src/export_trt.py --weights data/models/player_best.pt --format engine --half

⚠️  TENSORRT ENGINES ARE NOT PORTABLE ⚠️
A .engine is compiled for a specific GPU architecture + TensorRT/CUDA version.
Rule of thumb:
    * Build the FP16 engine **ON the Jetson** where it will run.
    * On the laptop, export to ONNX, then (optionally) compile the engine on
      the Jetson, OR copy the .pt to the Jetson and run this script there.

RTX 3050 Ti Laptop (sm_86) and Jetson Orin Nano (sm_87) are close but not
identical — an engine built on one may fail to load on the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export(weights: str, fmt: str, imgsz: int, half: bool, batch: int,
           int8: bool, simplify: bool, workspace: int) -> None:
    from ultralytics import YOLO

    if not Path(weights).exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(weights)

    common = dict(
        format=fmt,
        imgsz=imgsz,
        half=half,
        int8=int8,
        batch=batch,
        dynamic=False,        # static shapes -> faster, smaller, Jetson-friendly
        simplify=simplify,    # onnx graph simplification
        workspace=workspace,  # TensorRT workspace GB (build-time)
    )

    # Ultralytics builds FP16 engines internally when half=True + format='engine'.
    path = model.export(**common)
    print(f"[export] {fmt.upper()} saved -> {path}")

    if fmt == "engine":
        print("\n[export] Remember: this .engine must run on the same GPU+TRT version.")
        print("         To deploy on Jetson, copy the .pt and re-run this on the Jetson.")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Export YOLOv26 -> ONNX / TensorRT.")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--format", default="engine", choices=["onnx", "engine"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 (recommended for Jetson).")
    p.add_argument("--int8", action="store_true", help="INT8 (needs calibration data).")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    p.set_defaults(simplify=True)
    p.add_argument("--workspace", type=int, default=4, help="TensorRT workspace (GB).")
    args = p.parse_args()

    export(
        weights=args.weights, fmt=args.format, imgsz=args.imgsz,
        half=args.half, batch=args.batch, int8=args.int8,
        simplify=args.simplify, workspace=args.workspace,
    )


if __name__ == "__main__":
    main()
