#!/usr/bin/env python3
"""
export_tracknet_trt.py
======================
Export the TrackNetV3 ball-detection model to ONNX / TensorRT (.engine).

TrackNetV3 is a vanilla PyTorch nn.Module (NOT Ultralytics), so it needs its
own export path — torch.onnx.export + optional trtexec.

Usage:
    # ONNX only (run anywhere, then copy .onnx to Jetson):
    python scripts/export_tracknet_trt.py --weights data/models/ball_best.pt --format onnx

    # TensorRT engine (run ON the Jetson):
    python scripts/export_tracknet_trt.py --weights data/models/ball_best.pt --format engine --half

    # If trtexec is not on PATH, the script exports ONNX and prints the
    # exact trtexec command to run manually.

⚠️  ENGINES ARE NOT PORTABLE — build on the Jetson where the model will run.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export(weights: str, fmt: str, half: bool, seq_len: int, bg_mode: str) -> None:
    import torch
    from src.tracknet import HEIGHT, WIDTH, get_model

    wpath = Path(weights)
    if not wpath.exists():
        raise SystemExit(f"Weights not found: {weights}")

    ckpt = torch.load(str(wpath), map_location="cpu")
    param_dict = ckpt.get("param_dict", {})
    seq_len = int(param_dict.get("seq_len", seq_len))
    bg_mode = param_dict.get("bg_mode", bg_mode)

    model = get_model("TrackNet", seq_len, bg_mode)
    ms, un = model.load_state_dict(ckpt["model"], strict=False)
    if ms or un:
        raise RuntimeError(f"Weight mismatch: missing={ms} unexpected={un}")
    model.eval()

    in_dim = (seq_len + (1 if bg_mode == "concat" else 0)) * 3
    dummy = torch.randn(1, in_dim, HEIGHT, WIDTH)

    onnx_path = wpath.with_suffix(".onnx")
    print(f"[export] TrackNet seq_len={seq_len} bg_mode={bg_mode} in_dim={in_dim}")
    print(f"[export] exporting ONNX -> {onnx_path}")

    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f"[export] ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if fmt != "engine":
        return

    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    engine_path = wpath.with_suffix(".engine")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspacePoolSize=4",
        "--useCudaGraph",
    ]
    if half:
        cmd.append("--fp16")

    if not Path(trtexec).exists():
        print(f"\n[export] trtexec not found at {trtexec}")
        print("[export] Run this command on the Jetson to build the engine:\n")
        print(" ".join(cmd))
        return

    print(f"[export] building TensorRT engine ({'FP16' if half else 'FP32'})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[export] engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[export] trtexec failed:\n{result.stderr}")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Export TrackNetV3 -> ONNX / TensorRT.")
    p.add_argument("--weights", required=True, help="Path to TrackNet .pt checkpoint.")
    p.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    p.add_argument("--half", action="store_true", help="FP16 (recommended for Jetson).")
    p.add_argument("--seq_len", type=int, default=8, help="Frames per input window.")
    p.add_argument("--bg_mode", default="concat", choices=["concat", "sub"],
                   help="Background mode (concat=prepend median, sub=subtract).")
    args = p.parse_args()

    export(args.weights, args.format, args.half, args.seq_len, args.bg_mode)


if __name__ == "__main__":
    main()
