#!/usr/bin/env python3
"""
build_tracknet_dataset.py
=========================
Convert the padelTracker100 ball trajectories (COCO JSON) into the label format
expected by the vendored TrackNetV3 (third_party/tracknetv3/), for fine-tuning
the pretrained TrackNet ball model on padel.

TrackNet is trajectory-based: it stacks ``seq_len`` CONSECUTIVE frames as one
input (bg_mode='concat' -> 8 RGB frames + 1 median background = 27 channels)
and predicts one ball-center heatmap per frame. Unlike the YOLO ball dataset,
we therefore CANNOT subsample (stride) — frames must stay consecutive within
each "rally". We chunk each match's continuous trajectory into fixed-length
rallies of consecutive frames and emit, per rally:

    data/datasets/ball_tracknet/{train,val}/<rally_id>/
        frames/0.png 1.png ... N-1.png     (symlinks to the source jpgs; PIL
                                            reads by content so .png -> .jpg is fine)
        label.csv                           (Frame,Visibility,X,Y; orig-pixel center)
        median.npz                          (key 'median', uint8 RGB at 288x512 —
                                             the model input size, so the dataset's
                                             resize is a no-op)
    manifest_{train,val}.txt                 (one rally_id per line)

A frame with no ball annotation gets a ``0,0,0`` row (Visibility=0); the
vendored dataset renders a zero heatmap for (0,0) centres, which is exactly the
"ball not visible" signal TrackNet learns from.

Rallies whose ball-visibility ratio is below --min_ball_ratio are dropped
(they teach nothing and waste GPU). The last --val_frac of each match's rallies
form the val split (trajectory-disjoint from train, so no leakage).

Usage:
    python scripts/build_tracknet_dataset.py [--rally_len 256] [--val_frac 0.15]
                                             [--min_ball_ratio 0.05] [--median_samples 64]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PT100_RAW = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "raw"
PT100_FRAMES = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "frames"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "ball_tracknet"

# (coco ball json, frame subdir stem) — matches scripts/build_ball_dataset.py
SOURCES = [
    ("2022_BCN_FinalF_1_ball.json", "FinalF"),
    ("2022_BCN_FinalM_1_ball.json", "FinalM"),
]

MEDIAN_W, MEDIAN_H = 512, 288  # TrackNet input resolution (WIDTH x HEIGHT)


def parse_frame_index(file_name: str) -> int:
    return int(Path(file_name).stem.split("_")[1])


def load_ball_centres(coco_path: Path) -> tuple[int, int, dict[int, tuple[float, float]]]:
    """Return (width, height, {frame_idx: (cx, cy)}) for the 'Ball' category."""
    data = json.loads(coco_path.read_text())
    w = int(data["images"][0]["width"])
    h = int(data["images"][0]["height"])
    ball_cat_id = next((c["id"] for c in data["categories"] if c["name"].lower() == "ball"), None)
    if ball_cat_id is None:
        raise RuntimeError(f"no 'Ball' category in {coco_path.name}")
    frame_of = {im["id"]: parse_frame_index(im["file_name"]) for im in data["images"]}
    centres: dict[int, tuple[float, float]] = {}
    for a in data["annotations"]:
        if a["category_id"] != ball_cat_id:
            continue
        bx, by, bw, bh = a["bbox"]
        centres[frame_of[a["image_id"]]] = (bx + bw / 2.0, by + bh / 2.0)
    return w, h, centres


def build_rally(
    rally_id: str,
    out_split_dir: Path,
    frame_src_dir: Path,
    frames: list[int],
    centres: dict[int, tuple[float, float]],
    median_samples: int,
) -> tuple[int, int]:
    """Materialise one rally dir. Returns (n_frames, n_visible)."""
    rally_dir = out_split_dir / rally_id
    frames_dir = rally_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, fidx in enumerate(frames):
        src = frame_src_dir / f"frame_{fidx:06d}.jpg"
        dst = frames_dir / f"{i}.png"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            os.symlink(os.path.relpath(src, dst.parent), dst)
        except OSError:
            cv2.imwrite(str(dst), cv2.imread(str(src)))

    rows = []
    n_vis = 0
    for i, fidx in enumerate(frames):
        c = centres.get(fidx)
        if c is None:
            rows.append(f"{i},0,0,0")
        else:
            rows.append(f"{i},1,{c[0]:.2f},{c[1]:.2f}")
            n_vis += 1
    (rally_dir / "label.csv").write_text("Frame,Visibility,X,Y\n" + "\n".join(rows) + "\n")

    step = max(1, len(frames) // median_samples)
    sample_idx = frames[::step][:median_samples]
    stack = []
    for fidx in sample_idx:
        img = cv2.imread(str(frame_src_dir / f"frame_{fidx:06d}.jpg"))
        if img is None:
            continue
        stack.append(cv2.resize(img, (MEDIAN_W, MEDIAN_H)))
    if stack:
        median = np.median(np.stack(stack), axis=0).astype(np.uint8)
    else:
        median = np.zeros((MEDIAN_H, MEDIAN_W, 3), dtype=np.uint8)
    np.savez_compressed(rally_dir / "median.npz", median=median)
    return len(frames), n_vis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rally_len", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--min_ball_ratio", type=float, default=0.05)
    ap.add_argument("--median_samples", type=int, default=64)
    args = ap.parse_args()

    if not PT100_FRAMES.exists():
        raise SystemExit(f"padelTracker100 frames not found: {PT100_FRAMES}")

    manifests = {"train": [], "val": []}
    totals = {"train": [0, 0], "val": [0, 0]}  # [frames, visible]

    for json_name, stem in SOURCES:
        coco_path = PT100_RAW / "labels" / json_name
        frame_src_dir = PT100_FRAMES / stem
        if not coco_path.exists() or not frame_src_dir.is_dir():
            print(f"  WARNING: missing {coco_path} or {frame_src_dir}, skipping")
            continue
        _, _, centres = load_ball_centres(coco_path)
        frames = sorted(set(range(max(centres) + 1)) | set(centres.keys()))
        n_chunks = (len(frames) + args.rally_len - 1) // args.rally_len
        n_val_chunks = max(1, int(round(n_chunks * args.val_frac)))
        print(f"{stem}: {len(frames)} frames -> {n_chunks} rallies of <= {args.rally_len}")

        for ci in range(n_chunks):
            chunk = frames[ci * args.rally_len:(ci + 1) * args.rally_len]
            if len(chunk) < 8:
                continue
            vis_ratio = sum(1 for f in chunk if f in centres) / len(chunk)
            if vis_ratio < args.min_ball_ratio:
                continue
            split = "val" if ci >= n_chunks - n_val_chunks else "train"
            rally_id = f"{stem}_r{ci:04d}"
            nf, nv = build_rally(
                rally_id, OUT_DIR / split, frame_src_dir, chunk, centres, args.median_samples
            )
            manifests[split].append(rally_id)
            totals[split][0] += nf
            totals[split][1] += nv

    for split in ("train", "val"):
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"manifest_{split}.txt").write_text("\n".join(manifests[split]) + "\n")
        tf, tv = totals[split]
        ratio = (tv / tf * 100) if tf else 0
        print(f"  {split}: {len(manifests[split])} rallies | {tf} frames | {tv} ball frames ({ratio:.1f}%)")

    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
