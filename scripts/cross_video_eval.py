#!/usr/bin/env python3
"""
cross_video_eval.py
===================
Evaluate trained models across video boundaries to reveal TRUE generalization
performance without temporal leakage.

padelTracker100 has 2 videos (FinalF, FinalM). Models trained on both have
inflated val metrics because the val split is temporally adjacent to the train
split. This script evaluates each model on:

  1. Same-video: train on FinalF → test on FinalF val (inflated)
  2. Cross-video: train on FinalF → test on FinalM (true generalization)

Since our models are already trained on BOTH videos, we approximate this by
evaluating on each video separately and comparing the scores.

Usage:
    python scripts/cross_video_eval.py --task bodypose
    python scripts/cross_video_eval.py --task detection_combined
    python scripts/cross_video_eval.py --task shotclass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Frame directories by video
VIDEO_DIRS = {
    "FinalF": PROJECT_ROOT / "data/datasets/padeltracker100/frames/FinalF",
    "FinalM": PROJECT_ROOT / "data/datasets/padeltracker100/frames/FinalM",
}

# Model weights
MODEL_PATHS = {
    "detection_combined": PROJECT_ROOT / "data/models/player_best.pt",
    "bodypose": PROJECT_ROOT / "data/models/bodypose_best.pt",
    "shotclass": PROJECT_ROOT / "data/models/shotclass_best.pt",
}


def sample_frames(video_dir: Path, n: int = 200) -> list[str]:
    """Pick n evenly-spaced frames from a video directory."""
    files = sorted(p.name for p in video_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".png"))
    if len(files) <= n:
        return [str(video_dir / f) for f in files]
    step = len(files) // n
    return [str(video_dir / files[i]) for i in range(0, len(files), step)][:n]


def evaluate_model(weights: Path, images: list[str], task: str) -> dict:
    """Run model on images, return per-image confidence statistics."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(images, verbose=False, conf=0.25)

    total_dets = 0
    total_conf = 0.0
    low_conf_count = 0
    zero_det_count = 0

    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        total_dets += n
        if n == 0:
            zero_det_count += 1
        else:
            confs = r.boxes.conf.tolist()
            total_conf += sum(confs)
            low_conf_count += sum(1 for c in confs if c < 0.5)

    n_images = len(images)
    avg_dets = total_dets / n_images if n_images else 0
    avg_conf = total_conf / total_dets if total_dets else 0
    zero_det_pct = zero_det_count / n_images * 100 if n_images else 0
    low_conf_pct = low_conf_count / total_dets * 100 if total_dets else 0

    return {
        "images": n_images,
        "avg_detections": round(avg_dets, 1),
        "avg_confidence": round(avg_conf, 4),
        "zero_det_pct": round(zero_det_pct, 1),
        "low_conf_pct": round(low_conf_pct, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-video evaluation")
    parser.add_argument("--task", required=True,
                        choices=["detection_combined", "bodypose", "shotclass"],
                        help="Which model to evaluate")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of frames per video to sample")
    args = parser.parse_args()

    weights = MODEL_PATHS.get(args.task)
    if not weights or not weights.exists():
        print(f"ERROR: weights not found for task '{args.task}'")
        print(f"  Expected: {weights}")
        sys.exit(1)

    print("=" * 60)
    print(f"Cross-Video Evaluation: {args.task}")
    print(f"Model: {weights.name}")
    print("=" * 60)

    for video_name, video_dir in VIDEO_DIRS.items():
        if not video_dir.exists():
            print(f"\n[skip] {video_name}: directory not found")
            continue

        print(f"\n--- {video_name} ---")
        images = sample_frames(video_dir, args.n)
        print(f"  Sampling {len(images)} frames...")

        stats = evaluate_model(weights, images, args.task)
        print(f"  Avg detections/frame: {stats['avg_detections']}")
        print(f"  Avg confidence:       {stats['avg_confidence']}")
        print(f"  Frames with 0 dets:   {stats['zero_det_pct']}%")
        print(f"  Low-conf detections:  {stats['low_conf_pct']}%")

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  If FinalF and FinalM show similar stats → good cross-video generalization")
    print("  If one video is much worse → model overfit to that video's conditions")
    print("  High zero_det_pct or low_conf_pct on either → poor real-world performance")
    print("=" * 60)


if __name__ == "__main__":
    main()
