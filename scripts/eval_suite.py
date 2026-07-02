#!/usr/bin/env python3
"""
eval_suite.py
=============
Comprehensive evaluation suite for all trained padel analytics models.

Runs each available model on held-out test data and produces a unified
report with per-model metrics, cross-video comparison, and class-wise
analysis for shot classification.

Usage:
    python scripts/eval_suite.py
    python scripts/eval_suite.py --models player court bodypose shotclass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "player": {
        "weights": PROJECT_ROOT / "data/models/player_best.pt",
        "config": "configs/detection_combined.yaml",
        "type": "detect",
    },
    "court": {
        "weights": PROJECT_ROOT / "data/models/court_best.pt",
        "config": "configs/pose.yaml",
        "type": "pose",
    },
    "bodypose": {
        "weights": PROJECT_ROOT / "data/models/bodypose_best.pt",
        "config": "configs/bodypose.yaml",
        "type": "pose",
    },
    "shotclass": {
        "weights": PROJECT_ROOT / "data/models/shotclass_best.pt",
        "config": "configs/shotclass.yaml",
        "type": "detect",
    },
}

VIDEO_DIRS = {
    "FinalF": PROJECT_ROOT / "data/datasets/padeltracker100/frames/FinalF",
    "FinalM": PROJECT_ROOT / "data/datasets/padeltracker100/frames/FinalM",
}


def sample_frames(video_dir: Path, n: int = 150) -> list[str]:
    files = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in (".jpg",))
    if len(files) <= n:
        return [str(f) for f in files]
    step = len(files) // n
    return [str(files[i]) for i in range(0, len(files), step)][:n]


def eval_detect(weights: Path, images: list[str]) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    results = model.predict(images, verbose=False, conf=0.25)

    total_dets = 0
    total_conf = 0.0
    zero_det = 0
    class_counts: dict[str, int] = {}

    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        total_dets += n
        if n == 0:
            zero_det += 1
        else:
            total_conf += float(r.boxes.conf.sum())
            if hasattr(r, "names") and r.boxes.cls is not None:
                for cls_id in r.boxes.cls.tolist():
                    name = r.names.get(int(cls_id), str(int(cls_id)))
                    class_counts[name] = class_counts.get(name, 0) + 1

    n = len(images)
    return {
        "images": n,
        "avg_dets": round(total_dets / n, 2) if n else 0,
        "avg_conf": round(total_conf / total_dets, 4) if total_dets else 0,
        "zero_det_pct": round(zero_det / n * 100, 1) if n else 0,
        "class_distribution": class_counts,
    }


def eval_pose(weights: Path, images: list[str]) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    results = model.predict(images, verbose=False, conf=0.25)

    total_dets = 0
    total_conf = 0.0
    total_kpt_vis = 0
    total_kpt = 0
    zero_det = 0

    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        total_dets += n
        if n == 0:
            zero_det += 1
        else:
            total_conf += float(r.boxes.conf.sum())
            if r.keypoints is not None:
                for i in range(len(r.keypoints)):
                    kp = r.keypoints[i]
                    if kp is not None and hasattr(kp, "shape"):
                        total_kpt += kp.shape[0]
                        if kp.shape[1] > 2:
                            total_kpt_vis += int((kp[:, 2] > 0.5).sum())

    n = len(images)
    return {
        "images": n,
        "avg_dets": round(total_dets / n, 2) if n else 0,
        "avg_conf": round(total_conf / total_dets, 4) if total_dets else 0,
        "zero_det_pct": round(zero_det / n * 100, 1) if n else 0,
        "kpt_visibility": round(total_kpt_vis / total_kpt, 4) if total_kpt else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation suite")
    parser.add_argument("--models", nargs="*",
                        default=list(MODELS.keys()),
                        help="Which models to evaluate")
    parser.add_argument("--n", type=int, default=150,
                        help="Frames per video to sample")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: print to stdout)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PADEL ANALYTICS — EVALUATION SUITE")
    print("=" * 70)

    report: dict = {}

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"  [skip] unknown model: {model_name}")
            continue

        info = MODELS[model_name]
        weights = info["weights"]

        if not weights.exists():
            print(f"\n  [{model_name}] SKIP — weights not found: {weights.name}")
            report[model_name] = {"status": "missing", "weights": str(weights)}
            continue

        print(f"\n  [{model_name}] evaluating {weights.name}...")
        model_report: dict = {"weights": weights.name, "type": info["type"]}

        for video_name, video_dir in VIDEO_DIRS.items():
            if not video_dir.exists():
                continue
            images = sample_frames(video_dir, args.n)
            print(f"    {video_name}: {len(images)} frames...")

            if info["type"] == "pose":
                stats = eval_pose(weights, images)
            else:
                stats = eval_detect(weights, images)
            model_report[video_name] = stats

            print(f"      avg_dets={stats['avg_dets']}  avg_conf={stats['avg_conf']}"
                  f"  zero_det={stats['zero_det_pct']}%")

        # Cross-video comparison
        ff = model_report.get("FinalF", {})
        fm = model_report.get("FinalM", {})
        if ff and fm:
            conf_gap = abs(ff.get("avg_conf", 0) - fm.get("avg_conf", 0))
            det_gap = abs(ff.get("avg_dets", 0) - fm.get("avg_dets", 0))
            model_report["cross_video_gap"] = {
                "conf_gap": round(conf_gap, 4),
                "det_gap": round(det_gap, 2),
                "assessment": "good" if conf_gap < 0.05 else "moderate" if conf_gap < 0.15 else "poor",
            }
            print(f"    cross-video: conf_gap={conf_gap:.4f} det_gap={det_gap:.2f}"
                  f"  → {model_report['cross_video_gap']['assessment']}")

        report[model_name] = model_report

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<15s} {'Type':<7s} {'FinalF conf':>12s} {'FinalM conf':>12s} {'Gap':>8s} {'Verdict':>10s}")
    print(f"  {'-'*15} {'-'*7} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

    for name, r in report.items():
        if r.get("status") == "missing":
            print(f"  {name:<15s} {'—':<7s} {'—':>12s} {'—':>12s} {'—':>8s} {'MISSING':>10s}")
            continue
        ff = r.get("FinalF", {}).get("avg_conf", "—")
        fm = r.get("FinalM", {}).get("avg_conf", "—")
        gap = r.get("cross_video_gap", {})
        gap_val = gap.get("conf_gap", "—")
        verdict = gap.get("assessment", "—")
        ff_s = f"{ff:.4f}" if isinstance(ff, float) else str(ff)
        fm_s = f"{fm:.4f}" if isinstance(fm, float) else str(fm)
        gap_s = f"{gap_val:.4f}" if isinstance(gap_val, float) else str(gap_val)
        print(f"  {name:<15s} {r.get('type','?'):<7s} {ff_s:>12s} {fm_s:>12s} {gap_s:>8s} {verdict:>10s}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  Report saved to: {out_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
