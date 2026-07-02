"""
train.py
========
Unified training entry point for both tasks:

    # Player detection (YOLOv26 detection head)
    python src/train.py --task detection
    python src/train.py --task detection --epochs 200 --batch 4

    # Court keypoints (YOLOv26 pose head)
    python src/train.py --task pose

Behavior:
    * Reads base weights + hyperparameter defaults from configs/*.yaml.
    * Lets you override any hyperparameter on the CLI.
    * Saves to runs/{detect,pose}/train/weights/best.pt

CLI overrides are passed straight to Ultralytics model.train(**kwargs),
so anything Ultralytics supports works here (imgsz, lr0, cos_lr, mosaic, ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_latest_best(taskdir: str) -> Path | None:
    """Find the most recently modified best.pt under runs/<taskdir>/."""
    base = PROJECT_ROOT / "runs" / taskdir
    if not base.is_dir():
        return None
    candidates = sorted(
        (d / "weights" / "best.pt" for d in base.iterdir() if d.is_dir()),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def load_config(task: str) -> dict:
    cfg_path = PROJECT_ROOT / "configs" / f"{task}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_train_kwargs(cfg: dict, cli_overrides: list[str]) -> dict:
    """
    Merge config `hyp:` defaults with CLI `KEY=VALUE` overrides.
    Example override: ["epochs=200", "batch=4", "lr0=0.001"]
    """
    kwargs = dict(cfg.get("hyp", {}))
    for item in cli_overrides:
        if "=" not in item:
            raise SystemExit(f"Override must be KEY=VALUE, got: {item!r}")
        key, val = item.split("=", 1)
        # try to coerce numerics/bools
        try:
            val_parsed: float | bool | str
            if val.lower() in ("true", "false"):
                val_parsed = val.lower() == "true"
            else:
                val_parsed = float(val)
                if val_parsed.is_integer():
                    val_parsed = int(val_parsed)
        except ValueError:
            val_parsed = val
        kwargs[key.strip()] = val_parsed
    return kwargs


def train(task: str, overrides: list[str]) -> None:
    from ultralytics import YOLO

    cfg = load_config(task)
    base = cfg.get("base")
    print(f"[train] task={task} base={base}")

    model = YOLO(base)

    # `data` must point at the data.yaml for this task.
    data_yaml = PROJECT_ROOT / "configs" / f"{task}.yaml"
    kwargs = build_train_kwargs(cfg, overrides)
    kwargs["data"] = str(data_yaml)

    print("[train] kwargs:", kwargs)
    results = model.train(**kwargs)

    # Convenience: copy best weights into data/models with a friendly name.
    is_pose = "pose" in task
    taskdir = "pose" if is_pose else "detect"
    weight_names = {
        # Nano variants (original training)
        "pose": "court_best.pt",
        "bodypose": "bodypose_best.pt",
        "ball": "ball_best.pt",
        "shotclass": "shotclass_best.pt",
        # Small variants (never overwrite nano — separate _s suffix)
        "detection_combined_s": "player_s_best.pt",
        "pose_s": "court_s_best.pt",
        "bodypose_s": "bodypose_s_best.pt",
        "shotclass_s": "shotclass_s_best.pt",
    }
    friendly_name = weight_names.get(task, "player_best.pt")
    best = _find_latest_best(taskdir)
    if best is not None:
        models_dir = PROJECT_ROOT / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        friendly = models_dir / friendly_name
        best.replace(friendly)
        print(f"[train] copied best weights -> {friendly}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv26 (detection or pose).")
    parser.add_argument("--task", required=True,
                        help="Config stem in configs/, e.g. detection, pose, detection_combined.")
    parser.add_argument(
        "overrides", nargs="*",
        help="Hyperparameter overrides as KEY=VALUE (e.g. epochs=200 batch=4).",
    )
    args = parser.parse_args()
    train(args.task, args.overrides)


if __name__ == "__main__":
    main()
