#!/usr/bin/env python3
"""
build_ball_dataset.py
=====================
Build the unified ball detection dataset from:
  1. Roboflow: padel/padel-ball-detector (already in data/datasets/ball_roboflow/)
  2. padelTracker100 ball JSONs (subsampled every Nth frame)

Output: data/datasets/ball/ with single class "ball" (id 0), YOLO format.

Usage:
    python scripts/build_ball_dataset.py [--stride 5]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PT100_RAW = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "raw"
PT100_FRAMES = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "frames"
ROBOFLOW_BALL = PROJECT_ROOT / "data" / "datasets" / "ball_roboflow"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "ball"

SOURCES = [
    ("2022_BCN_FinalF_1_ball.json", "FinalF"),
    ("2022_BCN_FinalM_1_ball.json", "FinalM"),
]


def parse_frame_index(file_name: str) -> int:
    return int(Path(file_name).stem.split("_")[1])


def add_padeltracker(stride: int, stats: dict):
    """Convert padelTracker100 ball JSONs → YOLO labels, symlink frames."""
    for json_name, stem in SOURCES:
        coco_path = PT100_RAW / "labels" / json_name
        if not coco_path.exists():
            print(f"  WARNING: {coco_path} not found")
            continue
        data = json.loads(coco_path.read_text())
        w = data["images"][0]["width"]
        h = data["images"][0]["height"]

        # Find the "Ball" category id
        ball_cat_id = None
        for c in data["categories"]:
            if c["name"].lower() == "ball":
                ball_cat_id = c["id"]
                break
        if ball_cat_id is None:
            print(f"  WARNING: no 'Ball' category in {json_name}")
            continue

        # Group annotations by image
        anns_by_img: dict[int, list] = defaultdict(list)
        for a in data["annotations"]:
            if a["category_id"] == ball_cat_id:
                anns_by_img[a["image_id"]].append(a)

        images = sorted(data["images"], key=lambda im: parse_frame_index(im["file_name"]))
        split_idx = int(len(images) * 0.85)
        frames_subdir = PT100_FRAMES / stem

        count = 0
        for i, im in enumerate(images):
            frame_no = parse_frame_index(im["file_name"])
            if frame_no % stride != 0:
                continue
            anns = anns_by_img.get(im["id"], [])
            if not anns:
                continue  # skip frames without ball

            split = "train" if i < split_idx else "val"
            img_stem = f"{stem}_frame_{frame_no:06d}"

            # Build label
            lines = []
            for a in anns:
                bx, by, bw, bh = a["bbox"]
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            lbl_dst = OUT_DIR / "labels" / split / f"{img_stem}.txt"
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("\n".join(lines) + "\n")

            # Symlink image
            img_dst = OUT_DIR / "images" / split / f"{img_stem}.jpg"
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            src = frames_subdir / f"frame_{frame_no:06d}.jpg"
            if img_dst.is_symlink() or img_dst.exists():
                img_dst.unlink()
            os.symlink(os.path.relpath(src, img_dst.parent), img_dst)

            stats[split] += 1
            count += 1
        print(f"  {stem}: {count} frames with ball (stride {stride})")


def add_roboflow(stats: dict):
    """Copy Roboflow ball dataset, remap all classes to single 'ball' (0)."""
    if not ROBOFLOW_BALL.exists():
        print("  WARNING: Roboflow ball dataset not found, skipping")
        return

    for split_in, split_out in [("train", "train"), ("valid", "val"), ("test", "train")]:
        img_src = ROBOFLOW_BALL / split_in / "images"
        lbl_src = ROBOFLOW_BALL / split_in / "labels"
        if not img_src.is_dir():
            continue
        count = 0
        for img_file in sorted(img_src.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_file = lbl_src / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue

            # Remap all classes to 0 (ball)
            lines = []
            for line in lbl_file.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"  # force class 0
                    lines.append(" ".join(parts))

            if not lines:
                continue

            stem = f"rf_{img_file.stem}"
            lbl_dst = OUT_DIR / "labels" / split_out / f"{stem}.txt"
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.write_text("\n".join(lines) + "\n")

            img_dst = OUT_DIR / "images" / split_out / f"{stem}.jpg"
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            if img_dst.is_symlink() or img_dst.exists():
                img_dst.unlink()
            os.symlink(os.path.relpath(img_file, img_dst.parent), img_dst)

            stats[split_out] += 1
            count += 1
        print(f"  Roboflow {split_in}: {count} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5,
                    help="subsample padelTracker100 every N frames")
    args = ap.parse_args()

    print(f"==> Building unified ball dataset -> {OUT_DIR}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images" / "train").mkdir(parents=True)
    (OUT_DIR / "images" / "val").mkdir(parents=True)
    (OUT_DIR / "labels" / "train").mkdir(parents=True)
    (OUT_DIR / "labels" / "val").mkdir(parents=True)

    stats = {"train": 0, "val": 0}
    print("  -- padelTracker100 --")
    add_padeltracker(args.stride, stats)
    print("  -- Roboflow --")
    add_roboflow(stats)

    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        "# Auto-generated by scripts/build_ball_dataset.py\n"
        "# Ball detection: padelTracker100 + Roboflow padel-ball-detector\n"
        f"path: {OUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: ball\n"
    )
    print(f"\n==> Done. Train: {stats['train']:,}  Val: {stats['val']:,}")
    print(f"    Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
