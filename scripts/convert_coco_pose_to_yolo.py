#!/usr/bin/env python3
"""
convert_coco_pose_to_yolo.py
============================
Convert padelTracker100 COCO pose annotations into Ultralytics YOLO-pose format.

Output layout::

    data/datasets/bodypose/
        images/
            train/FinalF_frame_000000.jpg  (symlink -> ../../frames/FinalF/frame_000000.jpg)
            val/...
        labels/
            train/FinalF_frame_000000.txt
            val/...

Each label line (YOLO-pose):
    class cx cy w h  x0 y0 v0  x1 y1 v1  ...  x16 y16 v16
All coordinates normalised to [0, 1].

Split strategy: temporal — first ``--train-frac`` of each video's frames go to
train, the rest to val.  This avoids near-duplicate consecutive frames leaking
across the split (which a random split would cause at 30 fps).

Usage:
    python scripts/convert_coco_pose_to_yolo.py
    python scripts/convert_coco_pose_to_yolo.py --train-frac 0.85 --symlink
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "raw" / "labels"
FRAMES_DIR = PROJECT_ROOT / "data" / "datasets" / "padeltracker100" / "frames"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "bodypose"

# (json_name, video_stem)
SOURCES = [
    ("2022_BCN_FinalF_1_pose.json", "FinalF"),
    ("2022_BCN_FinalM_1_pose.json", "FinalM"),
]

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Horizontal-flip swap indices (left <-> right), nose stays.
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def parse_frame_index(file_name: str) -> int:
    """frame_000123.PNG -> 123"""
    stem = Path(file_name).stem  # frame_000123
    return int(stem.split("_")[1])


def convert_one(coco_path: Path, video_stem: str, train_frac: float,
                out_root: Path, symlink: bool, stats: dict):
    data = json.loads(coco_path.read_text())
    w = data["images"][0]["width"]
    h = data["images"][0]["height"]

    # Group annotations by image_id
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in data["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Sort images by frame index (temporal order)
    images = sorted(data["images"], key=lambda im: parse_frame_index(im["file_name"]))
    split_idx = int(len(images) * train_frac)

    frames_subdir = FRAMES_DIR / video_stem

    for i, im in enumerate(images):
        split = "train" if i < split_idx else "val"
        frame_no = parse_frame_index(im["file_name"])
        img_stem = f"{video_stem}_frame_{frame_no:06d}"

        img_dst = out_root / "images" / split / f"{img_stem}.jpg"
        lbl_dst = out_root / "labels" / split / f"{img_stem}.txt"

        # Build label lines
        lines = []
        for a in anns_by_img.get(im["id"], []):
            bx, by, bw, bh = a["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            kpts = a.get("keypoints", [])
            if len(kpts) != 51:
                continue  # need exactly 17×3
            kpt_strs = []
            for j in range(17):
                kx = kpts[j * 3] / w
                ky = kpts[j * 3 + 1] / h
                kv = int(kpts[j * 3 + 2])
                # COCO visibility: 0=not labeled, 1=labeled/not visible, 2=visible
                # YOLO format is the same.
                kpt_strs.append(f"{kx:.6f} {ky:.6f} {kv}")
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_strs))

        # Skip images with no annotations entirely
        if not lines:
            continue

        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("\n".join(lines) + "\n")

        # Link or copy the image (symlink resolves once frames are extracted)
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        src_frame = frames_subdir / f"frame_{frame_no:06d}.jpg"
        if not src_frame.exists():
            stats["missing_frames"] += 1
            # still create symlink — it'll resolve after frame extraction
        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        if symlink:
            os.symlink(os.path.relpath(src_frame, img_dst.parent), img_dst)
        else:
            os.link(src_frame, img_dst) if False else None  # hardlink not cross-fs safe
            import shutil
            shutil.copy2(src_frame, img_dst)

        stats[split] += 1


def main():
    ap = argparse.ArgumentParser(description="Convert COCO pose -> YOLO-pose format")
    ap.add_argument("--train-frac", type=float, default=0.85,
                    help="fraction of each video for train (rest=val)")
    ap.add_argument("--symlink", action="store_true", default=True,
                    help="symlink images instead of copying (saves disk)")
    ap.add_argument("--copy", dest="symlink", action="store_false",
                    help="copy images instead of symlinking")
    args = ap.parse_args()

    print(f"==> Converting padelTracker100 COCO pose -> YOLO-pose")
    print(f"    Output: {OUT_DIR}")
    print(f"    Train/val split: {args.train_frac:.0%}/{1-args.train_frac:.0%} (temporal)")
    print(f"    Images: {'symlink' if args.symlink else 'copy'}")
    print(f"    Keypoints: 17 (COCO), kpt_shape=[17,3]")
    print(f"    flip_idx: {FLIP_IDX}")

    stats = {"train": 0, "val": 0, "missing_frames": 0}
    for json_name, stem in SOURCES:
        coco_path = LABELS_DIR / json_name
        if not coco_path.exists():
            print(f"  WARNING: {coco_path} not found, skipping")
            continue
        print(f"  Processing {stem} ({coco_path.name})...")
        convert_one(coco_path, stem, args.train_frac, OUT_DIR, args.symlink, stats)

    print(f"\n==> Done.")
    print(f"    Train images: {stats['train']:,}")
    print(f"    Val images:   {stats['val']:,}")
    print(f"    Missing frames (video not extracted yet): {stats['missing_frames']:,}")

    # Write dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"# Auto-generated by scripts/convert_coco_pose_to_yolo.py\n"
        f"# Body pose (17 COCO keypoints) from padelTracker100\n"
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
        f"\n"
        f"# 17 COCO keypoints\n"
        f"kpt_shape: [17, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
    )
    print(f"    Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
