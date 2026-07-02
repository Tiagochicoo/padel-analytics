#!/usr/bin/env python3
"""
convert_court_26.py
===================
Convert Josh's 26-class / 1-keypoint-per-object court dataset into
standard YOLO-pose format: 1 class ("court") with 26 keypoints.

Input  (per line):  class_id cx cy w h  kpt_x kpt_y kpt_v     (26 lines per image)
Output (per line):  0 cx cy w h  x0 y0 v0 x1 y1 v1 ... x25 y25 v25  (1 line per image)

Missing landmarks are filled with (0, 0, 0) — not labeled.

Usage:
    python scripts/convert_court_26.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data" / "datasets" / "court_26_josh"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "court_keypoints_26"
NUM_KPTS = 26

# Josh's class ordering (0-25):
KPT_NAMES = [
    "cage_bottom_left_close",   # 0
    "cage_bottom_left_far",     # 1
    "cage_bottom_right_close",  # 2
    "cage_bottom_right_far",    # 3
    "cage_top_left_close",      # 4
    "cage_top_left_far",        # 5
    "cage_top_right_close",     # 6
    "cage_top_right_far",       # 7
    "court_bottom_left_close",  # 8
    "court_bottom_left_far",    # 9
    "court_bottom_right_close", # 10
    "court_bottom_right_far",   # 11
    "court_top_left_close",     # 12
    "court_top_left_far",       # 13
    "court_top_right_close",    # 14
    "court_top_right_far",      # 15
    "net_bottom_left",          # 16
    "net_bottom_right",         # 17
    "net_top_left",             # 18
    "net_top_right",            # 19
    "service_centre_close",     # 20
    "service_centre_far",       # 21
    "service_left_close",       # 22
    "service_left_far",         # 23
    "service_right_close",      # 24
    "service_right_far",        # 25
]

# Horizontal-flip swap: left <-> right, close/far/centre stay.
FLIP_IDX = [
    2, 3, 0, 1,     # cage bottom: L<->R
    6, 7, 4, 5,     # cage top: L<->R
    10, 11, 8, 9,   # court bottom: L<->R
    14, 15, 12, 13, # court top: L<->R
    17, 16,         # net bottom: L<->R
    19, 18,         # net top: L<->R
    20, 21,         # service centre: close<->far (centre line, depth swap)
    24, 25, 22, 23, # service left/right: L<->R
]


def convert_label(in_path: Path, out_path: Path) -> bool:
    """Convert one label file. Returns True if output was written."""
    kpts: list[tuple[float, float, int] | None] = [None] * NUM_KPTS
    bboxes: list[tuple[float, float, float, float]] = []

    for line in in_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        kx, ky, kv = float(parts[5]), float(parts[6]), int(float(parts[7]))
        if 0 <= cls < NUM_KPTS:
            kpts[cls] = (kx, ky, kv)
            bboxes.append((cx, cy, w, h))

    if not bboxes:
        return False

    # Union bounding box of all landmarks
    xs_min = [b[0] - b[2] / 2 for b in bboxes]
    xs_max = [b[0] + b[2] / 2 for b in bboxes]
    ys_min = [b[1] - b[3] / 2 for b in bboxes]
    ys_max = [b[1] + b[3] / 2 for b in bboxes]
    ux = (min(xs_min) + max(xs_max)) / 2
    uy = (min(ys_min) + max(ys_max)) / 2
    uw = max(xs_max) - min(xs_min)
    uh = max(ys_max) - min(ys_min)

    # Build keypoint string (fill missing with 0 0 0)
    kpt_strs = []
    for i in range(NUM_KPTS):
        if kpts[i] is not None:
            kx, ky, kv = kpts[i]
            kpt_strs.append(f"{kx:.6f} {ky:.6f} {kv}")
        else:
            kpt_strs.append("0.000000 0.000000 0")

    out_path.write_text(
        f"0 {ux:.6f} {uy:.6f} {uw:.6f} {uh:.6f} " + " ".join(kpt_strs) + "\n"
    )
    return True


def process_split(split_in: str, split_out: str, stats: dict):
    img_src = SRC_DIR / split_in / "images"
    lbl_src = SRC_DIR / split_in / "labels"
    if not lbl_src.is_dir():
        return

    img_dst = OUT_DIR / "images" / split_out
    lbl_dst = OUT_DIR / "labels" / split_out
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for lbl_file in sorted(lbl_src.glob("*.txt")):
        out_lbl = lbl_dst / lbl_file.name
        if convert_label(lbl_file, out_lbl):
            # Symlink the image
            stem = lbl_file.stem
            for ext in (".jpg", ".jpeg", ".png"):
                img_file = img_src / (stem + ext)
                if img_file.exists():
                    img_link = img_dst / img_file.name
                    if not img_link.exists():
                        img_link.symlink_to(img_file.resolve())
                    break
            stats[split_out] += 1


def main():
    print(f"==> Converting Josh's 26-keypoint court dataset -> {OUT_DIR}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images" / "train").mkdir(parents=True)
    (OUT_DIR / "images" / "val").mkdir(parents=True)
    (OUT_DIR / "labels" / "train").mkdir(parents=True)
    (OUT_DIR / "labels" / "val").mkdir(parents=True)

    stats = {"train": 0, "val": 0}
    process_split("train", "train", stats)
    process_split("valid", "val", stats)
    process_split("test", "val", stats)

    # Write dataset.yaml
    yaml = OUT_DIR / "dataset.yaml"
    yaml.write_text(
        "# Auto-generated by scripts/convert_court_26.py\n"
        "# 26-keypoint court detection (Josh's Workspace + cage/court/net/service)\n"
        f"path: {OUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: court\n"
        "\n"
        f"kpt_shape: [26, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "\n"
        "# Keypoint names (0-25):\n"
        + "\n".join(f"#   {i}: {KPT_NAMES[i]}" for i in range(NUM_KPTS))
        + "\n"
    )

    total_kpts = sum(
        1 for f in (OUT_DIR / "labels" / "train").glob("*.txt")
        for line in f.read_text().strip().split("\n")
        for i in range(26)
        if len(line.split()) > 5 + i * 3 and float(line.split()[5 + i * 3 + 2]) > 0
    )
    print(f"\n==> Done.")
    print(f"    Train: {stats['train']:,} images")
    print(f"    Val:   {stats['val']:,} images")
    print(f"    Total visible keypoints (train): {total_kpts:,}")
    print(f"    Dataset YAML: {yaml}")
    print(f"    flip_idx: {FLIP_IDX}")


if __name__ == "__main__":
    main()
