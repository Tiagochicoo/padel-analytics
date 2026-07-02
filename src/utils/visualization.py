"""
visualization.py
================
Drawing helpers for boxes, track IDs and pose keypoints.
Kept dependency-light (OpenCV only) so the same code runs on the Jetson.
"""

from __future__ import annotations

from typing import Sequence

import cv2

# Court keypoint connection order (matches configs/pose.yaml keypoint_names)
COURT_SKELETON = [(0, 1), (1, 2), (2, 3), (3, 0)]


def draw_box(img, xyxy, label: str | None, color=(0, 255, 0), thickness=2):
    """Draw a bounding box with an optional label."""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_track(img, xyxy, track_id: int, color=(0, 255, 0)):
    """Draw a tracked box with its persistent ID."""
    draw_box(img, xyxy, f"ID:{track_id}", color=color)


def draw_keypoints(img, kpts: Sequence[Sequence[float]],
                   conf: float = 0.5,
                   color=(0, 165, 255),
                   skeleton=COURT_SKELETON,
                   radius: int = 6,
                   draw_skeleton: bool = True):
    """
    Draw court keypoints and connect them into a polygon/skeleton.

    Args:
        kpts: array-like of shape (N, 2|3): x, y, [visibility].
        conf: visibility threshold (only used when 3rd value present).
    """
    points = []
    for k in kpts:
        x, y = float(k[0]), float(k[1])
        visible = True
        if len(k) >= 3:
            visible = float(k[2]) >= conf
        if visible and x >= 0 and y >= 0:
            points.append((int(x), int(y)))
            cv2.circle(img, (int(x), int(y)), radius, color, -1)

    if draw_skeleton and len(points) >= 2:
        # Connect consecutive skeleton pairs that are present.
        idx_map = [i for i, k in enumerate(kpts)
                   if (len(k) < 3 or float(k[2]) >= conf) and float(k[0]) >= 0]
        for a, b in skeleton:
            if a in idx_map and b in idx_map:
                pa = (int(kpts[a][0]), int(kpts[a][1]))
                pb = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(img, pa, pb, color, 2)


def draw_hud(img, fps: float, n_objects: int):
    """Draw a small heads-up display with FPS and detection count."""
    cv2.rectangle(img, (0, 0), (240, 40), (0, 0, 0), -1)
    cv2.putText(img, f"FPS: {fps:5.1f}  | obj: {n_objects}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)


def make_writer(path: str, fps: float, size):
    """Create an mp4 writer matching the frame size."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)
