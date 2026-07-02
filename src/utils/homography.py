"""
homography.py
=============
Maps image-space court keypoints to a canonical top-down court coordinate
system (20 m × 10 m). Supports the 26-keypoint court model trained from
joshs-workspace-p1aa0/padel-court-detection.

Reference padel court dimensions (official):
    - 20 m long × 10 m wide (play area)
    - Net at midline (10 m from each back wall)
    - Service lines at 6.95 m from the back wall (3.05 m from net)
    - Cage (metal mesh + glass enclosure) surrounds the court

Coordinate system (top-down):
    X: 0 (left) → 10 (right) metres
    Y: 0 (near/camera side) → 20 (far side) metres

Keypoint indices (0–25) match configs/pose.yaml + scripts/convert_court_26.py.
Positions marked (ground) are projections of elevated cage/wall points.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

COURT_LENGTH_M = 20.0
COURT_WIDTH_M = 10.0
SERVICE_LINE_M = 6.95          # from back wall
SERVICE_LINE_FAR_M = 13.05     # 20 - 6.95

# ---------------------------------------------------------------------------
# 26-point reference template (metres, top-down)
# Index → (X, Y) on a 20×10 m court.
#
# "close"  = near half (Y ∈ [0, 10])
# "far"    = far half  (Y ∈ [10, 20])
# "top"    = back wall end of each half
# "bottom" = net / service line end of each half
#
# NOTE: These are best-estimate positions. After the model trains, verify
# by projecting detected keypoints onto the court map and adjust if needed.
# ---------------------------------------------------------------------------
REFERENCE_COURT_26 = np.array([
    # --- Cage corners (outer enclosure, ground projection) ---
    [0.0,  0.0],     #  0: cage_bottom_left_close
    [0.0,  20.0],    #  1: cage_bottom_left_far
    [10.0, 0.0],     #  2: cage_bottom_right_close
    [10.0, 20.0],    #  3: cage_bottom_right_far
    [0.0,  0.0],     #  4: cage_top_left_close (ground)
    [0.0,  20.0],    #  5: cage_top_left_far (ground)
    [10.0, 0.0],     #  6: cage_top_right_close (ground)
    [10.0, 20.0],    #  7: cage_top_right_far (ground)
    # --- Court corners (playing surface) ---
    [0.0,  0.0],     #  8: court_bottom_left_close (back wall, near-left)
    [0.0,  20.0],    #  9: court_bottom_left_far (back wall, far-left)
    [10.0, 0.0],     # 10: court_bottom_right_close (back wall, near-right)
    [10.0, 20.0],    # 11: court_bottom_right_far (back wall, far-right)
    [0.0,  SERVICE_LINE_M],      # 12: court_top_left_close (service line, near-left)
    [0.0,  SERVICE_LINE_FAR_M],  # 13: court_top_left_far (service line, far-left)
    [10.0, SERVICE_LINE_M],      # 14: court_top_right_close (service line, near-right)
    [10.0, SERVICE_LINE_FAR_M],  # 15: court_top_right_far (service line, far-right)
    # --- Net ---
    [0.0,  10.0],    # 16: net_bottom_left
    [10.0, 10.0],    # 17: net_bottom_right
    [0.0,  10.0],    # 18: net_top_left (ground)
    [10.0, 10.0],    # 19: net_top_right (ground)
    # --- Service line centre / lateral ---
    [5.0,  SERVICE_LINE_M],      # 20: service_centre_close
    [5.0,  SERVICE_LINE_FAR_M],  # 21: service_centre_far
    [0.0,  SERVICE_LINE_M],      # 22: service_left_close
    [0.0,  SERVICE_LINE_FAR_M],  # 23: service_left_far
    [10.0, SERVICE_LINE_M],      # 24: service_right_close
    [10.0, SERVICE_LINE_FAR_M],  # 25: service_right_far
], dtype=np.float32)

# ---------------------------------------------------------------------------
# 4-point reference template (metres, top-down) for MANUAL calibration.
# The user clicks the four back-wall court corners on a frame; the click order
# MUST match this array (see src/utils/calibration.py):
#     0: top_left, 1: top_right, 2: bottom_right, 3: bottom_left
# ("top" = far side of the court from the camera, Y = COURT_LENGTH_M.)
# These are the same back-wall playing-surface corners as REFERENCE_COURT_26
# indices 9, 11, 10, 8, reordered into click order.
# ---------------------------------------------------------------------------
REFERENCE_CORNERS = np.array([
    [0.0,           COURT_LENGTH_M],   # 0: top_left     (far-left back wall)
    [COURT_WIDTH_M, COURT_LENGTH_M],   # 1: top_right    (far-right back wall)
    [COURT_WIDTH_M, 0.0],              # 2: bottom_right (near-right back wall)
    [0.0,           0.0],              # 3: bottom_left  (near-left back wall)
], dtype=np.float32)

# Indices of the 4 primary court corners used for the inside-court polygon.
# These are the back wall + service line corners that define the playable area.
COURT_POLYGON_INDICES = [8, 9, 10, 11, 12, 13, 14, 15]

# Indices of net keypoints (useful for serve-box detection in the rules engine)
NET_INDICES = [16, 17, 18, 19]


def compute_homography(
    image_points: Sequence[Sequence[float]],
    visible_mask: Sequence[bool] | None = None,
):
    """
    Compute the 3×3 homography H mapping image pixels → top-down court (metres).

    Uses all visible 26 keypoints (overdetermined system → robust).

    Args:
        image_points: 26 court keypoints in image coords (x, y), index 0–25.
        visible_mask: optional list of 26 booleans; False = skip that point.
                      If None, points at (0, 0) are treated as not visible.

    Returns:
        (H, mask) from cv2.findHomography, or (None, None) if too few visible.
    """
    pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)

    if visible_mask is not None:
        vis = np.asarray(visible_mask, dtype=bool)
    else:
        # A point at exactly (0, 0) is "not labeled" in YOLO-pose format
        vis = ~((pts[:, 0] == 0) & (pts[:, 1] == 0))

    if vis.sum() < 4:
        print(f"[homography] need ≥ 4 visible points, got {vis.sum()}")
        return None, None

    src = pts[vis]
    dst = REFERENCE_COURT_26[vis]

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H, mask


def project_points(H, image_points):
    """Transform image points to court coordinates (metres) using H."""
    if H is None:
        return None
    pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    court = cv2.perspectiveTransform(pts, H)
    return court.reshape(-1, 2)


def court_polygon_image_space(image_keypoints: np.ndarray) -> np.ndarray | None:
    """
    Extract the court play-area polygon from 26 image keypoints.

    Returns an array of (x, y) pixel coords forming the court boundary,
    or None if too few corners are visible.
    """
    pts = np.asarray(image_keypoints).reshape(-1, 2)
    poly_pts = pts[COURT_POLYGON_INDICES]
    # Filter out (0, 0) placeholders
    vis = ~((poly_pts[:, 0] == 0) & (poly_pts[:, 1] == 0))
    if vis.sum() < 3:
        return None
    return poly_pts[vis]


# ---------------------------------------------------------------------------
# Zone predicates — court-position reasoning for the rules engine.
# All operate in top-down court coordinates (metres): X ∈ [0, 10], Y ∈ [0, 20].
# ---------------------------------------------------------------------------

_TOLERANCE_M = 0.5  # grace margin for in/out calls (court-line detection jitter)


def is_in_play(court_xy: Sequence[float]) -> bool:
    """True if the point is inside the court playing area (with tolerance)."""
    if court_xy is None or len(court_xy) < 2:
        return False
    x, y = float(court_xy[0]), float(court_xy[1])
    return (-_TOLERANCE_M <= x <= COURT_WIDTH_M + _TOLERANCE_M and
            -_TOLERANCE_M <= y <= COURT_LENGTH_M + _TOLERANCE_M)


def is_out(court_xy: Sequence[float]) -> bool:
    """True if the point is outside the court playing area."""
    return not is_in_play(court_xy)


def court_half(court_xy: Sequence[float]) -> int | None:
    """Return 0 (near half, Y < 10) or 1 (far half, Y >= 10), or None if invalid."""
    if court_xy is None or len(court_xy) < 2:
        return None
    return 0 if float(court_xy[1]) < COURT_LENGTH_M / 2 else 1


def is_in_service_box(court_xy: Sequence[float], half: int = 0) -> bool:
    """True if the point is in the service box of the given half.

    Near half (0): between service line (Y=6.95) and net (Y=10.0).
    Far half  (1): between net (Y=10.0) and far service line (Y=13.05).
    """
    if court_xy is None or len(court_xy) < 2:
        return False
    y = float(court_xy[1])
    if half == 0:
        return SERVICE_LINE_M <= y <= COURT_LENGTH_M / 2
    else:
        return COURT_LENGTH_M / 2 <= y <= SERVICE_LINE_FAR_M


def is_near_net(court_xy: Sequence[float], threshold: float = 1.0) -> bool:
    """True if within *threshold* metres of the net (Y = 10)."""
    if court_xy is None or len(court_xy) < 2:
        return False
    return abs(float(court_xy[1]) - COURT_LENGTH_M / 2) <= threshold
