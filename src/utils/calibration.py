"""
calibration.py
==============
Manual court calibration: the user clicks the four court corners on a
representative frame, and we derive everything the pipeline needs from the
court-keypoint model:

    * a homography  H : image px -> top-down court (meters)  -> 2D minimap
    * a court polygon (convex hull of the corners)           -> inside-court
                                                                player filter

This lets the court-aware 4-slot PlayerTracker (src/player_tracker.py) run
with ONLY the player-detection model, before the court-keypoint model is
trained (see docs/ROADMAP.md Component 2).

Corner order MUST match src/utils/homography.py REFERENCE_CORNERS:
    0: top_left, 1: top_right, 2: bottom_right, 3: bottom_left
("top" = far side of the court from the camera.)

Calibrations are persisted per source under data/calibrations/<source_id>.json
so they survive server restarts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.utils.homography import REFERENCE_CORNERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIB_DIR = PROJECT_ROOT / "data" / "calibrations"


@dataclass
class Calibration:
    source_id: str
    image_corners: list[tuple[float, float]]   # 4 (x, y) px, REFERENCE_CORNERS order
    frame_width: int
    frame_height: int

    def corners_array(self) -> np.ndarray:
        return np.asarray(self.image_corners, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def source_id_for(source: str) -> str:
    """Stable, filesystem-safe short id for a source (filepath or URL)."""
    s = str(source)
    base = Path(s).name or s
    stem = Path(base).stem or base
    # keep only filesystem-safe chars
    safe = "".join(c if c.isalnum() or c in "-." else "_" for c in stem).strip("._") or "source"
    h = hashlib.md5(s.encode()).hexdigest()[:8]
    return f"{safe}_{h}"


def _path(source_id: str) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    return CALIB_DIR / f"{source_id}.json"


def load_calibration(source_id: str) -> Optional[Calibration]:
    p = _path(source_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return Calibration(**data)
    except Exception as e:  # pragma: no cover
        print(f"[calibration] could not read {p}: {e}")
        return None


def load_for_source(source: str) -> Optional[Calibration]:
    return load_calibration(source_id_for(source))


def save_calibration(cal: Calibration) -> Path:
    p = _path(cal.source_id)
    p.write_text(json.dumps(asdict(cal), indent=2))
    return p


def delete_calibration(source_id: str) -> None:
    p = _path(source_id)
    if p.exists():
        p.unlink()


# ─────────────────────────────────────────────────────────────────────────────
def homography_from_calibration(cal: Calibration) -> Optional[np.ndarray]:
    """3x3 H mapping image px -> court meters, or None if corners degenerate.

    Uses a direct 4-point solve against REFERENCE_CORNERS (exact, no RANSAC):
    the manual calibration supplies exactly the four back-wall corners, so the
    overdetermined 26-point compute_homography path does not apply here.
    """
    H, _ = cv2.findHomography(cal.corners_array(), REFERENCE_CORNERS)
    return H


def polygon_from_calibration(cal: Calibration) -> np.ndarray:
    """Convex hull of the 4 corners as (K,1,2) float32, for cv2.pointPolygonTest."""
    pts = cal.corners_array().reshape(-1, 1, 2)
    return cv2.convexHull(pts)


def project_image_to_court(H: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Map a list of image (x,y) points into court meters using H."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


__all__ = [
    "Calibration", "REFERENCE_CORNERS",
    "source_id_for", "load_calibration", "load_for_source",
    "save_calibration", "delete_calibration",
    "homography_from_calibration", "polygon_from_calibration",
    "project_image_to_court",
]
