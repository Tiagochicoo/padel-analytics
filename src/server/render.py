"""
render.py
=========
Shared drawing of an AnalysisResult onto a frame, used by both the live
pipeline and the offline analysis jobs so the on-screen overlay is identical.

Draws: the manual court outline + corner numbers, per-player boxes labelled by
their stable slot (P1..P4) with a colour-coded feet marker.
"""

from __future__ import annotations

import cv2

from src.utils.calibration import polygon_from_calibration
from src.utils.visualization import draw_box
from src.server.stats import slot_color_bgr


def render_result(frame, result, manual_calibration=None) -> int:
    """Draw the analysis overlay in-place. Returns the player count."""
    if manual_calibration is not None:
        poly = polygon_from_calibration(manual_calibration).reshape(-1, 2).astype(int)
        cv2.polylines(frame, [poly], True, (90, 90, 90), 2)
        for i, (x, y) in enumerate(poly):
            cv2.circle(frame, (int(x), int(y)), 5, (200, 200, 200), -1)
            cv2.putText(frame, str(i), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    n = 0
    for p in result.players:
        n += 1
        slot = int(p.track_id)
        color = slot_color_bgr(slot)
        label = f"P{slot}"
        if p.outside_court:
            label += " OUT"
            color = tuple(int(c * 0.4) for c in color)  # dim the color
        draw_box(frame, p.box, label, color=color)
        cx = int((p.box[0] + p.box[2]) / 2)
        cy = int(p.box[3])
        cv2.circle(frame, (cx, cy), 4, color, -1)
    return n
