"""
src/shot_classifier.py
======================
Shot-type classification per player hit.

A shot is detected when the ball enters a player's hitting zone. Two trigger
modes:

  * **Wrist proximity** (preferred): when body-pose keypoints are available,
    the ball must be within ``_WRIST_THRESHOLD`` px of the nearest wrist
    keypoint (COCO indices 9/10). This is far more precise than box proximity.
  * **Box proximity** (fallback): when keypoints are absent (model not yet
    trained, or every-3rd-frame off-cycle), the ball must be inside an
    expanded rectangle around the player's bounding box.

Once triggered, the trained ``shotclass`` YOLO detector (11 padel shot types)
runs on a crop of that player to label the shot.

Graceful degradation: if ``shotclass`` weights are absent the classifier is a
no-op (returns []), so PadelAnalyzer runs fine while shotclass is still
training (it is the last YOLO task in the queue).

Output per hit: ``{"slot": int, "type": str, "conf": float}`` — the rules engine
stamps the frame and folds it into the active rally.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.analyzer import Player

_PROXIMITY_MARGIN = 60      # px grown around the player box to count as contact
_BOX_TOP_EXTRA = 40         # extra grow upward (racket reach is above the feet)
_CROP_PAD = 40              # px context kept around the player when cropping
_CONF_THRESHOLD = 0.35      # min shot-class confidence to accept a label
_COOLDOWN = 15              # frames between two attributed shots for the same slot
_WRIST_THRESHOLD = 80       # px: ball within this distance of wrist keypoint = contact
_KPT_VIS_THRESHOLD = 0.3    # min keypoint visibility to trust the coordinate


class ShotClassifier:
    def __init__(self, shotclass_model, device: str = "0",
                 names: Optional[dict[int, str]] = None):
        self.model = shotclass_model                  # Ultralytics YOLO or None
        self.device = device
        self.names = names or (shotclass_model.names if shotclass_model is not None else {})
        self._last_hit: dict[int, int] = {}           # slot -> last attributed frame

    def classify(self, frame: np.ndarray, players: list[Player],
                 ball_xy: Optional[tuple[float, float]],
                 frame_idx: int = -1) -> list[dict]:
        if self.model is None or ball_xy is None:
            return []
        bx, by = float(ball_xy[0]), float(ball_xy[1])
        shots: list[dict] = []
        for p in players:
            if p.box is None:
                continue
            if frame_idx - self._last_hit.get(p.track_id, -10_000) < _COOLDOWN:
                continue
            if not self._is_contact(bx, by, p):
                continue
            label, conf = self._classify_crop(frame, p.box)
            if label is None or conf < _CONF_THRESHOLD:
                continue
            self._last_hit[p.track_id] = frame_idx
            shots.append({"slot": p.track_id, "type": label, "conf": float(conf)})
        return shots

    def _is_contact(self, bx: float, by: float, player: Player) -> bool:
        """Decide if the ball is in contact range of the player.

        Uses wrist-keypoint proximity when pose data is available (precise),
        falls back to box-proximity otherwise.
        Returns True only when contact is confirmed; returns True (fall through
        to box) when no keypoints are available.
        """
        kpts = getattr(player, "keypoints", None)
        if kpts is not None and len(kpts) > 10:
            wrists = []
            for idx in (9, 10):
                kx, ky, kv = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
                if kv >= _KPT_VIS_THRESHOLD:
                    wrists.append((kx, ky))
            if wrists:
                nearest = min(wrists, key=lambda w: (w[0] - bx) ** 2 + (w[1] - by) ** 2)
                dist = ((nearest[0] - bx) ** 2 + (nearest[1] - by) ** 2) ** 0.5
                return dist <= _WRIST_THRESHOLD
        # Fall back to box proximity
        return self._in_hitting_zone(bx, by, player.box)

    @staticmethod
    def _in_hitting_zone(bx: float, by: float, box: tuple) -> bool:
        x1, y1, x2, y2 = box
        return (x1 - _PROXIMITY_MARGIN) <= bx <= (x2 + _PROXIMITY_MARGIN) and \
               (y1 - _PROXIMITY_MARGIN - _BOX_TOP_EXTRA) <= by <= (y2 + _PROXIMITY_MARGIN)

    def _classify_crop(self, frame: np.ndarray, box: tuple) -> tuple[Optional[str], float]:
        h, w = frame.shape[:2]
        x1 = max(0, int(box[0]) - _CROP_PAD)
        y1 = max(0, int(box[1]) - _CROP_PAD - _BOX_TOP_EXTRA)
        x2 = min(w, int(box[2]) + _CROP_PAD)
        y2 = min(h, int(box[3]) + _CROP_PAD)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, 0.0
        crop = frame[y1:y2, x1:x2]
        try:
            res = self.model(crop, verbose=False, conf=_CONF_THRESHOLD, imgsz=640)
        except Exception:
            return None, 0.0
        boxes = getattr(res[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, 0.0
        confs = boxes.conf.cpu().numpy()
        idx = int(confs.argmax())
        cls = int(boxes.cls[idx].item())
        return self.names.get(cls, str(cls)), float(confs[idx])


__all__ = ["ShotClassifier"]
