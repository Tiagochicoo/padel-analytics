"""
PlayerTracker — 4-slot player tracker for padel courts.

Pure NumPy (no OpenCV). Designed for fixed-camera padel: exactly 4 players
on court for the entire match. Spectators, referees, or brief walk-throughs
are filtered by requiring sustained visibility before slot assignment.

SLOT LOGIC (the key difference from generic trackers):
- Every track accumulates `total_visible_frames` across its entire lifetime
  (including gaps where it was predicted but not matched).
- A track only becomes a slot candidate after `min_track_frames` total
  visible frames (default: 90 frames = 3 seconds at 30fps).
- The FIRST 4 tracks to cross that threshold get slots P1-P4 permanently.
- A 5th person who appears for 20 frames then leaves → never gets a slot.
- Once assigned, a slot is never freed or reassigned. If P2 walks off court
  and comes back 2 minutes later, the tracker (via IoU + Kalman prediction)
  re-acquires the same track_id and keeps slot P2.
- Confirmed (slotted) tracks are NEVER pruned — they survive the whole match.
- Unconfirmed tracks are pruned after `max_age` frames without a match.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class PlayerTrack:
    """A single tracked player."""
    track_id: int
    slot: int = 0  # 1-4 once assigned, 0 = unassigned
    kf_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kf_cov: np.ndarray = field(default_factory=lambda: np.eye(6) * 100)
    age: int = 0  # total frames since birth
    hits: int = 0  # total matched detections
    total_visible: int = 0  # accumulated visible frames (across gaps)
    time_since_update: int = 0  # frames since last detection match
    confirmed: bool = False  # True once slot assigned
    last_bbox: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def cx(self) -> float:
        return float(self.kf_state[0])

    @property
    def cy(self) -> float:
        return float(self.kf_state[1])

    @property
    def bbox(self) -> tuple:
        cx, cy, w, h = self.kf_state[:4]
        return (float(cx - w / 2), float(cy - h / 2),
                float(cx + w / 2), float(cy + h / 2))


# ── Kalman constants ──
_F = np.array([
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=np.float64)

_H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
], dtype=np.float64)

_Q = np.eye(6, dtype=np.float64) * 5.0
_R = np.eye(4, dtype=np.float64) * 8.0


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


class PlayerTracker:
    """
    Track exactly 4 padel players with permanent slot IDs.

    Slot assignment is TIME-BASED: a player must be visible for
    `min_track_frames` total frames before getting a slot. This
    filters out spectators and brief detections.

    Parameters
    ----------
    min_track_frames : int
        Total visible frames required before slot assignment.
        Default 90 = 3 seconds at 30fps. A person seen for only
        20 frames will never get a slot.
    max_players : int
        Hard cap on slots (4 for padel doubles).
    max_age : int
        Frames a confirmed track survives without any detection
        before being pruned. Set high (e.g. 600 = 20 seconds) so
        players who temporarily leave frame aren't lost.
    max_age_unconfirmed : int
        Frames an unconfirmed track survives without detection.
        Lower (e.g. 30) so ghost tracks from false detections
        are cleaned up quickly.
    min_box_h : float
        Minimum bbox height in pixels. Filters out small / distant
        non-player detections (spectators in the background).
    """

    def __init__(
        self,
        min_track_frames: int = 90,
        max_players: int = 4,
        max_age: int = 600,
        max_age_unconfirmed: int = 30,
        iou_threshold: float = 0.2,
        min_box_h: float = 50.0,
        img_w: int = 1280,
        img_h: int = 720,
    ):
        self.min_track_frames = min_track_frames
        self.max_players = max_players
        self.max_age = max_age
        self.max_age_unconfirmed = max_age_unconfirmed
        self.iou_threshold = iou_threshold
        self.min_box_h = min_box_h
        self.img_w = img_w
        self.img_h = img_h

        self._tracks: dict[int, PlayerTrack] = {}
        self._next_id = 1
        self._used_slots: set[int] = set()

    def predict(self):
        """Advance all tracks by one frame (Kalman predict step)."""
        for tr in self._tracks.values():
            tr.kf_state = _F @ tr.kf_state
            tr.kf_cov = _F @ tr.kf_cov @ _F.T + _Q
            tr.age += 1
            tr.time_since_update += 1

    def update(self, detections: list) -> list[dict]:
        """
        Match detections to tracks, update Kalman, return active players.

        Args:
            detections: [(x1, y1, x2, y2, confidence), ...] in pixel coords

        Returns:
            [{slot, track_id, confirmed, bbox, cx, cy, total_visible, hits}, ...]
        """
        # Filter: only accept boxes tall enough to be real players
        valid_dets = [
            (float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4] if len(d) > 4 else 0.5))
            for d in detections
            if (d[3] - d[1]) >= self.min_box_h
        ]

        track_list = list(self._tracks.values())
        n_t, n_d = len(track_list), len(valid_dets)

        if n_t == 0:
            for d in valid_dets:
                self._create(d)
            self._try_slot_assignment()
            return self._active()

        if n_d == 0:
            self._prune()
            return self._active()

        # IoU cost matrix
        cost = np.ones((n_t, n_d), dtype=np.float64)
        for i, tr in enumerate(track_list):
            tb = tr.bbox
            for j, d in enumerate(valid_dets):
                cost[i, j] = 1.0 - _iou(tb, d[:4])

        # Greedy matching
        matched_t, matched_d = set(), set()
        candidates = sorted(
            ((cost[i, j], i, j) for i in range(n_t) for j in range(n_d)
             if cost[i, j] < (1.0 - self.iou_threshold))
        )
        for _, i, j in candidates:
            if i in matched_t or j in matched_d:
                continue
            matched_t.add(i)
            matched_d.add(j)
            self._match(track_list[i], valid_dets[j])

        # Unmatched detections → new tracks
        for j in range(n_d):
            if j not in matched_d:
                self._create(valid_dets[j])

        # Try to assign slots to tracks that now have enough visibility
        self._try_slot_assignment()

        # Prune dead unconfirmed tracks
        self._prune()

        return self._active()

    def _create(self, det):
        x1, y1, x2, y2, _ = det[:5]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = PlayerTrack(
            track_id=tid,
            kf_state=np.array([cx, cy, max(w, 1), max(h, 1), 0, 0], dtype=np.float64),
            kf_cov=np.eye(6) * 50,
            age=1, hits=1, total_visible=1, time_since_update=0,
            last_bbox=(x1, y1, x2, y2),
        )

    def _match(self, tr: PlayerTrack, det):
        x1, y1, x2, y2, _ = det[:5]
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2,
                       max(x2 - x1, 1), max(y2 - y1, 1)], dtype=np.float64)
        S = _H @ tr.kf_cov @ _H.T + _R
        K = tr.kf_cov @ _H.T @ np.linalg.inv(S)
        tr.kf_state = tr.kf_state + K @ (z - _H @ tr.kf_state)
        tr.kf_cov = (np.eye(6) - K @ _H) @ tr.kf_cov
        tr.hits += 1
        tr.total_visible += 1
        tr.time_since_update = 0
        tr.last_bbox = (x1, y1, x2, y2)

    def _try_slot_assignment(self):
        """Assign slots to tracks that crossed the visibility threshold."""
        if len(self._used_slots) >= self.max_players:
            return  # all slots taken

        # Candidates: unconfirmed tracks with enough visibility
        candidates = [
            tr for tr in self._tracks.values()
            if not tr.confirmed and tr.total_visible >= self.min_track_frames
        ]
        # Sort by total_visible DESC (most-seen first)
        candidates.sort(key=lambda t: -t.total_visible)

        for tr in candidates:
            if len(self._used_slots) >= self.max_players:
                break
            # Find lowest free slot
            for s in range(1, self.max_players + 1):
                if s not in self._used_slots:
                    tr.slot = s
                    tr.confirmed = True
                    self._used_slots.add(s)
                    break

    def _prune(self):
        """Remove unconfirmed tracks that haven't been seen recently."""
        dead = [
            tid for tid, tr in self._tracks.items()
            if not tr.confirmed and tr.time_since_update > self.max_age_unconfirmed
        ]
        for tid in dead:
            del self._tracks[tid]

    def _active(self) -> list[dict]:
        """Return confirmed players + recently-seen tentative tracks."""
        result = []
        for tr in self._tracks.values():
            # Confirmed: always included (Kalman predicts position through gaps)
            # Unconfirmed: only if seen in last 5 frames
            if tr.confirmed or tr.time_since_update <= 5:
                bx1, by1, bx2, by2 = tr.bbox
                result.append({
                    "slot": tr.slot,
                    "track_id": tr.track_id,
                    "confirmed": tr.confirmed,
                    "bbox": [bx1, by1, bx2, by2],
                    "cx": round(tr.cx / self.img_w, 4),
                    "cy": round(tr.cy / self.img_h, 4),
                    "total_visible": tr.total_visible,
                    "hits": tr.hits,
                    "time_since_update": tr.time_since_update,
                })
        result.sort(key=lambda x: (x["slot"] if x["slot"] > 0 else 99, -x["total_visible"]))
        return result

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._used_slots.clear()
