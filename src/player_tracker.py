"""
player_tracker.py
=================
Court-aware player tracking with exactly 4 stable IDs, start to finish.

Two cooperating components:

1) CourtFilter (two-phase tracking)
   - ESTABLISHMENT phase: a new track must survive inside the court polygon
     for ``min_establish_frames`` with avg confidence ≥ ``min_confidence``
     and realistic bbox size.  This rejects glass reflections (short-lived),
     spectators (never inside court), and false positives (low confidence).
   - MAINTENANCE phase: once established, the player is tracked EVERYWHERE —
     inside and outside the court — for up to ``max_outside_frames`` (10 s).
     Stats are paused while outside but the ID and position trail continue.
   - If a player is outside longer than the grace period, the slot is
     vacated for re-acquisition.

2) PlayerRegistry
   - Locks the match to 4 stable slots (1..4).
   - Hungarian assignment on (HSV appearance + position).
   - Vacated-slot re-acquisition: if a player's track is lost and a new
     track appears with matching appearance, the same slot ID is reused.

Usage:
    tracker = PlayerTracker(player_model, court_model)
    players = tracker.track(frame)        # -> [{slot, box, raw_id, team?}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackMeta:
    """Per-track metadata for ghost filtering and two-phase establishment."""
    frames_seen: int = 0
    conf_sum: float = 0.0
    inside_consec: int = 0
    outside_consec: int = 0
    established: bool = False
    lost: bool = False

    @property
    def avg_conf(self) -> float:
        return self.conf_sum / max(self.frames_seen, 1)


@dataclass
class TrackedPlayer:
    slot: int                         # stable 1..4
    box: tuple[float, float, float, float]   # x1,y1,x2,y2 px
    raw_id: int                       # underlying BoT-SORT id
    court_xy: Optional[tuple[float, float]] = None  # feet projected (later)
    outside_court: bool = False       # True when established player is outside polygon
    established: bool = True          # True once past establishment phase


# ─────────────────────────────────────────────────────────────────────────────
class CourtFilter:
    """Two-phase court-aware filter: establishment → maintenance.

    ESTABLISHMENT (ghost filter):
        A new track must satisfy ALL of:
        - Inside court polygon for ≥ ``min_establish_frames`` consecutive frames
        - Average detection confidence ≥ ``min_confidence``
        - Bbox height within [``min_height_px``, ``max_height_px``]
        Until established, the track is invisible to the registry.

    MAINTENANCE (sticky ID):
        Once established, the player is tracked everywhere (inside + outside
        court) for up to ``max_outside_frames`` consecutive outside frames.
        Stats should be paused while outside but the ID persists.
    """

    def __init__(
        self,
        min_inside: int = 3,           # legacy param, maps to min_establish_frames
        margin_frac: float = 0.0,
        *,
        min_establish_frames: int = 15,
        min_confidence: float = 0.40,
        max_outside_frames: int = 300,   # 10 s at 30 fps
        min_height_px: float = 40.0,
        max_height_px: float = 300.0,
    ):
        self.min_inside = min_inside
        self.margin_frac = margin_frac
        self.min_establish_frames = max(min_establish_frames, min_inside)
        self.min_confidence = min_confidence
        self.max_outside_frames = max_outside_frames
        self.min_height_px = min_height_px
        self.max_height_px = max_height_px
        self._polygon: Optional[np.ndarray] = None
        self._frame_wh: tuple[int, int] = (1, 1)
        self._meta: dict[int, TrackMeta] = {}

    # ---- polygon from keypoints ----------------------------------------
    def set_keypoints(self, kpts_xy: np.ndarray, frame_wh: tuple[int, int]):
        """kpts_xy: (N,2) pixel coords of visible court keypoints."""
        self._frame_wh = frame_wh
        pts = np.asarray(kpts_xy, dtype=np.float32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        if len(pts) < 3:
            return
        hull = cv2.convexHull(pts)
        if self.margin_frac > 0:
            c = hull.reshape(-1, 2).mean(axis=0)
            hull = (c + (hull.reshape(-1, 2) - c) * (1 - self.margin_frac))
            hull = hull.reshape(-1, 1, 2).astype(np.float32)
        self._polygon = hull

    @property
    def polygon(self) -> Optional[np.ndarray]:
        return self._polygon

    # ---- geometry helpers ------------------------------------------------
    def _feet(self, box):
        return ((box[0] + box[2]) / 2.0, box[3])

    def feet_inside(self, box) -> bool:
        if self._polygon is None:
            return True
        return cv2.pointPolygonTest(self._polygon, self._feet(box), False) >= 0

    @staticmethod
    def _box_height(box) -> float:
        return float(box[3] - box[1])

    def _valid_size(self, box) -> bool:
        h = self._box_height(box)
        return self.min_height_px <= h <= self.max_height_px

    # ---- per-track metadata ----------------------------------------------
    def _get_meta(self, raw_id: int) -> TrackMeta:
        if raw_id not in self._meta:
            self._meta[raw_id] = TrackMeta()
        return self._meta[raw_id]

    def prune(self, active_ids: set[int]):
        """Drop metadata for tracks no longer seen (prevents memory growth)."""
        stale = set(self._meta) - active_ids
        for sid in stale:
            del self._meta[sid]

    # ---- two-phase evaluation --------------------------------------------
    def evaluate(
        self, raw_id: int, box, confidence: float = 1.0
    ) -> tuple[bool, bool]:
        """Update track metadata and return (is_valid, is_outside).

        - ``is_valid``: True if the track should be visible to the registry
          (established or in active maintenance).
        - ``is_outside``: True if the player is currently outside the court
          polygon (stats should pause).
        """
        m = self._get_meta(raw_id)
        m.frames_seen += 1
        m.conf_sum += confidence

        inside = self.feet_inside(box)
        good_size = self._valid_size(box)

        if m.established:
            # ── MAINTENANCE PHASE — sticky ID ──
            if inside:
                m.outside_consec = 0
            else:
                m.outside_consec += 1
                if m.outside_consec > self.max_outside_frames:
                    m.established = False
                    m.lost = True
                    return False, True
            return True, (not inside)

        # ── ESTABLISHMENT PHASE — strict ghost filter ──
        if inside and good_size:
            m.inside_consec += 1
            m.outside_consec = 0
        else:
            m.inside_consec = 0

        if (
            m.frames_seen >= self.min_establish_frames
            and m.inside_consec >= self.min_establish_frames
            and m.avg_conf >= self.min_confidence
        ):
            m.established = True
            return True, False

        return False, False

    # ---- legacy compat ---------------------------------------------------
    def is_court_player(self, raw_id: int, box, confidence: float = 1.0) -> bool:
        """Legacy boolean API — use ``evaluate()`` for the full return."""
        valid, _ = self.evaluate(raw_id, box, confidence)
        return valid

    def is_outside(self, raw_id: int) -> bool:
        m = self._meta.get(raw_id)
        return bool(m and m.established and m.outside_consec > 0)


# ─────────────────────────────────────────────────────────────────────────────
class PlayerRegistry:
    """Pin the match to 4 stable slots via appearance + position matching.

    Vacated-slot re-acquisition: when a slot's track disappears, the slot
    enters a vacated state with the player's last appearance (HSV histogram)
    and position.  If a new track appears within ``max_vacated_frames`` that
    matches the vacated appearance (HSV correlation > ``reacquire_thresh``)
    and is within ``reacquire_dist`` pixels, the same slot ID is reused.
    """

    def __init__(self, max_players: int = 4, app_w: float = 0.6,
                 pos_w: float = 0.4, max_cost: float = 0.85,
                 max_vacated_frames: int = 600,
                 reacquire_thresh: float = 0.55,
                 reacquire_dist: float = 300.0):
        self.max_players = max_players
        self.app_w = app_w
        self.pos_w = pos_w
        self.max_cost = max_cost
        self.max_vacated_frames = max_vacated_frames
        self.reacquire_thresh = reacquire_thresh
        self.reacquire_dist = reacquire_dist
        self._slots: dict[int, dict] = {}           # slot -> {hist, pos, seen}
        self._vacated: dict[int, dict] = {}         # slot -> {hist, pos, frame_lost}
        self._diag = 1.0
        self._frame_idx = 0

    # ---- features ------------------------------------------------------
    def _appearance(self, frame, box) -> np.ndarray:
        """HSV hue-sat histogram of the torso band (jersey color)."""
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(0)
        # torso = central horizontal band (skip head & legs)
        ty1 = y1 + int(0.20 * (y2 - y1))
        ty2 = y1 + int(0.65 * (y2 - y1))
        crop = frame[ty1:ty2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist).flatten()
        n = np.linalg.norm(hist)
        return hist / n if n > 0 else hist

    def _feet(self, box):
        return np.array([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)

    @staticmethod
    def _app_dist(a, b) -> float:
        if a.size == 0 or b.size == 0:
            return 1.0
        return 1.0 - float(cv2.compareHist(
            a.reshape(-1, 1), b.reshape(-1, 1), cv2.HISTCMP_CORREL))

    # ---- assignment ----------------------------------------------------
    def assign(self, frame, tracks) -> list[int]:
        """
        tracks: list of (raw_id, box). Returns a list of stable slot ids
        aligned with `tracks` (0 means: not assigned / overflow ignored).
        """
        self._frame_idx += 1
        H, W = frame.shape[:2]
        self._diag = float(np.hypot(H, W))
        feats = [(self._appearance(frame, b), self._feet(b)) for _, b in tracks]

        # Expire old vacated slots
        expired = [sid for sid, v in self._vacated.items()
                   if self._frame_idx - v["frame_lost"] > self.max_vacated_frames]
        for sid in expired:
            del self._vacated[sid]

        slot_ids = list(self._slots.keys())
        n_slots, n_tr = len(slot_ids), len(tracks)
        result = [0] * n_tr
        if n_tr == 0:
            self._check_lost_slots(set())
            return result

        if n_slots == 0:
            # bootstrap: first valid tracks take slots in arrival order
            for i in range(min(n_tr, self.max_players)):
                slot = i + 1
                self._slots[slot] = {"hist": feats[i][0], "pos": feats[i][1], "seen": 0}
                result[i] = slot
            return result

        # cost matrix (slots x tracks)
        cost = np.zeros((n_slots, n_tr), dtype=np.float32)
        for si, sid in enumerate(slot_ids):
            s = self._slots[sid]
            for ti, (hist, pos) in enumerate(feats):
                app = self._app_dist(s["hist"], hist)
                dpos = np.linalg.norm(s["pos"] - pos) / self._diag
                cost[si, ti] = self.app_w * app + self.pos_w * min(dpos, 1.0)

        rows, cols = linear_sum_assignment(cost)
        used_tracks = set()
        for si, ti in zip(rows, cols):
            if cost[si, ti] <= self.max_cost:
                sid = slot_ids[si]
                hist, pos = feats[ti]
                self._slots[sid]["pos"] = pos
                if hist.size:
                    old = self._slots[sid]["hist"]
                    if old.size:
                        self._slots[sid]["hist"] = 0.5 * old + 0.5 * hist
                    else:
                        self._slots[sid]["hist"] = hist
                result[ti] = sid
                used_tracks.add(ti)

        # unassigned valid tracks -> check vacated slots first, then open new
        active_sids = set(result) - {0}
        for ti in range(n_tr):
            if ti in used_tracks or result[ti] != 0:
                continue
            hist, pos = feats[ti]

            # 1. Try to re-acquire a vacated slot
            best_sid, best_dist = 0, self.reacquire_thresh
            for sid, vac in self._vacated.items():
                if sid in active_sids:
                    continue
                app_d = self._app_dist(vac["hist"], hist)
                pos_d = float(np.linalg.norm(vac["pos"] - pos))
                if app_d < best_dist and pos_d < self.reacquire_dist:
                    best_sid, best_dist = sid, app_d

            if best_sid:
                del self._vacated[best_sid]
                self._slots[best_sid] = {"hist": hist, "pos": pos, "seen": 0}
                result[ti] = best_sid
                active_sids.add(best_sid)
                continue

            # 2. Open a new slot if under the limit
            if len(self._slots) < self.max_players:
                sid = max(self._slots) + 1 if self._slots else 1
                self._slots[sid] = {"hist": hist, "pos": pos, "seen": 0}
                result[ti] = sid
                active_sids.add(sid)

        # Detect lost slots (active last frame, missing this frame)
        self._check_lost_slots(active_sids)
        return result

    def _check_lost_slots(self, active_sids: set[int]):
        """Move slots not seen this frame to vacated state."""
        lost = [sid for sid in list(self._slots) if sid not in active_sids]
        for sid in lost:
            s = self._slots.pop(sid)
            self._vacated[sid] = {
                "hist": s["hist"],
                "pos": s["pos"],
                "frame_lost": self._frame_idx,
            }


# ─────────────────────────────────────────────────────────────────────────────
class PlayerTracker:
    """End-to-end: court polygon -> BoT-SORT track -> inside filter -> 4 slots."""

    def __init__(self, player_model, court_model=None, *,
                 manual_calibration=None, max_players: int = 4,
                 tracker_cfg: str = "configs/botsort_reid.yaml",
                 min_inside: int = 3, player_classes=None,
                 court_every_n: int = 60):
        self.player = player_model
        self.court = court_model
        self.manual_calibration = manual_calibration
        self.max_players = max_players
        self.tracker_cfg = tracker_cfg
        self.player_classes = player_classes   # e.g. [0] to restrict COCO->person
        self.court_every_n = max(1, int(court_every_n))
        self.court_filter = CourtFilter(min_inside=min_inside)
        self.registry = PlayerRegistry(max_players=max_players)
        self.court_keypoints: Optional[np.ndarray] = None  # last frame's kpts
        self._court_frame_count = 0

        # With manual calibration the court polygon is constant; seed it once
        # so the inside-court filter works without the keypoint model.
        self._manual_kpts: Optional[np.ndarray] = None
        if manual_calibration is not None:
            from src.utils.calibration import polygon_from_calibration
            self._manual_kpts = manual_calibration.corners_array()
            self.court_filter.set_keypoints(
                self._manual_kpts,
                (manual_calibration.frame_width, manual_calibration.frame_height),
            )
            # bypass the (absent) polygon shrink path; set hull directly
            self.court_filter._polygon = polygon_from_calibration(manual_calibration)

    def _update_court(self, frame):
        # Manual calibration -> court geometry is constant across frames.
        if self._manual_kpts is not None:
            return self._manual_kpts
        if self.court is None:
            return None
        self._court_frame_count += 1
        # Cadence: the court model is expensive and its output barely changes,
        # so only re-run it on frame 1 and every court_every_n-th frame after;
        # reuse the last detected keypoints + polygon in between.
        if self.court_keypoints is not None and not (
            self._court_frame_count == 1 or self._court_frame_count % self.court_every_n == 0
        ):
            return self.court_keypoints
        H, W = frame.shape[:2]
        res = self.court.predict(frame, verbose=False, imgsz=640)
        for r in res:
            if r.keypoints is not None and len(r.keypoints):
                kpts = r.keypoints.xy[0].cpu().numpy()      # (N,2)
                self.court_filter.set_keypoints(kpts, (W, H))
                return kpts
        return self.court_keypoints      # keep last good if detection failed

    def track(self, frame) -> list[TrackedPlayer]:
        court_kpts = self._update_court(frame)
        self.court_keypoints = court_kpts

        raw = []
        track_kw = dict(persist=True, tracker=self.tracker_cfg,
                        conf=0.3, iou=0.5, verbose=False, imgsz=640)
        if self.player_classes is not None:
            track_kw["classes"] = self.player_classes
        for r in self.player.track(frame, **track_kw):
            if r.boxes is None:
                continue
            for b in r.boxes:
                if b.id is None:
                    continue
                raw_id = int(b.id[0])
                box = tuple(b.xyxy[0].cpu().numpy())
                conf = float(b.conf[0]) if b.conf is not None else 1.0
                raw.append((raw_id, box, conf))

        # Two-phase evaluation: establishment (ghost filter) + maintenance (sticky)
        valid = []
        outside_flags: dict[int, bool] = {}
        for raw_id, box, conf in raw:
            is_valid, is_outside = self.court_filter.evaluate(raw_id, box, conf)
            if is_valid:
                valid.append((raw_id, box))
                outside_flags[raw_id] = is_outside

        # Prune stale metadata (tracks no longer seen)
        active_ids = {rid for rid, _, _ in raw}
        self.court_filter.prune(active_ids)

        slots = self.registry.assign(frame, valid)
        out: list[TrackedPlayer] = []
        for (tid, box), slot in zip(valid, slots):
            if slot:
                out.append(TrackedPlayer(
                    slot=slot,
                    box=box,
                    raw_id=tid,
                    outside_court=outside_flags.get(tid, False),
                    established=True,
                ))
        return out
