"""
stats.py
========
Accumulates per-player statistics from AnalysisResult over time and produces
a JSON-serialisable snapshot for the web UI.

Each player occupies one stable slot (1..4) assigned by PlayerTracker, so the
stats are keyed by slot and stay consistent across the whole video even when
the underlying BoT-SORT id churns.

Tracked per slot:
    * shots          total + breakdown by type (drive/lob/bandeja/...)  [Phase 3]
    * distance_m     cumulative court distance travelled (meters)
    * time_on_court  seconds visible on court
    * last position  box (px) + projected court_xy (m) -> drives the minimap
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

# Stable per-slot colours. BGR for OpenCV drawing, hex for the frontend.
# Both maps are derived from the same RGB values to guarantee consistency.
SLOT_COLORS_HEX = {
    1: "#3fb950",   # green
    2: "#a371f7",   # purple
    3: "#f778ba",   # pink
    4: "#79c0ff",   # light blue
}
def _hex_to_bgr(h: str) -> tuple:
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return (b, g, r)
SLOT_COLORS_BGR = {s: _hex_to_bgr(h) for s, h in SLOT_COLORS_HEX.items()}
SLOT_NAMES = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}


@dataclass
class _SlotState:
    frames_seen: int = 0
    last_box: tuple = ()
    last_court_xy: tuple = ()
    prev_court_xy: tuple = ()
    distance_m: float = 0.0
    shots_total: int = 0
    shots_by_type: dict = field(default_factory=dict)
    team: object = None
    _last_ts: float = 0.0
    _time_sec: float = 0.0


class StatsAccumulator:
    """Thread-safe per-slot stats aggregator."""

    def __init__(self, max_slots: int = 4, court_fps: float = 30.0):
        self.max_slots = max_slots
        self.court_fps = court_fps
        self._lock = threading.Lock()
        self._slots: dict[int, _SlotState] = {
            i: _SlotState() for i in range(1, max_slots + 1)
        }

    def reset(self) -> None:
        with self._lock:
            for s in self._slots.values():
                s.frames_seen = 0
                s.last_box = ()
                s.last_court_xy = ()
                s.prev_court_xy = ()
                s.distance_m = 0.0
                s.shots_total = 0
                s.shots_by_type = {}
                s.team = None
                s._last_ts = 0.0
                s._time_sec = 0.0

    def update(self, players, shots=None) -> None:
        """
        players: list of AnalysisResult.Player (has track_id=slot, box, court_xy)
        shots:   list of shot dicts [{slot, type}, ...] (Phase 3; may be empty)
        """
        with self._lock:
            seen_slots = set()
            now = time.time()
            for p in players:
                slot = int(p.track_id)
                if slot < 1 or slot > self.max_slots:
                    continue
                st = self._slots[slot]
                st.frames_seen += 1
                if st._last_ts > 0:
                    dt = now - st._last_ts
                    if dt < 5.0:
                        st._time_sec += dt
                st._last_ts = now
                st.last_box = tuple(float(v) for v in p.box)
                if getattr(p, "team", None) is not None:
                    st.team = int(p.team)
                cxy = p.court_xy
                if cxy is not None:
                    cxy = (float(cxy[0]), float(cxy[1]))
                    st.last_court_xy = cxy
                    if st.prev_court_xy:
                        # ignore huge jumps (re-acquisition noise)
                        d = math.hypot(cxy[0] - st.prev_court_xy[0],
                                       cxy[1] - st.prev_court_xy[1])
                        if d < 5.0:  # meters: a player can't teleport
                            st.distance_m += d
                    st.prev_court_xy = cxy
                seen_slots.add(slot)

            if shots:
                for sh in shots:
                    slot = int(sh.get("slot", 0))
                    stype = sh.get("type", "unknown")
                    if slot in self._slots:
                        self._slots[slot].shots_total += 1
                        self._slots[slot].shots_by_type[stype] = \
                            self._slots[slot].shots_by_type.get(stype, 0) + 1

    def snapshot(self) -> list[dict]:
        with self._lock:
            out = []
            for slot in range(1, self.max_slots + 1):
                st = self._slots[slot]
                out.append({
                    "slot": slot,
                    "name": SLOT_NAMES.get(slot, f"Player {slot}"),
                    "color": SLOT_COLORS_HEX.get(slot, "#cccccc"),
                    "team": st.team,
                    "active": bool(st.last_court_xy or st.last_box),
                    "frames_seen": st.frames_seen,
                    "time_on_court": round(st._time_sec, 1),
                    "distance_m": round(st.distance_m, 1),
                    "shots": st.shots_total,
                    "shots_by_type": dict(st.shots_by_type),
                    "box": list(st.last_box),
                    "court_xy": list(st.last_court_xy),
                })
            return out


def slot_color_bgr(slot: int):
    return SLOT_COLORS_BGR.get(slot, (0, 255, 0))
