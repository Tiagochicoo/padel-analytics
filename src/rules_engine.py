"""
src/rules_engine.py
===================
v1 padel match rules engine — rally/point segmentation + shot attribution.

v1 is deliberately SCORING-FREE: it groups frames into rallies from ball
presence/motion, forwards shot events (attributed to players by the shot
classifier) into their rally, and emits a COARSE point winner (the team that
touched the ball last — the default heuristic when bounce/in-out reasoning is
unavailable, since most points in padel are won by putting the ball away and
the opponent failing to return). Full official scoring (serve boxes, faults,
golden point, tie-break) is scoped in docs/scoring_spec.md and lands once the
ball + bounce detectors exist.

It runs as a streaming state machine: call ``update(result, frame_idx)`` once
per analysed frame and collect the events it emits.

    IDLE  --ball becomes active-->  RALLY  --ball lost K frames-->  IDLE
                                                       (emits RallyEnd + PointScored)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.analyzer import AnalysisResult, Player


@dataclass
class Event:
    kind: str          # "rally_start" | "rally_end" | "shot" | "point"


@dataclass
class RallyStart(Event):
    kind: str = "rally_start"
    frame: int = 0


@dataclass
class Shot(Event):
    slot: int = 0
    shot_type: str = ""
    frame: int = 0
    kind: str = "shot"


@dataclass
class RallyEnd(Event):
    duration_frames: int = 0
    shots_in_rally: int = 0
    last_hitter_slot: Optional[int] = None
    winner_team: Optional[int] = None
    start_frame: int = 0
    frame: int = 0
    kind: str = "rally_end"


# tuning (frames) — ball @ ~30 fps analysis cadence
_BALL_WINDOW = 10        # look-back window for "ball active"
_BALL_ACTIVE_MIN = 3     # detections in window to consider the ball in play
_MISS_FRAMES = 12        # consecutive ball misses -> rally over (~0.4 s)
_MIN_RALLY_LEN = 5       # discard rallies shorter than this (noise)
_SHOT_COOLDOWN = 8       # min frames between attributed shots per slot


def _team_of(slot: int, players: list[Player]) -> Optional[int]:
    for p in players:
        if p.track_id == slot:
            return p.team
    return None


class RulesEngine:
    def __init__(self) -> None:
        self._state = "IDLE"                       # IDLE | RALLY
        self._ball_window: deque[bool] = deque(maxlen=_BALL_WINDOW)
        self._misses = 0
        self._rally_start: int = 0
        self._rally_shots: int = 0
        self._last_hitter: Optional[int] = None
        self._last_shot_frame: dict[int, int] = {}
        self.total_rallies = 0
        self.total_shots = 0

    def update(self, result: AnalysisResult, frame_idx: int) -> list[Event]:
        events: list[Event] = []
        ball_seen = result.ball_xy is not None
        self._ball_window.append(ball_seen)
        active = sum(self._ball_window) >= _BALL_ACTIVE_MIN

        # forward attributed shots into the active rally (with per-slot cooldown)
        for s in result.shots:
            slot = int(s.get("slot", -1))
            if slot < 0 or "type" not in s:
                continue
            if frame_idx - self._last_shot_frame.get(slot, -10_000) < _SHOT_COOLDOWN:
                continue
            self._last_shot_frame[slot] = frame_idx
            self._last_hitter = slot
            if self._state == "RALLY":
                self._rally_shots += 1
            self.total_shots += 1
            events.append(Shot(slot=slot, shot_type=str(s["type"]), frame=frame_idx))

        if self._state == "IDLE":
            if active:
                self._state = "RALLY"
                self._rally_start = frame_idx
                self._rally_shots = 0
                self._misses = 0
                self._last_hitter = None
                events.append(RallyStart(frame=frame_idx))
        else:  # RALLY
            if ball_seen:
                self._misses = 0
            else:
                self._misses += 1
            if self._misses >= _MISS_FRAMES:
                duration = frame_idx - self._rally_start
                winner = None
                if self._last_hitter is not None:
                    lh_team = _team_of(self._last_hitter, result.players)
                    winner = lh_team if lh_team is not None else None
                if duration >= _MIN_RALLY_LEN:
                    self.total_rallies += 1
                    events.append(RallyEnd(
                        duration_frames=duration, shots_in_rally=self._rally_shots,
                        last_hitter_slot=self._last_hitter, winner_team=winner,
                        start_frame=self._rally_start, frame=frame_idx,
                    ))
                self._state = "IDLE"
                self._rally_shots = 0
                self._last_hitter = None
        return events


__all__ = ["RulesEngine", "Event", "RallyStart", "RallyEnd", "Shot"]
