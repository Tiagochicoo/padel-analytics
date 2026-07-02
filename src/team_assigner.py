"""
src/team_assigner.py
====================
Automatic team assignment for padel doubles.

In padel each pair occupies one half of the court (split by the net at Y = 10 m
in court-plane metres), so a player's team is just the half their feet live on.
We track a per-slot EMA of court-Y and, once it has stabilised over a few
hundred frames (the warmup window), lock each canonical slot P1..P4 to team 0
(near half) or team 1 (far half).

This complements the online Re-ID resolver (Phase 4): once P1..P4 are stable,
their teams lock too — feeding the per-team stats panels (Phase 6) and the rules
engine's coarse point winner (``rules_engine._team_of`` reads ``Player.team``).

Court-side split: P1+P2 vs P3+P4 (the canonical ordering groups the two near
players as P1/P2 and the two far players as P3/P4), but the assignment is driven
by the actual mean court-Y, so it stays correct even if the canonical ordering
is noisy.
"""

from __future__ import annotations

from typing import Optional

NET_Y_M = 10.0          # net at court midline (metres)
_DEFAULT_EMA = 0.02     # slow EMA -> stable home-half estimate
_DEFAULT_MIN = 240      # frames before locking (~8 s @ 30 fps)


class TeamAssigner:
    def __init__(self, net_y: float = NET_Y_M, ema_alpha: float = _DEFAULT_EMA,
                 min_frames: int = _DEFAULT_MIN):
        self.net_y = net_y
        self.ema_alpha = ema_alpha
        self.min_frames = min_frames
        self._mean_y: dict[int, float] = {}
        self._counts: dict[int, int] = {}
        self._team: dict[int, int] = {}      # slot -> 0/1, locked once assigned

    def update(self, slot: int, court_xy: Optional[tuple]) -> Optional[int]:
        """Feed one frame for a slot; return its locked team (0/1) or None."""
        if court_xy is None or slot is None:
            return self._team.get(slot)
        y = float(court_xy[1])
        cur = self._mean_y.get(slot)
        self._mean_y[slot] = y if cur is None else (1 - self.ema_alpha) * cur + self.ema_alpha * y
        self._counts[slot] = self._counts.get(slot, 0) + 1
        if slot not in self._team and self._counts[slot] >= self.min_frames:
            self._team[slot] = 0 if self._mean_y[slot] < self.net_y else 1
        return self._team.get(slot)

    @property
    def teams(self) -> dict[int, int]:
        return dict(self._team)

    @property
    def status(self) -> dict:
        return {
            "locked": {int(s): t for s, t in self._team.items()},
            "mean_y": {int(s): round(v, 2) for s, v in self._mean_y.items()},
            "net_y": self.net_y,
        }


__all__ = ["TeamAssigner", "NET_Y_M"]
