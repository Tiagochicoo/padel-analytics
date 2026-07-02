"""
src/match_state.py
==================
Match lifecycle state machine: WARMUP -> LIVE -> ENDED.

Governs when stats/heatmaps start counting. The pre-match warmup is the online
Re-ID calibration window (see src/reid_online.py): the resolver clusters during
WARMUP and locks at LIVE.

Start trigger (per the agreed rule):
    * the user clicks "Start Match", OR
    * serve auto-detect fires,
    whichever comes first — BUT the button is AUTHORITATIVE: if auto-detect
    already started LIVE and the user later clicks Start, the stats counted
    since auto-detect are DISCARDED as warmup and LIVE re-anchors to the button
    moment. (If the user never clicks, auto-detect's time stands.)

Serve auto-detect depends on the ball model (Phase 1c, not trained yet), so
``detect_serve_start`` is a stub returning False until then — only the button
can start the match for now. Wire the real detector in once ball_best.pt lands.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

WARMUP, LIVE, ENDED = "warmup", "live", "ended"


class MatchState:
    def __init__(self) -> None:
        self.state: str = WARMUP
        self.live_started_at: Optional[int] = None   # authoritative LIVE frame
        self.started_by: Optional[str] = None        # "button" | "auto" | None
        self.auto_start_frame: Optional[int] = None  # frame auto-detect fired (for discard report)
        self.ended_at: Optional[int] = None
        self.discarded_warmup_frames: int = 0        # stats thrown away by a button override

    def start_by_button(self, frame_idx: int) -> bool:
        """Authoritative match start (also overrides a prior auto-detect)."""
        if self.state == ENDED:
            return False
        if self.state == LIVE and self.started_by == "auto":
            self.discarded_warmup_frames = frame_idx - (self.auto_start_frame or frame_idx)
        self.state = LIVE
        self.live_started_at = frame_idx
        self.started_by = "button"
        return True

    def start_by_autodetect(self, frame_idx: int) -> bool:
        """Auto-detected first serve. Only fires from WARMUP (never overrides)."""
        if self.state != WARMUP:
            return False
        self.state = LIVE
        self.live_started_at = frame_idx
        self.started_by = "auto"
        self.auto_start_frame = frame_idx
        return True

    def end(self, frame_idx: int) -> None:
        if self.state == LIVE:
            self.state = ENDED
            self.ended_at = frame_idx

    def reset(self) -> None:
        self.state = WARMUP
        self.live_started_at = None
        self.started_by = None
        self.auto_start_frame = None
        self.ended_at = None
        self.discarded_warmup_frames = 0

    @property
    def counting(self) -> bool:
        """True on frames whose stats should be accumulated (i.e. during LIVE)."""
        return self.state == LIVE

    @property
    def status(self) -> dict:
        return {
            "state": self.state,
            "started_by": self.started_by,
            "live_started_at": self.live_started_at,
            "auto_start_frame": self.auto_start_frame,
            "ended_at": self.ended_at,
            "discarded_warmup_frames": self.discarded_warmup_frames,
        }


# Ball-absence window that precedes a serve (frames). Conservative so warmup
# noise does not trigger it; tune down once the ball model is accurate.
_SERVE_STILLNESS = 20      # consecutive ball-absent frames before a serve
_SERVE_MIN_HISTORY = 22    # need at least this many frames of history
_NET_Y_NORM = 0.5          # net at mid-court in normalised feet-y (future use)


def detect_serve_start(ball_xy_history: deque, frame_idx: int) -> bool:
    """
    Serve auto-detect: a stillness gap (ball absent for >= _SERVE_STILLNESS
    consecutive frames) immediately followed by the ball reappearing.

    Pattern:  [ball seen] [absent × N >= _SERVE_STILLNESS] [ball seen NOW]

    Fires only on the frame where the ball reappears. The match-state state
    machine ensures it only triggers once (WARMUP → LIVE on first fire, then
    detect is no longer called).

    Args:
        ball_xy_history: deque of Optional[tuple] — ball_xy per frame, or None.
        frame_idx: current frame index (unused but kept for API stability).

    Returns:
        True if the serve pattern is detected in the most recent frames.
    """
    if len(ball_xy_history) < _SERVE_MIN_HISTORY:
        return False
    history = list(ball_xy_history)
    # The most recent entry must be a ball detection (ball just reappeared)
    if history[-1] is None:
        return False
    # Count the stillness gap immediately before the reappearance
    gap = 0
    for i in range(len(history) - 2, -1, -1):
        if history[i] is None:
            gap += 1
        else:
            break
    return gap >= _SERVE_STILLNESS


__all__ = ["MatchState", "WARMUP", "LIVE", "ENDED", "detect_serve_start"]
