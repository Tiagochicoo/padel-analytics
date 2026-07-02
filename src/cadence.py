"""
src/cadence.py
==============
Per-component frame cadence for ``PadelAnalyzer``.

Not every model needs to run every frame. A ``Cadence`` policy gives each
component an ``every_n`` interval; the analyzer reuses the component's last
output on the frames it is skipped, so the per-frame ``AnalysisResult`` stays
complete while compute is spent only where it matters.

Defaults reflect the live-deployment plan (see docs/ROADMAP.md Component 5):

    player  every 1   BoT-SORT needs every frame for ID continuity
    ball    every 1   TrackNet keeps an internal seq_len window -> no skipping
    court   every 60  court geometry is effectively static (~2 s @ 30 fps)
    pose    every 3   body pose on player crops (Phase 4)
    shot    every 1   event-driven via ball proximity (+ the classifier cooldown)

The headline win is ``court``: the court keypoint model is expensive and its
output barely changes, so running it 1/60th as often is a large per-frame saving
once the court model is trained.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cadence:
    player_every: int = 1
    ball_every: int = 1
    court_every: int = 60
    pose_every: int = 3
    shot_every: int = 1

    def should_run(self, component: str, frame_idx: int) -> bool:
        """True if `component` should execute on this frame (0-indexed)."""
        n = getattr(self, f"{component}_every", 1)
        if n <= 1:
            return True
        return frame_idx % n == 0


__all__ = ["Cadence"]
