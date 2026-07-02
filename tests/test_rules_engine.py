"""Tests for RulesEngine — rally/point segmentation + shot attribution."""
import pytest
from src.rules_engine import RulesEngine, RallyStart, RallyEnd, Shot
from src.analyzer import AnalysisResult, Player


def _result(ball_xy=None, players=None, shots=None):
    """Minimal AnalysisResult factory for tests."""
    return AnalysisResult(
        players=players or [],
        ball_xy=ball_xy,
        shots=shots or [],
    )


def _player(slot, team=None):
    return Player(track_id=slot, box=(100, 100, 200, 300), team=team, court_xy=(5.0, 10.0))


class TestRallyTransitions:
    def test_idle_to_rally_on_ball_active(self):
        """Ball must appear in enough frames within the window to start a rally."""
        eng = RulesEngine()
        all_evs = []
        for i in range(3):
            all_evs += eng.update(_result(ball_xy=(500, 500)), i)
        assert eng._state == "RALLY"
        assert any(isinstance(e, RallyStart) for e in all_evs)

    def test_rally_ends_on_ball_missing(self):
        eng = RulesEngine()
        # Start a rally
        for i in range(3):
            eng.update(_result(ball_xy=(500, 500)), i)
        assert eng._state == "RALLY"
        # Ball goes missing
        events = []
        for i in range(3, 3 + 15):
            events += eng.update(_result(ball_xy=None), i)
        assert eng._state == "IDLE"
        assert any(isinstance(e, RallyEnd) for e in events)

    def test_short_rally_discarded(self):
        eng = RulesEngine()
        eng.update(_result(ball_xy=(500, 500)), 0)
        # Immediately lose ball — rally too short (< _MIN_RALLY_LEN=5)
        for i in range(1, 15):
            eng.update(_result(ball_xy=None), i)
        assert eng.total_rallies == 0


class TestLastHitterReset:
    def test_last_hitter_reset_on_rally_start(self):
        """_last_hitter must be None when a new rally starts (Tier 1 fix)."""
        eng = RulesEngine()
        # Emit a shot during IDLE (sets _last_hitter)
        eng.update(_result(ball_xy=None, shots=[{"slot": 2, "type": "drive"}]), 0)
        assert eng._last_hitter == 2
        # Start a rally
        for i in range(1, 4):
            eng.update(_result(ball_xy=(500, 500)), i)
        assert eng._state == "RALLY"
        assert eng._last_hitter is None  # Reset on RallyStart


class TestWinnerHeuristic:
    def test_winner_is_last_hitters_team(self):
        """Point goes to the team that hit last, NOT the opposing team (Tier 1 fix)."""
        eng = RulesEngine()
        # Start rally
        for i in range(3):
            eng.update(_result(ball_xy=(500, 500)), i)
        assert eng._state == "RALLY"
        # Player 2 (team 0) hits the ball
        eng.update(_result(ball_xy=(500, 500), shots=[{"slot": 2, "type": "drive"}]), 3)
        # Ball goes missing → rally ends
        events = []
        for i in range(4, 20):
            events += eng.update(
                _result(ball_xy=None, players=[_player(2, team=0)]), i
            )
        ends = [e for e in events if isinstance(e, RallyEnd)]
        assert len(ends) == 1
        assert ends[0].winner_team == 0  # last hitter's team wins, not 1-0=1


class TestShotCooldown:
    def test_cooldown_prevents_rapid_shots(self):
        eng = RulesEngine()
        shots_fired = []
        for i in range(20):
            evs = eng.update(
                _result(ball_xy=None, shots=[{"slot": 1, "type": "drive"}]), i
            )
            shots_fired += [e for e in evs if isinstance(e, Shot)]
        # With _SHOT_COOLDOWN=8, shots at frames 0, 8, 16 → 3 total
        assert len(shots_fired) == 3
