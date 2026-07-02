"""Tests for MatchState — WARMUP → LIVE → ENDED state machine."""
import pytest
from src.match_state import MatchState


class TestStateTransitions:
    def test_initial_state_is_warmup(self):
        ms = MatchState()
        assert ms.state == "warmup"
        assert not ms.counting

    def test_button_starts_live(self):
        ms = MatchState()
        ok = ms.start_by_button(100)
        assert ok
        assert ms.state == "live"
        assert ms.counting
        assert ms.status["live_started_at"] == 100

    def test_end_transitions_to_ended(self):
        ms = MatchState()
        ms.start_by_button(0)
        ms.end(500)
        assert ms.state == "ended"
        assert not ms.counting

    def test_reset_returns_to_warmup(self):
        ms = MatchState()
        ms.start_by_button(0)
        ms.end(100)
        ms.reset()
        assert ms.state == "warmup"
        assert not ms.counting

    def test_cannot_start_from_ended(self):
        ms = MatchState()
        ms.start_by_button(0)
        ms.end(100)
        ok = ms.start_by_button(200)
        assert not ok  # Must reset first

    def test_double_start_resets_live_started_at(self):
        ms = MatchState()
        ms.start_by_button(100)
        ms.start_by_button(200)
        assert ms.status["live_started_at"] == 200

    def test_detect_serve_start_empty_history(self):
        """Empty history → no serve detected."""
        from src.match_state import detect_serve_start
        from collections import deque
        assert detect_serve_start(deque(maxlen=50), 0) is False

    def test_detect_serve_start_pattern(self):
        """Stillness gap followed by ball reappearance → serve detected."""
        from src.match_state import detect_serve_start, _SERVE_STILLNESS
        from collections import deque
        hist = deque(maxlen=50)
        # Ball seen, then absent for _SERVE_STILLNESS frames, then reappears
        hist.append((100, 200))
        for _ in range(_SERVE_STILLNESS):
            hist.append(None)
        hist.append((150, 250))
        assert detect_serve_start(hist, 99) is True

    def test_detect_no_serve_without_gap(self):
        """Ball present throughout → no serve."""
        from src.match_state import detect_serve_start
        from collections import deque
        hist = deque(maxlen=50)
        for _ in range(30):
            hist.append((100, 200))
        assert detect_serve_start(hist, 99) is False

    def test_detect_no_serve_short_gap(self):
        """Gap shorter than threshold → no serve."""
        from src.match_state import detect_serve_start, _SERVE_STILLNESS
        from collections import deque
        hist = deque(maxlen=50)
        hist.append((100, 200))
        for _ in range(_SERVE_STILLNESS - 1):  # one short
            hist.append(None)
        hist.append((150, 250))
        assert detect_serve_start(hist, 99) is False
