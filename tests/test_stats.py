"""Tests for StatsAccumulator — per-player statistics."""
import time
import pytest
from src.server.stats import StatsAccumulator, SLOT_COLORS_HEX, SLOT_COLORS_BGR


def _player(slot, box=(100, 100, 200, 300), court_xy=(5.0, 10.0), team=None):
    """Create a minimal player-like object."""
    class P:
        pass
    p = P()
    p.track_id = slot
    p.box = box
    p.court_xy = court_xy
    p.team = team
    p.keypoints = None
    return p


class TestColorConsistency:
    def test_bgr_matches_hex(self):
        """BGR tuples must produce the same RGB as the hex values."""
        for slot in range(1, 5):
            b, g, r = SLOT_COLORS_BGR[slot]
            hex_str = SLOT_COLORS_HEX[slot]
            assert int(hex_str[1:3], 16) == r, f"R mismatch slot {slot}"
            assert int(hex_str[3:5], 16) == g, f"G mismatch slot {slot}"
            assert int(hex_str[5:7], 16) == b, f"B mismatch slot {slot}"


class TestDistance:
    def test_distance_accumulates(self):
        acc = StatsAccumulator()
        acc.update([_player(1, court_xy=(0.0, 0.0))])
        acc.update([_player(1, court_xy=(3.0, 3.0))])  # ~4.24m
        snap = acc.snapshot()
        assert abs(snap[0]["distance_m"] - 4.2) < 0.5

    def test_teleport_filtered(self):
        """Jumps >5m should not count as distance (re-acquisition noise)."""
        acc = StatsAccumulator()
        acc.update([_player(1, court_xy=(0.0, 0.0))])
        acc.update([_player(1, court_xy=(0.0, 10.0))])  # 10m — too far, ignored
        snap = acc.snapshot()
        assert snap[0]["distance_m"] == 0.0


class TestTimeOnCourt:
    def test_wall_clock_accumulation(self):
        """time_on_court should reflect real elapsed time, not frame count / 30."""
        acc = StatsAccumulator()
        acc.update([_player(1)])
        time.sleep(0.2)
        acc.update([_player(1)])
        snap = acc.snapshot()
        # Should be ~0.2s, definitely > 0 and < 5.0
        assert snap[0]["time_on_court"] > 0.0
        assert snap[0]["time_on_court"] < 5.0


class TestReset:
    def test_reset_clears_all(self):
        acc = StatsAccumulator()
        acc.update([_player(1, court_xy=(0.0, 0.0))], shots=[{"slot": 1, "type": "drive"}])
        acc.update([_player(1, court_xy=(3.0, 4.0))])
        acc.reset()
        snap = acc.snapshot()
        assert snap[0]["distance_m"] == 0.0
        assert snap[0]["shots"] == 0
        assert snap[0]["time_on_court"] == 0.0
        assert snap[0]["frames_seen"] == 0


class TestShots:
    def test_shot_counting(self):
        acc = StatsAccumulator()
        acc.update([_player(1)], shots=[{"slot": 1, "type": "drive"}])
        acc.update([_player(1)], shots=[{"slot": 1, "type": "drive"}, {"slot": 1, "type": "lob"}])
        snap = acc.snapshot()
        assert snap[0]["shots"] == 3
        assert snap[0]["shots_by_type"]["drive"] == 2
        assert snap[0]["shots_by_type"]["lob"] == 1
