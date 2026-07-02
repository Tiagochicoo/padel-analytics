"""Tests for OnlineReIDResolver — position fallback and slot assignment."""
import numpy as np
import pytest

try:
    from src.reid_online import OnlineReIDResolver
    _REID_AVAILABLE = True
except Exception:
    _REID_AVAILABLE = False


@pytest.mark.skipif(not _REID_AVAILABLE, reason="reid_online import deps unavailable")
class TestPositionFallback:
    def test_assigns_distinct_slots(self):
        """When warmup never converged, _position_fallback assigns P1..P4 by position."""
        r = OnlineReIDResolver(device="cpu", k=4)
        # Simulate 4 tracklets with different feet positions
        from src.reid import Tracklet
        positions = [
            (0.2, 0.1),   # near-left → should get low slot
            (0.8, 0.1),   # near-right
            (0.2, 0.9),   # far-left
            (0.8, 0.9),   # far-right → should get high slot
        ]
        for i, pos in enumerate(positions):
            t = Tracklet(track_id=i)
            t.frame_min = 0
            t.frame_max = 100
            t.feet_pos.append(pos)
            r._tracklets[i] = t
        # Force fallback (no clustering happened)
        assert r.mapping == {}
        r._position_fallback()
        assert len(r.mapping) == 4
        slots_used = set(r.mapping.values())
        assert slots_used == {1, 2, 3, 4}

    def test_nearest_slot_no_pileup(self):
        """After all 4 slots are used, _nearest_slot should not return 1 for everyone."""
        r = OnlineReIDResolver(device="cpu", k=4)
        # Fill slots 1-4
        for raw_id in range(1, 5):
            r.mapping[raw_id] = raw_id
        # 5th raw_id: all slots used — should still return something 1-4
        result = r._nearest_slot(99)
        assert 1 <= result <= 4
