"""Tests for ShotClassifier — contact detection and cooldown."""
import numpy as np
import pytest

from src.analyzer import Player


def _player(slot, box=(100, 100, 200, 300), keypoints=None):
    return Player(track_id=slot, box=box, team=0, court_xy=(5.0, 10.0), keypoints=keypoints)


def _fake_kpts(lw=(150, 200, 1.0), rw=(180, 200, 1.0)):
    """Build a (17,3) keypoint array with only wrists set."""
    k = np.zeros((17, 3))
    if lw:
        k[9] = lw
    if rw:
        k[10] = rw
    return k


class TestHittingZone:
    def test_ball_inside_expanded_box(self):
        from src.shot_classifier import ShotClassifier
        assert ShotClassifier._in_hitting_zone(150, 200, (100, 100, 200, 300))

    def test_ball_outside_box(self):
        from src.shot_classifier import ShotClassifier
        assert not ShotClassifier._in_hitting_zone(500, 500, (100, 100, 200, 300))


class TestWristProximity:
    def test_ball_near_wrist_triggers(self):
        """When keypoints are available, ball near wrist = contact."""
        from src.shot_classifier import ShotClassifier
        sc = ShotClassifier(None)  # no model — we test _is_contact only
        p = _player(1, keypoints=_fake_kpts(lw=(150, 200, 1.0)))
        assert sc._is_contact(155, 205, p)   # 7px from wrist → contact

    def test_ball_far_from_wrist_no_trigger(self):
        from src.shot_classifier import ShotClassifier
        sc = ShotClassifier(None)
        p = _player(1, keypoints=_fake_kpts(lw=(150, 200, 1.0)))
        assert not sc._is_contact(400, 400, p)  # far from wrist

    def test_invisible_wrists_fall_back_to_box(self):
        """When wrists are not visible, fall back to box proximity."""
        from src.shot_classifier import ShotClassifier
        sc = ShotClassifier(None)
        p = _player(1, keypoints=_fake_kpts(lw=(150, 200, 0.0), rw=(180, 200, 0.0)))
        assert sc._is_contact(150, 200, p)  # inside box → contact via fallback

    def test_no_keypoints_falls_back_to_box(self):
        from src.shot_classifier import ShotClassifier
        sc = ShotClassifier(None)
        p = _player(1, keypoints=None)
        assert sc._is_contact(150, 200, p)      # inside box
        assert not sc._is_contact(500, 500, p)  # outside box


class TestGracefulDegradation:
    def test_no_model_returns_empty(self):
        from src.shot_classifier import ShotClassifier
        sc = ShotClassifier(None)
        result = sc.classify(np.zeros((480, 640, 3)), [_player(1)], (150, 200))
        assert result == []

    def test_no_ball_returns_empty(self):
        from src.shot_classifier import ShotClassifier
        sc = ShotClassifier(None)
        result = sc.classify(np.zeros((480, 640, 3)), [_player(1)], None)
        assert result == []
