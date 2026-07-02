"""Tests for homography utilities — court projection math."""
import numpy as np
import pytest

from src.utils.homography import compute_homography, project_points, REFERENCE_COURT_26


class TestComputeHomography:
    def test_valid_homography_from_26_keypoints(self):
        """26-point court keypoints with 4 visible corners → valid H."""
        pts = np.zeros((26, 2), dtype=np.float32)
        pts[8]  = [100, 800]   # court_bottom_left_close → ref (0,0)
        pts[9]  = [100, 200]   # court_bottom_left_far → ref (0,20)
        pts[10] = [900, 800]   # court_bottom_right_close → ref (10,0)
        pts[11] = [900, 200]   # court_bottom_right_far → ref (10,20)
        H, mask = compute_homography(pts)
        assert H is not None
        # Project image center → should land inside court [0,10]×[0,20]
        projected = project_points(H, [[500, 500]])
        assert projected is not None
        cx, cy = projected[0]
        assert 0 <= cx <= 10
        assert 0 <= cy <= 20

    def test_corner_round_trip(self):
        """Projected corners should approximately match reference court corners."""
        pts = np.zeros((26, 2), dtype=np.float32)
        img_corners = {8: [100, 800], 9: [100, 200], 10: [900, 800], 11: [900, 200]}
        for idx, val in img_corners.items():
            pts[idx] = val
        H, _ = compute_homography(pts)
        assert H is not None
        for idx, img_pt in img_corners.items():
            proj = project_points(H, [img_pt])
            ref = REFERENCE_COURT_26[idx]
            assert abs(proj[0][0] - ref[0]) < 1.0, f"X mismatch for kpt {idx}"
            assert abs(proj[0][1] - ref[1]) < 1.0, f"Y mismatch for kpt {idx}"

    def test_degenerate_input_returns_none(self):
        """Collinear points cannot produce a valid homography."""
        pts = np.zeros((26, 2), dtype=np.float32)
        pts[8]  = [0, 0]
        pts[9]  = [1, 1]
        pts[10] = [2, 2]
        pts[11] = [3, 3]
        H, mask = compute_homography(pts)
        assert H is None or mask is None or mask.sum() < 4

    def test_too_few_visible_returns_none(self):
        """Only 2 visible points → None."""
        pts = np.zeros((26, 2), dtype=np.float32)
        pts[8] = [100, 200]
        pts[9] = [900, 200]
        H, _ = compute_homography(pts)
        assert H is None


class TestProjectPoints:
    def test_none_homography(self):
        assert project_points(None, [[5, 5]]) is None


class TestZonePredicates:
    def test_is_in_play_center(self):
        from src.utils.homography import is_in_play
        assert is_in_play([5.0, 10.0])

    def test_is_in_play_corner(self):
        from src.utils.homography import is_in_play
        assert is_in_play([0.0, 0.0])
        assert is_in_play([10.0, 20.0])

    def test_is_out_beyond_bounds(self):
        from src.utils.homography import is_out
        assert is_out([15.0, 10.0])
        assert is_out([5.0, 25.0])
        assert not is_out([5.0, 10.0])

    def test_is_in_play_tolerance(self):
        from src.utils.homography import is_in_play
        assert is_in_play([-0.3, -0.3])   # within 0.5m tolerance
        assert not is_in_play([-1.0, -1.0])

    def test_court_half(self):
        from src.utils.homography import court_half
        assert court_half([5.0, 3.0]) == 0   # near
        assert court_half([5.0, 17.0]) == 1  # far
        assert court_half([5.0, 10.0]) == 1  # exactly at net → far

    def test_service_box_near(self):
        from src.utils.homography import is_in_service_box
        assert is_in_service_box([5.0, 8.0], half=0)
        assert not is_in_service_box([5.0, 3.0], half=0)
        assert not is_in_service_box([5.0, 12.0], half=0)

    def test_service_box_far(self):
        from src.utils.homography import is_in_service_box
        assert is_in_service_box([5.0, 12.0], half=1)
        assert not is_in_service_box([5.0, 17.0], half=1)

    def test_near_net(self):
        from src.utils.homography import is_near_net
        assert is_near_net([5.0, 9.5])
        assert is_near_net([5.0, 10.8])
        assert not is_near_net([5.0, 5.0])
