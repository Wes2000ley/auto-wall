from collections import Counter

import cv2
import numpy as np
import pytest

from src.core.fast_pipeline import DetectionCancelled, FastSettings, detect_map
from src.core.wall_geometry import attach_centerline_doors, centerline_segments


pytest.importorskip("skimage.morphology")


def test_centerline_junction_shares_endpoint_without_duplicate_edges():
    mask = np.zeros((100, 100), np.uint8)
    cv2.line(mask, (10, 50), (90, 50), 255, 7)
    cv2.line(mask, (50, 10), (50, 50), 255, 7)
    walls = centerline_segments(mask, 1)
    degrees = Counter(tuple(p) for segment in walls for p in segment)
    assert sum(value == 3 for value in degrees.values()) == 1
    assert sum(value == 1 for value in degrees.values()) == 3
    assert len({tuple(sorted(map(tuple, segment))) for segment in walls}) == len(walls)


def test_centerline_loop_stays_closed_and_cancellation_is_honored():
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (10, 10), (90, 90), 255, 7)
    walls = centerline_segments(mask, 1)
    degrees = Counter(tuple(p) for segment in walls for p in segment)
    assert walls and set(degrees.values()) == {2}
    with pytest.raises(DetectionCancelled):
        centerline_segments(mask, 1, cancelled=lambda: True)


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    import builtins

    original_import = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == "skimage.morphology":
            raise ImportError("Not installed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ValueError, match="requirements-centerline.txt"):
        centerline_segments(np.zeros((20, 20), np.uint8), 1)


def test_centerline_doors_join_wall_middles_without_moving_walls():
    walls = np.array([[[10, 10], [10, 80]], [[50, 10], [50, 80]]], dtype=float)
    door = np.array([[13, 45], [47, 45]], dtype=float)
    mask = np.zeros((90, 70), np.uint8)
    cv2.rectangle(mask, (6, 10), (14, 80), 255, -1)
    cv2.rectangle(mask, (46, 10), (54, 80), 255, -1)
    before = mask.copy()
    connected, doors, rejected = attach_centerline_doors(walls, [door], 1, mask)
    assert not rejected and len(connected) == 4
    np.testing.assert_array_equal(connected[:2], walls)
    np.testing.assert_array_equal(doors[0], door)
    np.testing.assert_array_equal(mask, before)
    # A nearby line across an open floor is insufficient evidence for a jamb.
    mask[:, 45:55] = 0
    connected, doors, rejected = attach_centerline_doors(walls, [door], 1, mask)
    assert rejected == [0] and not doors
    np.testing.assert_array_equal(connected, walls)


def test_collinear_centerline_door_closes_a_real_opening():
    image = np.zeros((384, 512, 3), np.uint8)
    cv2.line(image, (30, 100), (480, 100), (160, 160, 160), 9)
    cv2.rectangle(image, (250, 95), (268, 105), (144, 32, 112), -1)
    result = detect_map(image, FastSettings(wall_geometry="centerline", lights=False))
    assert len(result.doors) == 1 and not result.warnings
    assert result.doors[0][0][1] == pytest.approx(result.doors[0][1][1])
    # Every portal endpoint touches an exported wall endpoint.
    endpoints = np.asarray(result.walls).reshape(-1, 2)
    for point in result.doors[0]:
        assert np.linalg.norm(endpoints - point, axis=1).min() < 1e-6


def test_rejected_centerline_candidate_restores_wall_geometry(monkeypatch):
    from src.core import fast_pipeline

    image = np.zeros((384, 512, 3), np.uint8)
    cv2.line(image, (30, 100), (480, 100), (160, 160, 160), 9)
    settings = FastSettings(wall_geometry="centerline", doors=False, lights=False)
    baseline = detect_map(image, settings)
    # A false candidate's rectangle intersects the wall, but its center is on floor.
    monkeypatch.setattr(fast_pipeline, "_detect_doors", lambda *args: (
        [np.array([[200.0, 80], [240.0, 80]])],
        [np.array([[198, 70], [242, 70], [242, 110], [198, 110]])]))
    from dataclasses import replace
    result = detect_map(image, replace(settings, doors=True))
    assert result.doors == [] and "Skipped 1" in result.warnings[0]
    np.testing.assert_array_equal(result.walls, baseline.walls)
