import numpy as np
import pytest

from benchmarks.check_geometry import evaluate_geometry, match_instances, ray_blocked
from src.core.fast_pipeline import DetectionResult, to_uvtt


def test_assignment_counts_duplicates_and_avoids_greedy_mismatch():
    result = match_instances([[1, 2, 50], [2, 8, 50]], 3)
    assert result["matched"] == 2 and result["extra"] == [2]
    assert result["precision"] == 2 / 3 and result["recall"] == 1
    assert match_instances(np.empty((0, 2)), 3)["extra"] == [0, 1]
    assert match_instances(np.empty((2, 0)), 3)["missed"] == [0, 1]


def test_ray_crossing_collinear_overlap_touch_and_clear_path():
    walls = [((10, 10), (10, 20))]
    assert ray_blocked((0, 15), (20, 15), walls)
    assert ray_blocked((10, 15), (10, 25), walls)
    assert ray_blocked((0, 10), (10, 10), walls)
    assert not ray_blocked((10, 30), (10, 40), walls)
    assert not ray_blocked((0, 25), (20, 25), walls)
    assert not ray_blocked((0, 10), (10, 10), [])
    with pytest.raises(ValueError, match="distinct finite"):
        ray_blocked((0, 0), (float("nan"), 1), [])


def test_export_metrics_scale_endpoints_and_check_closed_and_open_doors():
    result = DetectionResult(200, 100, [((100, 0), (100, 40)), ((100, 60), (100, 100))],
                             [((100, 60), (100, 40))],
                             [{"position": {"x": 20, "y": 40}}])
    spec = {"format": "auto-wall-geometry-checks-v1", "reference_size": [100, 50],
            "doors": [{"bounds": [[50, 20], [50, 30]]}], "lights": [{"position": [10, 20]}],
            "rays": [{"from": [40, 25], "to": [60, 25], "blocked": False},
                     {"from": [40, 25], "to": [60, 25], "blocked": True, "closed_doors": True},
                     {"from": [40, 5], "to": [60, 5], "blocked": True}]}
    metrics = evaluate_geometry(to_uvtt(result, 70), spec)
    assert metrics["doors"]["matched"] == metrics["lights"]["matched"] == 1
    assert metrics["doors"]["mean_error"] == metrics["lights"]["mean_error"] == 0
    assert metrics["rays_passed"] == metrics["rays_total"] == 3
    assert metrics["lights"]["precision"] is None  # no completeness claim supplied
    spec["complete_instances"] = {"doors": True, "lights": True}
    assert evaluate_geometry(to_uvtt(result), spec)["lights"]["precision"] == 1
    # A short fragment at the correct center is not a complete door.
    result.doors = [((100, 49), (100, 51))]
    spec["door_tolerance"] = 2
    assert evaluate_geometry(to_uvtt(result), spec)["doors"]["matched"] == 0
