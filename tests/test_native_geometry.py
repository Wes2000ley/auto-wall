import numpy as np
import pytest

from src.core.fast_pipeline import DetectionCancelled
from src.core import native_geometry
from src.core.wall_geometry import _trace_python, centerline_backend, centerline_segments


def test_backend_selection_and_missing_library(monkeypatch):
    monkeypatch.setattr(native_geometry, "native_available", lambda: False)
    assert centerline_backend("auto") == "python"
    assert centerline_backend("python") == "python"
    with pytest.raises(ValueError, match="build_native.py"):
        centerline_backend("cpp")
    with pytest.raises(ValueError, match="must be"):
        centerline_backend("unknown")


@pytest.fixture
def compiled():
    if not native_geometry.native_available():
        pytest.skip("Optional C++ tracer not built; run python scripts/build_native.py")


@pytest.mark.parametrize("seed", range(25))
def test_native_paths_match_python_on_random_graphs(compiled, seed):
    # Different densities cover isolated points, diagonals, cycles, and junctions.
    rng = np.random.default_rng(seed)
    skeleton = rng.random((23, 31)) < (0.05 + (seed % 10) / 10)
    before = skeleton.copy()
    expected = _trace_python(skeleton)
    result = native_geometry.trace_skeleton(skeleton)
    assert len(result) == len(expected)
    for actual, reference in zip(result, expected):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(skeleton, before)


def test_native_geometry_and_empty_masks_match(compiled):
    pytest.importorskip("skimage.morphology")
    import cv2

    mask = np.zeros((120, 150), np.uint8)
    cv2.rectangle(mask, (10, 10), (80, 80), 255, 9)
    cv2.line(mask, (80, 45), (140, 110), 255, 7)
    for tolerance in (0, 0.5, 2, 5):
        np.testing.assert_array_equal(centerline_segments(mask, tolerance, backend="cpp"),
                                      centerline_segments(mask, tolerance, backend="python"))
    assert native_geometry.trace_skeleton(np.zeros((3, 7), bool)) == []
    assert native_geometry.trace_skeleton(np.ones((1, 1), bool)) == []


def test_native_cancellation_and_callback_errors(compiled):
    mask = np.ones((100, 100), bool)
    with pytest.raises(DetectionCancelled):
        native_geometry.trace_skeleton(mask, cancelled=lambda: True)
    def failed_callback():
        raise ValueError("Callback failed")
    with pytest.raises(ValueError, match="Callback failed"):
        native_geometry.trace_skeleton(mask, cancelled=failed_callback)
    # Cancellation/errors leave the library usable and own no leaked output buffers.
    np.testing.assert_array_equal(native_geometry.trace_skeleton(np.ones((1, 3), bool))[0],
                                  [[0, 0], [1, 0], [2, 0]])


def test_native_accepts_noncontiguous_masks_and_rejects_bad_shapes(compiled):
    source = np.eye(10, dtype=bool)[::2]
    result = native_geometry.trace_skeleton(source)
    expected = _trace_python(source)
    assert len(result) == len(expected)
    for actual, reference in zip(result, expected):
        np.testing.assert_array_equal(actual, reference)
    for shape in ((0, 10), (10,), (2, 3, 4)):
        with pytest.raises(ValueError, match="two-dimensional"):
            native_geometry.trace_skeleton(np.zeros(shape, np.uint8))


def test_native_paths_remain_valid_after_another_call(compiled):
    result = native_geometry.trace_skeleton(np.ones((1, 100), bool))[0]
    expected = result.copy()
    native_geometry.trace_skeleton(np.ones((100, 100), bool))
    np.testing.assert_array_equal(result, expected)


def test_native_can_cancel_inside_a_long_path(compiled):
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        # Two neighbor scans, one branch scan, then the first long-path check.
        return calls == 4
    with pytest.raises(DetectionCancelled):
        native_geometry.trace_skeleton(np.ones((1, 8192), bool), cancelled)
    assert calls == 4
