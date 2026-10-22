import base64
from copy import deepcopy
from dataclasses import replace
import json

import cv2
import numpy as np
import pytest

from src.core.fast_pipeline import (
    DetectionCancelled, DetectionResult, FastSettings, _attach_doors, _clip_doorways,
    detect_map, export_image_bytes, load_map, to_uvtt, write_uvtt,
)


@pytest.fixture
def dungeon():
    image = np.full((384, 512, 3), 12, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (480, 350), (131, 131, 131), 7)
    # A doorway with a bright rectangular door, plus a separate torch.
    cv2.rectangle(image, (250, 26), (263, 34), (65, 180, 255), -1)
    cv2.circle(image, (150, 180), 1, (0, 255, 255), -1)
    # A large bright patch must not become a point light or a door.
    cv2.rectangle(image, (280, 200), (380, 300), (0, 220, 255), -1)
    return image


def test_detects_door_and_torch_but_not_lava(dungeon):
    before = dungeon.copy()
    result = detect_map(dungeon)
    assert result.walls
    assert len(result.doors) == 1
    assert len(result.lights) == 1
    assert result.lights[0]["position"] == {"x": 150.0, "y": 180.0}
    start, end = result.doors[0]
    assert start[1] == pytest.approx(30, abs=5)
    assert end[1] == pytest.approx(30, abs=5)
    assert abs(end[0] - start[0]) > 13
    np.testing.assert_array_equal(before, dungeon)


def test_door_requires_wall_jambs(dungeon):
    cv2.rectangle(dungeon, (220, 180), (233, 188), (65, 180, 255), -1)
    assert len(detect_map(dungeon).doors) == 1
    assert len(detect_map(dungeon, FastSettings(door_wall_support=False)).doors) == 2


def test_quality_preserves_original_coordinates(dungeon):
    baseline = detect_map(dungeon)
    large = cv2.resize(dungeon, (1024, 768), interpolation=cv2.INTER_NEAREST)
    result = detect_map(large, FastSettings(max_dimension=512))
    assert result.working_size == (512, 384)
    assert (result.width, result.height) == (1024, 768)
    assert result.lights[0]["position"] == {"x": 300.0, "y": 360.0}
    assert result.doors[0][0][1] == pytest.approx(60, abs=10)
    np.testing.assert_allclose(result.doors, np.asarray(baseline.doors) * 2)


def test_warm_rectangle_beside_parallel_wall_is_not_a_door(dungeon):
    cv2.rectangle(dungeon, (200, 340), (213, 348), (65, 180, 255), -1)
    result = detect_map(dungeon)
    assert len(result.doors) == 1


def test_portal_endpoints_attach_to_wall_geometry():
    walls = [((0, 0), (0, 20)), ((30, 0), (30, 20))]
    doors = [np.array([[2.0, 10], [27.0, 10]])]
    attached, disconnected = _attach_doors(walls, doors, 1)
    assert disconnected == []
    np.testing.assert_allclose(attached, [[[0, 10], [30, 10]]])
    far = [np.array([[100.0, 10], [130.0, 10]])]
    attached, disconnected = _attach_doors(walls, far, 1)
    assert disconnected == [0]
    np.testing.assert_allclose(attached, far)


def test_unattached_candidate_is_rejected_without_cutting_walls(dungeon, monkeypatch):
    from src.core import fast_pipeline

    baseline = detect_map(dungeon, FastSettings(doors=False, lights=False))
    # A candidate parallel to this wall has no actual jamb intersections. Its
    # footprint overlaps masonry, so retaining its cut would damage the wall.
    monkeypatch.setattr(fast_pipeline, "_detect_doors", lambda *args: (
        [np.array([[100.0, 26], [120.0, 26]])],
        [np.array([[98, 20], [122, 20], [122, 36], [98, 36]])]))
    result = detect_map(dungeon, FastSettings(lights=False))
    assert result.doors == []
    np.testing.assert_allclose(result.walls, baseline.walls)
    assert any("Skipped 1" in warning for warning in result.warnings)


def test_clipping_preserves_jambs_and_straight_portal():
    walls = [((0, -5), (0, 5)), ((30, -5), (30, 5)), ((0, 4), (30, 4))]
    footprint = np.array([[-2, -6], [32, -6], [32, 6], [-2, 6]])
    clipped = _clip_doorways(walls, [footprint], preserve_jambs=True)
    np.testing.assert_allclose(clipped, walls[:2])
    doors, disconnected = _attach_doors(clipped, [np.array([[-1.0, 0], [31.0, 0]])], 1)
    assert disconnected == []
    np.testing.assert_allclose(doors, [[[0, 0], [30, 0]]])


def test_disabling_doors_keeps_highlights_out_of_lights(dungeon):
    settings = FastSettings(doors=False, light_threshold=160, light_max_area=3000)
    result = detect_map(dungeon, settings)
    assert not result.doors
    assert len(result.lights) == 1
    assert not detect_map(dungeon, replace(settings, lights=False)).lights


@pytest.mark.parametrize("degrees", [0, 25, 45, 90])
def test_clipping_opens_doorway_and_retains_wall_tails(degrees):
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    polygon = np.array([[4, -2], [6, -2], [6, 2], [4, 2]]) @ rotation.T
    wall = np.array([[0, 0], [10, 0]]) @ rotation.T
    clipped = _clip_doorways([wall], [polygon])
    expected = np.array([[[0, 0], [4, 0]], [[6, 0], [10, 0]]]) @ rotation.T
    np.testing.assert_allclose(clipped, expected, atol=1e-6)


def test_clipping_handles_parallel_and_fully_covered_segments():
    polygon = np.array([[4, -2], [6, -2], [6, 2], [4, 2]])
    walls = [((5, -1), (5, 1)), ((0, 3), (10, 3)), ((0, 0), (3, 0))]
    assert len(_clip_doorways(walls, [polygon])) == 2
    assert _clip_doorways([], [polygon]) == []


def test_uvtt_scales_walls_portals_and_lights_once():
    result = DetectionResult(400, 300, [((20, 40), (80, 40))], [((80, 40), (100, 40))],
                             [{"position": {"x": 100, "y": 150}, "range": 6,
                               "color": "ffffbb44", "intensity": 1, "shadows": True}])
    original = deepcopy(result)
    data = to_uvtt(result, 50, b"existing map bytes")
    assert data["line_of_sight"] == [[{"x": 0.4, "y": 0.8}, {"x": 1.6, "y": 0.8}]]
    assert data["portals"][0]["position"] == {"x": 1.8, "y": 0.8}
    assert data["portals"][0]["bounds"][1] == {"x": 2, "y": 0.8}
    assert data["lights"][0]["position"] == {"x": 2, "y": 3}
    assert data["resolution"]["map_size"] == {"x": 8, "y": 6}
    assert base64.b64decode(data["image"]) == b"existing map bytes"
    assert result == original
    assert "_preview_pixels" not in data


def test_cancellation_before_and_between_stages(dungeon):
    with pytest.raises(DetectionCancelled):
        detect_map(dungeon, cancelled=lambda: True)
    stages = []
    with pytest.raises(DetectionCancelled):
        detect_map(dungeon, progress=stages.append, cancelled=lambda: len(stages) == 2)
    assert stages == ["Preparing image", "Detecting walls"]


@pytest.mark.parametrize("settings", [FastSettings(grid_size=0), FastSettings(grid_size=float("nan")),
    FastSettings(door_max_length=1), FastSettings(light_min_area=500, light_max_area=10),
    FastSettings(max_dimension=1), FastSettings(wall_mode="unknown"), FastSettings(simplify=-1)])
def test_invalid_settings_fail_before_processing(settings, dungeon):
    with pytest.raises(ValueError):
        detect_map(dungeon, settings)


def test_blank_and_invalid_images():
    blank = detect_map(np.zeros((400, 401, 3), dtype=np.uint8))
    assert not blank.walls and not blank.doors and not blank.lights
    assert blank.warnings
    with pytest.raises(ValueError):
        detect_map(np.zeros((0, 0, 3), dtype=np.uint8))


def test_unicode_path_and_png_preservation(tmp_path):
    path = tmp_path / "地図.bmp"
    image = np.full((12, 16, 3), (30, 70, 90), dtype=np.uint8)
    path.write_bytes(cv2.imencode(".bmp", image)[1].tobytes())
    loaded, source = load_map(path)
    np.testing.assert_array_equal(image, loaded)
    png = export_image_bytes(loaded, source)
    assert png.startswith(b"\x89PNG")
    assert export_image_bytes(loaded, png) is png
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png, np.uint8), 1), image)


def test_atomic_export_preserves_existing_file_on_error(tmp_path):
    path = tmp_path / "map.dd2vtt"
    path.write_text("previous export", encoding="utf-8")
    with pytest.raises(ValueError):
        write_uvtt(path, {"invalid": float("nan")})
    assert path.read_text() == "previous export"
    assert list(tmp_path.iterdir()) == [path]
    write_uvtt(path, {"format": 0.3})
    assert json.loads(path.read_text()) == {"format": 0.3}
