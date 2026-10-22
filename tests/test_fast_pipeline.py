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
    cv2.rectangle(image, (30, 30), (480, 350), (160, 160, 160), 7)
    # A doorway with a dark purple rectangular door, plus a separate torch.
    cv2.rectangle(image, (250, 26), (263, 34), (144, 32, 112), -1)
    cv2.circle(image, (150, 180), 2, (0, 255, 255), -1)
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
    cv2.rectangle(dungeon, (220, 180), (233, 188), (144, 32, 112), -1)
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


def test_colored_rectangle_beside_parallel_wall_is_not_a_door(dungeon):
    cv2.rectangle(dungeon, (200, 340), (213, 348), (144, 32, 112), -1)
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


def test_portal_stops_at_inner_faces_when_endpoints_extend_into_thick_posts():
    walls = [((x, 0), (x, 20)) for x in (0, 4, 30, 34)]
    door = np.array([[1.0, 10], [33.0, 10]])
    attached, disconnected = _attach_doors(walls, [door], 1)
    assert not disconnected
    np.testing.assert_allclose(attached, [[[4, 10], [30, 10]]])
    # The rule is independent of the door's point ordering.
    attached, disconnected = _attach_doors(walls, [door[::-1]], 1)
    assert not disconnected
    np.testing.assert_allclose(attached, [[[30, 10], [4, 10]]])


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


@pytest.mark.parametrize("wall,door", [((160, 160, 160), (112, 32, 144)),
    ((30, 110, 180), (100, 45, 15)), ((210, 80, 40), (180, 20, 25))])
def test_dark_doors_and_alternative_palettes(wall, door):
    image = np.zeros((384, 512, 3), np.uint8)
    cv2.rectangle(image, (30, 30), (480, 350), wall[::-1], 9)
    cv2.rectangle(image, (240, 25), (258, 35), door[::-1], -1)
    result = detect_map(image, FastSettings(wall_color=wall, door_color=door, lights=False))
    assert len(result.doors) == 1 and result.walls


def test_exact_palette_preserves_openings_voids_and_original_rgb():
    from src.core.fast_pipeline import _wall_mask, _detect_doors
    from src.core.detection_profiles import preset_settings

    image = np.full((384, 512, 3), 20, np.uint8)
    cv2.rectangle(image, (30, 30), (480, 350), (160, 160, 160), 7)
    image[25:36, 100:103] = 20  # intentional three-pixel opening
    image[150:180, 150:180] = 0  # exact void must never enter the wall mask
    image[100:120, 200:220] = 159  # one channel off is not an exact wall
    cv2.rectangle(image, (250, 26), (263, 34), (144, 32, 112), -1)
    settings = preset_settings("exact")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = _wall_mask(image, hsv, settings, 0.25)
    assert not mask[25:36, 100:103].any()
    assert not mask[150:180, 150:180].any()
    assert not mask[100:120, 200:220].any()
    assert mask[30, 50]
    # RGB->HSV->RGB is lossy for this purple; exact matching uses source pixels.
    doors, _ = _detect_doors(hsv, mask, settings, 0.25, image)
    assert len(doors) == 1
    broad = _wall_mask(image, hsv, replace(settings, wall_tolerance=255), 0.25)
    assert not broad[150:180, 150:180].any()


def test_wide_shaded_gate_is_one_portal():
    image = np.zeros((1024, 1536, 3), np.uint8)
    cv2.rectangle(image, (30, 80), (1500, 900), (160, 160, 160), 17)
    cv2.rectangle(image, (600, 70), (710, 90), (144, 32, 112), -1)
    image[70:91, 654:656] = 15  # dark seam between the gate's leaves
    result = detect_map(image, FastSettings(lights=False))
    assert len(result.doors) == 1
    assert abs(result.doors[0][1][0] - result.doors[0][0][0]) >= 110


def test_lights_group_highlights_and_reject_sparse_marks_and_glints():
    image = np.zeros((512, 512, 3), np.uint8)
    # One orange source, with two separate yellow cores, is one light.
    cv2.rectangle(image, (50, 50), (60, 60), (0, 90, 240), -1)
    cv2.circle(image, (53, 54), 2, (0, 255, 255), -1)
    cv2.circle(image, (57, 57), 2, (0, 255, 255), -1)
    # Another neighboring source is distinct, not distance-merged into the first.
    cv2.circle(image, (80, 55), 3, (0, 255, 255), -1)
    # Sparse bright diagram and a glint in a large colored patch are not sources.
    cv2.rectangle(image, (150, 50), (190, 90), (0, 255, 255), 1)
    cv2.rectangle(image, (250, 50), (264, 64), (0, 90, 240), -1)
    image[55, 255] = (0, 255, 255)
    result = detect_map(image, FastSettings(doors=False))
    assert len(result.lights) == 2
    positions = sorted(light["position"]["x"] for light in result.lights)
    assert positions == pytest.approx([55, 80])


def test_cool_and_sampled_sources_support_other_art_palettes():
    image = np.zeros((512, 512, 3), np.uint8)
    cv2.circle(image, (80, 80), 3, (255, 255, 80), -1)
    assert not detect_map(image, FastSettings()).lights
    for settings in (FastSettings(light_mode="cool"),
                     FastSettings(light_mode="color", light_color=(80, 255, 255))):
        result = detect_map(image, settings)
        assert len(result.lights) == 1
        assert result.lights[0]["position"] == {"x": 80, "y": 80}


@pytest.mark.parametrize("door_match", ["rgb", "hue"])
def test_exact_downsampling_does_not_invent_wall_colors(door_match):
    from src.core.detection_profiles import preset_settings

    # Area averaging would turn these non-wall colors into exactly gray 160.
    image = np.empty((1024, 1024, 3), np.uint8)
    image[::2] = 140
    image[1::2] = 180
    result = detect_map(image, replace(preset_settings("exact"), max_dimension=512,
                                       door_match=door_match, lights=False))
    assert not result.walls
