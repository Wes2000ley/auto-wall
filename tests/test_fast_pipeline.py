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


def test_outline_smoothing_keeps_a_small_opening_and_closed_boundaries():
    from collections import Counter

    from benchmarks.check_geometry import ray_blocked
    from src.core.wall_geometry import outline_segments

    mask = np.zeros((120, 320), np.uint8)
    # A shallow opening that ordinary coarse simplification would seal, with
    # unrelated one-pixel boundary noise along the far side of the wall.
    points = [(20, 30), (135, 30), (135, 35), (143, 35), (143, 30), (280, 30)]
    points += [(x, 80 + (x // 4) % 2) for x in range(280, 19, -4)]
    cv2.fillPoly(mask, [np.asarray(points, np.int32)], 255)
    detailed = outline_segments(mask, .25, 1)
    coarse = outline_segments(mask, 8, 1)
    protected = outline_segments(mask, 8, 1, protected_regions=[(133, 28, 145, 36)],
                                 detail_tolerance=.25)
    assert not ray_blocked((139, 20), (139, 33), detailed)
    assert ray_blocked((139, 20), (139, 33), coarse)
    assert not ray_blocked((139, 20), (139, 33), protected)
    assert len(protected) < len(detailed) / 3
    # The protected arc and simplified remainder still share every endpoint.
    degrees = Counter(tuple(point) for segment in protected for point in segment)
    assert set(degrees.values()) == {2}
    assert len({tuple(sorted(map(tuple, edge))) for edge in protected}) == len(protected)
    distant = outline_segments(mask, 8, 1, protected_regions=[(400, 400, 410, 410)],
                               detail_tolerance=.25)
    np.testing.assert_array_equal(distant, coarse)


def test_learned_outline_smoothing_preserves_detailed_door_jambs():
    labels = np.zeros((1024, 1536), np.uint8)
    cv2.rectangle(labels, (150, 300), (1350, 316), 1, -1)
    cv2.rectangle(labels, (700, 300), (750, 316), 2, -1)
    # A narrow shaded indentation at the inner jamb is significant to the door
    # intersection even when the rest of the wall is safe to simplify.
    labels[307:310, 694:700] = 0
    prediction = (labels, np.ones(labels.shape, np.float32))
    image = np.zeros((*labels.shape, 3), np.uint8)
    settings = FastSettings(lights=False, ml_recover_jambs=True)
    detailed = detect_map(image, settings, prediction=prediction)
    simplified = detect_map(image, replace(settings, simplify=6), prediction=prediction)
    assert len(detailed.doors) == 1
    np.testing.assert_array_equal(simplified.doors, detailed.doors)


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


@pytest.mark.parametrize("degrees", [0, 25, 45, 90])
def test_clipping_preserves_sloped_jamb_intersections_without_blocking_opening(degrees):
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    jambs = np.array([[[-2, -2], [1, 1]], [[29, -2], [32, 1]]], dtype=float)
    # Remove both a central obstruction and a collinear fragment at the end.
    walls = np.concatenate((jambs, [[[15, -2], [15, 2]], [[0, 0], [2, 0]]])) @ rotation.T
    footprint = np.array([[-3, -6], [33, -6], [33, 6], [-3, 6]]) @ rotation.T
    candidate = np.array([[-1, 0], [31, 0]]) @ rotation.T
    clipped = _clip_doorways(list(walls), [footprint], preserve_jambs=True)
    np.testing.assert_allclose(clipped, jambs @ rotation.T, atol=1e-6)
    doors, disconnected = _attach_doors(clipped, [candidate], 1)
    assert not disconnected
    np.testing.assert_allclose(doors, np.array([[[0, 0], [31, 0]]]) @ rotation.T, atol=1e-6)
    from benchmarks.check_geometry import ray_blocked
    ray = np.array([[15, -10], [15, 10]]) @ rotation.T
    assert not ray_blocked(*ray, clipped)
    assert ray_blocked(*ray, clipped + doors)


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


def test_unicode_path_and_webp_85_export_preserve_source_and_dimensions(tmp_path):
    path = tmp_path / "地図.bmp"
    image = np.full((12, 16, 3), (30, 70, 90), dtype=np.uint8)
    path.write_bytes(cv2.imencode(".bmp", image)[1].tobytes())
    loaded, source = load_map(path)
    np.testing.assert_array_equal(image, loaded)
    webp = export_image_bytes(loaded)
    assert webp.startswith(b'RIFF') and webp[8:12] == b'WEBP'
    decoded = cv2.imdecode(np.frombuffer(webp, np.uint8), 1)
    assert decoded.shape == image.shape
    assert np.abs(decoded.astype(float) - image).mean() < 3
    assert path.read_bytes() == source
    assert webp == cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 85])[1].tobytes()


def test_webp_export_reports_oversized_image_without_resizing():
    with pytest.raises(ValueError, match='16383'):
        export_image_bytes(np.zeros((1, 16384, 3), np.uint8))


def test_atomic_export_preserves_existing_file_on_error(tmp_path):
    path = tmp_path / "map.dd2vtt"
    path.write_text("previous export", encoding="utf-8")
    with pytest.raises(ValueError):
        write_uvtt(path, {"invalid": float("nan")})
    assert path.read_text() == "previous export"
    assert list(tmp_path.iterdir()) == [path]
    write_uvtt(path, {"format": 0.3})
    assert json.loads(path.read_text()) == {"format": 0.3}


def test_public_uvtt_windows_do_not_include_editor_metadata(tmp_path):
    result = DetectionResult(100, 100, [], [], [], windows=[((50., 20.), (50., 80.))])
    assert to_uvtt(result, preview=True)["portals"][0]["_auto_wall_window"]
    public = to_uvtt(result)
    assert public["portals"][0]["closed"] is False
    assert not any(key.startswith("_") for portal in public["portals"] for key in portal)
    path = tmp_path / "map.dd2vtt"
    write_uvtt(path, public)
    assert json.loads(path.read_text()) == public


def test_cli_preview_and_report_include_window_faces_and_movement_barriers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import fast_auto_wall

    image = np.zeros((64, 64, 3), np.uint8)
    source = tmp_path / "source.bmp"
    assert cv2.imwrite(str(source), image)
    result = DetectionResult(64, 64, [], [], [], timings={"total": 0.0}, working_size=(64, 64),
                             windows=[((23., 10.), (23., 45.))],
                             window_faces=[[((20., 10.), (20., 45.)), ((26., 10.), (26., 45.))]],
                             invisible_walls=[((50., 10.), (50., 45.))])
    monkeypatch.setattr(fast_auto_wall, "detect_map", lambda *args: result)
    args = SimpleNamespace(image=source, benchmark=None, export=None,
                           preview=tmp_path / "preview.bmp", report=tmp_path / "report.json")
    assert fast_auto_wall.process(args, FastSettings()) == 0
    report = json.loads(args.report.read_text())
    assert report["windows"] == report["movement_only_segments"] == 1
    preview = cv2.imread(str(args.preview))
    assert preview[25, 20].any() and preview[25, 26].any() and preview[25, 50].any()
    np.testing.assert_array_equal(cv2.imread(str(source)), image)


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
