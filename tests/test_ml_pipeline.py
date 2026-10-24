"""ML mask-to-geometry integration stays testable without optional torch imports."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.core import fast_pipeline
from src.core.fast_pipeline import DetectionResult, FastSettings, detect_map, to_uvtt


def test_learned_masks_replace_palette_and_omit_transparent_barriers(monkeypatch):
    image = np.full((384, 512, 3), (31, 97, 69), np.uint8)
    wall = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(wall, (30, 30), (480, 350), 255, 7)
    door, light, transparent = (np.zeros_like(wall) for _ in range(3))
    cv2.rectangle(door, (250, 26), (263, 34), 255, -1)
    wall[door > 0] = 0
    cv2.circle(light, (150, 180), 3, 255, -1)
    cv2.line(transparent, (210, 160), (350, 160), 255, 5)
    copies = [m.copy() for m in (wall, door, light, transparent)]
    monkeypatch.setattr(fast_pipeline, "_learned_masks", lambda *args: ([wall, door, light, transparent], "torch-cpu"))
    settings = replace(FastSettings(), model_path="model.safetensors")
    result = detect_map(image, settings)
    assert result.detection_backend == "torch-cpu"
    assert len(result.doors) == len(result.lights) == 1
    assert result.transparent_barriers
    assert result.lights[0]["position"] == {"x": 150.0, "y": 180.0}
    data = to_uvtt(result)
    assert data["objects_line_of_sight"] == []
    assert len(data["line_of_sight"]) == len(result.walls)
    for a, b in result.walls:
        assert min(a[1], b[1]) < 40 or max(a[1], b[1]) > 340
    for original, after in zip(copies, (wall, door, light, transparent)):
        np.testing.assert_array_equal(original, after)


def test_transparent_only_result_never_becomes_opaque_or_door():
    result = DetectionResult(100, 100, [], [], [], transparent_barriers=[((20, 30), (80, 30))])
    exported = to_uvtt(result)
    assert exported["line_of_sight"] == exported["objects_line_of_sight"] == exported["portals"] == []


def test_precomputed_prediction_uses_same_masks_and_checks_working_dimensions(monkeypatch):
    image = np.zeros((512, 512, 3), np.uint8)
    labels = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(labels, (20, 20), (480, 480), 1, 8)
    cv2.line(labels, (100, 80), (200, 80), 4, 5)
    confidence = np.full(labels.shape, .99, np.float32)

    def unexpected_inference(*args):
        pytest.fail("An existing prediction must not reload or run a model.")

    monkeypatch.setattr(fast_pipeline, "_learned_masks", unexpected_inference)
    result = detect_map(image, prediction=(labels, confidence))
    assert result.detection_backend == "precomputed" and result.walls and result.transparent_barriers
    with pytest.raises(ValueError, match="working image"):
        detect_map(image, replace(FastSettings(), max_dimension=256), prediction=(labels, confidence))
    confidence[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite probabilities"):
        detect_map(image, prediction=(labels, confidence))


def test_narrow_learned_jambs_survive_area_filter_without_preserving_specks(monkeypatch):
    image = np.zeros((512, 512, 3), np.uint8)
    wall, door, light, transparent = (np.zeros(image.shape[:2], np.uint8) for _ in range(4))
    # Each post is smaller than the ordinary 1000 px reference-area cutoff.
    cv2.rectangle(wall, (100, 96), (102, 108), 255, -1)
    cv2.rectangle(wall, (124, 96), (126, 108), 255, -1)
    cv2.rectangle(wall, (210, 96), (212, 108), 255, -1)  # unsupported noise
    cv2.rectangle(door, (104, 98), (122, 106), 255, -1)
    monkeypatch.setattr(fast_pipeline, "_learned_masks", lambda *args: ([wall, door, light, transparent], "torch-cpu"))
    result = detect_map(image, replace(FastSettings(), model_path="model.safetensors", lights=False))
    assert len(result.doors) == 1 and result.walls
    assert max(point[0] for segment in result.walls for point in segment) < 130
    # Both exported endpoints meet a wall, with no synthetic bridge to a speck.
    doors, disconnected = fast_pipeline._attach_doors(result.walls, [np.array(result.doors[0])], 1)
    assert not disconnected
    np.testing.assert_allclose(doors, result.doors)


def test_learned_rectangle_next_to_parallel_wall_does_not_protect_noise(monkeypatch):
    image = np.zeros((512, 512, 3), np.uint8)
    wall, door, light, transparent = (np.zeros(image.shape[:2], np.uint8) for _ in range(4))
    cv2.rectangle(wall, (95, 110), (128, 111), 255, -1)
    cv2.rectangle(door, (104, 98), (122, 106), 255, -1)
    monkeypatch.setattr(fast_pipeline, "_learned_masks", lambda *args: ([wall, door, light, transparent], "torch-cpu"))
    result = detect_map(image, replace(FastSettings(), model_path="model.safetensors", min_wall_area=2000, lights=False))
    assert not result.doors and not result.walls


@pytest.mark.parametrize("change", [dict(model_path=42), dict(ml_device="magic"),
                                    dict(ml_wall_confidence=.49), dict(ml_door_confidence=float("nan")),
                                    dict(ml_light_confidence=1.1), dict(ml_recover_jambs=1),
                                    dict(ml_profile="new"), dict(ml_profile="windows", ml_window_confidence=1.1),
                                    dict(ml_barrier_confidence=float("nan")),
                                    dict(ml_light_center_threshold=.01), dict(ml_light_center_threshold=float("nan"))])
def test_invalid_ml_settings(change):
    with pytest.raises(ValueError):
        replace(FastSettings(), **change).validate()


def test_model_class_metadata_must_match_explicit_profile():
    legacy = ("background", "wall", "door", "light", "transparent_barrier")
    windows = legacy + ("window",)
    fast_pipeline._validate_model_profile(legacy, "legacy")
    fast_pipeline._validate_model_profile(windows, "windows")
    with pytest.raises(ValueError, match="does not match"):
        fast_pipeline._validate_model_profile(windows, "legacy")
    with pytest.raises(ValueError, match="does not match"):
        fast_pipeline._validate_model_profile(legacy, "windows")


def test_windows_profile_exports_supported_window_and_only_long_thin_fences():
    image = np.zeros((512, 512, 3), np.uint8)
    labels = np.zeros(image.shape[:2], np.uint8)
    confidence = np.full(labels.shape, .99, np.float32)

    # A broken wall band with a confident window-shaped opening.
    labels[200:211, 40:150] = 1
    labels[200:211, 150:191] = 5
    labels[200:211, 191:480] = 1
    # A legitimate, straight fence stroke.
    labels[80:85, 80:201] = 4
    # Compact and very short props must not become movement walls.
    cv2.circle(labels, (320, 120), 12, 4, -1)
    labels[160:163, 250:255] = 4
    # Shape-like fence pixels below the strict confidence threshold are ignored.
    labels[300:305, 250:420] = 4
    confidence[300:305, 250:420] = .96
    # A plausible but unsupported door is retained only because door support is
    # explicitly disabled; it must not stop a separate supported window.
    labels[280:289, 40:60] = 2

    settings = replace(FastSettings(), ml_profile="windows", lights=False,
                       door_wall_support=False, ml_window_confidence=.93,
                       ml_barrier_confidence=.97)
    result = detect_map(image, settings, prediction=(labels, confidence))
    assert len(result.windows) == 1
    assert len(result.window_faces) == 1 and len(result.window_faces[0]) == 2
    assert not any(frozenset(face) == frozenset(wall)
                   for group in result.window_faces for face in group for wall in result.walls)
    assert len(result.doors) == 1
    assert len(result.invisible_walls) == 1
    # A straight, thick fence is exported as one movement-only centerline.
    start, end = result.invisible_walls[0]
    assert abs(start[1] - end[1]) <= 1
    assert abs(end[0] - start[0]) > 100
    assert not result.transparent_barriers

    data = to_uvtt(result, grid_size=70, image_bytes=fast_pipeline.export_image_bytes(image), preview=True)
    window_portals = [portal for portal in data["portals"] if not portal["closed"]]
    doors = [portal for portal in data["portals"] if portal["closed"]]
    assert len(window_portals) == len(result.window_faces[0]) == 2
    assert all(portal["_auto_wall_window"] for portal in window_portals)
    assert len(doors) == 1
    assert len(data["_invisible_walls"]) == 1
    from src.core.foundry_export import to_foundry_scene
    scene = to_foundry_scene(data)
    foundry_windows = [wall for wall in scene["walls"] if wall["sight"] == 30]
    foundry_doors = [wall for wall in scene["walls"] if wall["door"] == 1]
    foundry_invisible = [wall for wall in scene["walls"] if wall["sight"] == 0]
    assert len(foundry_windows) == 2
    assert len(foundry_doors) == len(foundry_invisible) == 1
    assert all(wall["door"] == 0 and wall["move"] == 20 for wall in foundry_windows)
    assert all(wall["sight"] == wall["light"] == wall["sound"] == 30 for wall in foundry_windows)
    assert any(wall["sight"] == wall["light"] == wall["sound"] == 20 for wall in scene["walls"])

    strict = detect_map(image, replace(settings, door_wall_support=True), prediction=(labels, confidence))
    assert not strict.doors and len(strict.windows) == 1


def test_fence_centerline_trace_terminates_and_collapses_a_thick_stroke():
    mask = np.zeros((64, 256), np.uint8)
    mask[30:35, 20:230] = 255
    segments = fast_pipeline._fence_segments(mask, 1.0)
    assert len(segments) == 1
    start, end = segments[0]
    assert abs(start[1] - end[1]) <= 1
    assert end[0] - start[0] > 180

    enclosure = np.zeros((240, 240), np.uint8)
    cv2.rectangle(enclosure, (30, 30), (210, 210), 255, 5)
    accepted = fast_pipeline._filter_fence_components(enclosure, .12)
    outline = fast_pipeline._fence_segments(accepted, 1.0)
    assert len(outline) == 4  # One centerline loop, not paired outer contours.


def test_windows_trace_long_glazing_runs_and_cornered_panes():
    image = np.zeros((512, 1024, 3), np.uint8)
    labels = np.zeros(image.shape[:2], np.uint8)
    confidence = np.full(labels.shape, .99, np.float32)

    # A continuous 350px glazing run and a connected L-shaped corner run.
    labels[90:101, 30:80] = 1
    labels[90:101, 80:430] = 5
    labels[90:101, 430:480] = 1
    labels[300:311, 30:60] = 1
    labels[160:200, 190:201] = 1
    labels[300:311, 60:201] = 5
    labels[200:311, 190:201] = 5

    # A separate multi-pane strip with wall mullions and partial panes.
    labels[400:411, 30:60] = 1
    labels[400:411, 60:430] = 5
    labels[400:411, 190:195] = 1
    labels[400:411, 260:265] = 1
    labels[400:411, 330:335] = 1
    labels[400:411, 430:480] = 1

    # This plausible door has no wall support and must remain rejected.
    labels[450:459, 600:620] = 2

    settings = replace(FastSettings(), ml_profile="windows", lights=False, min_wall_area=0)
    result = detect_map(image, settings, prediction=(labels, confidence))
    assert len(result.windows) >= 8
    assert len(result.window_faces) == len(result.windows)
    assert all(len(faces) == 2 for faces in result.window_faces)
    assert all(10 <= np.linalg.norm(np.asarray(end) - start) <= 165
               for start, end in result.windows)
    assert not result.doors  # Window tracing does not relax the door validator.


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("frame_gaps", [False, True])
def test_deep_window_assembly_uses_actual_extent_and_requires_two_axial_jambs(transpose, frame_gaps):
    from benchmarks.check_geometry import ray_blocked

    labels = np.zeros((512, 512), np.uint8)
    # A deep glazed bay is shorter and thicker than a long glazing strip.
    # Its thinned skeleton cannot establish the full opening extent.
    labels[180:245, 100:180] = 1
    labels[180:245, 180:290] = 5
    labels[180:245, 290:390] = 1
    if frame_gaps:
        # Perspective trims the edge; narrow internal frame/reflection gaps do
        # not make the whole glazed assembly opaque or shorten its opening.
        cv2.fillConvexPoly(labels, np.array([[180, 180], [195, 180], [180, 235]]), 0)
        labels[190:235, 210:225] = 0
        labels[190:235, 250:265] = 0
    # A filled rectangular prop beside a parallel wall has no axial jambs.
    labels[320:350, 100:170] = 5
    labels[355:365, 100:170] = 1
    # A sparse outline is not a filled glazing prediction.
    cv2.rectangle(labels, (340, 70), (400, 110), 5, 2)
    if transpose:
        labels = labels.T.copy()
    original = labels.copy()
    settings = replace(FastSettings(), ml_profile="windows", lights=False, min_wall_area=0)
    result = detect_map(np.zeros((*labels.shape, 3), np.uint8), settings,
                        prediction=(labels, np.full(labels.shape, .99, np.float32)))
    assert len(result.windows) == 1
    start, end = result.windows[0]
    assert np.linalg.norm(np.asarray(end) - start) >= 109
    ray = [(235, 150), (235, 275)]
    if transpose:
        ray = [(y, x) for x, y in ray]
    assert not ray_blocked(*ray, result.walls)
    assert ray_blocked(*ray, result.walls + result.window_faces[0])
    np.testing.assert_array_equal(labels, original)


def test_cancel_interrupts_thinning_inside_a_single_large_window_component():
    mask = np.zeros((512, 512), np.uint8)
    mask[80:430, 80:430] = 255
    original = mask.copy()
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) >= 2

    with pytest.raises(fast_pipeline.DetectionCancelled):
        fast_pipeline._detect_windows(np.zeros_like(mask), mask, .25, cancelled)
    np.testing.assert_array_equal(mask, original)


def test_window_faces_reach_real_jambs_across_unclassified_frame_gaps():
    from benchmarks.check_geometry import ray_blocked

    labels = np.zeros((512, 512), np.uint8)
    labels[200:261, 40:141] = 1
    labels[200:261, 260:400] = 1
    labels[210:251, 160:241] = 5
    result = detect_map(np.zeros((*labels.shape, 3), np.uint8),
                        replace(FastSettings(), ml_profile="windows", lights=False, min_wall_area=0),
                        prediction=(labels, np.full(labels.shape, .99, np.float32)))
    assert len(result.windows) == 1
    for x in (145, 200, 255):
        ray = ((x, 180), (x, 280))
        assert not ray_blocked(*ray, result.walls)
        assert ray_blocked(*ray, result.window_faces[0])


def test_long_glazing_trace_reaches_jambs_instead_of_stopping_at_the_pane():
    from benchmarks.check_geometry import ray_blocked

    labels = np.zeros((512, 1024), np.uint8)
    labels[200:211, 30:60] = 1
    labels[200:211, 64:420] = 5
    labels[200:211, 424:480] = 1
    result = detect_map(np.zeros((*labels.shape, 3), np.uint8),
                        replace(FastSettings(), ml_profile="windows", lights=False, min_wall_area=0),
                        prediction=(labels, np.full(labels.shape, .99, np.float32)))
    assert len(result.windows) >= 3
    faces = [span for group in result.window_faces for span in group]
    for x in (62, 200, 422):
        ray = ((x, 185), (x, 225))
        assert not ray_blocked(*ray, result.walls)
        assert ray_blocked(*ray, faces)


def test_partly_predicted_glazing_uses_building_axis_only_with_full_jamb_support():
    from benchmarks.check_geometry import ray_blocked

    labels = np.zeros((512, 512), np.uint8)
    labels[100:178, 170:231] = 1
    labels[300:400, 170:231] = 1
    pane = np.rint(cv2.boxPoints(((200., 240.), (40., 68.), 9.))).astype(np.int32)
    cv2.fillConvexPoly(labels, pane, 5)
    settings = replace(FastSettings(), ml_profile="windows", lights=False, min_wall_area=0)
    image = np.zeros((*labels.shape, 3), np.uint8)
    confidence = np.full(labels.shape, .99, np.float32)
    result = detect_map(image, settings, prediction=(labels, confidence))
    assert len(result.windows) == 1
    start, end = result.windows[0]
    assert abs(start[0] - end[0]) < 1e-6
    for y in (190, 240, 290):
        assert ray_blocked((150, y), (250, y), result.window_faces[0])
        assert not ray_blocked((150, y), (250, y), result.walls)

    # A single jamb does not establish an opening, regardless of box alignment.
    labels[300:400] = 0
    assert not detect_map(image, settings, prediction=(labels, confidence)).windows


def test_supported_diagonal_glazing_keeps_its_real_orientation():
    labels = np.zeros((512, 512), np.uint8)
    axis = np.array([np.cos(np.deg2rad(55)), np.sin(np.deg2rad(55))])
    center = np.array([256., 256.])
    for offset, length, cls in ((-90, 100, 1), (0, 80, 5), (90, 100, 1)):
        position = tuple(center + offset * axis)
        polygon = np.rint(cv2.boxPoints((position, (length, 20.), 55.))).astype(np.int32)
        cv2.fillConvexPoly(labels, polygon, cls)
    result = detect_map(np.zeros((*labels.shape, 3), np.uint8),
                        replace(FastSettings(), ml_profile="windows", lights=False, min_wall_area=0),
                        prediction=(labels, np.full(labels.shape, .99, np.float32)))
    assert len(result.windows) == 1
    start, end = np.asarray(result.windows[0])
    direction = (end - start) / np.linalg.norm(end - start)
    assert abs(np.dot(direction, axis)) > .999


def test_windows_profile_requires_model_or_precomputed_prediction():
    with pytest.raises(ValueError, match="requires a selected six-class"):
        detect_map(np.zeros((256, 256, 3), np.uint8), replace(FastSettings(), ml_profile="windows"))


def test_model_controls_roundtrip_and_clear(qt_app, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from src.gui.fast_setup import FastSetupDialog
    settings = replace(FastSettings(), model_path="test.safetensors", ml_device="cuda",
                       ml_wall_confidence=.94, ml_light_confidence=.95, ml_door_confidence=.96,
                       ml_profile="windows", ml_window_confidence=.97, ml_barrier_confidence=.98,
                       ml_recover_jambs=True, ml_light_center_threshold=.65)
    dialog = FastSetupDialog()
    dialog.apply_settings(settings)
    assert dialog.settings() == settings
    dialog.clear_model()
    assert not dialog.settings().model_path
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("new.safetensors", ""))
    dialog.choose_model()
    assert dialog.settings().model_path == "new.safetensors"
    result = DetectionResult(100, 100, [], [], [], transparent_barriers=[((20, 30), (80, 30))])
    dialog.view.set_result(result)
    assert dialog.view.layers["transparent"][0].isVisible()
    dialog.view.show_layer("transparent", False)
    assert not dialog.view.layers["transparent"][0].isVisible()
    dialog.close()


def test_jamb_recovery_preserves_openings_and_requires_confident_wall_anchors():
    image = np.zeros((512, 512, 3), np.uint8)
    labels = np.zeros(image.shape[:2], np.uint8)
    confidence = np.full(labels.shape, .99, np.float32)
    cv2.rectangle(labels, (40, 100), (98, 116), 1, -1)
    cv2.rectangle(labels, (132, 100), (190, 116), 1, -1)
    cv2.rectangle(labels, (100, 102), (130, 114), 2, -1)
    confidence[100:117, 92:99] = .6
    confidence[100:117, 132:139] = .6
    labels[108, 101] = 1  # unanchored weak speck within the doorway
    confidence[108, 101] = .6
    labels[108, 99] = 4  # never turn a transparent class into an opaque jamb
    labels[180:190, 50:180] = 1  # weak wall elsewhere must not grow
    confidence[180:190, 50:180] = .6
    original_labels, original_confidence = labels.copy(), confidence.copy()
    settings = replace(FastSettings(), lights=False)
    strict = detect_map(image, settings, prediction=(labels, confidence))
    assert not strict.doors
    settings = replace(settings, ml_recover_jambs=True)
    masks = fast_pipeline._prediction_masks((labels, confidence), labels.shape, settings)
    assert masks[0][108, 96] and masks[0][108, 135]
    assert not masks[0][108, 99] and not masks[0][108, 101]
    assert not masks[0][180:190].any()
    recovered = detect_map(image, settings, prediction=(labels, confidence))
    assert len(recovered.doors) == 1
    from benchmarks.check_geometry import ray_blocked
    assert not ray_blocked((115, 80), (115, 135), recovered.walls)
    assert ray_blocked((115, 80), (115, 135), recovered.walls + recovered.doors)
    np.testing.assert_array_equal(labels, original_labels)
    np.testing.assert_array_equal(confidence, original_confidence)
    # Without either confident anchor, a plausible rectangle is insufficient.
    confidence[100:117, 40:99] = .6
    unsupported = detect_map(image, settings, prediction=(labels, confidence))
    assert not unsupported.doors
    no_doors = replace(settings, doors=False)
    masks = fast_pipeline._prediction_masks((labels, original_confidence), labels.shape, no_doors)
    assert not masks[0][108, 96]
    # Enabling windows must not recover a disabled door's weak wall anchors.
    windows_without_doors = replace(no_doors, ml_profile="windows")
    masks = fast_pipeline._prediction_masks((labels, original_confidence), labels.shape, windows_without_doors)
    assert not masks[0][108, 96]
