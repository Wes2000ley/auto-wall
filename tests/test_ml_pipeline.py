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
                                    dict(ml_light_center_threshold=.01), dict(ml_light_center_threshold=float("nan"))])
def test_invalid_ml_settings(change):
    with pytest.raises(ValueError):
        replace(FastSettings(), **change).validate()


def test_model_controls_roundtrip_and_clear(qt_app, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from src.gui.fast_setup import FastSetupDialog
    settings = replace(FastSettings(), model_path="test.safetensors", ml_device="cuda",
                       ml_wall_confidence=.94, ml_light_confidence=.95, ml_door_confidence=.96,
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
