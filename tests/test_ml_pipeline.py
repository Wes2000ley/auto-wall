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


@pytest.mark.parametrize("change", [dict(model_path=42), dict(ml_device="magic"),
                                    dict(ml_wall_confidence=.49), dict(ml_door_confidence=float("nan")),
                                    dict(ml_light_confidence=1.1)])
def test_invalid_ml_settings(change):
    with pytest.raises(ValueError):
        replace(FastSettings(), **change).validate()


def test_model_controls_roundtrip_and_clear(qt_app, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from src.gui.fast_setup import FastSetupDialog
    settings = replace(FastSettings(), model_path="test.safetensors", ml_device="cuda",
                       ml_wall_confidence=.94, ml_light_confidence=.95, ml_door_confidence=.96)
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
