from copy import deepcopy
from dataclasses import replace
import time

import cv2
import numpy as np

from src.gui.fast_setup import FastSetupDialog
from src.core.fast_pipeline import FastSettings


def wait_until(app, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.002)
    raise AssertionError("Timed out waiting for the Qt worker")


def make_image():
    image = np.zeros((384, 512, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (480, 350), (160, 160, 160), 7)
    cv2.rectangle(image, (250, 26), (263, 34), (144, 32, 112), -1)
    cv2.circle(image, (150, 180), 2, (0, 255, 255), -1)
    return image


def test_worker_result_layers_removal_and_undo(qt_app):
    dialog = FastSetupDialog(image=make_image())
    dialog.apply_settings(FastSettings())
    assert dialog.settings() == FastSettings()
    dialog.show()
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    assert dialog.export_button.isEnabled()
    assert len(dialog.result.doors) == 1
    original = deepcopy(dialog.result.doors)
    dialog.remove_candidate("doors", 0)
    assert not dialog.result.doors
    dialog.undo_removal()
    assert dialog.result.doors == original
    dialog.view.show_layer("doors", False)
    assert not dialog.view.layers["doors"][0].isVisible()
    dialog.close()


def test_latest_request_wins(qt_app):
    dialog = FastSetupDialog(image=make_image())
    dialog.run_detection()
    dialog.light_range.setValue(12)
    dialog.run_detection()
    dialog.light_range.setValue(15)
    dialog.run_detection()
    wait_until(qt_app, lambda: dialog.worker is None and dialog.result_settings is not None
               and dialog.result_settings.light_range == 15 and not dialog.timer.isActive())
    assert dialog.result.lights[0]["range"] == 15
    dialog.close()


def test_closing_cancels_worker_without_destroying_running_thread(qt_app):
    dialog = FastSetupDialog(image=make_image())
    dialog.show()
    dialog.run_detection()
    dialog.close()
    wait_until(qt_app, lambda: dialog.worker is None)
    assert dialog.closing
    assert not dialog.isVisible()


def test_export_roundtrip(qt_app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog
    import base64
    import json

    dialog = FastSetupDialog(image=make_image())
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    path = tmp_path / "map.dd2vtt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    dialog.export_map()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    data = json.loads(path.read_text())
    assert len(data["portals"]) == 1 and len(data["lights"]) == 1
    image = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), np.uint8), 1)
    np.testing.assert_array_equal(image, dialog.image)
    dialog.close()


def test_profile_roundtrip_and_preset_selection(qt_app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog
    from src.core.detection_profiles import load_settings, preset_settings, save_settings

    settings = replace(FastSettings(), wall_tolerance=160, grid_size=123.125, light_mode="cool",
                       wall_geometry="centerline", min_wall_area=7000, light_core_fraction=0.08,
                       door_match="rgb", door_color_tolerance=9, door_min_length=0.5)
    dialog = FastSetupDialog()
    dialog.apply_settings(settings)
    assert dialog.settings() == settings  # controls must not silently clamp imports
    path = tmp_path / "settings.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    dialog.save_current_settings()
    assert load_settings(path) == settings
    dialog.preset.setCurrentIndex(dialog.preset.findData("demon"))
    assert dialog.preset.currentData() == "demon"
    assert dialog.settings() == replace(preset_settings("demon"), grid_size=settings.grid_size)
    save_settings(path, settings)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    dialog.import_settings()
    assert dialog.settings() == settings
    path.write_text('{"grid_size": "wrong type"}')
    dialog.import_settings()
    assert "Could not import" in dialog.status.text() and dialog.settings() == settings
    dialog.close()
