from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np

from src.gui.export_panel import ExportPanel
from src.core.contour_processor import ContourProcessor


def test_undo_includes_doors_lights_and_batch_edits():
    app = SimpleNamespace(uvtt_preview_active=True, uvtt_walls_preview={
        "line_of_sight": [[{"x": 0, "y": 0}, {"x": 1, "y": 1}]],
        "_preview_pixels": [[{"x": 0, "y": 0}, {"x": 70, "y": 70}]],
        "portals": [{"closed": True}], "lights": [{"position": {"x": 1, "y": 1}}]},
        wall_edit_history=[], setStatusTip=lambda text: None)
    panel = ExportPanel(app)
    panel.display_uvtt_preview = lambda: None
    initial = deepcopy(app.uvtt_walls_preview)
    panel.save_wall_state_for_undo(force=True)
    panel.save_wall_state_for_undo(force=True)
    app.uvtt_walls_preview["lights"].clear()
    panel.save_wall_state_for_undo()
    assert len(app.wall_edit_history) == 3  # light-only changes count
    panel.undo_wall_edit()
    assert app.uvtt_walls_preview == initial
    # Undoing at the baseline must not disable subsequent before-only edits.
    panel.undo_wall_edit()
    panel.undo_wall_edit()
    panel.save_wall_state_for_undo()
    app.uvtt_walls_preview["line_of_sight"].clear()
    panel.undo_wall_edit()
    assert app.uvtt_walls_preview == initial
    # A batch operation saves only its prior state.
    panel.save_wall_state_for_undo(force=True)
    app.uvtt_walls_preview["portals"].clear()
    panel.undo_wall_edit()
    assert app.uvtt_walls_preview == initial


def test_redrawing_never_rescales_shared_light_positions():
    image = np.zeros((80, 100, 3), np.uint8)
    app = SimpleNamespace(current_image=image, original_image=image, scale_factor=0.25,
                          current_contours=[], current_lights=[{
                              "position": {"x": 1.0, "y": 0.5}, "range": 1.0,
                              "intensity": 1.0, "color": "ffffffff"}],
                          enable_light_detection=SimpleNamespace(isChecked=lambda: True),
                          image_processor=SimpleNamespace(_get_display_base_image=lambda fallback: image),
                          refresh_display=lambda: None)
    before = deepcopy(app.current_lights)
    processor = ContourProcessor(app)
    processor.update_display_from_contours()
    processor.update_display_from_contours()
    assert app.current_lights == before


def test_fast_setup_imports_into_existing_editor(qt_app, monkeypatch, tmp_path):
    import base64
    import json
    import src.gui.app as app_module
    from src.gui.fast_setup import FastSetupDialog
    from tests.test_fast_setup import make_image, wait_until
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(app_module, "check_for_updates", lambda app: None)
    app = app_module.WallDetectionApp(version="test-dev-fast")
    dialog = FastSetupDialog(app, image=make_image())
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    expected = deepcopy(dialog.result)
    dialog.use_in_editor()
    assert app.uvtt_preview_active
    assert len(app.uvtt_walls_preview["portals"]) == len(expected.doors)
    assert len(app.uvtt_walls_preview["lights"]) == len(expected.lights)
    assert len(app.wall_edit_history) == 2
    path = tmp_path / "editor.uvtt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    app.export_panel.save_uvtt_preview()
    data = json.loads(path.read_text())
    assert "_preview_pixels" not in data
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), np.uint8), 1)
    np.testing.assert_array_equal(pixels, app.original_image)
    assert data["portals"] and data["lights"]
    app.image_processor.debounced_update.timer.stop()
    app.close()


def test_detection_cache_separates_maps_with_same_border(qt_app, monkeypatch):
    import src.gui.app as app_module
    import src.core.image_processor as processor_module

    monkeypatch.setattr(app_module, "check_for_updates", lambda app: None)
    app = app_module.WallDetectionApp(version="test-dev-fast")
    app.refresh_display = lambda: None
    app.merge_contours.setChecked(False)
    calls = []
    real_detect = processor_module.detect_walls
    def record(image, **kwargs):
        calls.append(kwargs["min_contour_area"])
        return real_detect(image, **kwargs)
    monkeypatch.setattr(processor_module, "detect_walls", record)
    first = np.zeros((200, 200, 3), dtype=np.uint8)
    second = first.copy()
    cv2.rectangle(first, (40, 40), (80, 80), (255, 255, 255), 3)
    cv2.rectangle(second, (100, 100), (180, 180), (255, 255, 255), 3)
    assert first.tobytes()[:1000] == second.tobytes()[:1000]
    app.current_image = first
    app.original_image = first
    app.image_processor._update_image_internal()
    app.image_processor._update_image_internal()
    assert len(calls) == 1
    app.current_image = second
    app.original_image = second
    app.image_processor._update_image_internal()
    assert len(calls) == 2
    app.image_processor.debounced_update.timer.stop()
    app.close()
