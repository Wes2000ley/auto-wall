from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

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


@pytest.mark.parametrize('invisible', [False, True])
def test_fast_setup_imports_into_existing_editor(qt_app, monkeypatch, tmp_path, invisible):
    import base64
    import json
    import src.gui.app as app_module
    from src.gui.fast_setup import FastSetupDialog
    from tests.test_fast_setup import make_image, wait_until
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(app_module, "check_for_updates", lambda app: None)
    app = app_module.WallDetectionApp(version="test-dev-fast")
    dialog = FastSetupDialog(app, image=make_image(), source_path=tmp_path / 'editor.png')
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    dialog.add_shape("walls", (100., 150.), (200., 250.))
    dialog.add_shape("doors", (140., 150.), (160., 150.))
    window_center = ((300., 100.), (340., 100.))
    window_faces = [[((300., 97.), (340., 97.)), ((300., 103.), (340., 103.))]]
    dialog.apply_edit(windows=[window_center], window_faces=window_faces)
    if invisible:
        dialog.add_shape('invisible_walls', (300., 150.), (400., 250.))
    assert dialog.undo_states
    expected = deepcopy(dialog.result)
    dialog.use_in_editor()
    assert app.uvtt_preview_active
    assert len(app.uvtt_walls_preview["portals"]) == len(expected.doors) + 2
    preview_windows = [portal for portal in app.uvtt_walls_preview["portals"] if not portal["closed"]]
    assert len(preview_windows) == 2
    assert all(portal.get("_auto_wall_window") for portal in preview_windows)
    assert len(app.uvtt_walls_preview["lights"]) == len(expected.lights)
    assert app.uvtt_walls_preview["_preview_pixels"] == [
        [{"x": float(x), "y": float(y)} for x, y in segment] for segment in expected.walls]
    assert len(app.wall_edit_history) == 2
    path = tmp_path / "editor.dd2vtt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: pytest.fail('Export must not open a save dialog'))
    app.export_panel.save_uvtt_preview()
    data = json.loads(path.read_text())
    assert "_preview_pixels" not in data
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), np.uint8), 1)
    from src.core.fast_pipeline import export_image_bytes
    assert pixels.shape == app.original_image.shape
    assert base64.b64decode(data['image']) == export_image_bytes(app.original_image)
    assert (tmp_path / 'editor.foundry.webp').read_bytes() == base64.b64decode(data['image'])
    assert data["portals"] and data["lights"]
    assert data["portals"] == [
        {key: value for key, value in portal.items() if not key.startswith("_")}
        for portal in app.uvtt_walls_preview["portals"]]
    assert all("_auto_wall_window" not in portal for portal in data["portals"])
    scene = json.loads((tmp_path / 'editor.foundry.json').read_text())
    foundry_windows = [wall for wall in scene["walls"] if wall["sight"] == 30]
    assert len(foundry_windows) == 2
    assert all(wall["door"] == 0 and wall["move"] == 20 and wall["light"] == 30
               and wall["sound"] == 30 for wall in foundry_windows)
    if invisible:
        invisible_walls = [wall for wall in scene['walls'] if wall['sight'] == 0]
        assert len(invisible_walls) == 4 and all(wall['move'] == 20 for wall in invisible_walls)
        app.export_panel.save_wall_state_for_undo(force=True)
        app.uvtt_walls_preview['_invisible_walls'].clear()
        app.export_panel.undo_wall_edit()
        assert len(app.uvtt_walls_preview['_invisible_walls']) == 4
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
