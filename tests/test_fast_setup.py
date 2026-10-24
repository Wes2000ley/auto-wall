from copy import deepcopy
from dataclasses import replace
import time

import cv2
import numpy as np
import pytest

from src.gui.fast_setup import FastSetupDialog
from src.core.fast_pipeline import DetectionResult, FastSettings


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


def test_window_face_candidate_editing_restore_delete_and_manual_door(qt_app):
    dialog = FastSetupDialog()
    settings = FastSettings()
    dialog.apply_settings(settings)
    center = ((40., 50.), (60., 50.))
    faces = [((40., 47.), (60., 47.)), ((40., 53.), (60., 53.))]
    result = DetectionResult(100, 100, walls=[], doors=[], lights=[], timings={"total": 0.0},
                             working_size=(100, 100), windows=[center], window_faces=[faces])
    dialog.result = result
    dialog.result_settings = settings
    dialog.view.set_result(result)
    assert len(dialog.view.layers["windows"]) == 2
    assert [item.data(1) for item in dialog.view.layers["windows"]] == [0, 0]

    # Removing a false-positive window turns its paired boundaries back into
    # ordinary walls, while undo restores the original candidate and geometry.
    dialog.remove_candidate("windows", 0)
    assert not result.windows and not result.window_faces and result.walls == faces
    dialog.undo_removal()
    assert result.windows == [center] and result.window_faces == [faces] and not result.walls

    # A partial manual door cuts only its own interval. Remaining glazing must
    # still pass sight, block movement and export as windows rather than walls.
    dialog.add_shape("doors", (45., 50.), (55., 50.))
    assert len(result.doors) == 1 and not result.walls
    assert result.windows == [((40., 50.), (45., 50.)), ((55., 50.), (60., 50.))]
    assert result.window_faces == [
        [((40., 47.), (45., 47.)), ((40., 53.), (45., 53.))],
        [((55., 47.), (60., 47.)), ((55., 53.), (60., 53.))]]
    from benchmarks.check_geometry import ray_blocked
    assert not ray_blocked((42., 40.), (42., 60.), result.walls + result.doors)
    assert ray_blocked((42., 40.), (42., 60.), result.windows)
    from src.core.fast_pipeline import to_uvtt
    assert len([portal for portal in to_uvtt(result)["portals"] if not portal["closed"]]) == 4
    dialog.undo_removal()
    assert not result.doors and result.windows == [center] and result.window_faces == [faces]

    # Rectangle deletion removes a candidate when either displayed face is hit.
    dialog.delete_walls_in_rectangle((45., 46.), (55., 48.))
    assert not result.windows and not result.window_faces
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

    source_path = tmp_path / "map.name.雪.bmp"
    success, encoded = cv2.imencode('.bmp', make_image())
    assert success
    source_path.write_bytes(encoded.tobytes())
    dialog = FastSetupDialog(path=str(source_path))
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    path = source_path.with_suffix('.dd2vtt')
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: pytest.fail("Export must not open a save-location dialog"))
    from src.core.fast_pipeline import export_image_bytes
    encoding_calls = []
    def encode_once(image):
        encoding_calls.append(True)
        return export_image_bytes(image)
    monkeypatch.setattr('src.gui.fast_setup.export_image_bytes', encode_once)
    assert dialog.source_path == source_path and dialog.path is None
    dialog.export_map()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    data = json.loads(path.read_text())
    assert len(data["portals"]) == 1 and len(data["lights"]) == 1
    image = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), np.uint8), 1)
    webp = source_path.with_suffix('.foundry.webp').read_bytes()
    assert webp == export_image_bytes(dialog.image) == base64.b64decode(data['image'])
    assert image.shape == dialog.image.shape
    assert source_path.read_bytes() == encoded.tobytes()
    scene_path = source_path.with_suffix('.foundry.json')
    assert scene_path.exists()
    scene = json.loads(scene_path.read_text())
    assert len(scene['lights']) == 1 and len([wall for wall in scene['walls'] if wall['door'] == 1]) == 1
    assert scene['levels'][0]['background']['src'] == 'data:image/webp;base64,' + data['image']
    # A second export replaces the same file, including edits, without prompting.
    dialog.remove_candidate('lights', 0)
    dialog.export_map()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    assert json.loads(path.read_text())['lights'] == []
    assert json.loads(scene_path.read_text())['lights'] == []
    assert len(encoding_calls) == 1
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


def test_opening_another_map_changes_automatic_export_destination(qt_app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    success, encoded = cv2.imencode('.bmp', make_image())
    assert success
    first, second = tmp_path / 'first.bmp', tmp_path / 'second.bmp'
    for path in (first, second):
        path.write_bytes(encoded.tobytes())
    dialog = FastSetupDialog(path=str(first))
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    dialog.run_detection()
    wait_until(qt_app, lambda: dialog.worker is None)
    assert dialog.source_path == first  # Redetecting does not discard the filename.
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: pytest.fail('No export popup'))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(second), ''))
    dialog.open_map()
    wait_until(qt_app, lambda: dialog.worker is None)
    assert dialog.source_path == second
    assert dialog.export_image_cache is None
    dialog.export_map()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    assert second.with_suffix('.dd2vtt').exists() and not first.with_suffix('.dd2vtt').exists()
    dialog.close()


def test_export_errors_are_visible_and_never_trigger_a_save_dialog(qt_app, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: pytest.fail('No export popup'))
    dialog = FastSetupDialog(image=make_image())
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    dialog.export_map()
    assert 'Open a local map image first' in dialog.status.text()
    def fail(*args):
        raise PermissionError('folder is read-only')
    monkeypatch.setattr('src.gui.fast_setup.write_map_exports', fail)
    dialog.source_path = tmp_path / 'map.png'
    dialog.export_map()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    assert 'Export failed: folder is read-only' in dialog.status.text()
    assert dialog.export_button.isEnabled() and dialog.view.editing_enabled
    assert list(tmp_path.iterdir()) == []
    dialog.close()
