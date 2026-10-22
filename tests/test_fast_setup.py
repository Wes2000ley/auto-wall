from copy import deepcopy
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
    cv2.rectangle(image, (30, 30), (480, 350), (131, 131, 131), 7)
    cv2.rectangle(image, (250, 26), (263, 34), (65, 180, 255), -1)
    cv2.circle(image, (150, 180), 1, (0, 255, 255), -1)
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
