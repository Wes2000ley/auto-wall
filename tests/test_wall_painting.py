from copy import deepcopy
import json

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog

from benchmarks.check_geometry import ray_blocked
from tests.test_fast_setup import wait_until
from tests.test_wall_editing import drag, scene_point, wall_dialog


def wheel(view, delta, modifiers=Qt.KeyboardModifier.NoModifier):
    position = view.viewport().rect().center()
    event = QWheelEvent(QPointF(position), QPointF(view.viewport().mapToGlobal(position)),
                        QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton, modifiers,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view.viewport(), event)


def blank(dialog):
    dialog.apply_edit(walls=[], doors=[], invisible_walls=[])
    dialog.undo_states.clear()


def paint(dialog, layer, points):
    dialog.view.set_paint_enabled(True)
    dialog.view.set_drawing_mode(layer)
    first, *remaining = [scene_point(dialog.view, *point) for point in points]
    QTest.mousePress(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=first)
    for point in remaining:
        QTest.mouseMove(dialog.view.viewport(), point)
    QTest.mouseRelease(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=remaining[-1] if remaining else first)


@pytest.mark.parametrize("zoom", [.5, 2.])
def test_paint_toggle_material_shortcuts_and_wheel_width_are_independent_of_zoom(wall_dialog, qt_app, zoom):
    dialog, view = wall_dialog, wall_dialog.view
    before = dialog.settings()
    view.resetTransform()
    view.scale(zoom, zoom)
    QTest.keyClick(view, Qt.Key.Key_B)
    assert view.paint_enabled and dialog.paint_button.isChecked() and view.drawing_mode == "walls"
    QTest.keyClick(view, Qt.Key.Key_2)
    assert view.drawing_mode == "doors" and dialog.draw_buttons["doors"].isChecked()
    QTest.keyClick(view, Qt.Key.Key_3)
    assert view.drawing_mode == "invisible_walls"
    diameter = view.brush_diameter
    wheel(view, 120)
    assert view.brush_diameter == round(diameter * 1.15, 1)
    assert dialog.brush_size.value() == view.brush_diameter
    assert view.transform().m11() == zoom
    wheel(view, -120)
    assert view.brush_diameter == pytest.approx(diameter, abs=.1)
    wheel(view, 120, Qt.KeyboardModifier.ControlModifier)
    assert view.transform().m11() == pytest.approx(zoom * 1.2)
    dialog.brush_size.setValue(40)
    assert view.brush_diameter == 40 and dialog.settings() == before and not dialog.timer.isActive()
    QTest.keyClick(view, Qt.Key.Key_B)
    assert not view.paint_enabled and not dialog.paint_button.isChecked()
    assert view.drawing_mode == "invisible_walls"
    assert "rectangle" in dialog.draw_buttons["invisible_walls"].text()


def test_freehand_stroke_is_live_then_one_undo_and_fast_motion_has_no_gaps(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.set_paint_enabled(True)
    view.set_brush_diameter(20)
    start = scene_point(view, 40, 140)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), scene_point(view, 220, 140))
    QTest.mouseMove(view.viewport(), scene_point(view, 220, 220))
    assert view.has_wall_gesture() and view.paint_preview is not None
    assert not dialog.result.walls and not dialog.undo_states
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=scene_point(view, 220, 220))
    assert dialog.result.walls and len(dialog.undo_states) == 1 and not view.has_wall_gesture()
    assert dialog.paint_geometry.walls.contains(QPointF(120, 140))
    assert dialog.paint_geometry.walls.contains(QPointF(220, 180))
    assert not ray_blocked((210., 140.), (220., 150.), dialog.result.walls)  # no interior seam
    dialog.undo_removal()
    assert not dialog.result.walls and dialog.paint_geometry is None and not dialog.undo_states


@pytest.mark.parametrize("cancel", ["escape", "pan", "material", "settings"])
def test_unfinished_paint_cancels_without_committing(wall_dialog, cancel):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result)
    view.set_paint_enabled(True)
    start, end = scene_point(view, 40, 140), scene_point(view, 200, 140)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    assert view.has_wall_gesture()
    if cancel == "escape":
        QTest.keyClick(view, Qt.Key.Key_Escape)
    elif cancel == "pan":
        QTest.mousePress(view.viewport(), Qt.MouseButton.RightButton, pos=end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.RightButton, pos=end + QPoint(10, 0))
    elif cancel == "material":
        QTest.keyClick(view, Qt.Key.Key_2)
    else:
        dialog.live.setChecked(False)
        dialog.grid.setValue(80)
        assert not view.editing_enabled and not dialog.paint_button.isEnabled()
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert not view.has_wall_gesture() and view.paint_preview is None
    assert dialog.result == initial and not dialog.undo_states and dialog.paint_geometry is None


def test_repainting_merges_without_duplicate_undo_and_other_edits_preserve_paint(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.set_brush_diameter(20)
    points = [(40., 140.), (240., 140.)]
    paint(dialog, "walls", points)
    first = deepcopy(dialog.result.walls)
    paint(dialog, "walls", list(reversed(points)))
    assert len(dialog.undo_states) == 1 and len(dialog.result.walls) == len(first)
    dialog.remove_candidate("lights", 0)
    assert dialog.paint_geometry is not None and not dialog.result.lights
    view.set_paint_enabled(False)
    dialog.add_shape("walls", (300., 30.), (360., 90.))
    assert dialog.paint_geometry is not None
    paint(dialog, "walls", [(200., 140.), (300., 140.)])
    assert not ray_blocked((205., 135.), (235., 145.), dialog.result.walls)
    assert ray_blocked((320., 10.), (320., 50.), dialog.result.walls)
    dialog.undo_removal()
    dialog.undo_removal()
    dialog.undo_removal()
    assert dialog.result.walls == first and dialog.result.lights


def test_painted_doors_open_both_faces_merge_and_undo_together(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.set_brush_diameter(20)
    paint(dialog, "walls", [(30., 140.), (350., 140.)])
    before = deepcopy(dialog.result)
    view.set_brush_diameter(40)
    paint(dialog, "doors", [(130., 140.), (180., 140.)])
    assert len(dialog.result.doors) == 1 and len(dialog.undo_states) == 2
    assert not ray_blocked((155., 100.), (155., 180.), dialog.result.walls)
    assert ray_blocked((155., 100.), (155., 180.), dialog.result.walls + dialog.result.doors)
    paint(dialog, "doors", [(165., 140.), (200., 140.)])
    assert len(dialog.result.doors) == 1
    dialog.undo_removal()
    dialog.undo_removal()
    assert dialog.result == before
    paint(dialog, "doors", [(130., 140.)])
    assert dialog.result == before  # A door click is not an accidental new portal.


def test_middle_erase_hidden_layers_and_undo_keep_future_paint_consistent(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.set_brush_diameter(20)
    paint(dialog, "walls", [(30., 140.), (350., 140.)])
    paint(dialog, "invisible_walls", [(30., 220.), (350., 220.)])
    before = deepcopy(dialog.result)
    dialog.layer_checks["invisible_walls"].setChecked(False)
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 140, 80), scene_point(view, 180, 260))
    assert dialog.result.invisible_walls == before.invisible_walls
    assert not ray_blocked((160., 100.), (160., 180.), dialog.result.walls)
    paint(dialog, "walls", [(280., 100.), (280., 140.)])
    assert not ray_blocked((160., 100.), (160., 180.), dialog.result.walls)
    dialog.undo_removal()
    dialog.undo_removal()
    assert dialog.result == before


def test_paint_export_has_doors_and_invisible_geometry_beside_source_without_popup(wall_dialog, qt_app, monkeypatch, tmp_path):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.set_brush_diameter(20)
    paint(dialog, "walls", [(30., 140.), (350., 140.)])
    view.set_brush_diameter(40)
    paint(dialog, "doors", [(130., 140.), (180., 140.)])
    paint(dialog, "invisible_walls", [(40., 240.), (300., 240.)])
    source = tmp_path / "paint.png"
    source.write_bytes(b"source must remain unchanged")
    dialog.source_path = source
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: pytest.fail("No export popup"))
    expected = deepcopy(dialog.result)
    dialog.export_map()
    assert not view.editing_enabled and not dialog.paint_button.isEnabled()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    assert source.read_bytes() == b"source must remain unchanged"
    data = json.loads(source.with_suffix(".dd2vtt").read_text())
    scene = json.loads(source.with_suffix(".foundry.json").read_text())
    assert len(data["portals"]) == len(expected.doors) == 1
    assert len(data["line_of_sight"]) == len(expected.walls)
    assert len([wall for wall in scene["walls"] if wall["sight"] == 0]) == len(expected.invisible_walls)
    assert len([wall for wall in scene["walls"] if wall["door"] == 1]) == 1
    assert source.with_suffix(".foundry.webp").exists()
    dialog.run_detection()
    wait_until(qt_app, lambda: dialog.worker is None)
    assert dialog.paint_geometry is None and not dialog.undo_states


def test_low_zoom_repainting_snaps_without_growing_existing_ink(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.resetTransform()
    view.scale(.5, .5)
    view.set_brush_diameter(20)
    paint(dialog, "walls", [(40., 140.), (240., 140.)])
    area = dialog.paint_geometry.walls
    paint(dialog, "walls", [(240., 140.), (40., 140.)])
    assert dialog.paint_geometry.walls.subtracted(area).isEmpty()
    assert len(dialog.undo_states) == 1


def test_paint_wheel_during_stroke_updates_preview_and_committed_width(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    blank(dialog)
    view.set_paint_enabled(True)
    view.set_brush_diameter(20)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=scene_point(view, 40, 140))
    QTest.mouseMove(view.viewport(), scene_point(view, 240, 140))
    wheel(view, 120)
    assert view.paint_preview.pen().widthF() == view.brush_diameter == 23
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=scene_point(view, 240, 140))
    assert dialog.paint_geometry.walls.boundingRect().height() == pytest.approx(23)
    assert len(dialog.undo_states) == 1


def test_ordinary_door_tool_keeps_its_narrow_cut_after_using_the_brush(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    nearby = ((20., 110.), (300., 110.))
    dialog.apply_edit(walls=[((20., 100.), (300., 100.)), nearby], doors=[], invisible_walls=[])
    paint(dialog, "walls", [(40., 220.), (300., 220.)])
    view.set_paint_enabled(False)
    dialog.add_shape("doors", (100., 100.), (150., 100.))
    assert nearby in dialog.result.walls
    assert ((20., 100.), (100., 100.)) in dialog.result.walls
    assert ((150., 100.), (300., 100.)) in dialog.result.walls
    assert dialog.result.doors == [((100., 100.), (150., 100.))]
