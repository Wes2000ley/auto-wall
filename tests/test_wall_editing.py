from copy import deepcopy
import json

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QFileDialog

from src.core.fast_pipeline import DetectionResult
from src.core.wall_editing import cut_door_opening, incident_segments, move_vertex, nearest_wall, walls_in_rectangle
from src.gui.fast_setup import FastSetupDialog
from tests.test_fast_setup import wait_until


def test_nearest_wall_snaps_to_vertices_or_segment_interiors_with_a_bounded_radius():
    walls = [((0, 0), (100, 0)), ((200, 20), (200, 100))]
    assert nearest_wall(walls, (97, 4), 5) == (0, (100., 0.))
    assert nearest_wall(walls, (45, 4), 5) == (0, (45., 0.))
    assert nearest_wall(walls, (203, 60), 5) == (1, (200., 60.))
    assert nearest_wall(walls, (206, 60), 5) is None
    assert nearest_wall([], (0, 0), 5) is None
    assert nearest_wall([((2, 3), (2, 3))], (2, 3), 5) == (0, (2., 3.))


def test_moving_a_junction_preserves_connections_and_input_and_drops_collapsed_segments():
    walls = np.array([[[0., 0.], [10., 10.]], [[10., 10.], [20., 20.]],
                      [[10.0000001, 10.], [10., 20.]], [[10.01, 10.], [0., 20.]]])
    original = walls.copy()
    np.testing.assert_array_equal(incident_segments(walls, (10, 10)), [0, 1, 2])
    moved = move_vertex(walls, (10, 10), (0, 0))
    assert moved == [((0., 0.), (20., 20.)), ((0., 0.), (10., 20.)),
                     ((10.01, 10.), (0., 20.))]
    np.testing.assert_array_equal(walls, original)
    assert move_vertex([], (0, 0), (1, 1)) == []


def test_rectangle_hits_crossings_and_boundaries_but_not_only_overlapping_bounding_boxes():
    walls = [((2, 2), (4, 4)), ((-5, 5), (15, 5)), ((5, -5), (5, 15)),
             ((0, 0), (0, 10)), ((0, 11), (10, 11)), ((8, 13), (13, 8)),
             ((9, 11), (11, 9)), ((4, 4), (4, 4))]
    expected = [0, 1, 2, 3, 6, 7]
    assert walls_in_rectangle(walls, (0, 0), (10, 10)) == expected
    assert walls_in_rectangle(walls, (10, 10), (0, 0)) == expected
    assert walls_in_rectangle([], (0, 0), (10, 10)) == []


@pytest.mark.parametrize("door", [((4., 4.), (6., 6.)), ((6., 6.), (4., 4.))])
def test_manual_door_opens_only_collinear_walls_and_preserves_unrelated_geometry(door):
    walls = [((0., 0.), (10., 10.)), ((4.5, 4.5), (5.5, 5.5)),
             ((0., 1.), (10., 11.)), ((5., 0.), (5., 10.)), ((20., 20.), (30., 30.))]
    original = deepcopy(walls)
    cut = cut_door_opening(walls, door)
    assert len(cut) == 5
    ordered = sorted(sorted(segment) for segment in cut[:2])
    np.testing.assert_allclose(ordered, [[(0., 0.), (4., 4.)], [(6., 6.), (10., 10.)]], atol=1e-12)
    np.testing.assert_allclose(cut[2:], walls[2:])
    assert walls == original
    assert cut_door_opening(walls, ((4., 4.), (4., 4.))) == original
    assert cut_door_opening([], door) == []


@pytest.fixture
def wall_dialog(qt_app, monkeypatch):
    result = DetectionResult(400, 300,
                             walls=[((40., 100.), (120., 100.)), ((120., 100.), (120., 40.)),
                                    ((220., 40.), (220., 220.)), ((50., 250.), (180., 250.))],
                             doors=[((120., 100.), (160., 100.)), ((215., 120.), (225., 120.))],
                             lights=[{"position": {"x": 220., "y": 180.}, "range": 6.,
                                      "intensity": 1., "color": "ffffffff"}],
                             timings={"total": 0.}, working_size=(400, 300),
                             transparent_barriers=[((230., 10.), (230., 280.))])
    monkeypatch.setattr("src.gui.fast_setup.detect_map", lambda *args: deepcopy(result))
    dialog = FastSetupDialog(image=np.zeros((300, 400, 3), np.uint8))
    dialog.show()
    wait_until(qt_app, lambda: dialog.result is not None and dialog.worker is None)
    dialog.view.fit_map()
    qt_app.processEvents()
    yield dialog
    dialog.close()
    wait_until(qt_app, lambda: dialog.worker is None and dialog.export_worker is None)


def drag(view, button, first, second):
    QTest.mousePress(view.viewport(), button, pos=first)
    QTest.mouseMove(view.viewport(), second)
    QTest.mouseRelease(view.viewport(), button, pos=second)


def scene_point(view, x, y):
    return view.mapFromScene(QPointF(x, y))


def test_right_drag_live_preview_shared_junction_snapping_and_mixed_undo(wall_dialog, qt_app):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result)
    start, end = scene_point(view, 120, 100), scene_point(view, 220, 220) + QPoint(4, -3)
    QTest.mousePress(view.viewport(), Qt.MouseButton.RightButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    assert view.drag_target == (220., 220.)
    assert view.drag_preview.path().elementCount() > 0
    assert dialog.result.walls == initial.walls and not dialog.undo_states
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.RightButton, pos=end)
    assert dialog.result.walls[:2] == [((40., 100.), (220., 220.)), ((220., 220.), (120., 40.))]
    assert dialog.result.doors[0] == ((220., 220.), (160., 100.))
    assert dialog.result.doors[1] == initial.doors[1]
    assert dialog.result.lights == initial.lights
    assert dialog.result.transparent_barriers == initial.transparent_barriers
    assert len(dialog.undo_states) == 1 and dialog.export_button.isEnabled()
    assert view.drag_preview is None and view.snap_marker is None
    dialog.remove_candidate("lights", 0)
    assert not dialog.result.lights
    dialog.undo_removal()
    assert dialog.result.lights == initial.lights and len(dialog.undo_states) == 1
    QTest.keyClick(view.viewport(), Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qt_app.processEvents()
    assert dialog.result.walls == initial.walls and dialog.result.doors == initial.doors
    assert not dialog.undo_states and not dialog.undo_button.isEnabled()


@pytest.mark.parametrize("zoom", [.5, 2.])
def test_drag_pick_and_snap_radius_stay_in_screen_pixels_at_different_zooms(wall_dialog, zoom):
    dialog, view = wall_dialog, wall_dialog.view
    view.resetTransform()
    view.scale(zoom, zoom)
    view.centerOn(160, 150)
    start = scene_point(view, 120, 100) + QPoint(0, 7)
    end = scene_point(view, 220, 160) + QPoint(7, 0)
    drag(view, Qt.MouseButton.RightButton, start, end)
    target = dialog.result.walls[0][1]
    assert target[0] == 220.
    assert abs(target[1] - 160) <= 1 / zoom
    dialog.undo_removal()
    end = scene_point(view, 220, 160) + QPoint(14, 0)
    drag(view, Qt.MouseButton.RightButton, start, end)
    assert dialog.result.walls[0][1][0] > 220. + 10 / zoom


def test_middle_drag_reverse_rectangle_deletes_crossing_walls_only_and_undoes_as_one_edit(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result)
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 240, 210), scene_point(view, 200, 60))
    assert dialog.result.walls == initial.walls[:2] + initial.walls[3:]
    assert dialog.result.doors == initial.doors and dialog.result.lights == initial.lights
    assert dialog.result.transparent_barriers == initial.transparent_barriers
    assert len(dialog.undo_states) == 1 and view.erase_box is None
    dialog.undo_removal()
    assert dialog.result.walls == initial.walls
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 270, 210), scene_point(view, 280, 220))
    assert not dialog.undo_states  # Empty selection does not consume an undo step.
    QTest.mouseClick(view.viewport(), Qt.MouseButton.MiddleButton, pos=scene_point(view, 120, 100))
    assert dialog.result.walls == initial.walls and not dialog.undo_states


@pytest.mark.parametrize("button", [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton])
def test_escape_cancels_gestures_without_changing_geometry_or_closing_dialog(wall_dialog, button):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result.walls)
    QTest.mousePress(view.viewport(), button, pos=scene_point(view, 120, 100))
    QTest.mouseMove(view.viewport(), scene_point(view, 220, 150))
    QTest.keyClick(view.viewport(), Qt.Key.Key_Escape)
    QTest.mouseRelease(view.viewport(), button, pos=scene_point(view, 220, 150))
    assert dialog.isVisible() and not dialog.closing
    assert dialog.result.walls == initial and not dialog.undo_states
    assert view.drag_preview is None and view.erase_box is None


def test_stale_settings_and_hidden_layers_prevent_edits_and_cancel_active_drag(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result.walls)
    start, end = scene_point(view, 120, 100), scene_point(view, 220, 150)
    QTest.mousePress(view.viewport(), Qt.MouseButton.RightButton, pos=start)
    assert view.drag_source is not None
    dialog.live.setChecked(False)
    dialog.grid.setValue(80)
    assert not view.editing_enabled and view.drag_source is None
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.RightButton, pos=end)
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 0, 0), scene_point(view, 300, 280))
    assert dialog.result.walls == initial and not dialog.undo_states
    dialog.apply_settings(dialog.result_settings)
    assert view.editing_enabled
    view.show_layer("walls", False)
    drag(view, Qt.MouseButton.RightButton, start, end)
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 0, 0), scene_point(view, 300, 280))
    assert dialog.result.walls == initial and not dialog.undo_states


def test_edited_geometry_is_exported_and_redetection_starts_a_fresh_history(wall_dialog, qt_app, monkeypatch, tmp_path):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result.walls)
    drag(view, Qt.MouseButton.RightButton, scene_point(view, 120, 100), scene_point(view, 220, 220))
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 45, 240), scene_point(view, 190, 260))
    path = tmp_path / "edited.dd2vtt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    dialog.export_map()
    assert not view.editing_enabled and not dialog.undo_button.isEnabled()
    dialog.undo_removal()  # Cannot alter the geometry while its export is running.
    assert len(dialog.undo_states) == 2
    wait_until(qt_app, lambda: dialog.export_worker is None)
    data = json.loads(path.read_text())
    assert len(data["line_of_sight"]) == 3
    assert data["line_of_sight"][0][1] == {"x": 220 / 70, "y": 220 / 70}
    assert data["portals"][0]["bounds"][0] == data["line_of_sight"][0][1]
    assert len(data["lights"]) == 1 and not data["objects_line_of_sight"]
    dialog.run_detection()
    assert not view.editing_enabled
    wait_until(qt_app, lambda: dialog.worker is None)
    assert dialog.result.walls == initial and not dialog.undo_states
    assert view.editing_enabled and not dialog.undo_button.isEnabled()


def test_number_keys_and_buttons_toggle_drawing_without_stealing_numeric_input(wall_dialog, qt_app):
    dialog, view = wall_dialog, wall_dialog.view
    assert view.drawing_mode is None
    QTest.keyClick(view.viewport(), Qt.Key.Key_1)
    assert view.drawing_mode == "walls" and dialog.draw_buttons["walls"].isChecked()
    QTest.keyClick(view.viewport(), Qt.Key.Key_2)
    assert view.drawing_mode == "doors" and dialog.draw_buttons["doors"].isChecked()
    assert not dialog.draw_buttons["walls"].isChecked()
    QTest.keyClick(view.viewport(), Qt.Key.Key_2)
    assert view.drawing_mode is None and not dialog.draw_buttons["doors"].isChecked()
    QTest.mouseClick(dialog.draw_buttons["walls"], Qt.MouseButton.LeftButton)
    qt_app.processEvents()
    assert view.drawing_mode == "walls" and view.hasFocus()
    QTest.mouseClick(dialog.draw_buttons["walls"], Qt.MouseButton.LeftButton)
    assert view.drawing_mode is None
    QTest.mouseClick(dialog.draw_buttons["doors"], Qt.MouseButton.LeftButton)
    QTest.keyClick(view.viewport(), Qt.Key.Key_Escape)
    assert view.drawing_mode is None and dialog.isVisible()
    dialog.live.setChecked(False)
    dialog.grid.lineEdit().setFocus()
    dialog.grid.lineEdit().selectAll()
    QTest.keyClicks(dialog.grid.lineEdit(), "123")
    QTest.keyClick(dialog.grid.lineEdit(), Qt.Key.Key_Tab)
    assert dialog.grid.value() == 123 and view.drawing_mode is None
    assert not any(button.isEnabled() for button in dialog.draw_buttons.values())


def test_wall_drawing_snaps_both_ends_and_keeps_stretch_and_erase_available(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result.walls)
    QTest.keyClick(view.viewport(), Qt.Key.Key_1)
    start = scene_point(view, 40, 100) + QPoint(3, 3)
    end = scene_point(view, 50, 250) + QPoint(4, -3)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end)
    assert view.draw_origin == (40., 100.) and view.draw_target == (50., 250.)
    assert view.drag_preview.path().elementCount() == 2
    assert dialog.result.walls == initial and not dialog.undo_states
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert dialog.result.walls == initial + [((40., 100.), (50., 250.))]
    assert len(dialog.undo_states) == 1 and view.drawing_mode == "walls"
    drag(view, Qt.MouseButton.RightButton, scene_point(view, 40, 100), scene_point(view, 220, 220))
    assert dialog.result.walls[-1] == ((220., 220.), (50., 250.))
    dialog.undo_removal()
    drag(view, Qt.MouseButton.MiddleButton, scene_point(view, 200, 60), scene_point(view, 240, 210))
    assert initial[2] not in dialog.result.walls and view.drawing_mode == "walls"
    dialog.undo_removal()
    dialog.undo_removal()
    assert dialog.result.walls == initial and not dialog.undo_states


def test_door_drawing_cuts_an_existing_wall_and_exports_an_openable_portal(wall_dialog, qt_app, monkeypatch, tmp_path):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result)
    QTest.keyClick(view.viewport(), Qt.Key.Key_2)
    drag(view, Qt.MouseButton.LeftButton, scene_point(view, 220, 80) + QPoint(3, 0),
         scene_point(view, 220, 160) + QPoint(3, 0))
    start, end = dialog.result.doors[-1]
    assert start[0] == end[0] == 220.
    assert abs(start[1] - 80) < 1 and abs(end[1] - 160) < 1
    assert len(dialog.result.walls) == len(initial.walls) + 1
    assert ((220., 40.), start) in dialog.result.walls
    assert (end, (220., 220.)) in dialog.result.walls
    assert len(dialog.undo_states) == 1 and view.drawing_mode == "doors"
    path = tmp_path / "drawn-door.dd2vtt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    dialog.export_map()
    wait_until(qt_app, lambda: dialog.export_worker is None)
    data = json.loads(path.read_text())
    assert data["portals"][-1]["closed"] is True
    assert data["portals"][-1]["bounds"] == [{"x": point[0] / 70, "y": point[1] / 70} for point in (start, end)]
    assert len(data["line_of_sight"]) == len(initial.walls) + 1
    dialog.undo_removal()
    assert dialog.result.walls == initial.walls and dialog.result.doors == initial.doors


@pytest.mark.parametrize("key", [Qt.Key.Key_1, Qt.Key.Key_2])
def test_cancel_short_or_duplicate_drawing_does_not_change_geometry(wall_dialog, key):
    dialog, view = wall_dialog, wall_dialog.view
    initial = deepcopy(dialog.result)
    QTest.keyClick(view.viewport(), key)
    point = scene_point(view, 40, 100)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    drag(view, Qt.MouseButton.LeftButton, point, point + QPoint(1, 0))
    # Both positions snap onto the same existing junction.
    drag(view, Qt.MouseButton.LeftButton, point + QPoint(-4, 0), point + QPoint(4, 0))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(view.viewport(), scene_point(view, 300, 200))
    QTest.keyClick(view.viewport(), Qt.Key.Key_Escape)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=scene_point(view, 300, 200))
    assert view.drawing_mode is not None and not view.has_wall_gesture()
    assert dialog.result.walls == initial.walls and dialog.result.doors == initial.doors
    assert not dialog.undo_states and dialog.isVisible()
    # Drawing an existing segment in reverse does not create a duplicate.
    first, second = (initial.walls if key == Qt.Key.Key_1 else initial.doors)[0]
    drag(view, Qt.MouseButton.LeftButton, scene_point(view, *second), scene_point(view, *first))
    assert not dialog.undo_states


def test_empty_map_drawing_clips_to_bounds_and_switching_modes_cancels_preview(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    dialog.apply_edit(walls=[], doors=[])
    dialog.undo_states.clear()
    QTest.keyClick(view.viewport(), Qt.Key.Key_1)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=scene_point(view, 100, 100))
    QTest.mouseMove(view.viewport(), scene_point(view, 150, 150))
    QTest.keyClick(view.viewport(), Qt.Key.Key_2)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=scene_point(view, 150, 150))
    assert not dialog.undo_states and view.drawing_mode == "doors" and view.drag_preview is None
    drag(view, Qt.MouseButton.LeftButton, scene_point(view, 100, 100), scene_point(view, 430, 330))
    assert dialog.result.doors[-1][1] == (399, 299) and not dialog.result.walls
    assert len(dialog.undo_states) == 1


def test_drawing_reveals_its_layer_and_does_not_conflict_with_picking_or_removing(wall_dialog):
    dialog, view = wall_dialog, wall_dialog.view
    dialog.layer_checks["walls"].setChecked(False)
    dialog.remove_check.setChecked(True)
    QTest.keyClick(view.viewport(), Qt.Key.Key_1)
    assert view.visibility["walls"] and dialog.layer_checks["walls"].isChecked()
    assert not view.removing and not dialog.remove_check.isChecked()
    dialog.start_pick("wall")
    assert view.picking and view.drawing_mode is None
    QTest.keyClick(view.viewport(), Qt.Key.Key_2)
    assert not view.picking and dialog.pick_target is None and view.drawing_mode == "doors"
    dialog.layer_checks["doors"].setChecked(False)
    assert view.drawing_mode is None
    QTest.keyClick(view.viewport(), Qt.Key.Key_2)
    dialog.remove_check.setChecked(True)
    assert view.drawing_mode is None and view.removing
