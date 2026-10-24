from collections import Counter

import numpy as np
import pytest
from PyQt6.QtCore import QPointF

from benchmarks.check_geometry import ray_blocked
from src.core.fast_pipeline import DetectionResult
from src.gui.paint_geometry import PaintGeometry, brush_area, path_segments, split_segments


def empty_geometry():
    return PaintGeometry.from_result(DetectionResult(400, 300, [], [], []))


def assert_clean(segments, *, closed=False):
    keys = [tuple(sorted(tuple(round(float(v), 6) for v in p) for p in s)) for s in segments]
    assert len(keys) == len(set(keys))
    assert all(a != b for a, b in keys)
    assert np.isfinite(np.asarray(segments)).all()
    if closed:
        assert set(Counter(p for segment in keys for p in segment).values()) == {2}


def test_brush_dots_fast_motion_clipping_and_invalid_input():
    dot = brush_area([(1., 1.)], 20, (400, 300))
    assert dot.contains(QPointF(5, 5)) and not dot.contains(QPointF(15, 15))
    assert dot.boundingRect().left() == 0 and dot.boundingRect().top() == 0
    stroke = brush_area([(20., 100.), (380., 100.)], 20, (400, 300))
    assert stroke.contains(QPointF(200, 100))
    assert not stroke.contains(QPointF(200, 112))
    assert brush_area([(20., 100.)], 20, (400, 300), door=True).isEmpty()
    assert brush_area([], 20, (400, 300)).isEmpty()
    with pytest.raises(ValueError, match="finite XY"):
        brush_area([(float("nan"), 1.)], 20, (400, 300))


def test_overlapping_and_reversed_passes_have_one_outline_without_seams():
    first = empty_geometry().paint("walls", [(30., 100.), (180., 100.)], 20.)
    joined = first.paint("walls", [(140., 100.), (300., 100.)], 20.)
    result, _ = joined.compose()
    assert len(result["walls"]) < 40
    assert_clean(result["walls"], closed=True)
    assert not ray_blocked((140., 95.), (180., 105.), result["walls"])
    assert ray_blocked((200., 80.), (200., 120.), result["walls"])
    repeated = joined.paint("walls", [(300., 100.), (140., 100.)], 20.)
    repeated_result, _ = repeated.compose()
    assert len(repeated_result["walls"]) == len(result["walls"])
    assert joined.walls.subtracted(repeated.walls).isEmpty()
    assert repeated.walls.subtracted(joined.walls).isEmpty()
    assert first.walls.boundingRect().right() < joined.walls.boundingRect().right()  # immutable undo state


def test_closed_room_retains_empty_interior_and_junctions_share_one_boundary():
    geometry = empty_geometry().paint("walls", [(50., 50.), (250., 50.), (250., 250.),
                                                (50., 250.), (50., 50.)], 20.)
    assert geometry.walls.contains(QPointF(50, 100))
    assert not geometry.walls.contains(QPointF(150, 150))
    result, _ = geometry.compose()
    assert_clean(result["walls"], closed=True)
    assert not ray_blocked((100., 100.), (200., 200.), result["walls"])
    branch = geometry.paint("walls", [(150., 50.), (150., 150.)], 20.)
    result, _ = branch.compose()
    assert_clean(result["walls"], closed=True)
    assert not ray_blocked((150., 45.), (150., 60.), result["walls"])


def test_semantic_overpaint_converts_covered_walls_without_double_barriers():
    geometry = empty_geometry().paint("walls", [(40., 100.), (300., 100.)], 20.)
    changed = geometry.paint("invisible_walls", [(100., 100.), (200., 100.)], 30.)
    result, _ = changed.compose()
    assert not ray_blocked((150., 70.), (150., 130.), result["walls"])
    assert ray_blocked((150., 70.), (150., 130.), result["invisible_walls"])
    assert ray_blocked((250., 70.), (250., 130.), result["walls"])
    assert_clean(result["walls"], closed=True)
    assert_clean(result["invisible_walls"], closed=True)


def test_doors_cut_both_faces_and_share_exact_jamb_endpoints_after_repaint():
    geometry = empty_geometry().paint("walls", [(30., 100.), (350., 100.)], 20.)
    geometry = geometry.paint("doors", [(130., 100.), (170., 100.)], 30.)
    for diameter in (20., 40.):
        geometry = geometry.paint("walls", [(60., 100.), (320., 100.)], diameter)
        result, owners = geometry.compose()
        assert len(result["doors"]) == 1 and len(owners) == 1
        assert not ray_blocked((150., 50.), (150., 150.), result["walls"])
        assert ray_blocked((150., 50.), (150., 150.), result["walls"] + result["doors"])
        assert ray_blocked((200., 50.), (200., 150.), result["walls"])
        endpoints = np.asarray(result["walls"]).reshape(-1, 2)
        for door in result["doors"]:
            for endpoint in door:
                assert np.linalg.norm(endpoints - endpoint, axis=1).min() < 1e-6
        assert_clean(result["walls"])
        assert_clean(result["doors"])


def test_door_overlapping_passes_do_not_fragment_or_duplicate_portals():
    geometry = empty_geometry().paint("walls", [(30., 100.), (350., 100.)], 20.)
    geometry = geometry.paint("doors", [(130., 100.), (170., 100.)], 30.)
    geometry = geometry.paint("doors", [(155., 100.), (200., 100.)], 30.)
    geometry = geometry.paint("doors", [(200., 100.), (155., 100.)], 30.)
    result, _ = geometry.compose()
    assert len(geometry.doors) == 1 and len(result["doors"]) == 1
    assert all(sorted((first[0], last[0])) == pytest.approx([130, 200]) for first, last in result["doors"])
    restored = geometry.remove_door(0)
    assert not restored.doors and not restored.compose()[0]["doors"]
    assert ray_blocked((150., 50.), (150., 150.), restored.compose()[0]["walls"])


def test_brush_door_cuts_existing_diagonal_walls_and_preserves_other_geometry():
    source = DetectionResult(400, 300,
                             [((20., 20.), (280., 280.)), ((20., 100.), (100., 100.))], [], [])
    geometry = PaintGeometry.from_result(source).paint("doors", [(100., 100.), (150., 150.)], 20.)
    result, _ = geometry.compose()
    assert len(result["doors"]) == 1 and len(result["walls"]) == 3
    np.testing.assert_allclose(result["doors"][0], ((100., 100.), (150., 150.)), atol=1e-7)
    assert source.walls == [((20., 20.), (280., 280.)), ((20., 100.), (100., 100.))]
    empty = empty_geometry().paint("doors", [(40., 40.), (80., 40.)], 20.)
    assert empty.compose()[0]["doors"] == [((40., 40.), (80., 40.))]


def test_erase_paint_does_not_return_on_a_later_stroke_and_respects_hidden_layers():
    geometry = empty_geometry().paint("walls", [(30., 100.), (350., 100.)], 20.)
    geometry = geometry.paint("invisible_walls", [(30., 200.), (350., 200.)], 20.)
    erased = geometry.erase((140., 0.), (180., 290.), {"walls": True, "invisible_walls": False})
    assert erased.invisible == geometry.invisible
    erased = erased.paint("walls", [(250., 50.), (250., 100.)], 20.)
    result, _ = erased.compose()
    assert not ray_blocked((160., 70.), (160., 130.), result["walls"])
    assert ray_blocked((160., 170.), (160., 230.), result["invisible_walls"])
    assert ray_blocked((160., 70.), (160., 130.), geometry.compose()[0]["walls"])


def test_split_handles_collinear_edges_crossings_holes_and_reversed_segments():
    area = brush_area([(50., 100.), (150., 100.)], 20., (400, 300), door=True)
    outside, inside = split_segments([((20., 90.), (200., 90.)), ((200., 100.), (20., 100.))], area)
    assert len(outside) == 4 and len(inside) == 2
    np.testing.assert_allclose(inside[0], ((50., 90.), (150., 90.)), atol=1e-7)
    np.testing.assert_allclose(inside[1], ((150., 100.), (50., 100.)), atol=1e-7)
    assert path_segments(area)


def test_brush_joins_existing_wall_at_exact_shared_boundary_vertices():
    source = DetectionResult(400, 300, [((10., 100.), (390., 100.))], [], [])
    geometry = PaintGeometry.from_result(source).paint("walls", [(100., 95.), (200., 95.)], 20.)
    result, _ = geometry.compose()
    degrees = Counter(tuple(round(float(v), 6) for v in point) for segment in result["walls"] for point in segment)
    assert degrees[(10., 100.)] == degrees[(390., 100.)] == 1
    assert list(degrees.values()).count(3) == 2  # both clipped ends join the brush boundary
    assert list(degrees.values()).count(1) == 2
    assert not ray_blocked((130., 90.), (170., 100.), result["walls"])


def test_hidden_other_layer_is_preserved_by_paint_and_later_composition():
    source = DetectionResult(400, 300, [], [], [], invisible_walls=[((10., 100.), (390., 100.))])
    geometry = PaintGeometry.from_result(source).paint("invisible_walls", [(40., 200.), (300., 200.)], 20.)
    geometry = geometry.paint("walls", [(40., 100.), (300., 100.)], 30., replace_other=False)
    geometry = geometry.paint("walls", [(40., 200.), (300., 200.)], 30., replace_other=False)
    result, _ = geometry.compose()
    assert source.invisible_walls[0] in result["invisible_walls"]
    assert ray_blocked((200., 170.), (200., 230.), result["invisible_walls"])


def test_brushing_existing_door_merges_it_and_removal_does_not_reveal_a_duplicate():
    source = DetectionResult(400, 300, [((10., 100.), (80., 100.)), ((120., 100.), (300., 100.))],
                             [((80., 100.), (120., 100.))], [])
    geometry = PaintGeometry.from_result(source).paint("doors", [(80., 100.), (150., 100.)], 20.)
    result, _ = geometry.compose()
    assert result["doors"] == [((80., 100.), (150., 100.))]
    assert not geometry.remove_door(0).compose()[0]["doors"]


def test_perpendicular_doors_with_touching_brush_areas_stay_separate():
    geometry = empty_geometry().paint("doors", [(80., 100.), (120., 100.)], 30.)
    geometry = geometry.paint("doors", [(125., 110.), (125., 160.)], 30.)
    result, _ = geometry.compose()
    assert len(geometry.doors) == len(result["doors"]) == 2
