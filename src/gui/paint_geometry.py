"""Vector brush areas and their wall/portal boundaries for the desktop editor.

Qt performs the area unions in native code. Geometry is rebuilt on release, not
for every mouse sample; image dimensions do not allocate a corresponding mask.
Instances are replaced rather than mutated so undo can retain previous states.
"""

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPainterPath, QPainterPathStroker, QPolygonF

from src.core.wall_editing import cut_door_opening, rectangle_walls, walls_in_rectangle


def brush_area(points, diameter, size, *, door=False):
    """Make continuous, clipped ink even when mouse samples are far apart."""
    if not points or not np.isfinite(diameter) or diameter <= 0:
        return QPainterPath()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Brush points must be finite XY pairs.")
    path = QPainterPath(QPointF(*points[0]))
    dot = not door and np.linalg.norm(points - points[0], axis=1).max() < 1e-6
    if door:
        # A door has one straight span; wobbling the pointer cannot add leaves.
        path.lineTo(*points[-1])
    elif dot:
        path.addEllipse(QPointF(*points[0]), diameter / 2, diameter / 2)
    else:
        closed = np.linalg.norm(points[-1] - points[0]) < 1e-6
        points = cv2.approxPolyDP(points.astype(np.float32), min(2., diameter * .04), closed).reshape(-1, 2)
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(*point)
        if closed:
            path.closeSubpath()
    if not dot:
        stroker = QPainterPathStroker()
        stroker.setWidth(diameter)
        stroker.setCapStyle(Qt.PenCapStyle.FlatCap if door else Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        path = stroker.createStroke(path)
    bounds = QPainterPath()
    bounds.addRect(QRectF(0, 0, size[0] - 1, size[1] - 1))
    path = path.intersected(bounds).simplified()
    for index in range(path.elementCount()):
        point = path.elementAt(index)
        path.setElementPositionAt(index, min(max(point.x, 0), size[0] - 1),
                                   min(max(point.y, 0), size[1] - 1))
    return path


def path_segments(path):
    """Keep hole boundaries separate; toFillPolygons would add bridge edges."""
    segments = []
    for polygon in path.toSubpathPolygons():
        points = [(point.x(), point.y()) for point in polygon]
        if points and points[-1] != points[0]:
            points.append(points[0])
        segments.extend((first, last) for first, last in zip(points, points[1:])
                        if np.linalg.norm(np.subtract(last, first)) > 1e-6)
    return segments


def _clean_area(path, diameter):
    """Remove subpixel pointer/curve detail before computing shared jambs."""
    cleaned = QPainterPath()
    cleaned.setFillRule(Qt.FillRule.OddEvenFill)
    for polygon in path.toSubpathPolygons():
        points = np.array([(p.x(), p.y()) for p in polygon], np.float32)
        points = cv2.approxPolyDP(points, min(1., diameter * .02), True).reshape(-1, 2)
        if len(points) >= 3:
            cleaned.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
            cleaned.closeSubpath()
    return cleaned.simplified()


def split_segments(segments, area):
    """Split lines exactly at area boundaries, returning (outside, inside)."""
    if area.isEmpty() or not segments:
        return list(segments), []
    edges = np.asarray(path_segments(area), dtype=float)
    if not len(edges):
        return list(segments), []
    edge_start, edge_delta = edges[:, 0], edges[:, 1] - edges[:, 0]
    edge_length = np.maximum(np.sum(edge_delta ** 2, axis=1), 1e-12)
    outside, inside = [], []
    bounds = area.boundingRect()
    for first, last in segments:
        start, end = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
        delta = end - start
        length_squared = float(delta @ delta)
        if length_squared < 1e-12:
            continue
        if not bounds.intersects(QRectF(QPointF(*start), QPointF(*end)).normalized().adjusted(-1e-6, -1e-6, 1e-6, 1e-6)):
            outside.append((tuple(start), tuple(end)))
            continue
        offset = edge_start - start
        denominator = delta[0] * edge_delta[:, 1] - delta[1] * edge_delta[:, 0]
        crossing = np.abs(denominator) > 1e-10
        safe = np.where(crossing, denominator, 1.)
        along = (offset[:, 0] * edge_delta[:, 1] - offset[:, 1] * edge_delta[:, 0]) / safe
        across = (offset[:, 0] * delta[1] - offset[:, 1] * delta[0]) / safe
        hits = crossing & (along > 0) & (along < 1) & (across >= -1e-8) & (across <= 1 + 1e-8)
        fractions = [0., 1., *along[hits]]
        # Collinear boundary overlaps also have to split at both endpoints.
        collinear = ~crossing & (np.abs(offset[:, 0] * delta[1] - offset[:, 1] * delta[0]) < 1e-8)
        fractions.extend(np.clip(((edges[collinear] - start) @ delta / length_squared).ravel(), 0, 1))
        fractions = np.unique(np.round(fractions, 12))
        for near, far in zip(fractions, fractions[1:]):
            if (far - near) ** 2 * length_squared < 1e-12:
                continue
            midpoint = start + delta * ((near + far) / 2)
            projection = np.clip(np.sum((midpoint - edge_start) * edge_delta, axis=1) / edge_length, 0, 1)
            on_edge = np.min(np.sum((edge_start + projection[:, None] * edge_delta - midpoint) ** 2, axis=1)) < 1e-12
            target = inside if on_edge or area.contains(QPointF(*midpoint)) else outside
            target.append((tuple(start + delta * near), tuple(start + delta * far)))
    return outside, inside


def _unique(segments):
    seen, values = set(), []
    for segment in segments:
        key = tuple(sorted(tuple(round(float(v), 7) for v in point) for point in segment))
        if key not in seen and key[0] != key[1]:
            seen.add(key)
            values.append(segment)
    return values


def _jambs(removed, span):
    """Connect cut faces to one portal, rather than exporting paired doors."""
    first, last = np.asarray(span, dtype=float)
    axis = last - first
    length = np.linalg.norm(axis)
    axis /= length
    additions = []
    for segment in removed:
        for point in map(np.asarray, segment):
            distance = float((point - first) @ axis)
            target = first if abs(distance) < 1e-5 else last if abs(distance - length) < 1e-5 else None
            if target is not None and np.linalg.norm(point - target) > 1e-6:
                additions.append((tuple(point), tuple(target)))
    return additions


def _split_at_points(segments, points):
    """Give an endpoint meeting the middle of a segment a shared vertex."""
    if not len(points):
        return segments
    endpoints = np.asarray(points, dtype=float).reshape(-1, 2)
    result = []
    for first, last in segments:
        start, end = np.asarray(first), np.asarray(last)
        delta = end - start
        squared = float(delta @ delta)
        if squared < 1e-12:
            continue
        fractions = (endpoints - start) @ delta / squared
        projections = start + fractions[:, None] * delta
        touching = (fractions > 1e-7) & (fractions < 1 - 1e-7) & (np.sum((projections - endpoints) ** 2, axis=1) < 1e-12)
        cuts = np.unique([0., 1., *fractions[touching]])
        result.extend((tuple(start + near * delta), tuple(start + far * delta))
                      for near, far in zip(cuts, cuts[1:]))
    return result


@dataclass(frozen=True)
class DoorPaint:
    area: QPainterPath
    span: tuple


@dataclass(frozen=True)
class PaintGeometry:
    size: tuple
    base_walls: list
    base_invisible: list
    base_doors: list
    walls: QPainterPath = field(default_factory=QPainterPath)
    invisible: QPainterPath = field(default_factory=QPainterPath)
    wall_coverage: QPainterPath = field(default_factory=QPainterPath)
    invisible_coverage: QPainterPath = field(default_factory=QPainterPath)
    doors: tuple = ()
    diameter: float = 1.

    @classmethod
    def from_result(cls, result):
        return cls((result.width, result.height), list(result.walls),
                   list(result.invisible_walls), list(result.doors))

    def paint(self, layer, points, diameter, *, replace_other=True):
        area = brush_area(points, diameter, self.size, door=layer == "doors")
        if area.isEmpty():
            return self
        if layer == "doors":
            span = (tuple(points[0]), tuple(points[-1]))
            remaining, endpoints = list(self.doors), list(span)
            axis = np.subtract(span[1], span[0], dtype=float)
            axis /= np.linalg.norm(axis)
            base_doors = []
            for existing in self.base_doors:
                offset = np.asarray(existing) - span[0]
                along = offset @ axis
                perpendicular = offset - along[:, None] * axis
                if (np.linalg.norm(perpendicular, axis=1).max() < 1e-6
                        and min(along) <= np.linalg.norm(np.subtract(span[1], span[0]))
                        and max(along) >= 0):
                    endpoints.extend(existing)
                    area = area.united(brush_area(existing, diameter, self.size, door=True))
                else:
                    base_doors.append(existing)
            # Combine touching passes before clipping walls, avoiding overlapping
            # portal leaves and a new fragment for every brush pass.
            while True:
                touching = []
                for door in remaining:
                    offset = np.asarray(door.span) - span[0]
                    perpendicular = offset - (offset @ axis)[:, None] * axis
                    if np.linalg.norm(perpendicular, axis=1).max() < 1e-5 and area.intersects(door.area):
                        touching.append(door)
                if not touching:
                    break
                for door in touching:
                    area = area.united(door.area)
                    endpoints.extend(door.span)
                    remaining.remove(door)
            fractions = (np.asarray(endpoints) - span[0]) @ axis
            span = tuple(tuple(np.asarray(span[0]) + fraction * axis) for fraction in (min(fractions), max(fractions)))
            return replace(self, base_doors=base_doors, doors=(*remaining, DoorPaint(area, span)))
        field_name, other = ("walls", "invisible") if layer == "walls" else ("invisible", "walls")
        area = _clean_area(area, diameter)
        combined = getattr(self, field_name).united(area).simplified()
        changes = {field_name: combined}
        own_coverage = "wall_coverage" if layer == "walls" else "invisible_coverage"
        other_coverage = "invisible_coverage" if layer == "walls" else "wall_coverage"
        changes[own_coverage] = getattr(self, own_coverage).united(area)
        if replace_other:
            changes[other] = getattr(self, other).subtracted(area)
            changes[other_coverage] = getattr(self, other_coverage).united(area)
        return replace(self, **changes,
                       diameter=max(self.diameter, diameter))

    def add_rectangle(self, layer, source, target):
        name = "base_walls" if layer == "walls" else "base_invisible"
        values = _unique([*getattr(self, name), *rectangle_walls(source, target)])
        return replace(self, **{name: values})

    def erase(self, first, last, visible):
        area = QPainterPath()
        area.addRect(QRectF(QPointF(*first), QPointF(*last)).normalized())
        changes = {}
        for layer, base_name, area_name in (("walls", "base_walls", "walls"),
                                             ("invisible_walls", "base_invisible", "invisible")):
            if not visible[layer]:
                continue
            base = getattr(self, base_name)
            removed = set(walls_in_rectangle(base, first, last))
            changes[base_name] = [wall for index, wall in enumerate(base) if index not in removed]
            changes[area_name] = getattr(self, area_name).subtracted(area)
        return replace(self, **changes)

    def compose(self):
        """Bake paint into ordinary result fields; keep source geometry for undo."""
        cuts = QPainterPath()
        # Existing spans protect openings when later wall strokes cross them.
        for span in self.base_doors:
            area = brush_area(span, max(2., self.diameter * 2), self.size, door=True)
            cuts = cuts.united(area)
        for door in self.doors:
            # Widen only the painted-material cut when a later wall pass is
            # thicker. Existing unpainted walls still use the actual brush area.
            protected = brush_area(door.span, max(2., self.diameter * 2), self.size, door=True)
            cuts = cuts.united(door.area).united(protected)
        explicit_cuts = QPainterPath()
        for door in self.doors:
            explicit_cuts = explicit_cuts.united(door.area)
        changes = {}
        for layer, base, area, coverage in (("walls", self.base_walls, self.walls, self.wall_coverage),
                                            ("invisible_walls", self.base_invisible, self.invisible, self.invisible_coverage)):
            for door in self.base_doors:
                base = cut_door_opening(base, door)
            outside, _ = split_segments(base, coverage)
            outside, replaced = split_segments(outside, explicit_cuts)
            jambs = []
            for door in self.doors:
                _, selected = split_segments(replaced, door.area)
                jambs.extend(_jambs(selected, door.span))
            boundary = path_segments(area.subtracted(cuts))
            boundary = _split_at_points(boundary, np.asarray(outside + jambs).reshape(-1, 2))
            changes[layer] = _unique(outside + jambs + boundary)
        portals = list(self.base_doors) + [door.span for door in self.doors]
        owners = [("base", index) for index in range(len(self.base_doors))]
        owners.extend(("paint", index) for index in range(len(self.doors)))
        unique_portals, unique_owners, seen = [], [], set()
        for portal, owner in zip(portals, owners):
            key = tuple(sorted(tuple(round(float(v), 7) for v in point) for point in portal))
            if key not in seen and key[0] != key[1]:
                seen.add(key)
                unique_portals.append(portal)
                unique_owners.append(owner)
        changes["doors"] = unique_portals
        for layer in ("walls", "invisible_walls"):
            changes[layer] = _unique(_split_at_points(changes[layer], np.asarray(unique_portals).reshape(-1, 2)))
        return changes, unique_owners

    def remove_door(self, index):
        _, owners = self.compose()
        kind, selected = owners[index]
        if kind == "paint":
            return replace(self, doors=tuple(door for index, door in enumerate(self.doors) if index != selected))
        return replace(self, base_doors=[door for index, door in enumerate(self.base_doors) if index != selected])
