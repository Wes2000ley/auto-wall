"""Geometry operations for manually correcting detected wall segments."""

import numpy as np


VERTEX_TOLERANCE = 1e-6


def incident_segments(segments, vertex):
    """Find segments sharing this exact junction, allowing roundoff only."""
    values = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    return np.flatnonzero(np.any(np.linalg.norm(values - vertex, axis=2) <= VERTEX_TOLERANCE, axis=1))


def nearest_wall(segments, point, radius):
    """Return (index, snapped XY), preferring endpoints within the given radius."""
    if not len(segments):
        return None
    values = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    distances = np.sum((values - point) ** 2, axis=2)
    closest = int(distances.argmin())
    index, endpoint = divmod(closest, 2)
    if distances[index, endpoint] <= radius ** 2:
        return index, tuple(values[index, endpoint])
    starts = values[:, 0]
    directions = values[:, 1] - starts
    lengths = np.sum(directions ** 2, axis=1)
    fractions = np.sum((point - starts) * directions, axis=1) / np.maximum(lengths, 1e-12)
    projections = starts + np.clip(fractions, 0, 1)[:, None] * directions
    distances = np.sum((projections - point) ** 2, axis=1)
    index = int(distances.argmin())
    if distances[index] <= radius ** 2:
        return index, tuple(projections[index])
    return None


def move_vertex(segments, source, target):
    """Move a shared junction together and discard segments collapsed by the edit."""
    values = np.asarray(segments, dtype=float).reshape(-1, 2, 2).copy()
    shared = np.linalg.norm(values - source, axis=2) <= VERTEX_TOLERANCE
    values[shared] = target
    keep = np.linalg.norm(values[:, 1] - values[:, 0], axis=1) > VERTEX_TOLERANCE
    return [(tuple(start), tuple(end)) for start, end in values[keep]]


def cut_door_opening(walls, door):
    """Replace only wall portions directly under a manually drawn door."""
    origin, end = np.asarray(door, dtype=float)
    direction = end - origin
    length = np.linalg.norm(direction)
    if length <= VERTEX_TOLERANCE:
        return list(walls)
    direction /= length
    result = []
    for start, end in walls:
        offsets = np.asarray((start, end), dtype=float) - origin
        distances = offsets @ direction
        perpendicular = offsets - distances[:, None] * direction
        if np.any(np.linalg.norm(perpendicular, axis=1) > VERTEX_TOLERANCE):
            result.append((start, end))
            continue
        first, last = sorted(distances)
        if last <= 0 or first >= length:
            result.append((start, end))
            continue
        if first < -VERTEX_TOLERANCE:
            result.append((tuple(origin + first * direction), tuple(origin)))
        if last > length + VERTEX_TOLERANCE:
            result.append((tuple(origin + length * direction), tuple(origin + last * direction)))
    return result


def walls_in_rectangle(segments, first, second):
    """Select whole segments touching the box, including crossings and its boundary."""
    minimum, maximum = np.minimum(first, second), np.maximum(first, second)
    selected = []
    for index, (start, end) in enumerate(segments):
        # Clip the segment's [0, 1] parameter range against both rectangle axes.
        enter, leave = 0., 1.
        for axis in (0, 1):
            direction = end[axis] - start[axis]
            if abs(direction) <= 1e-12:
                if not minimum[axis] <= start[axis] <= maximum[axis]:
                    break
            else:
                near, far = sorted(((minimum[axis] - start[axis]) / direction,
                                    (maximum[axis] - start[axis]) / direction))
                enter, leave = max(enter, near), min(leave, far)
                if enter > leave:
                    break
        else:
            selected.append(index)
    return selected
