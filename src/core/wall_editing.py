"""Geometry operations for manually correcting detected wall segments."""

import numpy as np


VERTEX_TOLERANCE = 1e-6


def rectangle_walls(first, second):
    """Return four axis-aligned perimeter walls, or nothing for a flat box."""
    left, right = sorted((first[0], second[0]))
    top, bottom = sorted((first[1], second[1]))
    if right - left <= VERTEX_TOLERANCE or bottom - top <= VERTEX_TOLERANCE:
        return []
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return list(zip(corners, corners[1:] + corners[:1]))


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
