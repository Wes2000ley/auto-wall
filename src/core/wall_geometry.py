"""Convert a structural mask to connected vector geometry, independent of color."""

import cv2
import numpy as np


def outline_segments(mask, tolerance, minimum_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    walls = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        points = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
        walls.extend((start.astype(float), end.astype(float))
                     for start, end in zip(points, np.roll(points, -1, axis=0)) if np.any(start != end))
    return walls


def centerline_segments(mask, tolerance, cancelled=None):
    """Trace each skeleton edge once; keep junctions shared and closed loops closed."""
    try:
        from skimage.morphology import skeletonize
    except ImportError as error:
        raise ValueError("Centerlines need scikit-image. Install requirements-centerline.txt.") from error

    skeleton = skeletonize(mask != 0)
    pixels = set(zip(*np.nonzero(skeleton)))
    neighbors = {}
    for index, (row, column) in enumerate(pixels):
        if index % 1024 == 0 and cancelled and cancelled():
            from src.core.fast_pipeline import DetectionCancelled
            raise DetectionCancelled()
        adjacent = []
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            candidate = (row + dy, column + dx)
            if candidate not in pixels:
                continue
            # Avoid triangular shortcuts at orthogonal corners and junctions.
            if dx and dy and ((row + dy, column) in pixels or (row, column + dx) in pixels):
                continue
            adjacent.append(candidate)
        neighbors[row, column] = adjacent
    visited = set()
    walls = []

    def trace(start, next_pixel):
        path = [start]
        previous, current = start, next_pixel
        visited.add(tuple(sorted((previous, current))))
        while True:
            path.append(current)
            if current == start or len(neighbors[current]) != 2:
                break
            following = next(point for point in neighbors[current] if point != previous)
            edge = tuple(sorted((current, following)))
            if edge in visited:
                break
            visited.add(edge)
            previous, current = current, following
        closed = path[-1] == path[0]
        points = np.asarray(path[:-1] if closed else path, dtype=np.float32)[:, ::-1]
        points = cv2.approxPolyDP(points, tolerance, closed).reshape(-1, 2)
        ends = np.roll(points, -1, axis=0) if closed else points[1:]
        starts = points if closed else points[:-1]
        walls.extend((a.astype(float), b.astype(float)) for a, b in zip(starts, ends) if np.any(a != b))

    # Branches first, then any remaining closed cycles without branch pixels.
    ordered = sorted(pixels, key=lambda point: (len(neighbors[point]) == 2, point))
    for index, point in enumerate(ordered):
        if index % 1024 == 0 and cancelled and cancelled():
            from src.core.fast_pipeline import DetectionCancelled
            raise DetectionCancelled()
        for adjacent in neighbors[point]:
            if tuple(sorted((point, adjacent))) not in visited:
                trace(point, adjacent)
    return walls


def attach_centerline_doors(walls, doors, reference_scale, wall_mask):
    """Join a portal to nearby centerlines with short, wall-supported jambs.

    Corner doors can meet the middle of a continuous wall. A small connector
    preserves that junction without dragging walls or slanting the door itself.
    No geometry is added unless both connectors stay within the masonry mask.
    """
    if len(walls) == 0:
        return walls, [], list(range(len(doors)))
    segments = np.asarray(walls, dtype=float)
    starts = segments[:, 0]
    vectors = segments[:, 1] - starts
    squared = np.sum(vectors * vectors, axis=1)
    squared = np.maximum(squared, 1e-12)
    # Permit the same two reference pixels of antialiasing used by the door
    # footprint. Without this, a single shaded jamb pixel rejects a valid door.
    radius = max(1, int(np.ceil(2 * reference_scale)))
    support = cv2.dilate(wall_mask, np.ones((radius * 2 + 1,) * 2, np.uint8))
    height, width = wall_mask.shape
    attached, rejected, additions = [], [], []
    for index, door in enumerate(doors):
        door = np.asarray(door, dtype=float)
        length = np.linalg.norm(door[1] - door[0])
        reach = max(4 * reference_scale, length * 0.65)
        connectors = []
        for endpoint in door:
            fraction = np.clip(np.sum((endpoint - starts) * vectors, axis=1) / squared, 0, 1)
            projections = starts + fraction[:, None] * vectors
            distances = np.linalg.norm(projections - endpoint, axis=1)
            candidates = np.flatnonzero(distances <= reach)
            for nearest in candidates[np.argsort(distances[candidates])]:
                target = projections[nearest]
                samples = np.rint(np.linspace(endpoint, target, max(2, int(distances[nearest]) + 2))).astype(int)
                if (np.any(samples < 0) or np.any(samples[:, 0] >= width)
                        or np.any(samples[:, 1] >= height) or not np.all(support[samples[:, 1], samples[:, 0]])):
                    continue
                connectors.append((endpoint, target))
                break
            else:
                break
        if len(connectors) != 2:
            rejected.append(index)
            continue
        additions.extend((a, b) for a, b in connectors if np.linalg.norm(b - a) > 1e-6)
        attached.append(door)
    return list(segments) + additions, attached, rejected
