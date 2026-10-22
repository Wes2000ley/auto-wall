"""Bounded-resolution detection with geometry kept in original-image pixels.

OpenCV performs the image kernels in native code. No GUI or model downloads are
needed, so this pipeline is shared by the desktop worker, CLI, and benchmarks.
"""

from dataclasses import dataclass, field
import base64
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


class DetectionCancelled(Exception):
    """A cooperative cancellation between native processing stages."""


@dataclass(frozen=True)
class FastSettings:
    max_dimension: int = 2048  # 0 processes the original pixels
    grid_size: float = 70.0  # original-image pixels per grid square
    wall_mode: str = "stone"  # stone, color, or edges
    wall_color: tuple = (131, 131, 131)  # RGB
    wall_tolerance: int = 24
    min_wall_area: float = 1000.0  # area at a 2048-pixel longest edge
    simplify: float = 1.25  # pixels at a 2048-pixel longest edge
    doors: bool = True
    door_color: tuple = (255, 160, 65)  # RGB; hue plus shape and wall support
    door_hue_tolerance: int = 16
    door_min_length: float = 12.0  # lengths at a 2048-pixel longest edge
    door_max_length: float = 60.0
    door_rectangularity: float = 0.78
    door_wall_support: bool = True
    lights: bool = True
    light_threshold: int = 190
    light_min_area: float = 3.0  # areas at a 2048-pixel longest edge
    light_max_area: float = 180.0
    light_saturation: int = 60
    light_range: float = 6.0  # grid squares

    def validate(self):
        if self.max_dimension != 0 and self.max_dimension < 256:
            raise ValueError("Detection resolution must be 0 (original) or at least 256.")
        numeric = (self.grid_size, self.min_wall_area, self.simplify,
                   self.door_min_length, self.door_max_length, self.door_rectangularity,
                   self.light_min_area, self.light_max_area, self.light_range)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Detection settings must be finite numbers.")
        if self.grid_size <= 0 or self.light_range <= 0:
            raise ValueError("Grid size and light range must be positive.")
        if self.wall_mode not in ("stone", "color", "edges"):
            raise ValueError("Unknown wall detection mode.")
        if self.min_wall_area < 0 or self.simplify < 0:
            raise ValueError("Wall area and simplification cannot be negative.")
        if not 0 < self.door_min_length <= self.door_max_length:
            raise ValueError("Door minimum length must be positive and no larger than the maximum.")
        if not 0 < self.door_rectangularity <= 1:
            raise ValueError("Door rectangularity must be between 0 and 1.")
        if not 0 < self.light_min_area <= self.light_max_area:
            raise ValueError("Light minimum area must be positive and no larger than the maximum.")
        for color in (self.wall_color, self.door_color):
            if len(color) != 3 or any(not 0 <= value <= 255 for value in color):
                raise ValueError("Colors must contain three RGB values between 0 and 255.")
        for value in (self.wall_tolerance, self.light_threshold, self.light_saturation):
            if not 0 <= value <= 255:
                raise ValueError("Color and brightness thresholds must be between 0 and 255.")
        if not 0 <= self.door_hue_tolerance <= 90:
            raise ValueError("Door hue tolerance must be between 0 and 90.")


@dataclass
class DetectionResult:
    width: int
    height: int
    walls: list  # pairs of (x, y), in original-image pixels
    doors: list  # pairs of (x, y), in original-image pixels
    lights: list  # position in original-image pixels, range in grid units
    timings: dict = field(default_factory=dict)
    working_size: tuple = (0, 0)
    warnings: list = field(default_factory=list)


def _stage(name, progress, cancelled):
    if cancelled and cancelled():
        raise DetectionCancelled()
    if progress:
        progress(name)


def _color_mask(image, rgb, tolerance):
    """Native channel bounds avoid three full float images and square roots."""
    bgr = np.asarray(rgb[::-1], dtype=np.int16)
    return cv2.inRange(image, np.maximum(0, bgr - tolerance).astype(np.uint8),
                       np.minimum(255, bgr + tolerance).astype(np.uint8))


def _keep_components(mask, minimum):
    """Filter every component in a single label lookup, not N image scans."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[stats[:, cv2.CC_STAT_AREA] >= minimum] = 255
    keep[0] = 0
    return keep[labels]


def _wall_mask(image, hsv, settings, reference_scale):
    if settings.wall_mode == "stone":
        # Estimate neutral masonry from the map; ignore dark voids and highlights.
        neutral = cv2.inRange(hsv, (0, 0, 65), (179, 38, 220))
        histogram = cv2.calcHist([hsv], [2], neutral, [256], [0, 256]).ravel()
        if not histogram.any():
            return np.zeros(image.shape[:2], np.uint8)
        peak = int(histogram.argmax())
        mask = cv2.inRange(hsv, (0, 0, max(0, peak - settings.wall_tolerance)),
                           (179, 38, min(255, peak + settings.wall_tolerance)))
    elif settings.wall_mode == "color":
        mask = _color_mask(image, settings.wall_color, settings.wall_tolerance)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    # Seal hairline stone seams, without joining separate wall runs or doorways.
    radius = max(1, round(reference_scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _keep_components(mask, max(1, settings.min_wall_area * reference_scale ** 2))


def _detect_doors(hsv, wall_mask, settings, reference_scale):
    rgb = np.uint8([[settings.door_color]])
    hue, saturation, value = map(int, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0])
    if saturation < 40:
        mask = cv2.inRange(hsv, (0, 0, max(0, value - 35)), (179, 55, min(255, value + 35)))
    else:
        # Circular hue distance includes reds on both sides of OpenCV's hue seam.
        delta = cv2.absdiff(hsv[:, :, 0], hue)
        mask = ((np.minimum(delta, 180 - delta) <= settings.door_hue_tolerance)
                & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 170)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doors, footprints = [], []
    height, width = mask.shape
    for contour in contours:
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        length, thickness = max(rw, rh), min(rw, rh)
        if not settings.door_min_length * reference_scale <= length <= settings.door_max_length * reference_scale:
            continue
        if thickness < max(2, reference_scale * 2) or not 1.25 <= length / thickness <= 6:
            continue
        if cv2.contourArea(contour) / (rw * rh) < settings.door_rectangularity:
            continue
        angle += 90 if rh > rw else 0
        # Remove tiny minAreaRect tilts caused by antialiasing on drawn doors.
        cardinal = round(angle / 90) * 90
        if abs(angle - cardinal) <= 3:
            angle = cardinal
        theta = math.radians(angle)
        axis = np.array([math.cos(theta), math.sin(theta)])
        center = np.array([cx, cy])
        ends = [center - axis * length / 2, center + axis * length / 2]
        if settings.door_wall_support:
            # Require masonry along the door's axis at both ends. A rectangle
            # beside a parallel wall is furniture/decor, not a supported doorway.
            reach = max(3 * reference_scale, thickness * 0.7)
            offsets = np.linspace(-reference_scale, reach, max(3, math.ceil(reach) + 2))
            supported = []
            for endpoint, sign in zip(ends, (-1, 1)):
                samples = np.rint(endpoint + sign * offsets[:, None] * axis).astype(int)
                in_bounds = ((samples[:, 0] >= 0) & (samples[:, 0] < width)
                             & (samples[:, 1] >= 0) & (samples[:, 1] < height))
                samples = samples[in_bounds]
                supported.append(bool(len(samples) and np.any(wall_mask[samples[:, 1], samples[:, 0]])))
            if not all(supported):
                continue
        # Extend through antialiasing/outline pixels to meet the jambs.
        extension = max(2 * reference_scale, thickness * 0.18)
        ends = [ends[0] - axis * extension, ends[1] + axis * extension]
        ends = [np.clip(point, (0, 0), (width - 1, height - 1)) for point in ends]
        doors.append(ends)
        # The rectangle's angle describes its width, whereas angle now describes
        # its long axis. Keep the footprint aligned with the actual door plane.
        footprints.append(cv2.boxPoints(((cx, cy), (length + 4 * reference_scale, thickness + 4 * reference_scale), angle)))
    return doors, footprints


def _detect_lights(image, hsv, door_footprints, settings, reference_scale):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, settings.light_threshold, 255)
    if settings.light_saturation:
        mask = cv2.bitwise_and(mask, cv2.inRange(hsv[:, :, 1], settings.light_saturation, 255))
    # Door highlights are not torches. Exclude the whole door, not just its center.
    for footprint in door_footprints:
        cv2.fillConvexPoly(mask, np.rint(footprint).astype(np.int32), 0)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lights = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if not settings.light_min_area * reference_scale ** 2 <= area <= settings.light_max_area * reference_scale ** 2:
            continue
        # Only inspect this component's bounding box for its intensity/color.
        component = (labels[y:y + height, x:x + width] == index).astype(np.uint8)
        b, g, r, _ = cv2.mean(image[y:y + height, x:x + width], mask=component)
        lights.append({"position": {"x": float(centers[index][0]), "y": float(centers[index][1])},
                       "range": settings.light_range, "intensity": 1.0,
                       "color": f"ff{round(r):02x}{round(g):02x}{round(b):02x}", "shadows": True})
    return lights


def _clip_doorways(walls, footprints, preserve_jambs=False):
    """Remove wall fragments inside door footprints so an opened portal is clear.

    Convex clipping handles diagonals as well as horizontal and vertical doors.
    Bounding-box rejection keeps the number of exact intersection tests small.
    """
    if not walls or not footprints:
        return walls
    segments = np.asarray(walls, dtype=float)
    for polygon in footprints:
        polygon = np.asarray(polygon, dtype=float)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        if not len(segments):
            break
        outside = (np.any(segments.max(axis=1) < low, axis=1)
                   | np.any(segments.min(axis=1) > high, axis=1))
        if preserve_jambs:
            # Keep short caps across the wall thickness at either end of a
            # doorway. Erasing these jambs would force a slanted or leaking door.
            edges = (polygon[1] - polygon[0], polygon[3] - polygon[0])
            long_edge = max(edges, key=np.linalg.norm)
            length = np.linalg.norm(long_edge)
            axis = long_edge / length
            projections = (segments - polygon.mean(axis=0)) @ axis
            near_end = ((projections.min(axis=1) > length * 0.35)
                        | (projections.max(axis=1) < -length * 0.35))
            vectors = segments[:, 1] - segments[:, 0]
            perpendicular = abs(vectors @ axis) <= 0.35 * np.linalg.norm(vectors, axis=1)
            outside |= near_end & perpendicular
        clipped = []
        for start, end in segments[~outside]:
            # A rectangle's two orthogonal axes give four slab inequalities.
            direction = end - start
            enter, leave = 0.0, 1.0
            for edge in (polygon[1] - polygon[0], polygon[3] - polygon[0]):
                projection = polygon @ edge
                lower, upper = projection.min(), projection.max()
                origin, speed = start @ edge, direction @ edge
                if abs(speed) < 1e-9:
                    if origin < lower or origin > upper:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    t0, t1 = sorted(((lower - origin) / speed, (upper - origin) / speed))
                    enter, leave = max(enter, t0), min(leave, t1)
            if enter >= leave:
                clipped.append((start, end))
                continue
            if enter > 1e-6:
                clipped.append((start, start + direction * enter))
            if leave < 1 - 1e-6:
                clipped.append((start + direction * leave, end))
        # Keep unaffected geometry in one native array instead of constructing
        # thousands of Python segment objects for every door.
        fragments = np.asarray(clipped, dtype=float).reshape(-1, 2, 2)
        segments = np.concatenate((segments[outside], fragments))
    return list(segments)


def load_map(path):
    """Read local images, including Unicode Windows paths, and retain source bytes."""
    source = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError("Use an 8-bit PNG, JPEG, WebP, or BMP map.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3]
        if np.all(alpha == 255):
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            # Match the desktop loader's white matte without float64 temporaries.
            opacity = alpha.astype(np.float32)[:, :, None] / 255
            image = np.rint(image[:, :, :3] * opacity + 255 * (1 - opacity)).astype(np.uint8)
    return np.ascontiguousarray(image), source


def export_image_bytes(image, source=None):
    """UVTT expects PNG. Retain an existing PNG or encode with compression disabled."""
    if source and source.startswith(b"\x89PNG\r\n\x1a\n"):
        return source
    success, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not success:
        raise ValueError("Could not encode the map image for UVTT.")
    return buffer.tobytes()


def _attach_doors(walls, doors, reference_scale):
    """Return straight portals and indices that could not reach both jambs."""
    if not walls:
        return doors, list(range(len(doors)))
    segments = np.asarray(walls, dtype=float)
    starts = segments[:, 0]
    vectors = segments[:, 1] - starts
    attached, disconnected = [], []
    for index, door in enumerate(doors):
        length = np.linalg.norm(door[1] - door[0])
        axis = (door[1] - door[0]) / length
        center = (door[1] + door[0]) / 2
        tolerance = max(3 * reference_scale, length * 0.4)
        offset = starts - center
        cross = axis[0] * vectors[:, 1] - axis[1] * vectors[:, 0]
        valid = abs(cross) > 1e-9
        denominator = np.where(valid, cross, 1.0)
        along = (offset[:, 0] * vectors[:, 1] - offset[:, 1] * vectors[:, 0]) / denominator
        fraction = (offset[:, 0] * axis[1] - offset[:, 1] * axis[0]) / denominator
        valid &= (fraction >= -1e-6) & (fraction <= 1 + 1e-6)
        ends, missing = [], False
        for endpoint in door:
            endpoint_distance = (endpoint - center) @ axis
            distances = np.where(valid, abs(along - endpoint_distance), np.inf)
            nearest = int(distances.argmin())
            if distances[nearest] <= tolerance:
                ends.append(center + along[nearest] * axis)
            else:
                ends.append(endpoint)
                missing = True
        attached.append(ends)
        if missing:
            disconnected.append(index)
    return attached, disconnected


def detect_map(image, settings=None, progress=None, cancelled=None):
    """Detect a map without mutating the input or touching any Qt objects."""
    settings = settings or FastSettings()
    settings.validate()
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Detection needs a nonempty uint8 BGR image.")
    started = time.perf_counter()
    timings = {}
    height, width = image.shape[:2]
    _stage("Preparing image", progress, cancelled)
    scale = min(1.0, settings.max_dimension / max(width, height)) if settings.max_dimension else 1.0
    working = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else image
    work_h, work_w = working.shape[:2]
    to_original = np.array([width / work_w, height / work_h])
    reference_scale = max(work_w, work_h) / 2048
    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    timings["prepare"] = time.perf_counter() - started
    _stage("Detecting walls", progress, cancelled)
    tick = time.perf_counter()
    mask = _wall_mask(working, hsv, settings, reference_scale)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    walls = []
    for contour in contours:
        if cv2.contourArea(contour) < max(2, settings.min_wall_area * reference_scale ** 2 * 0.1):
            continue
        points = cv2.approxPolyDP(contour, settings.simplify * reference_scale, True).reshape(-1, 2)
        for start, end in zip(points, np.roll(points, -1, axis=0)):
            if np.any(start != end):
                walls.append((start.astype(float), end.astype(float)))
    timings["walls"] = time.perf_counter() - tick
    _stage("Detecting doors", progress, cancelled)
    tick = time.perf_counter()
    # Detect footprints even when doors are off, to exclude highlights from lights.
    doors, footprints = _detect_doors(hsv, mask, settings, reference_scale) if settings.doors or settings.lights else ([], [])
    timings["doors"] = time.perf_counter() - tick
    _stage("Detecting lights", progress, cancelled)
    tick = time.perf_counter()
    lights = _detect_lights(working, hsv, footprints, settings, reference_scale) if settings.lights else []
    timings["lights"] = time.perf_counter() - tick
    _stage("Building geometry", progress, cancelled)
    tick = time.perf_counter()
    disconnected, rejected = [], 0
    if settings.doors:
        original_walls = walls
        walls = _clip_doorways(original_walls, footprints, preserve_jambs=True)
        doors, disconnected = _attach_doors(walls, doors, reference_scale)
        if disconnected and settings.door_wall_support:
            # Pixel-mask support alone is insufficient if simplification removed
            # a jamb. Reject that candidate and restore its uncut wall geometry.
            rejected_indices = set(disconnected)
            rejected = len(rejected_indices)
            doors = [door for index, door in enumerate(doors) if index not in rejected_indices]
            footprints = [box for index, box in enumerate(footprints) if index not in rejected_indices]
            walls = _clip_doorways(original_walls, footprints, preserve_jambs=True)
            doors, disconnected = _attach_doors(walls, doors, reference_scale)
    else:
        doors = []
    walls = [(tuple(start * to_original), tuple(end * to_original)) for start, end in walls]
    doors = [(tuple(start * to_original), tuple(end * to_original)) for start, end in doors]
    for light in lights:
        light["position"]["x"] *= to_original[0]
        light["position"]["y"] *= to_original[1]
    timings["geometry"] = time.perf_counter() - tick
    timings["total"] = time.perf_counter() - started
    warnings = []
    if not walls:
        warnings.append("No walls found. Try a wall color sample or Edge mode.")
    if rejected:
        warnings.append(f"Skipped {rejected} door candidate(s) without two connected wall jambs.")
    if disconnected:
        warnings.append(f"{len(disconnected)} door(s) could not connect to both wall jambs. Review these in the wall editor.")
    _stage("Ready", progress, cancelled)
    return DetectionResult(width, height, walls, doors, lights, timings, (work_w, work_h), warnings)


def to_uvtt(result, grid_size=70.0, image_bytes=None, preview=False):
    """Convert all geometry from original pixels to the same UVTT grid units."""
    if not math.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("Grid size must be a positive finite number.")
    def point(p):
        return {"x": float(p[0] / grid_size), "y": float(p[1] / grid_size)}
    portals = []
    for start, end in result.doors:
        portals.append({"position": point(((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)),
                        "bounds": [point(start), point(end)],
                        "rotation": math.atan2(end[1] - start[1], end[0] - start[0]),
                        "closed": True, "freestanding": False})
    lights = [{**light, "position": point((light["position"]["x"], light["position"]["y"]))} for light in result.lights]
    data = {"format": 0.3,
            "resolution": {"map_origin": {"x": 0.0, "y": 0.0},
                           "map_size": {"x": result.width / grid_size, "y": result.height / grid_size},
                           "pixels_per_grid": grid_size},
            "line_of_sight": [[point(a), point(b)] for a, b in result.walls],
            "objects_line_of_sight": [], "portals": portals, "lights": lights,
            "environment": {"baked_lighting": True, "ambient_light": "ff000000"}}
    if image_bytes:
        # Preserve the supplied image bytes. Never recompress the user's map.
        data["image"] = base64.b64encode(image_bytes).decode("ascii")
    if preview:
        data["_preview_pixels"] = [[{"x": float(x), "y": float(y)} for x, y in wall] for wall in result.walls]
    return data


def write_uvtt(path, data):
    """Write atomically so a failed export cannot destroy an existing map."""
    import os
    import tempfile

    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
