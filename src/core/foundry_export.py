"""Native Foundry v14 scenes, including movement-only invisible walls."""

import base64
import math
import os
from pathlib import Path
import secrets
import tempfile

from src.core.fast_pipeline import write_uvtt


# Foundry v14 CONST.EDGE_SENSE_TYPES and CONST.WALL_MOVEMENT_TYPES.
NONE = 0
NORMAL = 20
PROXIMITY = 30


def to_foundry_scene(data, name="Auto-Wall map"):
    """Convert the editor's UVTT preview into an importable single-level scene.

    The private _invisible_walls field uses the same grid units as UVTT geometry.
    Keep it out of standard UVTT: the installed importer treats open portals as
    proximity windows, which cannot represent movement-only walls accurately.
    """
    resolution = data["resolution"]
    grid = resolution["pixels_per_grid"]
    if not math.isfinite(grid) or grid < 20 or int(grid) != grid:
        raise ValueError("Foundry Scene export requires a whole Pixels / grid value of at least 20.")
    image = data.get("image")
    if not image:
        raise ValueError("Foundry Scene export requires the original map image.")
    header = base64.b64decode(image[:16])
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        mime = "image/webp"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ValueError("Foundry Scene export requires a PNG or WebP image.")
    origin = resolution["map_origin"]
    level_id = secrets.token_hex(8)

    def pixel_point(point):
        return [round((point[axis] - origin[axis]) * grid) for axis in ("x", "y")]

    walls = []

    def add_wall(start, end, *, invisible=False, door=False, window=False):
        first, last = pixel_point(start), pixel_point(end)
        if first == last:
            return  # Foundry uses integer pixels; a subpixel segment can collapse.
        sense = NONE if invisible else NORMAL
        wall = {"_id": secrets.token_hex(8), "c": first + last,
                "levels": [level_id], "move": NORMAL,
                "sight": sense, "light": sense, "sound": sense,
                "dir": 0, "door": 1 if door else 0, "ds": 0}
        if window:
            wall.update(sight=PROXIMITY, light=PROXIMITY,
                        threshold={"attenuation": True, "light": 10, "sight": 10})
        walls.append(wall)

    for key, invisible in (("line_of_sight", False), ("objects_line_of_sight", False), ("_invisible_walls", True)):
        for points in data.get(key, []):
            for start, end in zip(points, points[1:]):
                add_wall(start, end, invisible=invisible)
    for portal in data.get("portals", []):
        closed = portal.get("closed", True)
        add_wall(*portal["bounds"], door=closed, window=not closed)

    # Match the Universal Battlemap Importer's light mapping, with 5 ft squares.
    lights = []
    for light in data.get("lights", []):
        x, y = pixel_point(light["position"])
        radius = light["range"] * 5
        lights.append({"_id": secrets.token_hex(8), "x": x, "y": y,
                       "elevation": 0, "levels": [level_id], "walls": True,
                       "config": {"angle": 360, "color": "#" + light["color"][2:],
                                  "dim": radius, "bright": radius / 2,
                                  "alpha": min(1., max(0., .05 * light["intensity"]))}})

    return {"name": name, "width": round(resolution["map_size"]["x"] * grid),
            "height": round(resolution["map_size"]["y"] * grid), "padding": 0,
            "grid": {"type": 1, "size": int(grid), "distance": 5, "units": "ft"},
            "tokenVision": True, "initialLevel": level_id,
            "levels": [{"_id": level_id, "name": "Map", "elevation": {"bottom": 0, "top": None},
                        "background": {"src": f"data:{mime};base64," + image}}],
            "walls": walls, "lights": lights}


def write_map_exports(image_path, data):
    """Save UVTT, native scene JSON, and the same WebP beside the loaded image."""
    image_path = Path(image_path)
    uvtt_path = image_path.with_suffix(".dd2vtt")
    uvtt = {key: value for key, value in data.items() if not key.startswith("_")}
    invisible = data.get("_invisible_walls", [])
    foundry_path = image_path.with_suffix(".foundry.json")
    webp_path = image_path.with_suffix(".foundry.webp")
    # Convert before writing any file, so invalid grid settings write nothing.
    scene = to_foundry_scene(data, image_path.stem)
    image_bytes = base64.b64decode(data["image"], validate=True)
    if not image_bytes.startswith(b"RIFF") or image_bytes[8:12] != b"WEBP":
        raise ValueError("Map export requires a WebP image encoded at quality 85.")
    if invisible:
        uvtt["auto_wall"] = {"invisible_wall_segments": len(invisible),
                             "foundry_scene": foundry_path.name,
                             "notice": "Import the Foundry Scene JSON to retain invisible walls. UVTT contains ordinary walls, doors, and lights."}
    _write_image(webp_path, image_bytes)
    write_uvtt(foundry_path, scene)
    write_uvtt(uvtt_path, uvtt)
    return [uvtt_path, foundry_path, webp_path]


def _write_image(path, data):
    """Replace only a complete image, preserving an older export on write failure."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
