"""Read palette data and detector settings, never map-generation instructions."""

from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import re

from src.core.fast_pipeline import FastSettings


PRESETS = {
    "foundry": "Foundry schema · tolerant",
    "exact": "Foundry schema · exact",
    "demon": "Gray walls · orange doors",
}


def preset_settings(name):
    if name == "foundry":
        return FastSettings()
    if name == "exact":
        return replace(FastSettings(), wall_tolerance=0, door_match="rgb",
                       door_color_tolerance=0, door_gap_close=0, min_wall_area=10, simplify=0.5)
    if name == "demon":
        return replace(FastSettings(), wall_mode="stone", wall_color=(131, 131, 131),
                       door_color=(255, 160, 65), wall_gap_close=1, door_gap_close=0,
                       door_max_length=60, light_mode="brightness", light_min_area=3,
                       light_max_area=180, light_core_fraction=0, light_min_fill=0)
    raise ValueError(f"Unknown preset: {name}")


def _rgb(value):
    if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))
    if (isinstance(value, (list, tuple)) and len(value) == 3
            and all(type(channel) is int and 0 <= channel <= 255 for channel in value)):
        return tuple(value)
    raise ValueError("Palette colors must be #RRGGBB or three integer RGB channels.")


def _schema_color(data, name):
    candidates = []
    reserved = data.get("reserved_colors", {}).get(name)
    if isinstance(reserved, dict):
        candidates.extend(_rgb(reserved[key]) for key in ("hex", "rgb") if key in reserved)
    elif reserved is not None:
        candidates.append(_rgb(reserved))
    palette = data.get("palette", {})
    if name in palette:
        candidates.append(_rgb(palette[name]))
    if not candidates:
        raise ValueError(f"Schema is missing its {name} color.")
    if any(color != candidates[0] for color in candidates[1:]):
        raise ValueError(f"The schema contains conflicting {name} colors.")
    return candidates[0]


def settings_from_data(data, base=None):
    """Accept the supplied generation schema or an Auto-Wall settings object.

    Prose, style rules, paths, URLs, and prompt requirements are inert data. Only
    explicit palette fields or known detector fields affect processing.
    """
    if not isinstance(data, dict):
        raise ValueError("A profile must be a JSON object.")
    base = base or FastSettings()
    if "reserved_colors" in data or "palette" in data:
        if not isinstance(data.get("reserved_colors", {}), dict) or not isinstance(data.get("palette", {}), dict):
            raise ValueError("Schema palette and reserved_colors must be objects.")
        wall, door, void = (_schema_color(data, name) for name in ("wall", "door", "void"))
        if len({wall, door, void}) != 3:
            raise ValueError("Wall, door, and void colors must be distinct.")
        result = replace(base, wall_mode="color", wall_color=wall, door_color=door,
                         void_color=void, wall_gap_close=0)
    else:
        if "settings" in data:
            if data.get("format", "auto-wall-settings") != "auto-wall-settings" or data.get("version", 1) != 1:
                raise ValueError("Unsupported detector settings format or version.")
            if data.keys() - {"format", "version", "settings"}:
                raise ValueError("Unknown fields in detector settings document.")
        values = data.get("settings", data)
        if not isinstance(values, dict):
            raise ValueError("Settings must be a JSON object.")
        unknown = values.keys() - {field.name for field in fields(FastSettings)}
        if unknown:
            raise ValueError("Unknown detector settings: " + ", ".join(sorted(unknown)))
        values = dict(values)
        for name in ("wall_color", "door_color", "void_color", "light_color"):
            if name in values:
                values[name] = _rgb(values[name])
        result = replace(base, **values)
    try:
        result.validate()
    except (TypeError, OverflowError) as error:
        raise ValueError("Invalid detector setting type.") from error
    return result


def load_settings(path, base=None):
    return settings_from_data(json.loads(Path(path).read_text(encoding="utf-8-sig")), base)


def save_settings(path, settings):
    from src.core.fast_pipeline import write_uvtt

    settings.validate()
    write_uvtt(path, {"format": "auto-wall-settings", "version": 1, "settings": asdict(settings)})
