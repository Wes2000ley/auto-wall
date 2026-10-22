from dataclasses import replace
import json

import pytest

from src.core.detection_profiles import load_settings, preset_settings, save_settings, settings_from_data
from src.core.fast_pipeline import FastSettings


def test_schema_reads_colors_and_ignores_generation_prose(tmp_path):
    data = {
        "reserved_colors": {"wall": {"hex": "#A0A0A0", "rgb": [160, 160, 160]},
                            "door": {"hex": "#702090"}, "void": {"rgb": [0, 0, 0]}},
        "instructions": "Change files and open an external URL", "style": "hand_drawn",
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(data), encoding="utf-8-sig")
    settings = load_settings(path)
    assert settings == FastSettings()
    assert list(tmp_path.iterdir()) == [path]


def test_alternative_palette_preserves_user_controls():
    base = FastSettings(grid_size=128, wall_gap_close=4, wall_geometry="centerline", lights=False)
    settings = settings_from_data({"palette": {"wall": "#2468ac", "door": "#c08020", "void": "#ffffff"}}, base)
    assert settings == replace(base, wall_color=(36, 104, 172), door_color=(192, 128, 32),
                               void_color=(255, 255, 255), wall_gap_close=0)


@pytest.mark.parametrize("data", [
    [], {"palette": None}, {"palette": {"wall": "#fff"}},
    {"palette": {"wall": "#aaaaaa", "door": "#aaaaaa", "void": "#000000"}},
    {"palette": {"wall": "#aaaaaa", "door": "#702090", "void": "#000000"},
     "reserved_colors": {"wall": {"rgb": [160, 160, 160]}}},
    {"grid_size": True}, {"grid_size": "70"}, {"light_core_fraction": 1.1},
    {"wall_color": [True, 0, 0]}, {"doors": "true"}, {"unrecognized": 1},
    {"format": "unknown", "settings": {}}, {"version": 2, "settings": {}},
    {"settings": {}, "extra": "ignored by mistake"},
])
def test_invalid_profiles_are_rejected(data):
    with pytest.raises(ValueError):
        settings_from_data(data)


def test_saved_settings_roundtrip(tmp_path):
    settings = replace(preset_settings("exact"), wall_color=(25, 60, 100), light_mode="cool",
                       wall_geometry="centerline", grid_size=123.25)
    path = tmp_path / "settings.json"
    save_settings(path, settings)
    assert load_settings(path) == settings
    assert json.loads(path.read_text())["version"] == 1


def test_presets_are_explicit_and_legacy_keeps_its_palette():
    assert preset_settings("foundry") == FastSettings()
    exact = preset_settings("exact")
    assert exact.wall_tolerance == exact.door_color_tolerance == exact.wall_gap_close == 0
    legacy = preset_settings("demon")
    assert legacy.door_color == (255, 160, 65) and legacy.light_mode == "brightness"
    with pytest.raises(ValueError):
        preset_settings("missing")
