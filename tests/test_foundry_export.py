import base64
from copy import deepcopy
import json

import numpy as np
import pytest

from src.core.fast_pipeline import DetectionResult, export_image_bytes, to_uvtt
from src.core.foundry_export import to_foundry_scene, write_map_exports


@pytest.fixture
def preview():
    result = DetectionResult(400, 300, walls=[((40.25, 100.75), (120., 100.75))],
                             doors=[((120., 100.75), (160., 100.75))],
                             lights=[{'position': {'x': 220., 'y': 180.}, 'range': 6.,
                                      'intensity': 1., 'color': 'ffffaa44'}],
                             invisible_walls=[((200., 20.), (200., 200.))],
                             transparent_barriers=[((10., 10.), (20., 20.))])
    image = export_image_bytes(np.full((300, 400, 3), 100, np.uint8))
    return to_uvtt(result, 70, image, preview=True)


def test_native_scene_preserves_wall_types_pixel_alignment_image_and_lights(preview):
    before = deepcopy(preview)
    scene = to_foundry_scene(preview, 'Map name')
    assert scene['name'] == 'Map name'
    assert (scene['width'], scene['height'], scene['padding']) == (400, 300, 0)
    assert scene['grid'] == {'type': 1, 'size': 70, 'distance': 5, 'units': 'ft'}
    level = scene['levels'][0]
    assert scene['initialLevel'] == level['_id']
    assert level['elevation'] == {'bottom': 0, 'top': None}
    image = level['background']['src'].removeprefix('data:image/webp;base64,')
    assert base64.b64decode(image) == base64.b64decode(preview['image'])
    normal, invisible, door = scene['walls']
    assert normal['c'] == [40, 101, 120, 101]
    assert door['c'] == [120, 101, 160, 101] and door['door'] == 1 and door['ds'] == 0
    assert all(wall['move'] == 20 and wall['levels'] == [level['_id']] for wall in scene['walls'])
    assert all(normal[key] == door[key] == 20 and invisible[key] == 0 for key in ('sight', 'light', 'sound'))
    light = scene['lights'][0]
    assert (light['x'], light['y'], light['levels']) == (220, 180, [level['_id']])
    assert light['config'] == {'angle': 360, 'color': '#ffaa44', 'dim': 30., 'bright': 15., 'alpha': .05}
    ids = [level['_id'], *(wall['_id'] for wall in scene['walls']), light['_id']]
    assert len(ids) == len(set(ids)) and all(len(value) == 16 for value in ids)
    assert preview == before


@pytest.mark.parametrize('grid', [0, 19, 70.5, float('inf'), float('nan')])
def test_invalid_native_grid_is_rejected_without_writing_files(preview, tmp_path, grid):
    preview['resolution']['pixels_per_grid'] = grid
    with pytest.raises(ValueError, match='whole Pixels / grid'):
        write_map_exports(tmp_path / 'map.png', preview)
    assert list(tmp_path.iterdir()) == []


def test_native_export_requires_image_and_handles_origin_and_subpixel_segments(preview):
    image = preview.pop('image')
    with pytest.raises(ValueError, match='original map image'):
        to_foundry_scene(preview)
    preview['image'] = image
    preview['resolution']['map_origin'] = {'x': 1, 'y': 2}
    preview['line_of_sight'] = [[{'x': 1, 'y': 2}, {'x': 1.001, 'y': 2.001}]]
    scene = to_foundry_scene(preview)
    assert len(scene['walls']) == 2  # The collapsed subpixel wall is omitted.
    assert scene['lights'][0]['x'] == 150 and scene['lights'][0]['y'] == 40


def test_save_next_to_image_retains_invisible_walls_in_native_sidecar_and_explains_uvtt_limit(preview, tmp_path):
    image_path = tmp_path / 'map.name.雪.png'
    image_path.write_bytes(b'do not change the input')
    paths = write_map_exports(image_path, preview)
    assert paths == [tmp_path / 'map.name.雪.dd2vtt', tmp_path / 'map.name.雪.foundry.json',
                     tmp_path / 'map.name.雪.foundry.webp']
    uvtt, scene = (json.loads(path.read_text(encoding='utf-8')) for path in paths[:2])
    assert paths[2].read_bytes() == base64.b64decode(preview['image'])
    assert uvtt['image'] == preview['image'] and uvtt['line_of_sight'] == preview['line_of_sight']
    assert not any(key.startswith('_') for key in uvtt)
    assert uvtt['auto_wall']['invisible_wall_segments'] == 1
    assert uvtt['auto_wall']['foundry_scene'] == paths[1].name
    assert len(scene['walls']) == 3
    assert image_path.read_bytes() == b'do not change the input'


def test_uvtt_cannot_silently_discard_confirmed_invisible_walls():
    result = DetectionResult(100, 100, [], [], [], invisible_walls=[((0, 0), (50, 50))])
    with pytest.raises(ValueError, match='Invisible walls require Foundry'):
        to_uvtt(result)
    assert to_uvtt(result, preview=True)['_invisible_walls']


def test_normal_export_always_writes_both_formats_and_uvtt_has_no_editor_metadata(preview, tmp_path):
    preview.pop('_invisible_walls')
    paths = write_map_exports(tmp_path / 'map.png', preview)
    assert len(paths) == 3 and paths[0].suffix == '.dd2vtt'
    assert paths[1].name == 'map.foundry.json'
    scene = json.loads(paths[1].read_text())
    assert len(scene['walls']) == 2 and len(scene['lights']) == 1
    expected = {key: value for key, value in preview.items() if not key.startswith('_')}
    assert json.loads(paths[0].read_text()) == expected


def test_sidecar_is_updated_when_last_invisible_wall_is_deleted(preview, tmp_path):
    image = tmp_path / 'map.png'
    paths = write_map_exports(image, preview)
    preview.pop('_invisible_walls')
    assert write_map_exports(image, preview) == paths
    scene = json.loads(paths[1].read_text())
    assert all(wall['sight'] == 20 for wall in scene['walls'])
    assert 'auto_wall' not in json.loads(paths[0].read_text())


def test_legacy_object_walls_and_windows_keep_their_importer_behavior(preview):
    preview['objects_line_of_sight'] = [[{'x': 1, 'y': 1}, {'x': 2, 'y': 1}]]
    preview['portals'][0]['closed'] = False
    scene = to_foundry_scene(preview)
    assert len(scene['walls']) == 4
    window = scene['walls'][-1]
    assert window['move'] == 20 and window['sight'] == window['light'] == 30
    assert window['door'] == 0 and window['sound'] == 20
    assert window['threshold'] == {'attenuation': True, 'light': 10, 'sight': 10}


def test_model_windows_use_proximity_for_sound_and_keep_private_marker_out_of_uvtt(preview, tmp_path):
    preview['portals'].append({
        'position': {'x': 2.5, 'y': 2.0},
        'bounds': [{'x': 2, 'y': 2}, {'x': 3, 'y': 2}],
        'rotation': 0, 'closed': False, 'freestanding': False,
        '_auto_wall_window': True,
    })
    scene = to_foundry_scene(preview)
    model_window = scene['walls'][-1]
    assert model_window['door'] == 0 and model_window['move'] == 20
    assert model_window['sight'] == model_window['light'] == model_window['sound'] == 30
    assert model_window['threshold']['sound'] == model_window['threshold']['sight'] == 10
    paths = write_map_exports(tmp_path / 'map.png', preview)
    uvtt = json.loads(paths[0].read_text())
    assert not any(key.startswith('_') for portal in uvtt['portals'] for key in portal)
    exported_scene = json.loads(paths[1].read_text())
    assert exported_scene['walls'][-1]['sound'] == 30


def test_write_failure_is_reported_and_does_not_damage_existing_export(preview, tmp_path, monkeypatch):
    import src.core.foundry_export as export_module

    path = tmp_path / 'map.dd2vtt'
    path.write_text('existing')
    def fail(*args):
        raise PermissionError('folder is read-only')
    monkeypatch.setattr(export_module, 'write_uvtt', fail)
    with pytest.raises(PermissionError, match='read-only'):
        write_map_exports(tmp_path / 'map.png', preview)
    assert path.read_text() == 'existing'
