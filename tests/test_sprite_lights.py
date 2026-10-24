import json
import struct

import cv2
import numpy as np
import pytest

from scripts.generate_sprite_lights import generate, load_sprites, place_sprite, render
from src.ml.light_data import LightCropSampler, read_light_dataset


def sprite_inputs(tmp_path):
    import hashlib

    pixels = np.zeros((16, 32, 4), np.uint8)
    pixels[3:12, 3:12] = [60, 140, 240, 255]
    pixels[6:9, 6:9] = 0  # A luminous ring can have a transparent labeled center.
    pixels[3:12, 19:28] = [200, 120, 50, 128]
    sheet = tmp_path / "sprites.tiff"
    assert cv2.imwrite(str(sheet), pixels, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # Explicitly uncompressed.
    spec = {"format": "auto-wall-light-sprites-v1", "id": "fixture", "sheet_size": [32, 16],
            "sheet_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(), "families": [
                {"name": "ring", "emits_light": True, "center": [7.5, 7.5], "rectangles": [[0, 0, 16, 16]]},
                {"name": "metal", "emits_light": False, "center": [7.5, 7.5], "rectangles": [[16, 0, 16, 16]]}]}
    path = tmp_path / "sprites.json"
    path.write_text(json.dumps(spec))
    return path, sheet, spec


def test_sprite_centers_follow_rotations_reflections_and_pixel_center_resizing():
    pixels = np.zeros((4, 6, 4), np.uint8)
    pixels[1, 4] = [0, 0, 255, 255]
    sprite = {"pixels": pixels, "center": np.array([4., 1.])}
    original = pixels.copy()
    for turns in range(4):
        for flip in (False, True):
            image = np.zeros((48, 48, 3), np.uint8)
            point, rectangle = place_sprite(image, sprite, 24, 24, 18, turns, flip)
            yy, xx = np.where(image[:, :, 2] == 255)
            np.testing.assert_allclose(point, [xx.mean(), yy.mean()], rtol=0, atol=1e-6)
            assert len(xx) == 9 and rectangle[2] * rectangle[3] == 216
    np.testing.assert_array_equal(pixels, original)
    with pytest.raises(ValueError, match="inside the scene"):
        place_sprite(np.zeros((4, 4, 3), np.uint8), sprite, 0, 0, 18, 0, False)


def test_sprite_blending_respects_straight_alpha_and_transparent_centers():
    pixels = np.zeros((4, 4, 4), np.uint8)
    pixels[1, 1] = [100, 150, 200, 128]
    pixels[2, 2] = [255, 0, 255, 0]
    image = np.full((8, 8, 3), 60, np.uint8)
    point, rectangle = place_sprite(image, {"pixels": pixels, "center": [2., 2.]}, 4, 4, 4, 0, False)
    np.testing.assert_array_equal(image[3, 3], np.rint(60 * (127 / 255) + np.array([100, 150, 200]) * (128 / 255)))
    np.testing.assert_array_equal(image[4, 4], [60, 60, 60])
    assert point == [4., 4.] and rectangle == [2, 2, 4, 4]


def test_sprite_generation_keeps_complete_nonoverlapping_objects_and_reserved_seeds(tmp_path):
    path, sheet, spec = sprite_inputs(tmp_path)
    loaded, families = load_sprites(path, sheet)
    assert loaded == spec
    first, points, objects = render(921000, families)
    second, repeated_points, repeated_objects = render(921000, families)
    np.testing.assert_array_equal(first, second)
    assert points == repeated_points and objects == repeated_objects
    assert not np.array_equal(first, render(921001, families)[0])
    assert {o["kind"] for o in objects} == {"ring", "metal"}
    coverage = np.zeros(first.shape[:2], np.uint8)
    for item in objects:
        x, y, w, h = item["placed_rectangle"]
        coverage[y:y + h, x:x + w] += 1
        assert x <= item["position"][0] < x + w and y <= item["position"][1] < y + h
    assert coverage.max() == 1
    manifest = generate(path, sheet, tmp_path / "scenes", count=12)
    samples = read_light_dataset(manifest)
    train = [s for s in samples if s["split"] == "train"]
    test = [s for s in samples if s["split"] == "test"]
    assert len(train) == 4 and len(test) == 8
    assert not {s["group"] for s in train} & {s["group"] for s in test}
    assert all(s["synthetic"] and s["valid"].all() and len(s["points"]) + len(s["negative_points"]) == 16 for s in samples)
    images, targets, masks = LightCropSampler(train).batch(8)
    assert images.shape == (8, 3, 384, 384) and masks.all() and targets.max() == 1
    for sample in samples:
        raw = (manifest.parent / sample["image"]).read_bytes()
        assert raw[:2] == b"BM" and struct.unpack_from("<I", raw, 30)[0] == 0
    metadata = json.loads(manifest.read_text())
    assert metadata["sprite_specification"] == spec and len(metadata["source_sha256"]) == 2
    with pytest.raises(FileExistsError):
        generate(path, sheet, manifest.parent, count=12)


def test_sprite_input_errors_do_not_create_a_dataset(tmp_path):
    path, sheet, spec = sprite_inputs(tmp_path)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="last eight"):
        generate(path, sheet, output, count=8)
    for field, value, error in (("sheet_sha256", "bad", "SHA-256"), ("sheet_size", [16, 16], "dimensions")):
        changed = {**spec, field: value}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=error):
            generate(path, sheet, output, count=12)
    for rectangle in ([0, 0, 17, 20], [0, 0, 0, 16], [0., 0, 16, 16]):
        spec["families"][0]["rectangles"] = [rectangle]
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match="rectangles"):
            generate(path, sheet, output, count=12)
    spec["families"][0]["rectangles"] = [[0, 0, 16, 16]]
    spec["families"][0]["center"] = [float("nan"), 7.5]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="finite XY"):
        generate(path, sheet, output, count=12)
    spec["families"][0].update(center=[.5, .5], rectangles=[[0, 0, 2, 2]])
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="entirely transparent"):
        generate(path, sheet, output, count=12)
    assert not output.exists()
