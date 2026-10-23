import json
import struct

import numpy as np
import pytest

from scripts.generate_light_maps import DECOY_KINDS, LIGHT_KINDS, generate, render
from src.ml.light_data import LightCropSampler, read_light_dataset


def test_procedural_light_annotations_are_deterministic_and_survive_training_crops(tmp_path):
    first, points, objects = render(901000)
    second, repeated_points, repeated_objects = render(901000)
    np.testing.assert_array_equal(first, second)
    assert points == repeated_points and objects == repeated_objects
    assert set(item["kind"] for item in objects) == set(LIGHT_KINDS + DECOY_KINDS)
    assert points == [item["position"] for item in objects if item["emits_light"]]
    assert not np.array_equal(first, render(901001)[0])

    manifest = generate(tmp_path / "maps", count=12)
    samples = read_light_dataset(manifest)
    train = [sample for sample in samples if sample["split"] == "train"]
    test = [sample for sample in samples if sample["split"] == "test"]
    assert len(train) == 4 and len(test) == 8
    assert not {s["group"] for s in train} & {s["group"] for s in test}
    assert all(s["synthetic"] and s["valid"].all() for s in samples)
    assert all(len(s["points"]) >= 5 and len(s["negative_points"]) >= 5 for s in samples)
    images, targets, masks = LightCropSampler(train).batch(12)
    assert images.shape == (12, 3, 384, 384)
    assert np.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
    assert np.all(np.any((targets == 1) & masks, axis=(1, 2)))
    for sample in samples:
        raw = (manifest.parent / sample["image"]).read_bytes()
        assert raw[:2] == b"BM" and struct.unpack_from("<I", raw, 30)[0] == 0
    data = json.loads(manifest.read_text())
    assert len(data["generator_sha256"]) == 64


def test_generation_preserves_existing_data_and_requires_reserved_seeds(tmp_path):
    with pytest.raises(ValueError, match="last eight seeds"):
        generate(tmp_path / "too-small", count=8)
    assert not (tmp_path / "too-small").exists()
    marker = tmp_path / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        generate(tmp_path)
    assert marker.read_text() == "keep"


def test_crop_sampler_retains_only_sampled_coordinates():
    valid = np.ones((513, 511), np.uint8)
    valid[30:90, 70:250] = 0
    sampler = LightCropSampler([{"valid": valid}])
    coordinates = sampler.valid_points[0]
    np.testing.assert_array_equal(coordinates, np.argwhere(valid)[::16])
    assert coordinates.flags.owndata and coordinates.base is None
    assert coordinates.nbytes < np.count_nonzero(valid) * 2  # Less than 1/8 of the full int64 XY array.
