"""Check source integrity and annotation preservation without network access."""

import hashlib

import cv2
import numpy as np
import pytest

from scripts.prepare_open_light_maps import prepare_dataset
from src.ml.light_data import read_light_dataset


def specification(tmp_path):
    image = np.full((8, 10, 3), (30, 70, 120), np.uint8)
    image[2, 2] = 255
    data = cv2.imencode(".bmp", image)[1].tobytes()
    (tmp_path / "source.bmp").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return {"annotation_review": "Test labels", "split_notes": "One validation layout",
            "sources": {"artist": {"file": "source.bmp", "sha256": digest, "license": "test"}},
            "samples": [{"id": "fixture", "source": "artist", "image_sha256": digest,
                         "source_size": [10, 8], "output_size": [40, 32], "interpolation": "nearest",
                         "group": "fixture", "split": "validation", "annotation": "Test fixture",
                         "points": [[2, 2]], "negative_points": [[5, 5]],
                         "ignore_rectangles": [[7, 0, 10, 8]]}]}


def test_preparation_preserves_centers_exclusions_and_original_download(tmp_path):
    recipe = specification(tmp_path)
    original = (tmp_path / "source.bmp").read_bytes()
    path = prepare_dataset(tmp_path, tmp_path / "prepared", recipe)
    sample, = read_light_dataset(path)
    assert sample["split"] == "validation" and not sample["synthetic"]
    np.testing.assert_array_equal(sample["points"], [[9.5, 9.5]])
    np.testing.assert_array_equal(sample["negative_points"], [[21.5, 21.5]])
    assert sample["pixels"].shape == (32, 40, 3)
    assert np.all(sample["pixels"][8:12, 8:12] == 255)
    assert sample["valid"][:, :28].all() and not sample["valid"][:, 28:].any()
    assert (tmp_path / "source.bmp").read_bytes() == original
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_dataset(tmp_path, path.parent, recipe)


@pytest.mark.parametrize("failure", ["source", "member", "dimensions"])
def test_preparation_rejects_changed_sources_before_writing_output(tmp_path, failure):
    recipe = specification(tmp_path)
    if failure == "source":
        (tmp_path / "source.bmp").write_bytes(b"changed download")
    elif failure == "member":
        recipe["samples"][0]["image_sha256"] = "0" * 64
    else:
        recipe["samples"][0]["source_size"] = [11, 8]
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="(SHA-256|dimensions)"):
        prepare_dataset(tmp_path, output, recipe)
    assert not output.exists()
