"""Partial supervision must not invent labels or overwrite source artwork."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ml.encounter_data import (
    ANNOTATION_FORMAT,
    prepare_dataset,
    rasterize_annotation,
    source_image,
    structure_checks,
)


def annotation():
    return {"format": ANNOTATION_FORMAT, "site": "small-site", "width": 48, "height": 32,
            "complete": False, "regions": [
                {"class": "background", "polygon": [[0, 0], [20, 0], [20, 20], [0, 20]]},
                {"class": "door", "polygon": [[12, 8], [16, 8], [16, 12], [12, 12]]},
                {"class": "window", "polygon": [[22, 8], [28, 8], [28, 12], [22, 12]]},
                {"class": "ignore", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
            "strokes": [{"class": "wall", "points": [[4, 10], [35, 10]], "width": 5}],
            "doors": [{"span": [12, 10, 16, 10]}],
            "windows": [{"span": [22, 10, 28, 10]}],
            "rays": [{"id": "floor", "from": [4, 4], "to": [16, 4], "expect": "open"}]}


def test_partial_raster_keeps_unreviewed_pixels_ignored_and_openings_over_walls():
    mask = rasterize_annotation(annotation(), (48, 32))
    assert mask.dtype == np.uint8 and mask.shape == (32, 48)
    assert mask[30, 47] == 255  # unreviewed space is never an assumed negative
    assert mask[4, 4] == 0
    assert mask[10, 6] == 1
    assert mask[10, 14] == 2  # a broad wall stroke cannot erase the door
    assert mask[10, 25] == 5
    assert mask[1, 1] == 255  # explicitly uncertain pixels win over all labels


@pytest.mark.parametrize("update", [
    {"width": 47},
    {"complete": "false"},
    {"regions": [{"class": "rock-wall", "polygon": [[1, 1], [2, 1], [2, 2]]}]},
    {"strokes": [{"class": "wall", "points": [[1, 1], [float("nan"), 2]], "width": 2}]},
    {"strokes": [{"class": "wall", "points": [[1, 1], [49, 2]], "width": 2}]},
    {"strokes": [{"class": "wall", "points": [[1, 1], [2, 2]], "width": 0}]},
])
def test_bad_annotations_fail_clearly(update):
    with pytest.raises(ValueError):
        rasterize_annotation({**annotation(), **update}, (48, 32))


def test_partial_instance_checks_never_claim_complete_precision():
    checks = structure_checks(annotation())
    assert checks["reference_size"] == [48, 32]
    assert checks["complete_instances"] == {"doors": False, "windows": False}
    assert checks["doors"][0]["span"] == [12, 10, 16, 10]
    with pytest.raises(ValueError, match="zero length"):
        structure_checks({**annotation(), "windows": [{"span": [1, 1, 1, 1]}]})


def test_movement_checks_use_the_benchmark_contract_and_require_unique_ids():
    item = annotation()
    item["rays"][0]["sense"] = "movement"
    checks = structure_checks(item)
    assert checks["rays"][0]["sense"] == "movement"
    item["rays"].append(dict(item["rays"][0]))
    with pytest.raises(ValueError, match="Ray ID"):
        structure_checks(item)
    item["rays"] = [{**item["rays"][0], "sense": "move"}]
    with pytest.raises(ValueError, match="sense"):
        structure_checks(item)


def test_polygon_holes_preserve_prior_background_without_filling_room_interior():
    item = annotation()
    item["strokes"] = []
    item["regions"] = [
        {"class": "background", "polygon": [[0, 0], [40, 0], [40, 28], [0, 28]]},
        {"class": "wall", "polygon": [[4, 4], [35, 4], [35, 25], [4, 25]],
         "holes": [[[8, 8], [31, 8], [31, 21], [8, 21]]]}]
    mask = rasterize_annotation(item, (48, 32))
    assert mask[4, 12] == 1 and mask[12, 4] == 1
    assert mask[12, 12] == 0
    assert mask[31, 47] == 255


def make_source(tmp_path):
    source = tmp_path / "sources"
    site = source / "small-site"
    site.mkdir(parents=True)
    # The loader decodes bytes, not extensions. Keep the fixture uncompressed BMP.
    ok, data = cv2.imencode(".bmp", np.full((32, 48, 3), 110, np.uint8))
    assert ok
    image = site / "source.png"
    image.write_bytes(data.tobytes())
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    item = annotation()
    item["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    path = annotations / "small-site.json"
    path.write_text(json.dumps(item), encoding="utf-8")
    approval = {"status": "accepted", "reviewer": "test fixture", "notes": "Explicit partial fixture.",
                "source_sha256": item["image_sha256"],
                "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(rasterize_annotation(item, (48, 32)).tobytes()).hexdigest()}
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"format": "auto-wall-annotation-review-v1",
                                 "samples": {"small-site": approval}}), encoding="utf-8")
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"small-site": "train"}), encoding="utf-8")
    return source, annotations, split, image, path, review


def test_prepare_preserves_source_and_records_labels_provenance_and_frozen_split(tmp_path):
    source, annotations, split, image, _, review = make_source(tmp_path)
    original = image.read_bytes()
    manifest = prepare_dataset(source, annotations, split, tmp_path / "prepared", review_file=review)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["classes"] == ["background", "wall", "door", "light", "transparent_barrier", "window"]
    sample = data["samples"][0]
    assert sample["split"] == "train" and sample["group"] == "encounter-small-site"
    assert sample["annotation_complete"] is False
    assert sample["provenance"]["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert (manifest.parent / sample["mask"]).read_bytes().startswith(b"BM")
    assert image.read_bytes() == original
    with pytest.raises(ValueError, match="new directory"):
        prepare_dataset(source, annotations, split, manifest.parent)
    with pytest.raises(ValueError, match="new directory"):
        prepare_dataset(source, annotations, split, source / "prepared")


def test_replay_weight_only_changes_training_sampling_and_preserves_layout_splits(tmp_path):
    source, annotations, split, image, _, review = make_source(tmp_path)
    mask = tmp_path / "replay-labels.bmp"
    assert cv2.imwrite(str(mask), np.ones((32, 48), np.uint8))
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"format": "auto-wall-training-v1",
        "classes": ["background", "wall", "door", "light", "transparent_barrier"],
        "samples": [{"id": "older-" + name, "group": "older-" + name,
                     "image": str(image), "mask": str(mask), "split": name,
                     "annotation": "Replay fixture."} for name in ("train", "validation")]}))
    original = replay.read_bytes()
    manifest = prepare_dataset(source, annotations, split, tmp_path / "weighted",
                               replay=replay, review_file=review, replay_weight=5)
    samples = {item["id"]: item for item in json.loads(manifest.read_text())["samples"]}
    assert samples["older-train"]["sampling_weight"] == 5
    assert "sampling_weight" not in samples["older-validation"]
    assert "sampling_weight" not in samples["small-site"]
    assert samples["older-validation"]["split"] == "validation"
    assert replay.read_bytes() == original
    with pytest.raises(ValueError, match="finite positive"):
        prepare_dataset(source, annotations, split, tmp_path / "bad-weight", replay_weight=0)
    with pytest.raises(ValueError, match="requires a replay"):
        prepare_dataset(source, annotations, split, tmp_path / "missing-replay", replay_weight=5)


def test_changed_source_is_rejected_before_any_dataset_files_are_written(tmp_path):
    source, annotations, split, image, _, review = make_source(tmp_path)
    image.write_bytes(image.read_bytes() + b"changed")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="hash"):
        prepare_dataset(source, annotations, split, output, review_file=review)
    assert not output.exists()
    with pytest.raises(ValueError, match="folder name"):
        source_image(source, "../small-site")


@pytest.mark.parametrize("failure", ["missing", "rejected", "annotation", "raster"])
def test_unreviewed_or_changed_labels_cannot_enter_dataset(tmp_path, failure):
    source, annotations, split, _, path, review = make_source(tmp_path)
    data = json.loads(review.read_text(encoding="utf-8"))
    if failure == "rejected":
        data["samples"]["small-site"]["status"] = "rejected"
    elif failure == "annotation":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif failure == "raster":
        data["samples"]["small-site"]["mask_sha256"] = "0" * 64
    review.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="review"):
        prepare_dataset(source, annotations, split, output,
                        review_file=None if failure == "missing" else review)
    assert not output.exists()


def test_committed_partial_seed_reviews_match_annotation_bytes_and_rasters():
    """Catch stale/corrupted dataset revisions; hashes do not certify semantics."""
    root = Path(__file__).resolve().parents[1] / "benchmarks/data/encounters/partial-source-v2"
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    paths = {path.stem: path for path in (root / "annotations").glob("*.json")}
    assert set(paths) == set(splits) == set(review["samples"])
    assert len(paths) == 31
    assert {split: list(splits.values()).count(split) for split in set(splits.values())} == {
        "train": 25, "validation": 3, "test": 3}
    glazing_layouts = set()
    for site, path in paths.items():
        item = json.loads(path.read_text(encoding="utf-8"))
        accepted = review["samples"][site]
        assert item["site"] == site and item["complete"] is False
        assert accepted["status"] == "accepted"
        assert item["image_sha256"] == accepted["source_sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == accepted["annotation_sha256"]
        mask = rasterize_annotation(item, (item["width"], item["height"]))
        assert hashlib.sha256(mask.tobytes()).hexdigest() == accepted["mask_sha256"]
        if np.any(mask == 5):
            glazing_layouts.add((site, splits[site]))
    assert glazing_layouts == {("gilded-snare", "train"), ("mothveil-lodge", "validation")}
