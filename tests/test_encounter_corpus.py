"""Source-only tests for the encounter image and annotation audit."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from scripts.prepare_encounter_corpus import build_corpus_inventory


def _write_bmp(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".bmp", image)
    assert ok
    encoded.tofile(path)
    return encoded.tobytes()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _fixture_tree(tmp_path):
    source_root = tmp_path / "encounters"
    site = source_root / "example-site"
    source_image = np.full((24, 36, 3), (42, 86, 130), dtype=np.uint8)
    source_bytes = _write_bmp(site / "source.bmp", source_image)
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    _write_bmp(site / "source-q85.bmp", source_image.copy())
    _write_bmp(source_root / "aethrune-hidden-sites" / "software-pass" / "example-site" / "source.bmp",
               source_image)
    _write_json(site / "site.json", {"name": "Example Site"})
    _write_json(site / "map.json", {"width": 36, "height": 24, "source_sha256": source_digest})
    walls = [
        {"c": [1, 2, 20, 2], "move": 20, "sight": 20, "light": 20, "sound": 20,
         "door": 0, "flags": {"world": {"feature": "Auto-Wall OpenCV candidate; visually review"}}},
        {"c": [20, 2, 20, 12], "move": 20, "sight": 20, "light": 20, "sound": 20, "door": 1},
        {"c": [25, 2, 25, 12], "move": 20, "sight": 0, "light": 0, "sound": 0},
        {"c": [30, 2, 30, 12], "move": 20, "sight": 20, "light": 20, "sound": 20,
         "window": True},
    ]
    _write_json(site / "walls.json", {"image_sha256": source_digest, "walls": walls})
    _write_json(site / "wall-review.json", {
        "accepted": False,
        "status": "provisional_auto_wall_user_manual_review_pending",
        "image_sha256": source_digest,
        "wall_count": len(walls),
        "checks": {"visual_source_reviewed": False, "retained_barriers_match_source": False},
    })
    _write_json(site / "primary-review.json", {
        "accepted": True,
        "observations": ["The scene contains a small structure."],
    })
    _write_json(site / "example-site-scene-v1.json", {
        "width": 36, "height": 24, "walls": walls,
    })
    archive_site = (source_root / "aethrune-hidden-sites" / "software-pass" / "before-staging"
                    / "assets" / "maps" / "encounters" / "example-site")
    archived_walls = [
        {"c": [2, 3, 8, 3], "move": 20, "sight": 20, "light": 20, "sound": 20, "door": 0,
         "flags": {"world": {"feature": "Pine Court low stone fence"}}},
        {"c": [8, 3, 8, 8], "move": 20, "sight": 0, "light": 0, "sound": 0, "door": 0,
         "flags": {"world": {"feature": "Lantern Loft blue-glass shutter"}}},
        {"c": [10, 3, 10, 8], "move": 20, "sight": 20, "light": 20, "sound": 20, "door": 1,
         "flags": {"world": {"feature": "Mothveil Lodge west hinged main door"}}},
        {"c": [12, 3, 12, 8], "move": 20, "sight": 0, "light": 0, "sound": 0, "door": 0,
         "flags": {"world": {"feature": "Pear tree trunk base"}}},
    ]
    _write_json(archive_site / "walls.json", {
        "image_sha256": source_digest, "walls": archived_walls,
    })
    _write_json(archive_site / "wall-review.json", {
        "accepted": True,
        "image_sha256": source_digest,
        "wall_count": len(archived_walls),
        "door_count": 1,
        "checks": {"visual_source_reviewed": True},
    })
    return source_root, site, source_bytes


def test_inventory_groups_variants_and_archived_copies_without_promoting_candidates(tmp_path):
    source_root, site, original_source = _fixture_tree(tmp_path)
    output = tmp_path / "review"

    report = build_corpus_inventory(source_root, output)

    assert report["canonical_layout_count"] == 1
    assert report["recursive_image_file_count"] == 3
    assert report["unique_image_file_sha256_count"] == 1
    assert report["exact_duplicate_image_file_count"] == 2
    assert report["wall_review_approved_site_count"] == 0
    sample, = report["sites"]
    assert sample["layout_group"] == "example-site"
    assert sample["split"] is None
    assert len(sample["image_variants"]) == 3
    annotations = sample["annotations"]
    assert annotations["wall_review"]["eligible_as_wall_supervision"] is False
    assert annotations["primary_review"]["accepted"] is True
    assert annotations["primary_review"]["wall_label_approval"] is False
    assert annotations["wall_geometry"]["segments"] == 4
    assert annotations["wall_geometry"]["doors"] == 1
    assert annotations["wall_geometry"]["invisible_sense_walls"] == 1
    assert annotations["wall_geometry"]["explicit_window_fields"] == 1
    archived = annotations["archived_reviewed_geometry"]
    assert archived["accepted"] is True
    assert archived["source_hash_matches_canonical_image"] is True
    assert archived["reviewed_counts_match_geometry"] is True
    assert archived["usable_as_reviewed_geometry_reference"] is True
    assert archived["geometry"]["segments"] == 4
    assert archived["keyword_segment_counts_not_labels"]["fence_or_railing"] == 1
    assert archived["keyword_segment_counts_not_labels"]["window_or_glazing"] == 1
    assert archived["keyword_segment_counts_not_labels"]["outside_object_terms"] == 1
    assert (site / "source.bmp").read_bytes() == original_source

    inventory_path = output / "encounter-corpus-inventory.json"
    saved = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert saved["sites"][0]["source_image"]["sha256"] == hashlib.sha256(original_source).hexdigest()
    image_records = saved["recursive_images"]
    assert len([item for item in image_records if item["duplicate_of"]]) == 2
    assert (output / "encounter-map-contact-sheet.bmp").read_bytes()[:2] == b"BM"
    assert (output / "provisional-wall-overlay-contact-sheet.bmp").read_bytes()[:2] == b"BM"
    assert (output / "archived-reviewed-geometry-contact-sheet.bmp").read_bytes()[:2] == b"BM"
    assert {path.suffix.lower() for path in output.iterdir()} == {".json", ".bmp"}


def test_inventory_refuses_output_inside_read_only_source_tree(tmp_path):
    source_root, _, _ = _fixture_tree(tmp_path)
    output = source_root / "review"

    with pytest.raises(ValueError, match="outside the read-only source tree"):
        build_corpus_inventory(source_root, output)

    assert not output.exists()


def test_identical_sources_share_layout_group_and_preserve_site_ids(tmp_path):
    source_root = tmp_path / "encounters"
    source_image = np.full((24, 36, 3), (42, 86, 130), dtype=np.uint8)
    for slug in ("first-site", "second-site"):
        _write_bmp(source_root / slug / "source.bmp", source_image.copy())

    report = build_corpus_inventory(source_root, tmp_path / "review")

    assert report["canonical_layout_count"] == 2
    assert report["unique_image_file_sha256_count"] == 1
    assert report["exact_duplicate_image_file_count"] == 1
    assert report["exact_duplicate_canonical_source_group_count"] == 1
    assert {site["id"] for site in report["sites"]} == {"first-site", "second-site"}
    layout_groups = {site["layout_group"] for site in report["sites"]}
    assert len(layout_groups) == 1
    layout_group, = layout_groups
    assert layout_group.startswith("sha256:")
    assert {site["split"] for site in report["sites"]} == {None}
    assert all(image["role"] == "canonical_primary_source"
               for image in report["recursive_images"])
    assert {image["layout_group"] for image in report["recursive_images"]} == {layout_group}
    assert {image["site_group"] for image in report["recursive_images"]} == {
        "first-site", "second-site"}
    assert sum(image["duplicate_of"] is not None for image in report["recursive_images"]) == 1
    assert "Near-duplicate" in report["layout_grouping_limitations"]
