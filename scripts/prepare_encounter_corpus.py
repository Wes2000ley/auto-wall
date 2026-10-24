"""Inventory encounter-map images and review the provenance of their wall data.

This is an audit tool, not a label generator. It reads the encounter source tree
without modifying it, groups image variants by site before any later split, and
writes a JSON inventory plus uncompressed BMP contact sheets to a separate
output directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ARCHIVE_DIRECTORIES = {"aethrune-hidden-sites"}
CONTACT_COLUMNS = 5
CONTACT_WIDTH = 320
OVERLAY_COLUMNS = 3
OVERLAY_WIDTH = 600
ARCHIVED_REVIEW_ROOT = Path("aethrune-hidden-sites/software-pass/before-staging/assets/maps/encounters")


def _within(path, parent):
    return path == parent or parent in path.parents


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    if path is None or not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _decode_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _direct_primary_image(site_dir):
    preferred = site_dir / "source.png"
    if preferred.is_file():
        return preferred
    versioned = sorted(site_dir.glob("*-v1.png"))
    if len(versioned) == 1:
        return versioned[0]
    direct = sorted(path for path in site_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    and "q85" not in path.stem.lower())
    return direct[0] if len(direct) == 1 else None


def _scene_wall_documents(scene):
    walls = scene.get("walls", []) if isinstance(scene, dict) else []
    return walls if isinstance(walls, list) else []


def _is_invisible_wall(wall):
    return (wall.get("move") == 20
            and all(wall.get(sense) == 0 for sense in ("sight", "light", "sound")))


def _is_explicit_window(wall):
    """Count an explicit window field only; ordinary Foundry walls have none."""
    for key, value in wall.items():
        if "window" in str(key).lower() and value not in (None, False, 0, ""):
            return True
        if isinstance(value, str) and value.lower() == "window":
            return True
    return False


def _wall_summary(document):
    walls = document.get("walls", []) if isinstance(document, dict) else []
    if not isinstance(walls, list):
        walls = []
    door_count = sum(bool(wall.get("door")) for wall in walls if isinstance(wall, dict))
    invisible_count = sum(_is_invisible_wall(wall) for wall in walls if isinstance(wall, dict))
    window_count = sum(_is_explicit_window(wall) for wall in walls if isinstance(wall, dict))
    features = {}
    invalid_geometry = 0
    for wall in walls:
        if not isinstance(wall, dict):
            invalid_geometry += 1
            continue
        feature = wall.get("flags", {}).get("world", {}).get("feature")
        if feature:
            features[feature] = features.get(feature, 0) + 1
        coords = wall.get("c")
        if (not isinstance(coords, list) or len(coords) != 4
                or not all(isinstance(value, (int, float)) for value in coords)):
            invalid_geometry += 1
    return {
        "segments": len(walls),
        "doors": door_count,
        "invisible_sense_walls": invisible_count,
        "explicit_window_fields": window_count,
        "invalid_segment_geometry": invalid_geometry,
        "feature_provenance_counts": features,
    }


def _feature_keyword_summary(walls):
    patterns = {
        "fence_or_railing": ("fence", "railing", "palisade", "windbreak"),
        "window_or_glazing": ("window", "shutter", "glazing", "glass"),
        "construction_terms": ("wall", "masonry", "building", "lodge", "refuge", "depot",
                               "chapel", "shrine", "fort", "gate", "jamb", "partition",
                               "kiln", "shed", "stonework"),
        "outside_object_terms": ("tree", "trunk", "root", "boulder", "rock", "cliff", "ledge",
                                 "log", "stump", "reef", "stone marker", "deadfall", "table",
                                 "barrel", "trough", "furnishing", "debris"),
    }
    counts = {name: 0 for name in patterns}
    counts["door_flags"] = 0
    counts["named_features"] = 0
    counts["unnamed_or_auto_candidate_features"] = 0
    feature_counts = {}
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        counts["door_flags"] += bool(wall.get("door"))
        flags = wall.get("flags", {})
        world = flags.get("world", {}) if isinstance(flags, dict) else {}
        feature = world.get("feature", "") if isinstance(world, dict) else ""
        if feature:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
            if "auto-wall" in feature.lower() or "palette candidate" in feature.lower():
                counts["unnamed_or_auto_candidate_features"] += 1
            else:
                counts["named_features"] += 1
            lower = feature.lower()
            for category, words in patterns.items():
                counts[category] += any(word in lower for word in words)
        else:
            counts["unnamed_or_auto_candidate_features"] += 1
    return {
        "keyword_segment_counts_not_labels": counts,
        "feature_name_counts": dict(sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _archive_site_dir(root, slug):
    archive = root / ARCHIVED_REVIEW_ROOT
    for candidate in (archive / slug, archive / "aethrune-hidden-sites" / slug):
        if candidate.is_dir():
            return candidate
    return None


def _review_text_hints(review):
    fields = ("limitations", "visual_observations", "no_door_reason", "native_correction")
    chunks = []
    for field in fields:
        value = review.get(field)
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list):
            chunks.extend(str(item) for item in value)
    text = " ".join(chunks)
    lower = text.lower()
    hints = {}
    for name, terms in {
        "fence": ("fence", "palisade", "railing"),
        "window": ("window", "shutter", "glazing", "glass"),
        "door": ("door", "gate", "sluice"),
    }.items():
        if any(term in lower for term in terms):
            hints[name] = True
    return hints


def _find_site_for_image(relative_path, site_slugs):
    parts = relative_path.parts
    if parts and parts[0] in site_slugs:
        return parts[0]
    for part in parts:
        if part in site_slugs:
            return part
    return None


def _source_hash_agreement(expected, actual):
    if not expected:
        return "not_recorded"
    return expected == actual


def _image_record(path, root, primary_paths_by_hash, site_slugs, layout_groups_by_site):
    relative = path.relative_to(root)
    relative_name = relative.as_posix()
    digest = _sha256(path)
    site_slug = _find_site_for_image(relative, site_slugs)
    matching_primaries = primary_paths_by_hash.get(digest, [])
    if relative_name in matching_primaries:
        role = "canonical_primary_source"
        duplicate_of = matching_primaries[0] if relative_name != matching_primaries[0] else None
    elif matching_primaries:
        same_site_primary = next((candidate for candidate in matching_primaries
                                  if Path(candidate).parts[0] == site_slug), None)
        duplicate_of = same_site_primary or matching_primaries[0]
        role = "exact_duplicate_of_canonical_source"
    elif site_slug and relative.parent == Path(site_slug):
        duplicate_of = None
        role = "canonical_site_variant"
    else:
        duplicate_of = None
        role = "nested_backup_or_auxiliary"
    try:
        image = _decode_image(path)
        width, height = image.shape[1], image.shape[0]
        decode_error = None
    except (OSError, ValueError, cv2.error) as error:
        width = height = None
        decode_error = f"{type(error).__name__}: {error}"
    return {
        "path": relative.as_posix(),
        "site_group": site_slug,
        "layout_group": layout_groups_by_site.get(site_slug),
        "role": role,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "dimensions": [width, height] if width is not None else None,
        "duplicate_of": duplicate_of,
        "decode_error": decode_error,
    }


def _validate_wall_coordinates(walls, width, height):
    outside = 0
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        coords = wall.get("c")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        if width and height and any(
                x < 0 or x > width or y < 0 or y > height
                for x, y in (coords[:2], coords[2:])):
            outside += 1
    return outside


def _site_inventory(site_dir, root, primary_image, layout_group):
    slug = site_dir.name
    source = _sha256(primary_image) if primary_image else None
    image = _decode_image(primary_image) if primary_image else None
    dimensions = [int(image.shape[1]), int(image.shape[0])] if image is not None else None
    site_data, site_error = _read_json(site_dir / "site.json")
    map_data, map_error = _read_json(site_dir / "map.json")
    wall_data, wall_error = _read_json(site_dir / "walls.json")
    review, review_error = _read_json(site_dir / "wall-review.json")
    primary_review, primary_review_error = _read_json(site_dir / "primary-review.json")

    archive_dir = _archive_site_dir(root, slug)
    archived_walls, archived_walls_error = _read_json(archive_dir / "walls.json" if archive_dir else None)
    archived_review, archived_review_error = _read_json(
        archive_dir / "wall-review.json" if archive_dir else None)

    scene_paths = sorted(site_dir.glob("*-scene-v1.json"))
    scene_path = scene_paths[0] if scene_paths else None
    scene_data, scene_error = _read_json(scene_path)
    wall_summary = _wall_summary(wall_data or {})
    scene_wall_summary = _wall_summary({"walls": _scene_wall_documents(scene_data or {})})
    map_width = (map_data or {}).get("width") or (scene_data or {}).get("width")
    map_height = (map_data or {}).get("height") or (scene_data or {}).get("height")
    walls = (wall_data or {}).get("walls", [])
    if not isinstance(walls, list):
        walls = []

    review_accepted = bool((review or {}).get("accepted"))
    checks = (review or {}).get("checks", {})
    wall_supervision_eligible = bool(
        review_accepted
        and checks.get("visual_source_reviewed")
        and checks.get("retained_barriers_match_source")
        and source
        and _source_hash_agreement((review or {}).get("image_sha256"), source) is True
    )
    map_expected_hash = ((map_data or {}).get("source_sha256")
                         or (map_data or {}).get("image_sha256"))
    observed_map_hash = _source_hash_agreement(map_expected_hash, source)
    observed_wall_hash = _source_hash_agreement((wall_data or {}).get("image_sha256"), source)
    observed_review_hash = _source_hash_agreement((review or {}).get("image_sha256"), source)

    archived_wall_list = (archived_walls or {}).get("walls", [])
    if not isinstance(archived_wall_list, list):
        archived_wall_list = []
    archived_checks = (archived_review or {}).get("checks", {})
    archived_source_match = _source_hash_agreement(
        (archived_review or {}).get("image_sha256"), source)
    archived_count_match = (
        (archived_review or {}).get("wall_count") == len(archived_wall_list)
        and (archived_review or {}).get("door_count") == sum(
            bool(wall.get("door")) for wall in archived_wall_list if isinstance(wall, dict)))
    archived_reviewed_reference = bool(
        (archived_review or {}).get("accepted") is True
        and archived_checks.get("visual_source_reviewed") is True
        and archived_source_match is True
        and archived_count_match
    )

    annotations = {
        "walls_file": "walls.json" if wall_data is not None else None,
        "wall_review_file": "wall-review.json" if review is not None else None,
        "primary_review_file": "primary-review.json" if primary_review is not None else None,
        "scene_file": scene_path.relative_to(root).as_posix() if scene_path else None,
        "wall_review": {
            "accepted": (review or {}).get("accepted"),
            "status": (review or {}).get("status"),
            "visual_source_reviewed": checks.get("visual_source_reviewed"),
            "retained_barriers_match_source": checks.get("retained_barriers_match_source"),
            "wall_count_recorded": (review or {}).get("wall_count"),
            "door_count_recorded": (review or {}).get("door_count"),
            "limitations": (review or {}).get("limitations", []),
            "eligible_as_wall_supervision": wall_supervision_eligible,
        },
        "primary_review": {
            "accepted": (primary_review or {}).get("accepted"),
            "observations": (primary_review or {}).get("observations", []),
            "limitations": (primary_review or {}).get("limitations", []),
            "wall_label_approval": False,
        },
        "wall_geometry": wall_summary,
        "scene_geometry": scene_wall_summary,
        "archived_reviewed_geometry": {
            "walls_file": (archive_dir / "walls.json").relative_to(root).as_posix()
            if archive_dir and (archive_dir / "walls.json").exists() else None,
            "review_file": (archive_dir / "wall-review.json").relative_to(root).as_posix()
            if archive_dir and (archive_dir / "wall-review.json").exists() else None,
            "accepted": (archived_review or {}).get("accepted"),
            "visual_source_reviewed": archived_checks.get("visual_source_reviewed"),
            "source_hash_matches_canonical_image": archived_source_match,
            "reviewed_counts_match_geometry": archived_count_match,
            "usable_as_reviewed_geometry_reference": archived_reviewed_reference,
            "limitations": (archived_review or {}).get("limitations", []),
            "geometry": _wall_summary(archived_walls or {}),
            **_feature_keyword_summary(archived_wall_list),
            "review_text_mentions": _review_text_hints(archived_review or {}),
        },
        "wall_file_coordinate_outliers": _validate_wall_coordinates(
            walls, map_width, map_height),
        "source_hash_checks": {
            "map_json": observed_map_hash,
            "walls_json": observed_wall_hash,
            "wall_review_json": observed_review_hash,
        },
        "read_errors": {name: error for name, error in (
            ("site.json", site_error), ("map.json", map_error),
            ("walls.json", wall_error), ("wall-review.json", review_error),
            ("primary-review.json", primary_review_error), ("scene", scene_error),
            ("archived walls", archived_walls_error), ("archived wall review", archived_review_error)) if error},
    }
    layout_plan = (map_data or {}).get("layout_plan", "")
    if not isinstance(layout_plan, str):
        layout_plan = json.dumps(layout_plan, ensure_ascii=False)
    return {
        "id": slug,
        "title": (site_data or {}).get("name") or (map_data or {}).get("name") or slug,
        "layout_group": layout_group,
        "split": None,
        "split_note": "Keep every source, upscale, recolor, and backup for this site in one future split.",
        "source_image": {
            "path": primary_image.relative_to(root).as_posix() if primary_image else None,
            "sha256": source,
            "dimensions": dimensions,
        },
        "art_metadata": {
            "generation_tool": (map_data or {}).get("generation_tool"),
            "layout_plan_excerpt": layout_plan[:1200],
        },
        "annotations": annotations,
    }


def _tile(image, width):
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _write_bmp(path, image):
    ok, encoded = cv2.imencode(".bmp", image)
    if not ok:
        raise OSError(f"Could not encode BMP contact sheet: {path}")
    encoded.tofile(path)


def _write_map_contact_sheet(sites, output_path):
    tile_height = round(CONTACT_WIDTH * 1024 / 1536)
    label_height = 34
    columns = CONTACT_COLUMNS
    rows = (len(sites) + columns - 1) // columns
    sheet = np.full((rows * (tile_height + label_height), columns * CONTACT_WIDTH, 3),
                    (38, 38, 38), dtype=np.uint8)
    for index, site in enumerate(sites):
        image_path = site["_primary_image"]
        if image_path is None:
            continue
        image = _tile(_decode_image(image_path), CONTACT_WIDTH)
        x = (index % columns) * CONTACT_WIDTH
        y = (index // columns) * (tile_height + label_height)
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        dimensions = site["source_image"]["dimensions"]
        label = f'{site["id"]}  {dimensions[0]}x{dimensions[1]}' if dimensions else site["id"]
        cv2.putText(sheet, label, (x + 5, y + tile_height + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (245, 245, 245), 1, cv2.LINE_AA)
    _write_bmp(output_path, sheet)


def _draw_wall_overlay(image, walls, source_dimensions):
    height, width = image.shape[:2]
    source_width, source_height = source_dimensions
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        coords = wall.get("c")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x0, y0, x1, y1 = coords
        points = (round(x0 * width / source_width), round(y0 * height / source_height),
                  round(x1 * width / source_width), round(y1 * height / source_height))
        if wall.get("door"):
            color = (0, 0, 255)  # red: door-marked candidate
        elif _is_invisible_wall(wall):
            color = (255, 0, 255)  # magenta: invisible-sense candidate
        else:
            color = (0, 255, 255)  # yellow: ordinary candidate wall
        cv2.line(image, points[:2], points[2:], color, 2, cv2.LINE_AA)
        cv2.circle(image, points[:2], 2, color, -1, cv2.LINE_AA)
        cv2.circle(image, points[2:], 2, color, -1, cv2.LINE_AA)
    return image


def _write_geometry_contact_sheet(sites, output_path):
    tile_height = round(OVERLAY_WIDTH * 1024 / 1536)
    header_height = 34
    label_height = 32
    selected = [site for site in sites if site["_walls"]]
    columns = OVERLAY_COLUMNS
    rows = (len(selected) + columns - 1) // columns
    sheet = np.full((header_height + rows * (tile_height + label_height), columns * OVERLAY_WIDTH, 3),
                    (38, 38, 38), dtype=np.uint8)
    cv2.putText(sheet, "PROVISIONAL UNREVIEWED GEOMETRY: yellow wall / red door / magenta invisible",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    for index, site in enumerate(selected):
        image = _tile(_decode_image(site["_primary_image"]), OVERLAY_WIDTH)
        image = _draw_wall_overlay(image, site["_walls"], site["source_image"]["dimensions"])
        x = (index % columns) * OVERLAY_WIDTH
        y = header_height + (index // columns) * (tile_height + label_height)
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        counts = site["annotations"]["wall_geometry"]
        label = f'{site["id"]}  |  {counts["segments"]} lines / {counts["doors"]} doors'
        cv2.putText(sheet, label, (x + 5, y + tile_height + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.49, (245, 245, 245), 1, cv2.LINE_AA)
    if selected:
        _write_bmp(output_path, sheet)
    return len(selected)


def _reviewed_feature_color(wall):
    flags = wall.get("flags", {})
    world = flags.get("world", {}) if isinstance(flags, dict) else {}
    feature = world.get("feature", "") if isinstance(world, dict) else ""
    lower = feature.lower()
    if wall.get("door"):
        return (0, 0, 255)  # red: door-marked geometry
    if any(word in lower for word in ("window", "shutter", "glazing", "glass")):
        return (255, 0, 0)  # blue: feature name identifies glass or a shutter
    if any(word in lower for word in ("fence", "railing", "palisade")):
        return (0, 255, 0)  # green: fence or railing
    if any(word in lower for word in (
            "tree", "trunk", "root", "boulder", "rock", "cliff", "ledge", "log", "stump",
            "reef", "stone marker", "deadfall", "table", "barrel", "trough", "debris")):
        return (255, 255, 0)  # cyan: scenery/object geometry to exclude from this target
    if any(word in lower for word in (
            "wall", "masonry", "building", "lodge", "refuge", "depot", "chapel", "shrine",
            "fort", "gate", "jamb", "partition", "kiln", "shed", "stonework")):
        return (0, 255, 255)  # yellow: likely constructed barrier
    if feature:
        return (255, 255, 255)  # white: named feature needs manual interpretation
    return (0, 165, 255)  # orange: no per-segment feature name


def _draw_reviewed_overlay(image, walls, source_dimensions):
    height, width = image.shape[:2]
    source_width, source_height = source_dimensions
    for wall in walls:
        if not isinstance(wall, dict):
            continue
        coords = wall.get("c")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x0, y0, x1, y1 = coords
        start = (round(x0 * width / source_width), round(y0 * height / source_height))
        end = (round(x1 * width / source_width), round(y1 * height / source_height))
        cv2.line(image, start, end, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(image, start, end, _reviewed_feature_color(wall), 2, cv2.LINE_AA)
    return image


def _write_reviewed_geometry_sheet(sites, output_path):
    tile_height = round(OVERLAY_WIDTH * 1024 / 1536)
    header_height = 34
    label_height = 32
    selected = [site for site in sites if site["_archived_walls"]]
    columns = OVERLAY_COLUMNS
    rows = (len(selected) + columns - 1) // columns
    sheet = np.full((header_height + rows * (tile_height + label_height), columns * OVERLAY_WIDTH, 3),
                    (38, 38, 38), dtype=np.uint8)
    cv2.putText(sheet,
                "ARCHIVE: reviewed geometry; keyword colors only, not labels: yellow construction / green fence / blue glass / red door / cyan objects",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (245, 245, 245), 1, cv2.LINE_AA)
    for index, site in enumerate(selected):
        image = _tile(_decode_image(site["_primary_image"]), OVERLAY_WIDTH)
        image = _draw_reviewed_overlay(
            image, site["_archived_walls"], site["source_image"]["dimensions"])
        x = (index % columns) * OVERLAY_WIDTH
        y = header_height + (index // columns) * (tile_height + label_height)
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        counts = site["annotations"]["archived_reviewed_geometry"]["geometry"]
        label = f'{site["id"]}  |  {counts["segments"]} lines / {counts["doors"]} doors'
        cv2.putText(sheet, label, (x + 5, y + tile_height + 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.49, (245, 245, 245), 1, cv2.LINE_AA)
    if selected:
        _write_bmp(output_path, sheet)
    return len(selected)


def build_corpus_inventory(source_root, output_dir):
    """Build a provenance inventory and BMP review sheets without changing sources."""
    root = Path(source_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if _within(output, root):
        raise ValueError("Output directory must be outside the read-only source tree.")

    site_dirs = sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name not in ARCHIVE_DIRECTORIES
        and _direct_primary_image(child) is not None
    )
    site_slugs = {path.name for path in site_dirs}
    primaries = {path.name: _direct_primary_image(path) for path in site_dirs}
    primary_paths_by_hash = {}
    primary_sites_by_hash = {}
    for path in primaries.values():
        if path is not None:
            digest = _sha256(path)
            primary_paths_by_hash.setdefault(digest, []).append(path.relative_to(root).as_posix())
            primary_sites_by_hash.setdefault(digest, []).append(path.parent.name)

    layout_groups_by_site = {}
    for digest, duplicate_sites in primary_sites_by_hash.items():
        layout_group = f"sha256:{digest}" if len(duplicate_sites) > 1 else duplicate_sites[0]
        for slug in duplicate_sites:
            layout_groups_by_site[slug] = layout_group

    image_paths = sorted(path for path in root.rglob("*")
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    image_records = [_image_record(path, root, primary_paths_by_hash, site_slugs,
                                   layout_groups_by_site)
                     for path in image_paths]
    records_by_site = {}
    for record in image_records:
        if record["site_group"]:
            records_by_site.setdefault(record["site_group"], []).append(record["path"])

    sites = []
    for site_dir in site_dirs:
        item = _site_inventory(site_dir, root, primaries[site_dir.name],
                               layout_groups_by_site[site_dir.name])
        item["image_variants"] = records_by_site.get(site_dir.name, [])
        item["_primary_image"] = primaries[site_dir.name]
        walls_data, _ = _read_json(site_dir / "walls.json")
        item["_walls"] = (walls_data or {}).get("walls", [])
        if not isinstance(item["_walls"], list):
            item["_walls"] = []
        archive_dir = _archive_site_dir(root, site_dir.name)
        archived_data, _ = _read_json(archive_dir / "walls.json" if archive_dir else None)
        item["_archived_walls"] = (archived_data or {}).get("walls", [])
        if not isinstance(item["_archived_walls"], list):
            item["_archived_walls"] = []
        sites.append(item)

    output.mkdir(parents=True, exist_ok=True)
    original_sheet = output / "encounter-map-contact-sheet.bmp"
    geometry_sheet = output / "provisional-wall-overlay-contact-sheet.bmp"
    reviewed_sheet = output / "archived-reviewed-geometry-contact-sheet.bmp"
    _write_map_contact_sheet(sites, original_sheet)
    overlay_site_count = _write_geometry_contact_sheet(sites, geometry_sheet)
    reviewed_site_count = _write_reviewed_geometry_sheet(sites, reviewed_sheet)

    public_sites = [{key: value for key, value in site.items() if not key.startswith("_")}
                    for site in sites]
    extension_counts = {}
    for record in image_records:
        suffix = Path(record["path"]).suffix.lower()
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
    approved = [site["id"] for site in public_sites
                if site["annotations"]["wall_review"]["eligible_as_wall_supervision"]]
    geometry_sites = [site for site in public_sites
                      if site["annotations"]["wall_geometry"]["segments"]]
    report = {
        "format": "auto-wall-encounter-corpus-audit-v1",
        "source_root": str(root),
        "source_read_only": True,
        "canonical_layout_count": len(public_sites),
        "recursive_image_file_count": len(image_records),
        "unique_image_file_sha256_count": len({record["sha256"] for record in image_records}),
        "exact_duplicate_image_file_count": len(image_records) - len({record["sha256"] for record in image_records}),
        "exact_duplicate_canonical_source_group_count": sum(
            len(slugs) > 1 for slugs in primary_sites_by_hash.values()),
        "image_extension_counts": extension_counts,
        "candidate_geometry_site_count": len(geometry_sites),
        "candidate_geometry_segment_count": sum(
            site["annotations"]["wall_geometry"]["segments"] for site in geometry_sites),
        "candidate_door_mark_count": sum(
            site["annotations"]["wall_geometry"]["doors"] for site in public_sites),
        "archive_reviewed_geometry_site_count": sum(
            site["annotations"]["archived_reviewed_geometry"]["usable_as_reviewed_geometry_reference"]
            for site in public_sites),
        "archive_reviewed_geometry_segment_count": sum(
            site["annotations"]["archived_reviewed_geometry"]["geometry"]["segments"]
            for site in public_sites),
        "archive_reviewed_door_mark_count": sum(
            site["annotations"]["archived_reviewed_geometry"]["geometry"]["doors"]
            for site in public_sites),
        "explicit_window_annotation_count": sum(
            site["annotations"]["wall_geometry"]["explicit_window_fields"] for site in public_sites),
        "wall_review_approved_site_count": len(approved),
        "wall_review_approved_sites": approved,
        "split_policy": "No split is assigned until labels are reviewed. Assign splits by layout_group so exact duplicate canonical sources and their variants stay together.",
        "layout_grouping_limitations": (
            "Canonical sources with identical SHA-256 hashes share one layout_group while retaining "
            "distinct site IDs. Near-duplicate upscales, recolors, and other visual derivatives are "
            "not inferred from byte hashes and need review before assigning splits."),
        "annotation_findings": [
            "primary-review.json is not wall-label approval; consult wall-review.json for barrier provenance.",
            "Candidate walls are not training truth unless wall-review explicitly accepts the image-matched geometry.",
            "The canonical wall schema has door flags, but no explicit window class or fence material labels.",
            "Keep uncertain scene regions ignored. Do not treat trees, rocks, water, tents, or scenery as walls.",
        ],
        "contact_sheets": {
            "canonical_images": original_sheet.name,
            "provisional_wall_geometry": geometry_sheet.name if overlay_site_count else None,
            "overlay_site_count": overlay_site_count,
            "archived_reviewed_geometry": reviewed_sheet.name if reviewed_site_count else None,
            "archived_reviewed_site_count": reviewed_site_count,
        },
        "sites": public_sites,
        "recursive_images": image_records,
    }
    manifest_path = output / "encounter-corpus-inventory.json"
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Read-only encounter maps directory")
    parser.add_argument("output", type=Path, help="Separate directory for JSON inventory and BMP contact sheets")
    args = parser.parse_args()
    report = build_corpus_inventory(args.source, args.output)
    summary = {key: report[key] for key in (
        "canonical_layout_count", "recursive_image_file_count", "unique_image_file_sha256_count",
        "candidate_geometry_site_count", "candidate_geometry_segment_count",
        "candidate_door_mark_count", "explicit_window_annotation_count",
        "wall_review_approved_site_count", "archive_reviewed_geometry_site_count",
        "archive_reviewed_geometry_segment_count", "archive_reviewed_door_mark_count",
        "contact_sheets")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
