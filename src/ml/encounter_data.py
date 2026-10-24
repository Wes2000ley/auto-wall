"""Rasterize explicit, partial structure annotations without running a detector."""

import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


ANNOTATION_FORMAT = "auto-wall-encounter-annotations-v1"
LABELS = {"background": 0, "wall": 1, "door": 2, "light": 3,
          "transparent_barrier": 4, "window": 5, "ignore": 255}
# Openings take precedence over wall strokes; uncertain pixels always stay ignored.
PAINT_ORDER = ("background", "wall", "transparent_barrier", "door", "window", "light", "ignore")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_points(values, size, minimum, description):
    points = np.asarray(values, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum
            or not np.isfinite(points).all() or np.any(points < 0)
            or np.any(points > size)):
        raise ValueError(f"{description} needs at least {minimum} finite XY points inside the image.")
    return np.rint(points).astype(np.int32)


def rasterize_annotation(annotation, size):
    """Paint only reviewed regions; the rest is 255, never assumed background."""
    width, height = size
    if (annotation.get("format") != ANNOTATION_FORMAT
            or [annotation.get("width"), annotation.get("height")] != [width, height]
            or type(annotation.get("complete")) is not bool):
        raise ValueError("Annotation format, dimensions, or completeness does not match the source.")
    regions, strokes = annotation.get("regions", []), annotation.get("strokes", [])
    if not isinstance(regions, list) or not isinstance(strokes, list):
        raise ValueError("Annotation regions and strokes must be lists.")
    shapes = []
    for items, key, minimum in ((regions, "polygon", 3), (strokes, "points", 2)):
        for item in items:
            if not isinstance(item, dict) or item.get("class") not in LABELS:
                raise ValueError("Unknown structure annotation class.")
            points = checked_points(item.get(key), size, minimum, key)
            thickness = item.get("width") if key == "points" else None
            if key == "points" and (type(thickness) is not int or not 1 <= thickness <= max(size)):
                raise ValueError("Stroke width must be a positive integer no larger than the image.")
            holes = item.get("holes", [])
            if not isinstance(holes, list) or (thickness is not None and holes):
                raise ValueError("Only polygon regions can contain a list of holes.")
            holes = [checked_points(hole, size, 3, "Polygon hole") for hole in holes]
            shapes.append((item["class"], points, thickness, holes))
    mask = np.full((height, width), 255, np.uint8)
    for name in PAINT_ORDER:
        for class_name, points, thickness, holes in shapes:
            if class_name != name:
                continue
            if thickness is None:
                if holes:
                    region = np.zeros_like(mask)
                    cv2.fillPoly(region, [points], 1, lineType=cv2.LINE_8)
                    cv2.fillPoly(region, holes, 0, lineType=cv2.LINE_8)
                    mask[region != 0] = LABELS[name]
                else:
                    cv2.fillPoly(mask, [points], LABELS[name], lineType=cv2.LINE_8)
            else:
                cv2.polylines(mask, [points], False, LABELS[name], thickness, lineType=cv2.LINE_8)
    if np.all(mask == 255):
        raise ValueError("Annotation contains no supervised pixels.")
    return mask


def structure_checks(annotation):
    """Keep instance and passage checks separate from segmentation masks."""
    size = (annotation["width"], annotation["height"])
    instances = {}
    for kind in ("doors", "windows"):
        instances[kind] = []
        for item in annotation.get(kind, []):
            span = np.asarray(item.get("span"), dtype=np.float64)
            if span.shape != (4,):
                raise ValueError(f"Each {kind} span needs four coordinates.")
            points = checked_points(span.reshape(2, 2), size, 2, kind)
            if np.array_equal(points[0], points[1]):
                raise ValueError(f"A {kind} span cannot have zero length.")
            instances[kind].append({**item, "span": span.tolist()})
    rays, ray_ids = [], set()
    for item in annotation.get("rays", []):
        points = checked_points([item.get("from"), item.get("to")], size, 2, "Ray")
        ray_id = item.get("id")
        if (not isinstance(ray_id, str) or not ray_id.strip() or ray_id in ray_ids
                or item.get("expect") not in ("open", "blocked")
                or item.get("sense", "sight") not in ("sight", "movement")
                or np.array_equal(points[0], points[1])):
            raise ValueError("Ray ID, endpoints, expectation, or sense is invalid.")
        ray_ids.add(ray_id)
        rays.append(item)
    complete = annotation.get("complete_instances", {})
    if not isinstance(complete, dict) or any(type(value) is not bool for value in complete.values()):
        raise ValueError("Instance completeness flags must be boolean.")
    return {"format": "auto-wall-structure-checks-v1", "reference_size": list(size),
            "complete_instances": {kind: complete.get(kind, False) for kind in instances},
            **instances, "rays": rays,
            "description": "Partial visual annotations; unreviewed instances are not false positives."}


def source_image(source_root, site):
    if not isinstance(site, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", site):
        raise ValueError("A site must be a simple lowercase folder name.")
    folder = Path(source_root).resolve() / site
    candidates = [folder / "source.png", folder / f"{site}-v1.png"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"Expected one canonical PNG source for {site}.")
    path = existing[0].resolve()
    if not path.is_relative_to(Path(source_root).resolve()):
        raise ValueError("Canonical image must remain inside the source directory.")
    return path


def prepare_dataset(source_root, annotation_dir, split_file, output, replay=None, review_file=None,
                    replay_weight=1.0):
    """Verify originals and frozen splits before writing a new BMP-only dataset."""
    source_root, annotation_dir = Path(source_root).resolve(), Path(annotation_dir).resolve()
    if type(replay_weight) not in (int, float) or not np.isfinite(replay_weight) or replay_weight <= 0:
        raise ValueError("Replay sampling weight must be a finite positive number.")
    if replay_weight != 1 and replay is None:
        raise ValueError("A replay sampling weight requires a replay manifest.")
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source_root):
        raise ValueError("Dataset output must be a new directory outside the original maps.")
    if review_file is None:
        raise ValueError("A separate, source- and annotation-matched review record is required.")
    review = read_json(review_file)
    if (not isinstance(review, dict) or review.get("format") != "auto-wall-annotation-review-v1"
            or not isinstance(review.get("samples"), dict)):
        raise ValueError("Invalid annotation review record.")
    splits = read_json(split_file)
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Frozen splits must map each site to train, validation, or test.")
    prepared, samples, hashes = [], [], {}
    for site, split in sorted(splits.items()):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {site}.")
        path = source_image(source_root, site)
        source_bytes = path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        previous_split = hashes.setdefault(source_hash, split)
        if previous_split != split:
            raise ValueError("Identical source images cannot appear in different splits.")
        image = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode source image: {site}.")
        annotation_path = annotation_dir / f"{site}.json"
        annotation = read_json(annotation_path)
        if annotation.get("site") != site or annotation.get("image_sha256", "").lower() != source_hash:
            raise ValueError(f"Source identity/hash does not match annotations: {site}.")
        mask = rasterize_annotation(annotation, image.shape[1::-1])
        annotation_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()
        approval = review["samples"].get(site, {})
        if (approval.get("status") != "accepted"
                or approval.get("source_sha256") != source_hash
                or approval.get("annotation_sha256") != annotation_hash
                or approval.get("mask_sha256") != mask_hash
                or not isinstance(approval.get("reviewer"), str) or not approval["reviewer"].strip()
                or not isinstance(approval.get("notes"), str) or not approval["notes"].strip()):
            raise ValueError(f"Missing, rejected, or stale annotation review: {site}.")
        checks = structure_checks(annotation)
        classes, counts = np.unique(mask, return_counts=True)
        sample = {"id": site, "group": "encounter-" + site, "split": split,
                  "image": str(path), "mask": site + "-labels.bmp",
                  "structure_annotations": site + "-checks.json",
                  "annotation": "Partial source annotations; uncertain pixels ignored. See the recorded review scope and limitations.",
                  "annotation_complete": annotation["complete"],
                  "provenance": {"source_sha256": source_hash,
                                 "annotation_sha256": annotation_hash, "mask_sha256": mask_hash},
                  "review": approval,
                  "labelled_pixels": {str(int(key)): int(value) for key, value in zip(classes, counts)}}
        prepared.append((site, mask, checks))
        samples.append(sample)
    if replay:
        manifest = Path(replay).resolve()
        old = read_json(manifest)
        if old.get("format") != "auto-wall-training-v1" or old.get("classes") != list(LABELS)[:5]:
            raise ValueError("Replay requires an existing five-class segmentation dataset.")
        for item in old["samples"]:
            # Synthetic generation has its own controlled split; replay only the real older layouts.
            if item.get("synthetic") or item["id"].startswith("synthetic-"):
                continue
            sample = {**item}
            if sample["split"] == "train":
                sample["sampling_weight"] = float(replay_weight)
            for key in ("image", "mask", "geometry", "structure_annotations"):
                if key in sample:
                    sample[key] = str((manifest.parent / sample[key]).resolve())
            sample["annotation"] += " Legacy replay: IDs 0–4 unchanged; window labels were not exhaustively reviewed."
            samples.append(sample)
    ids, groups = set(), {}
    for sample in samples:
        if sample["id"] in ids or groups.setdefault(sample["group"], sample["split"]) != sample["split"]:
            raise ValueError("Replay and encounter samples must have unique IDs and disjoint layout splits.")
        ids.add(sample["id"])
    output.mkdir(parents=True)
    for site, mask, checks in prepared:
        success, encoded = cv2.imencode(".bmp", mask)
        if not success:
            raise OSError(f"Cannot encode annotations: {site}.")
        (output / f"{site}-labels.bmp").write_bytes(encoded.tobytes())
        (output / f"{site}-checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"format": "auto-wall-training-v1", "classes": list(LABELS)[:6],
        "annotation_review": "Partial annotations admitted only with a separate review matching source, annotation, and raster hashes; not exhaustive human ground truth.",
        "review_sha256": hashlib.sha256(Path(review_file).read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(Path(split_file).read_bytes()).hexdigest(),
        "samples": samples}, indent=2) + "\n", encoding="utf-8")
    return manifest
