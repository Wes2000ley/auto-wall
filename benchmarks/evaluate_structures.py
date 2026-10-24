"""Evaluate structure labels and exported wall, door, window, and barrier geometry."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.check_geometry import match_instances, ray_blocked
from src.core.detection_profiles import load_settings
from src.core.fast_pipeline import FastSettings, detect_map
from src.ml.data import read_dataset
from src.ml.model import load_model, predict
from src.ml.network import LEGACY_CLASSES, WINDOW_CLASSES, validate_classes


FORMAT = "auto-wall-structure-checks-v1"
METRICS_FORMAT = "auto-wall-structure-evaluation-v1"
DEFAULT_APPEARANCE_SEED = 20261024
STRUCTURAL_CLASSES = {"wall", "door", "transparent_barrier", "window"}
OVERLAY_COLORS = {
    1: (60, 230, 60),       # wall
    2: (255, 70, 255),      # door
    3: (40, 220, 255),      # light
    4: (255, 180, 40),      # transparent barrier
    5: (40, 220, 255),      # window
}


def confusion_counts(truth, prediction, class_count):
    """Count supervised pixels only; label 255 is explicitly unreviewed."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if (truth.ndim != 2 or prediction.shape != truth.shape or truth.dtype != np.uint8
            or prediction.dtype != np.uint8 or type(class_count) is not int or class_count < 2):
        raise ValueError("Truth and prediction must be matching uint8 label maps.")
    if not np.isin(truth, [*range(class_count), 255]).all():
        raise ValueError("Truth contains a class outside the selected dataset profile.")
    if not np.isin(prediction, range(class_count)).all():
        raise ValueError("Prediction contains a class outside the evaluation profile.")
    reviewed = truth != 255
    encoded = truth[reviewed].astype(np.int64) * class_count + prediction[reviewed]
    return np.bincount(encoded, minlength=class_count ** 2).reshape(class_count, class_count)


def metrics_from_confusion(matrix, classes):
    """Report class metrics and a structural-only macro that excludes background."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or np.any(matrix < 0):
        raise ValueError("Confusion matrix does not match the class labels.")
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        actual = int(matrix[index, :].sum())
        false_positive, false_negative = predicted - true_positive, actual - true_positive
        union = true_positive + false_positive + false_negative
        f05_denominator = 1.25 * true_positive + 0.25 * false_negative + false_positive
        per_class[name] = {
            "precision": true_positive / predicted if predicted else None,
            "recall": true_positive / actual if actual else None,
            "f0_5": 1.25 * true_positive / f05_denominator if f05_denominator else None,
            "iou": true_positive / union if union else None,
            "true_positive_pixels": true_positive,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": false_negative,
            "predicted_pixels": predicted,
            "truth_pixels": actual,
        }
    macro = {}
    structural_names = [name for name in classes if name in STRUCTURAL_CLASSES]
    for metric in ("precision", "recall", "f0_5", "iou"):
        values = [per_class[name][metric] for name in structural_names
                  if per_class[name][metric] is not None]
        macro[metric] = float(np.mean(values)) if values else None
    return {
        "confusion_matrix": matrix.tolist(),
        "reviewed_pixels": int(matrix.sum()),
        "per_class": per_class,
        "structural_macro": macro,
    }


def evaluate_pixels(truth, prediction, classes):
    classes = validate_classes(classes)
    return metrics_from_confusion(confusion_counts(truth, prediction, len(classes)), classes)


def background_false_structural(truth, prediction, classes):
    """Count predicted structures only inside pixels explicitly labelled background."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    background = truth == 0
    counts = {name: int(np.count_nonzero(background & (prediction == index)))
              for index, name in enumerate(classes) if name in STRUCTURAL_CLASSES}
    total = int(background.sum())
    false_count = sum(counts.values())
    return {
        "explicit_background_pixels": total,
        "false_structural_pixels": false_count,
        "false_structural_rate": false_count / total if total else None,
        "per_class_pixels": counts,
    }


def match_spans(predicted, annotations, complete, tolerance):
    """Match straight spans; unmatched predictions stay unassessed for partial labels."""
    if type(complete) is not bool or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Span completeness must be boolean and tolerance positive.")
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2, 2)
    truth = np.asarray(annotations, dtype=float).reshape(-1, 2, 2)
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("Door and window spans must contain finite coordinates.")
    direct = np.linalg.norm(truth[:, None] - predicted[None], axis=-1).max(axis=-1)
    reverse = np.linalg.norm(truth[:, None] - predicted[None, :, ::-1], axis=-1).max(axis=-1)
    result = match_instances(np.minimum(direct, reverse), tolerance)
    precision = result["precision"] if complete else None
    recall = result["recall"]
    f05 = None
    if complete:
        false_positive = result["prediction_count"] - result["matched"]
        false_negative = result["truth_count"] - result["matched"]
        denominator = 1.25 * result["matched"] + 0.25 * false_negative + false_positive
        f05 = 1.25 * result["matched"] / denominator if denominator else None
    output = {
        "matched": result["matched"],
        "truth_count": result["truth_count"],
        "prediction_count": result["prediction_count"],
        "complete_annotation": complete,
        "tolerance_pixels": float(tolerance),
        "precision": precision,
        "recall": recall,
        "f0_5": f05,
        "mean_endpoint_error_pixels": result["mean_error"],
        "missed_annotated_indices": result["missed"],
    }
    if complete:
        output["false_positive_prediction_indices"] = result["extra"]
    else:
        output["unassessed_prediction_indices"] = result["extra"]
    return output


def _scaled_point(point, scale):
    value = np.asarray(point, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("Ray endpoints must be finite [x, y] coordinates.")
    return value * scale


def evaluate_structure_annotations(detection, specification, image_shape):
    """Evaluate optional span and sight/movement annotations in their reference space."""
    if not isinstance(specification, dict) or specification.get("format") != FORMAT:
        raise ValueError(f"Structure annotations must use format {FORMAT}.")
    structure_free = specification.get("structure_free", False)
    if type(structure_free) is not bool:
        raise ValueError("structure_free must be boolean.")
    height, width = image_shape[:2]
    reference = np.asarray(specification.get("reference_size", [width, height]), dtype=float)
    if (reference.shape != (2,) or not np.isfinite(reference).all()
            or np.any(reference <= 0)):
        raise ValueError("reference_size must be positive [width, height].")
    scale = np.asarray([width, height], dtype=float) / reference
    complete = specification.get("complete_instances", {})
    if not isinstance(complete, dict):
        raise ValueError("complete_instances must be an object.")
    spans = {}
    for name, detected in (("doors", detection.doors), ("windows", getattr(detection, "windows", []))):
        raw = specification.get(name, [])
        if not isinstance(raw, list):
            raise ValueError(f"{name} annotations must be a list.")
        expected = []
        for item in raw:
            if not isinstance(item, dict) or "span" not in item:
                raise ValueError(f"Each {name} annotation needs a span.")
            span = np.asarray(item["span"], dtype=float)
            if span.shape != (4,) or not np.isfinite(span).all():
                raise ValueError(f"Each {name} span must be [x1, y1, x2, y2].")
            if "note" in item and not isinstance(item["note"], str):
                raise ValueError("Span notes must be strings.")
            expected.append((span.reshape(2, 2) * scale).tolist())
        if name in complete and type(complete[name]) is not bool:
            raise ValueError(f"complete_instances.{name} must be boolean.")
        spans[name] = match_spans(detected, expected, complete.get(name, False),
                                  float(specification.get(f"{name[:-1]}_tolerance", 8)))

    rays = specification.get("rays", [])
    if not isinstance(rays, list):
        raise ValueError("rays must be a list.")
    seen_ids = set()
    closed_doors = detection.doors
    opaque = [*detection.walls, *closed_doors]
    faces = getattr(detection, "window_faces", [])
    window_barriers = ([face for group in faces for face in group]
                       if faces else getattr(detection, "windows", []))
    movement_barriers = [*opaque, *window_barriers,
                         *getattr(detection, "invisible_walls", [])]
    evaluated_rays = []
    for item in rays:
        if not isinstance(item, dict):
            raise ValueError("Each ray annotation must be an object.")
        ray_id, expected = item.get("id"), item.get("expect")
        sense = item.get("sense", "sight")
        if (not isinstance(ray_id, str) or not ray_id.strip() or ray_id in seen_ids
                or expected not in ("open", "blocked") or sense not in ("sight", "movement")):
            raise ValueError("Ray IDs must be unique; expect must be open/blocked and sense sight/movement.")
        seen_ids.add(ray_id)
        start = _scaled_point(item.get("from"), scale)
        end = _scaled_point(item.get("to"), scale)
        actual = ray_blocked(start, end, opaque if sense == "sight" else movement_barriers)
        expected_blocked = expected == "blocked"
        evaluated_rays.append({"id": ray_id, "sense": sense, "expect": expected,
                               "actual": "blocked" if actual else "open",
                               "passed": actual == expected_blocked})
    result = {
        "reference_size": reference.tolist(),
        "coordinate_size": [width, height],
        "spans": spans,
        "rays": evaluated_rays,
        "ray_summary": {
            sense: {"passed": sum(ray["passed"] for ray in evaluated_rays if ray["sense"] == sense),
                    "total": sum(ray["sense"] == sense for ray in evaluated_rays)}
            for sense in ("sight", "movement")
        },
        "ray_assumption": "Sight rays use opaque walls and assume doors are closed. Movement rays also use exported windows and movement-only barriers. Review-only transparent-barrier candidates are excluded because they are not exported. Window sight blocking follows Foundry proximity-window semantics only approximately; per-world proximity distance is not modeled.",
        "span_metric_caveat": "Recall is measured only over annotated spans. With partial span annotations, missing unannotated instances are unknown and unmatched detections are not treated as false positives.",
    }
    if structure_free:
        if any(specification.get(name, []) for name in ("doors", "windows")) or any(
                ray["expect"] == "blocked" for ray in rays):
            raise ValueError("A structure-free annotation cannot contain positive spans or blocked rays.")
        counts = {"opaque_wall_segments": len(detection.walls),
                  "door_spans": len(detection.doors),
                  "window_spans": len(getattr(detection, "windows", [])),
                  "window_face_segments": len(window_barriers),
                  "movement_only_segments": len(getattr(detection, "invisible_walls", []))}
        result["structure_free"] = {"passed": not any(counts.values()), "exported_counts": counts,
                                    "scope": "All exported structures across the full image, including pixels with unknown segmentation labels. Lights and review-only barrier candidates are not assessed."}
    return result


def _mean_defined(values):
    selected = [value for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def make_appearance_transform(mode="original", seed=DEFAULT_APPEARANCE_SEED):
    """Build one reproducible, image-only color transform for a paired run."""
    if mode == "original":
        return {"mode": "original", "seed": None, "operation": "identity"}
    if mode != "recolored":
        raise ValueError("Appearance must be original or recolored.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Appearance seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    mixing = np.array([[.55, .30, .15], [.15, .60, .25], [.25, .10, .65]], dtype=np.float32)
    mixing += rng.uniform(-.04, .04, size=(3, 3)).astype(np.float32)
    mixing = np.maximum(mixing, .04)
    mixing /= mixing.sum(axis=1, keepdims=True)
    gains = np.array([1.25, .72, 1.12], dtype=np.float32) * rng.uniform(.95, 1.05, size=3)
    tint = np.array([18, -8, 10], dtype=np.float32) + rng.integers(-4, 5, size=3)
    return {
        "mode": mode,
        "name": "seeded_bgr_mix_gain_tint_v1",
        "seed": seed,
        "channel_order_bgr": rng.permutation(3).tolist(),
        "mixing_matrix": mixing.tolist(),
        "gains_bgr": gains.tolist(),
        "tint_bgr": tint.tolist(),
        "formula": "clip((mix(permuted_BGR) * gains_BGR) + tint_BGR, 0, 255), rounded to uint8",
        "paired_test_caveat": "This deterministic global color transform tests color robustness on the same maps; it is not an independent art style or map-layout evaluation.",
    }


def recolor_image(image, transform):
    """Apply an affine per-pixel BGR transform without moving or painting labels."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size):
        raise ValueError("Appearance transforms require a nonempty uint8 BGR image.")
    if not isinstance(transform, dict) or transform.get("mode") not in ("original", "recolored"):
        raise ValueError("Invalid appearance transform.")
    if transform["mode"] == "original":
        return image.copy()
    order = np.asarray(transform.get("channel_order_bgr"), dtype=int)
    matrix = np.asarray(transform.get("mixing_matrix"), dtype=np.float32)
    gains = np.asarray(transform.get("gains_bgr"), dtype=np.float32)
    tint = np.asarray(transform.get("tint_bgr"), dtype=np.float32)
    if (order.shape != (3,) or sorted(order.tolist()) != [0, 1, 2]
            or matrix.shape != (3, 3) or gains.shape != (3,) or tint.shape != (3,)
            or not np.isfinite(matrix).all() or not np.isfinite(gains).all()
            or not np.isfinite(tint).all() or np.any(gains <= 0)):
        raise ValueError("Recolored appearance parameters are invalid.")
    mixed = np.einsum("...c,oc->...o", image[..., order].astype(np.float32), matrix)
    return np.clip(mixed * gains + tint, 0, 255).round().astype(np.uint8)


def separate_output_directory(output, samples, manifest_path, appearance):
    """Keep generated evaluation artifacts away from image and label sources."""
    output = Path(output).resolve()
    for sample in samples:
        for key in ("image", "mask"):
            source = Path(sample[key])
            if not source.is_absolute():
                source = manifest_path.parent / source
            source = source.resolve(strict=True)
            if output == source.parent or output.is_relative_to(source.parent) or source.is_relative_to(output):
                raise ValueError("Choose a separate output directory outside the source image and label folders.")
    report_path = output / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_appearance = previous.get("experiment", {}).get("appearance", {"mode": "original"})
        if (previous.get("format") == METRICS_FORMAT
                and (previous_appearance.get("mode") != appearance.get("mode")
                     or previous_appearance.get("seed") != appearance.get("seed"))):
            raise ValueError("This output directory contains another appearance run; choose a separate output directory.")
    return output


def summarize_maps(sample_results, classes, metric_name):
    matrices = [np.asarray(sample["segmentation"][metric_name]["confusion_matrix"], dtype=np.int64)
                for sample in sample_results]
    aggregate = metrics_from_confusion(np.sum(matrices, axis=0), classes)
    grouped = {}
    for sample, matrix in zip(sample_results, matrices):
        group = sample.get("group", sample["id"])
        grouped[group] = grouped.get(group, np.zeros_like(matrix)) + matrix
    group_metrics = [metrics_from_confusion(matrix, classes) for matrix in grouped.values()]
    per_class = {}
    for name in classes:
        per_class[name] = {
            metric: _mean_defined([result["per_class"][name][metric] for result in group_metrics])
            for metric in ("precision", "recall", "f0_5", "iou")
        }
    structural_macro = {
        metric: _mean_defined([result["structural_macro"][metric] for result in group_metrics])
        for metric in ("precision", "recall", "f0_5", "iou")
    }
    return {"map_macro": {"per_class": per_class, "structural_macro": structural_macro,
                           "maps": len(grouped), "samples": len(sample_results)},
            "aggregate": aggregate}


def summarize_background_false_structural(sample_results, metric_name, classes):
    entries = [sample["segmentation"]["background_false_structural"][metric_name]
               for sample in sample_results]
    background_pixels = sum(item["explicit_background_pixels"] for item in entries)
    structural_names = [name for name in classes if name in STRUCTURAL_CLASSES]
    per_class = {name: sum(item["per_class_pixels"][name] for item in entries)
                 for name in structural_names}
    false_pixels = sum(per_class.values())
    return {"explicit_background_pixels": background_pixels,
            "false_structural_pixels": false_pixels,
            "false_structural_rate": false_pixels / background_pixels if background_pixels else None,
            "per_class_pixels": per_class}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_structure_annotations(sample, manifest_path):
    reference = sample.get("structure_annotations")
    if reference is None:
        return None, None, None
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Invalid structure_annotations path for sample {sample['id']}.")
    path = Path(reference)
    if not path.is_absolute():
        path = manifest_path.parent / path
    content = path.read_bytes()
    return json.loads(content.decode("utf-8-sig")), str(path.resolve()), hashlib.sha256(content).hexdigest()


def read_regression_checks(path, samples):
    """Read extra manual checks without changing the training labels or splits."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (not isinstance(data, dict) or data.get("format") != "auto-wall-structure-regressions-v1"
            or not isinstance(data.get("samples"), dict) or not data["samples"]):
        raise ValueError("Regression checks need the auto-wall-structure-regressions-v1 format and samples.")
    by_id = {sample["id"]: sample for sample in samples}
    result = {}
    for sample_id, item in data["samples"].items():
        if sample_id not in by_id:
            raise ValueError(f"Regression checks reference an unknown sample: {sample_id}.")
        if (not isinstance(item, dict)
                or item.get("source_pixels_sha256") != by_id[sample_id]["pixels_sha256"]):
            raise ValueError(f"Regression checks do not match source pixels: {sample_id}.")
        checks = item.get("checks")
        if not isinstance(checks, dict) or checks.get("format") != FORMAT:
            raise ValueError(f"Invalid structure regression checks: {sample_id}.")
        result[sample_id] = checks
    return result


def _settings_for_model(settings_path, profile, threshold):
    base = FastSettings(ml_profile=profile)
    settings = load_settings(settings_path, base=base) if settings_path else base
    settings = replace(settings, model_path="", ml_profile=profile)
    if threshold is not None:
        settings = replace(settings, ml_wall_confidence=threshold, ml_door_confidence=threshold,
                           ml_light_confidence=threshold, ml_window_confidence=threshold,
                           ml_barrier_confidence=threshold)
    settings.validate()
    return settings


def _working_image(image, settings):
    height, width = image.shape[:2]
    scale = min(1.0, settings.max_dimension / max(width, height)) if settings.max_dimension else 1.0
    if scale == 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    exact_palette = ((settings.wall_mode == "color" and settings.wall_tolerance == 0)
                     or (settings.door_match == "rgb" and settings.door_color_tolerance == 0))
    interpolation = cv2.INTER_NEAREST if exact_palette else cv2.INTER_AREA
    return cv2.resize(image, size, interpolation=interpolation)


def _write_bmp(path, image):
    success, encoded = cv2.imencode(".bmp", image)
    if not success:
        raise ValueError(f"Could not write review bitmap: {path.name}")
    path.write_bytes(encoded.tobytes())


def _preview_banner(image, title, detail):
    preview = np.full((image.shape[0] + 84, max(image.shape[1], 1080), 3), 24, np.uint8)
    preview[84:, :image.shape[1]] = image
    cv2.putText(preview, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, detail, (16, 63), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (220, 220, 220), 1, cv2.LINE_AA)
    return preview


def _write_sample_visuals(output, sample_id, image, truth, raw_labels, confidence, threshold,
                          classes, detection=None):
    size = (image.shape[1], image.shape[0])
    labels = cv2.resize(raw_labels, size, interpolation=cv2.INTER_NEAREST)
    scores = cv2.resize(confidence, size, interpolation=cv2.INTER_NEAREST)
    strict = np.where(scores >= threshold, labels, 0).astype(np.uint8)
    overlay = image.copy()
    for class_id, color in OVERLAY_COLORS.items():
        if class_id < len(classes):
            overlay[strict == class_id] = color
    review = cv2.addWeighted(image, 0.35, overlay, 0.65, 0)
    errors = image.copy()
    reviewed = truth != 255
    errors[reviewed & (truth == strict)] = (50, 200, 60)
    errors[reviewed & (truth != strict)] = (50, 50, 220)
    _write_bmp(output / f"{sample_id}-raw-labels.bmp", labels)
    _write_bmp(output / f"{sample_id}-thresholded-labels.bmp", strict)
    _write_bmp(output / f"{sample_id}-review.bmp", _preview_banner(
        review, "MODEL PREDICTION - not training labels",
        f"{sample_id} | confidence cutoff {threshold:g} | green wall, purple door, cyan fence, yellow window"))
    _write_bmp(output / f"{sample_id}-pixel-errors.bmp", _preview_banner(
        errors, "MODEL PREDICTION ERRORS - partial annotation coverage",
        f"{sample_id} | green correct, red incorrect on reviewed pixels; unreviewed pixels unchanged"))
    if detection is not None:
        geometry = image.copy()
        window_spans = ([span for group in detection.window_faces for span in group]
                        if detection.window_faces else detection.windows)
        for segments, color in ((detection.walls, OVERLAY_COLORS[1]),
                                (detection.doors, OVERLAY_COLORS[2]),
                                (detection.invisible_walls, OVERLAY_COLORS[4]),
                                (window_spans, OVERLAY_COLORS[5])):
            for start, end in segments:
                cv2.line(geometry, tuple(round(v) for v in start), tuple(round(v) for v in end),
                         color, 2, cv2.LINE_AA)
        _write_bmp(output / f"{sample_id}-geometry.bmp", _preview_banner(
            geometry, "EXTRACTED MODEL GEOMETRY - experimental output",
            f"{sample_id} | green wall, purple door, cyan movement barrier, yellow exported window faces"))
    with (output / f"{sample_id}-confidence.npy").open("wb") as stream:
        np.save(stream, scores.astype(np.float32), allow_pickle=False)


def evaluate_sample(model, sample, classes, settings, threshold, runs, device):
    image, labels = sample["pixels"], sample["labels"]
    working = _working_image(image, settings)
    working_truth = cv2.resize(labels, (working.shape[1], working.shape[0]),
                               interpolation=cv2.INTER_NEAREST) if labels.shape != working.shape[:2] else labels
    timings = []
    first_seconds = None
    prediction = None
    light_centers = settings.lights
    for run in range(runs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = predict(model, working, include_light_centers=light_centers)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        if run == 0:
            first_seconds = duration
        else:
            timings.append(duration)
    raw_labels, confidence = prediction[:2]
    thresholded = np.where(confidence >= threshold, raw_labels, 0).astype(np.uint8)
    pixel_metrics = {
        "argmax": evaluate_pixels(working_truth, raw_labels, classes),
        "thresholded": evaluate_pixels(working_truth, thresholded, classes),
        "background_false_structural": {
            "argmax": background_false_structural(working_truth, raw_labels, classes),
            "thresholded": background_false_structural(working_truth, thresholded, classes),
        },
    }
    detect_started = time.perf_counter()
    detection = detect_map(image, settings, prediction=prediction)
    detection_seconds = time.perf_counter() - detect_started
    result = {
        "id": sample["id"], "split": sample["split"], "group": sample["group"],
        "source_size": [image.shape[1], image.shape[0]],
        "detection_size": [working.shape[1], working.shape[0]],
        "timings": {"first_inference_seconds": first_seconds, "warm_inference_seconds": timings,
                    "median_warm_inference_seconds": statistics.median(timings),
                    "geometry_seconds": detection_seconds, "pipeline_timings": detection.timings},
        "segmentation": pixel_metrics,
        "geometry": {"opaque_wall_segments": len(detection.walls),
                     "door_spans": len(detection.doors),
                     "window_spans": len(getattr(detection, "windows", [])),
                     "transparent_barrier_segments": len(getattr(detection, "transparent_barriers", [])),
                     "movement_only_segments": len(getattr(detection, "invisible_walls", [])),
                     "lights": len(detection.lights)},
        "class_labels": list(classes),
        "output_threshold": threshold,
    }
    return result, detection, raw_labels, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runs", type=int, default=3, help="Warm inference runs per map (plus one first run).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Pixel cutoff; also applies to geometry unless --geometry-thresholds profile.")
    parser.add_argument("--geometry-thresholds", choices=("common", "profile"), default="common",
                        help="Use the common cutoff or preserve the runtime profile's per-class cutoffs.")
    parser.add_argument("--settings", type=Path,
                        help="Detector settings used for geometry after model predictions are cached.")
    parser.add_argument("--regression-checks", type=Path,
                        help="Optional extra manual passage checks matched to source-pixel hashes.")
    parser.add_argument("--appearance", choices=("original", "recolored"), default="original",
                        help="Optional paired color-shift stress run; use a separate output directory.")
    parser.add_argument("--appearance-seed", type=int, default=DEFAULT_APPEARANCE_SEED,
                        help="Seed for the fixed image-wide channel-mixing transform.")
    args = parser.parse_args()
    if min(args.runs, args.threads) < 1:
        parser.error("Runs and threads must be positive.")
    if not np.isfinite(args.threshold) or not .5 <= args.threshold <= 1:
        parser.error("Threshold must be between 0.5 and 1.")
    torch.set_num_threads(args.threads)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("format") != "auto-wall-training-v1":
        parser.error("Manifest must use format auto-wall-training-v1.")
    try:
        dataset_classes = validate_classes(manifest.get("classes", ()))
    except ValueError as error:
        parser.error(str(error))
    samples = read_dataset(manifest_path, classes=dataset_classes)
    try:
        regressions = read_regression_checks(args.regression_checks, samples) if args.regression_checks else {}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.split != "all":
        samples = [sample for sample in samples if sample["split"] == args.split]
    if not samples:
        parser.error(f"No samples found for split {args.split}.")

    load_started = time.perf_counter()
    model, model_details = load_model(args.model, args.device)
    model_load_seconds = time.perf_counter() - load_started
    model_classes = validate_classes(model.classes)
    if model_classes[:len(LEGACY_CLASSES)] != LEGACY_CLASSES:
        parser.error("The model and dataset must use the canonical Auto-Wall class order.")
    classes = WINDOW_CLASSES if WINDOW_CLASSES in (dataset_classes, model_classes) else LEGACY_CLASSES
    profile = "windows" if model_classes == WINDOW_CLASSES else "legacy"
    settings = _settings_for_model(args.settings, profile,
                                   args.threshold if args.geometry_thresholds == "common" else None)
    try:
        appearance = make_appearance_transform(args.appearance, args.appearance_seed)
    except ValueError as error:
        parser.error(str(error))
    try:
        args.output = separate_output_directory(args.output, samples, manifest_path, appearance)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "format": METRICS_FORMAT,
        "model": {"path": str(args.model.resolve()), "sha256": _sha256(args.model),
                  "classes": list(model_classes), "details": model_details,
                  "device": str(device), "load_seconds": model_load_seconds},
        "dataset": {"path": str(manifest_path), "sha256": _sha256(manifest_path),
                    "classes": list(dataset_classes), "evaluated_classes": list(classes)},
        "experiment": {"split": args.split, "threshold": args.threshold,
                       "geometry_thresholds": args.geometry_thresholds,
                       "runs": args.runs, "threads": args.threads,
                       "runtime_profile": profile, "settings": asdict(settings),
                       "regression_checks_sha256": _sha256(args.regression_checks) if args.regression_checks else None,
                       "appearance": appearance},
        "software": {"python": sys.version, "torch": torch.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__,
                     "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): _sha256(path)
                         for path in [Path(__file__),
                                      *(Path(__file__).resolve().parents[1] / "src/ml").glob("*.py"),
                                      *(Path(__file__).resolve().parents[1] / "src/core").glob("*.py")]}},
        "metric_caveat": "Metrics cover only pixels explicitly labelled in the supplied masks; 255 pixels are ignored. Precision and map-wide accuracy are not claims about unreviewed parts of each map.",
        "confidence_caveat": "The confidence cutoff is a selection threshold, not a calibrated probability of accuracy.",
        "samples": [],
    }
    selected_ids = {sample["id"] for sample in samples}
    for sample in samples:
        image_path = Path(sample["image"])
        label_path = Path(sample["mask"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not label_path.is_absolute():
            label_path = manifest_path.parent / label_path
        evaluated_image = recolor_image(sample["pixels"], appearance)
        evaluated_sample = {**sample, "pixels": evaluated_image}
        result, detection, raw_labels, confidence = evaluate_sample(
            model, evaluated_sample, classes, settings, args.threshold, args.runs, device)
        result["source_hashes"] = {
            "image_file_sha256": _sha256(image_path),
            "decoded_image_sha256": sample["pixels_sha256"],
            "label_file_sha256": _sha256(label_path),
            "decoded_label_sha256": sample["labels_sha256"],
            "evaluated_image_sha256": hashlib.sha256(evaluated_image.tobytes()).hexdigest(),
        }
        specification, annotation_path, annotation_hash = _read_structure_annotations(sample, manifest_path)
        if specification is not None:
            result["source_hashes"]["structure_annotations_sha256"] = annotation_hash
            result["structure_annotations_path"] = annotation_path
            result["geometry_annotations"] = evaluate_structure_annotations(
                detection, specification, evaluated_image.shape)
        if sample["id"] in regressions:
            result["regression_checks"] = evaluate_structure_annotations(
                detection, regressions[sample["id"]], evaluated_image.shape)
        _write_sample_visuals(args.output, sample["id"], evaluated_image, sample["labels"],
                              raw_labels, confidence, args.threshold, classes, detection)
        (args.output / f"{sample['id']}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["samples"].append(result)
        print(json.dumps({"id": result["id"], "split": result["split"],
                          "segmentation": result["segmentation"]["thresholded"]["structural_macro"],
                          "geometry": result["geometry"], "timings": result["timings"]}), flush=True)
    if len(selected_ids) != len(report["samples"]):
        raise ValueError("Selected manifest sample IDs are not unique.")
    report["summary"] = {
        "thresholded": summarize_maps(report["samples"], classes, "thresholded"),
        "argmax": summarize_maps(report["samples"], classes, "argmax"),
        "background_false_structural": {
            metric: summarize_background_false_structural(report["samples"], metric, classes)
            for metric in ("argmax", "thresholded")
        },
        "maps": len({sample["group"] for sample in report["samples"]}),
        "samples": len(report["samples"]),
    }
    if device.type == "cuda":
        report["model"]["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
