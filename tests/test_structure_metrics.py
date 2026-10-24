import numpy as np
import pytest
import cv2
import json
import hashlib

from benchmarks.evaluate_structures import (
    background_false_structural,
    DEFAULT_APPEARANCE_SEED,
    _settings_for_model,
    _write_sample_visuals,
    evaluate_sample,
    evaluate_pixels,
    evaluate_structure_annotations,
    make_appearance_transform,
    match_spans,
    metrics_from_confusion,
    recolor_image,
    read_regression_checks,
    summarize_maps,
)
from src.core.fast_pipeline import DetectionResult
from src.core.fast_pipeline import FastSettings, detect_map
from src.ml.network import BattlemapNet, LEGACY_CLASSES, WINDOW_CLASSES
import torch


def test_geometry_can_preserve_runtime_cutoffs_separately_from_pixel_metrics(tmp_path):
    common = _settings_for_model(None, "windows", .9)
    assert common.ml_window_confidence == common.ml_barrier_confidence == .9
    runtime = _settings_for_model(None, "windows", None)
    assert runtime.ml_wall_confidence == .9
    assert runtime.ml_window_confidence == .93 and runtime.ml_barrier_confidence == .97

    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"ml_window_confidence": .96, "ml_barrier_confidence": .98}), encoding="utf-8")
    custom = _settings_for_model(path, "windows", None)
    assert custom.ml_window_confidence == .96 and custom.ml_barrier_confidence == .98
    assert custom.ml_profile == "windows" and not custom.model_path


def test_pixel_metrics_ignore_unreviewed_pixels_and_score_nonstructural_false_positives():
    truth = np.array([[0, 1, 1, 255], [5, 0, 255, 255]], np.uint8)
    prediction = np.array([[2, 1, 0, 4], [0, 4, 2, 5]], np.uint8)
    result = evaluate_pixels(truth, prediction, WINDOW_CLASSES)

    assert result["reviewed_pixels"] == 5
    assert result["per_class"]["wall"]["precision"] == 1
    assert result["per_class"]["wall"]["recall"] == .5
    assert result["per_class"]["door"]["false_positive_pixels"] == 1
    assert result["per_class"]["window"]["false_negative_pixels"] == 1
    assert result["per_class"]["window"]["predicted_pixels"] == 0
    assert result["structural_macro"]["precision"] is not None
    false = background_false_structural(truth, prediction, WINDOW_CLASSES)
    assert false["explicit_background_pixels"] == 2
    assert false["false_structural_pixels"] == 2
    assert false["per_class_pixels"]["door"] == 1
    assert false["per_class_pixels"]["transparent_barrier"] == 1


def test_legacy_prediction_can_be_scored_against_six_class_truth_without_remapping():
    truth = np.array([[5, 0, 255]], np.uint8)
    legacy_prediction = np.array([[0, 1, 4]], np.uint8)
    result = evaluate_pixels(truth, legacy_prediction, WINDOW_CLASSES)

    assert result["per_class"]["window"]["truth_pixels"] == 1
    assert result["per_class"]["window"]["false_negative_pixels"] == 1
    assert result["per_class"]["window"]["predicted_pixels"] == 0
    with pytest.raises(ValueError, match="outside the selected"):
        evaluate_pixels(truth, np.array([[5, 0, 0]], np.uint8), LEGACY_CLASSES)


def test_metrics_undefined_when_a_class_has_no_truth_or_predictions():
    matrix = np.zeros((6, 6), dtype=np.int64)
    result = metrics_from_confusion(matrix, WINDOW_CLASSES)
    assert result["per_class"]["door"]["precision"] is None
    assert result["per_class"]["door"]["recall"] is None
    assert result["per_class"]["door"]["f0_5"] is None
    assert result["structural_macro"]["iou"] is None


def test_span_precision_is_unassessed_for_partial_labels():
    predicted = [((10, 0), (10, 20)), ((30, 0), (30, 20))]
    partial = match_spans(predicted, [[[10, 0], [10, 20]]], False, 2)
    assert partial["matched"] == 1 and partial["recall"] == 1
    assert partial["precision"] is None and partial["f0_5"] is None
    assert partial["unassessed_prediction_indices"] == [1]
    assert "false_positive_prediction_indices" not in partial


def test_complete_span_annotation_counts_extra_predictions_as_false_positives():
    result = match_spans([((10, 0), (10, 20)), ((30, 0), (30, 20))],
                         [[[10, 0], [10, 20]]], True, 2)
    assert result["precision"] == .5 and result["recall"] == 1
    assert result["false_positive_prediction_indices"] == [1]
    missed_all = match_spans([], [[[10, 0], [10, 20]]], True, 2)
    assert missed_all["precision"] is None and missed_all["recall"] == 0
    assert missed_all["f0_5"] == 0


def test_structure_evaluation_scales_spans_and_separates_sight_from_movement():
    detection = DetectionResult(
        200, 100,
        walls=[((100, 0), (100, 40)), ((100, 60), (100, 100))],
        doors=[((100, 40), (100, 60))],
        lights=[],
        invisible_walls=[((50, 0), (50, 100))],
        windows=[((150, 40), (150, 60))],
    )
    annotations = {
        "format": "auto-wall-structure-checks-v1",
        "reference_size": [100, 50],
        "doors": [{"span": [50, 20, 50, 30], "note": "door"}],
        "windows": [{"span": [75, 20, 75, 30]}],
        "rays": [
            {"id": "sight-wall", "from": [40, 10], "to": [60, 10], "expect": "blocked"},
            {"id": "sight-fence", "from": [20, 25], "to": [30, 25], "expect": "open"},
            {"id": "move-fence", "from": [20, 25], "to": [30, 25], "expect": "blocked", "sense": "movement"},
            {"id": "sight-window", "from": [70, 25], "to": [80, 25], "expect": "open"},
            {"id": "move-window", "from": [70, 25], "to": [80, 25], "expect": "blocked", "sense": "movement"},
        ],
    }
    result = evaluate_structure_annotations(detection, annotations, (100, 200, 3))

    assert result["spans"]["doors"]["matched"] == 1
    assert result["spans"]["windows"]["mean_endpoint_error_pixels"] == 0
    assert result["spans"]["doors"]["tolerance_pixels"] == 8
    assert result["ray_summary"]["sight"] == {"passed": 3, "total": 3}
    assert result["ray_summary"]["movement"] == {"passed": 2, "total": 2}


def test_review_only_barrier_candidates_do_not_block_exported_movement():
    detection = DetectionResult(100, 100, walls=[], doors=[], lights=[],
                                transparent_barriers=[((50, 0), (50, 100))])
    annotations = {"format": "auto-wall-structure-checks-v1", "reference_size": [100, 100],
                   "rays": [{"id": "review-only", "from": [40, 50], "to": [60, 50],
                             "sense": "movement", "expect": "open"}]}
    result = evaluate_structure_annotations(detection, annotations, (100, 100, 3))
    assert result["ray_summary"]["movement"] == {"passed": 1, "total": 1}


@pytest.mark.parametrize("field,segments", [
    ("walls", [((80, 80), (95, 95))]),
    ("doors", [((80, 80), (95, 95))]),
    ("windows", [((80, 80), (95, 95))]),
    ("window_faces", [[((80, 80), (95, 95))]]),
    ("invisible_walls", [((80, 80), (95, 95))]),
])
def test_structure_free_check_catches_exports_outside_sampled_rays(field, segments):
    detection = DetectionResult(100, 100, walls=[], doors=[], lights=[])
    setattr(detection, field, segments)
    annotations = {"format": "auto-wall-structure-checks-v1", "structure_free": True,
                   "rays": [{"id": "floor", "from": [10, 10], "to": [20, 10], "expect": "open"}]}
    result = evaluate_structure_annotations(detection, annotations, (100, 100, 3))
    assert result["ray_summary"]["sight"] == {"passed": 1, "total": 1}
    assert result["structure_free"]["passed"] is False
    assert any(result["structure_free"]["exported_counts"].values())


def test_structure_free_check_does_not_assess_lights_or_review_only_candidates():
    detection = DetectionResult(100, 100, walls=[], doors=[], lights=[{"x": 50, "y": 50}],
                                transparent_barriers=[((80, 80), (95, 95))])
    annotations = {"format": "auto-wall-structure-checks-v1", "structure_free": True}
    result = evaluate_structure_annotations(detection, annotations, (100, 100, 3))
    assert result["structure_free"]["passed"] is True
    assert not any(result["structure_free"]["exported_counts"].values())
    with pytest.raises(ValueError, match="boolean"):
        evaluate_structure_annotations(detection, {**annotations, "structure_free": "true"}, (100, 100, 3))
    with pytest.raises(ValueError, match="positive spans"):
        evaluate_structure_annotations(detection, {**annotations, "windows": [{"span": [1, 1, 2, 2]}]},
                                       (100, 100, 3))


def test_movement_checks_use_exported_window_faces_instead_of_editing_centerline():
    detection = DetectionResult(100, 100, walls=[], doors=[], lights=[],
                                windows=[((50, 20), (50, 80))],
                                window_faces=[[((45, 20), (45, 80)), ((55, 20), (55, 80))]])
    annotations = {"format": "auto-wall-structure-checks-v1", "rays": [
        {"id": "cross-one-face", "from": [40, 50], "to": [47, 50],
         "sense": "movement", "expect": "blocked"},
        {"id": "between-faces", "from": [47, 50], "to": [53, 50],
         "sense": "movement", "expect": "open"}]}
    result = evaluate_structure_annotations(detection, annotations, (100, 100, 3))
    assert result["ray_summary"]["movement"] == {"passed": 2, "total": 2}


def test_extra_regression_checks_require_matching_source_pixels_and_known_samples(tmp_path):
    path = tmp_path / "checks.json"
    checks = {"format": "auto-wall-structure-checks-v1", "rays": [
        {"id": "floor", "from": [10, 10], "to": [20, 10], "expect": "open"}]}
    data = {"format": "auto-wall-structure-regressions-v1", "samples": {
        "room": {"source_pixels_sha256": "a" * 64, "checks": checks}}}
    path.write_text(json.dumps(data), encoding="utf-8")
    samples = [{"id": "room", "pixels_sha256": "a" * 64}]
    assert read_regression_checks(path, samples) == {"room": checks}
    samples[0]["pixels_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source pixels"):
        read_regression_checks(path, samples)
    with pytest.raises(ValueError, match="unknown sample"):
        read_regression_checks(path, [{"id": "different", "pixels_sha256": "a" * 64}])


def test_map_macro_and_aggregate_are_both_reported():
    classes = LEGACY_CLASSES
    first = evaluate_pixels(np.array([[1, 0]], np.uint8), np.array([[1, 2]], np.uint8), classes)
    second = evaluate_pixels(np.array([[1, 1]], np.uint8), np.array([[0, 1]], np.uint8), classes)
    samples = [{"id": "first", "group": "one", "segmentation": {"strict": first}},
               {"id": "second", "group": "two", "segmentation": {"strict": second}}]
    result = summarize_maps(samples, classes, "strict")

    assert result["map_macro"]["maps"] == 2
    assert result["aggregate"]["per_class"]["wall"]["recall"] == 2 / 3
    assert result["map_macro"]["per_class"]["wall"]["recall"] == .75


def test_layout_variants_are_combined_before_map_macro_weighting():
    first = evaluate_pixels(np.array([[1, 0]], np.uint8), np.array([[1, 2]], np.uint8), LEGACY_CLASSES)
    second = evaluate_pixels(np.array([[1, 1]], np.uint8), np.array([[0, 1]], np.uint8), LEGACY_CLASSES)
    samples = [{"id": "original", "group": "same-layout", "segmentation": {"strict": first}},
               {"id": "upscale", "group": "same-layout", "segmentation": {"strict": second}}]

    result = summarize_maps(samples, LEGACY_CLASSES, "strict")

    assert result["map_macro"]["maps"] == 1
    assert result["map_macro"]["samples"] == 2
    assert result["map_macro"]["per_class"]["wall"]["recall"] == 2 / 3


def test_sample_evaluation_reuses_predictions_for_pipeline_geometry_and_writes_review_files(tmp_path):
    image = np.full((128, 128, 3), 180, np.uint8)
    labels = np.zeros((128, 128), np.uint8)
    labels[40:88, 60:68] = 1
    sample = {"id": "fixture", "split": "validation", "group": "fixture",
              "pixels": image, "labels": labels}
    model = BattlemapNet(width=8, classes=WINDOW_CLASSES).eval()
    settings = FastSettings(max_dimension=256, ml_profile="windows", lights=False)

    result, detection, raw_labels, confidence = evaluate_sample(
        model, sample, WINDOW_CLASSES, settings, .9, 1, torch.device("cpu"))
    _write_sample_visuals(tmp_path, "fixture", image, labels, raw_labels, confidence, .9,
                          WINDOW_CLASSES, detection)

    assert result["timings"]["median_warm_inference_seconds"] >= 0
    assert result["geometry"]["opaque_wall_segments"] == len(detection.walls)
    assert raw_labels.shape == confidence.shape == (128, 128)
    assert np.load(tmp_path / "fixture-confidence.npy", allow_pickle=False).shape == image.shape[:2]
    assert (tmp_path / "fixture-raw-labels.bmp").read_bytes().startswith(b"BM")
    assert (tmp_path / "fixture-review.bmp").exists()
    assert cv2.imread(str(tmp_path / "fixture-raw-labels.bmp"), 0).shape == labels.shape
    assert cv2.imread(str(tmp_path / "fixture-review.bmp")).shape[0] > image.shape[0]
    assert (tmp_path / "fixture-geometry.bmp").read_bytes().startswith(b"BM")


def test_geometry_preview_shows_exported_window_faces_instead_of_editing_axis(tmp_path):
    image = np.full((100, 100, 3), 50, np.uint8)
    labels = np.zeros((100, 100), np.uint8)
    detection = DetectionResult(100, 100, [], [], [], windows=[((50, 20), (50, 80))],
                                window_faces=[[((45, 20), (45, 80)), ((55, 20), (55, 80))]])
    _write_sample_visuals(tmp_path, "window", image, labels, labels,
                          np.full(labels.shape, .99, np.float32), .9, WINDOW_CLASSES, detection)
    preview = cv2.imread(str(tmp_path / "window-geometry.bmp"))
    # The banner adds rows but never shifts the original map's X coordinates.
    assert not np.array_equal(preview[84 + 50, 45], image[50, 45])
    assert not np.array_equal(preview[84 + 50, 55], image[50, 55])
    np.testing.assert_array_equal(preview[84 + 50, 50], image[50, 50])


def test_downscaled_map_cache_replays_exact_geometry_and_preserves_source_coordinates(tmp_path):
    image = np.zeros((512, 768, 3), np.uint8)
    truth = np.full(image.shape[:2], 255, np.uint8)
    raw = np.zeros((256, 384), np.uint8)
    raw[32:160, 80:120] = 1
    raw[200:224, 120:320] = 4
    confidence = np.linspace(.90001, .99991, raw.size, dtype=np.float32).reshape(raw.shape)
    settings = FastSettings(max_dimension=384, ml_profile="windows", lights=False, min_wall_area=0)
    original = detect_map(image, settings, prediction=(raw, confidence))
    assert original.walls and original.invisible_walls

    artifacts = _write_sample_visuals(tmp_path, "scaled", image, truth, raw, confidence, .9,
                                      WINDOW_CLASSES, original)
    cache = artifacts["working_prediction"]
    saved_labels = cv2.imread(str(tmp_path / cache["labels"]["path"]), cv2.IMREAD_UNCHANGED)
    saved_confidence = np.load(tmp_path / cache["confidence"]["path"], allow_pickle=False)
    assert cache["size"] == [384, 256]
    np.testing.assert_array_equal(saved_labels, raw)
    np.testing.assert_array_equal(saved_confidence, confidence)
    assert saved_confidence.dtype == confidence.dtype
    assert cv2.imread(str(tmp_path / "scaled-raw-labels.bmp"), 0).shape == image.shape[:2]
    replay = detect_map(image, settings, prediction=(saved_labels, saved_confidence))
    geometry = json.loads((tmp_path / artifacts["source_geometry"]["path"]).read_text())
    assert geometry["coordinate_system"] == "source_image_pixels"
    assert [geometry["width"], geometry["height"]] == [768, 512]
    for field in ("walls", "doors", "windows", "window_faces", "invisible_walls"):
        np.testing.assert_array_equal(getattr(replay, field), getattr(original, field))
        np.testing.assert_array_equal(geometry[field], getattr(original, field))
    for item in (cache["labels"], cache["confidence"], artifacts["source_geometry"]):
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_recolored_appearance_is_seeded_and_turns_gray_and_purple_into_other_colors():
    image = np.array([[[144, 32, 112], [128, 128, 128]]], np.uint8)
    transform = make_appearance_transform("recolored", DEFAULT_APPEARANCE_SEED)
    repeated = make_appearance_transform("recolored", DEFAULT_APPEARANCE_SEED)
    output = recolor_image(image, transform)

    assert transform == repeated
    assert np.array_equal(output, recolor_image(image, repeated))
    assert output.shape == image.shape
    assert len(set(output[0, 1].tolist())) > 1  # neutral gray becomes tinted
    original_hue = int(cv2.cvtColor(image[:, :1], cv2.COLOR_BGR2HSV)[0, 0, 0])
    recolored_hue = int(cv2.cvtColor(output[:, :1], cv2.COLOR_BGR2HSV)[0, 0, 0])
    assert min(abs(original_hue - recolored_hue), 180 - abs(original_hue - recolored_hue)) >= 8
    assert transform["name"] == "seeded_bgr_mix_gain_tint_v1"


def test_recolored_appearance_is_pixel_local_and_keeps_label_alignment():
    image = np.full((7, 9, 3), 120, np.uint8)
    marker = image.copy()
    marker[3, 5] = (255, 0, 0)
    labels = np.zeros((7, 9), np.uint8)
    labels[3, 5] = 1
    original_labels = labels.copy()
    transform = make_appearance_transform("recolored", DEFAULT_APPEARANCE_SEED)
    base_output, marker_output = recolor_image(image, transform), recolor_image(marker, transform)

    changed_pixels = np.any(base_output != marker_output, axis=2)
    assert np.argwhere(changed_pixels).tolist() == [[3, 5]]
    assert marker_output.shape[:2] == labels.shape
    assert np.array_equal(labels, original_labels)
