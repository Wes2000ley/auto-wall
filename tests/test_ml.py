"""Optional runtime checks; no private maps, pretrained downloads, or GPU required."""
# Optional dependencies must be checked before importing the ML runtime.
# ruff: noqa: E402

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file
from torch.nn import functional as F

from src.core.fast_pipeline import DetectionCancelled
from src.ml.data import CropSampler, read_dataset
from src.ml.model import FORMAT, load_model, predict, save_model
from src.ml.network import (
    BattlemapNet,
    CLASSES,
    LEGACY_CLASSES,
    WINDOW_CLASSES,
    expand_legacy_head,
)
from scripts.train_ml import geometry_score, metrics, training_optimizer
import scripts.train_ml as train_ml


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dilation", [1, 4])
def test_tiling_matches_whole_image_at_seams_and_odd_edges(dilation):
    torch.manual_seed(42)
    model = BattlemapNet(8, dilation).eval()
    image = np.random.default_rng(42).integers(0, 256, (257, 289, 3), dtype=np.uint8)
    labels, confidence = predict(model, image, tile_size=64)
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
    tensor = F.pad(tensor, (0, 7, 0, 7), mode="replicate")
    with torch.inference_mode():
        expected_confidence, expected_labels = model(tensor)[0].softmax(0).max(0)
    np.testing.assert_array_equal(labels, expected_labels[:257, :289].numpy())
    np.testing.assert_allclose(confidence, expected_confidence[:257, :289].numpy(), atol=2e-6, rtol=0)


def test_safe_model_roundtrip_cache_invalidation_and_metadata(tmp_path):
    model = BattlemapNet(8, 4).eval()
    path = tmp_path / "壁.safetensors"
    save_model(path, model, {"training_groups": ["one"]})
    loaded, details = load_model(path, "cpu")
    assert details == {"training_groups": ["one"]}
    assert loaded.context_dilation == 4 and loaded.training is False
    assert load_model(path, "cpu")[0] is loaded
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key])
    with torch.no_grad():
        model.head.bias.fill_(3)
    save_model(path, model, {"training_groups": ["two"]})
    replacement, details = load_model(path, "cpu")
    assert replacement is not loaded
    assert details["training_groups"] == ["two"]
    assert torch.all(replacement.head.bias == 3)


def test_windows_checkpoint_roundtrip_has_six_classes_and_keeps_strict_shapes(tmp_path):
    model = BattlemapNet(8, 4, WINDOW_CLASSES).eval()
    path = tmp_path / "windows.safetensors"
    save_model(path, model, {"class_set": "windows"})
    loaded, details = load_model(path, "cpu")
    assert loaded.classes == WINDOW_CLASSES
    assert loaded.head.out_channels == 6
    assert details == {"class_set": "windows"}
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key], rtol=0, atol=0)

    tensors = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    tensors["head.weight"] = tensors["head.weight"][:5]
    save_file(tensors, path, metadata={
        "format": FORMAT, "classes": json.dumps(WINDOW_CLASSES), "width": "8",
        "context_dilation": "4", "details": "{}"})
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_model(path, "cpu")


def test_missing_checkpoint_buffer_fails_before_inference(tmp_path):
    model = BattlemapNet(8)
    state = model.state_dict()
    del state["enc1.1.running_mean"]
    path = tmp_path / "missing-buffer.safetensors"
    save_file(state, path, metadata={
        "format": FORMAT, "classes": json.dumps(LEGACY_CLASSES), "width": "8",
        "context_dilation": "1", "details": "{}"})
    with pytest.raises(RuntimeError, match="Missing key.*enc1.1.running_mean"):
        load_model(path, "cpu")


def test_legacy_head_expansion_copies_existing_class_logits_exactly():
    torch.manual_seed(12)
    model = BattlemapNet(8, 2).eval()
    image = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        old_logits = model(image)
    original_weight = model.head.weight.detach().clone()
    original_bias = model.head.bias.detach().clone()

    assert expand_legacy_head(model) is model
    assert model.classes == WINDOW_CLASSES
    assert model.head.out_channels == 6
    torch.testing.assert_close(model.head.weight[:5], original_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.head.bias[:5], original_bias, rtol=0, atol=0)
    with torch.inference_mode():
        expanded_logits = model(image)
    torch.testing.assert_close(expanded_logits[:, :5], old_logits, rtol=0, atol=0)
    assert torch.isfinite(expanded_logits[:, 5]).all()
    with pytest.raises(ValueError, match="Only a legacy"):
        expand_legacy_head(model)


def test_invalid_models_fail_instead_of_guessing(tmp_path):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="safetensors"):
        load_model(path, "cpu")
    save_file({"arbitrary": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="compatible"):
        load_model(path, "cpu")
    save_file({"arbitrary": torch.zeros(1)}, path, metadata={
        "format": FORMAT, "classes": json.dumps(CLASSES), "width": "900"})
    with pytest.raises(ValueError, match="width"):
        load_model(path, "cpu")
    model = BattlemapNet(8)
    with torch.no_grad():
        model.head.bias.fill_(float("nan"))
    save_model(path, model, {})
    with pytest.raises(ValueError, match="nonfinite"):
        load_model(path, "cpu")


def test_cancel_between_tiles_and_tiny_image():
    model = BattlemapNet(8).eval()
    completed = []
    with pytest.raises(DetectionCancelled):
        predict(model, np.zeros((65, 65, 3), np.uint8), cancelled=lambda: bool(completed),
                progress=completed.append, tile_size=64)
    assert len(completed) == 1
    labels, confidence = predict(model, np.zeros((1, 1, 3), np.uint8))
    assert labels.shape == confidence.shape == (1, 1)
    assert np.isfinite(confidence).all()


def test_dataset_rejects_layout_leakage_and_invalid_labels(tmp_path):
    cv2.imwrite(str(tmp_path / "image.bmp"), np.zeros((64, 64, 3), np.uint8))
    mask = np.zeros((64, 64), np.uint8)
    mask[10:20, 10:40] = 1
    cv2.imwrite(str(tmp_path / "mask.bmp"), mask)
    item = {"id": "one", "image": "image.bmp", "mask": "mask.bmp", "group": "same-layout", "split": "train"}
    data = {"format": "auto-wall-training-v1", "classes": CLASSES,
            "samples": [item, {**item, "id": "upscale", "split": "test"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly one split"):
        read_dataset(path)
    data["samples"] = [item]
    path.write_text(json.dumps(data))
    samples = read_dataset(path)
    images, labels = CropSampler(samples, crop_size=32).batch(3)
    assert images.shape == (3, 3, 32, 32) and labels.shape == (3, 32, 32)
    assert 0 <= images.min() <= images.max() <= 1
    assert set(np.unique(labels)).issubset({0, 1})
    with pytest.raises(ValueError, match="selected windows"):
        read_dataset(path, classes=WINDOW_CLASSES)
    data["classes"] = WINDOW_CLASSES
    path.write_text(json.dumps(data))
    upgraded_legacy_mask = read_dataset(path, classes=WINDOW_CLASSES)[0]["labels"]
    assert set(np.unique(upgraded_legacy_mask)) == {0, 1}
    mask[30:40, 30:40] = 5
    cv2.imwrite(str(tmp_path / "mask.bmp"), mask)
    path.write_text(json.dumps(data))
    windows_samples = read_dataset(path, classes=WINDOW_CLASSES)
    sampler = CropSampler(windows_samples, seed=8, crop_size=32, classes=WINDOW_CLASSES)
    assert sampler.classes == WINDOW_CLASSES
    assert len(sampler.locations[0]) == 6 and len(sampler.locations[0][5]) > 0
    data["classes"] = LEGACY_CLASSES
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid label"):
        read_dataset(path)
    data["classes"] = WINDOW_CLASSES
    data["samples"] = [{**item, "id": "../outside"}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Sample IDs"):
        read_dataset(path, classes=WINDOW_CLASSES)
    data["samples"] = [item]
    path.write_text(json.dumps(data))
    cv2.imwrite(str(tmp_path / "mask.bmp"), np.full((64, 64), 7, np.uint8))
    with pytest.raises(ValueError, match="Invalid label"):
        read_dataset(path, classes=WINDOW_CLASSES)
    cv2.imwrite(str(tmp_path / "mask.bmp"), np.full((64, 64), 255, np.uint8))
    with pytest.raises(ValueError, match="No labelled pixels"):
        read_dataset(path, classes=WINDOW_CLASSES)


def test_metrics_distinguish_undefined_classes_from_false_positives():
    truth = np.array([[0, 1, 255]], np.uint8)
    prediction = np.array([[3, 1, 2]], np.uint8)
    result = metrics(truth, prediction)
    assert result["light"]["precision"] == 0 and result["light"]["recall"] is None
    assert result["door"]["iou"] is None and result["door"]["predicted_pixels"] == 0
    assert result["wall"]["iou"] == 1


def test_windows_metrics_report_window_pixels_separately():
    truth = np.array([[5, 0, 255]], np.uint8)
    prediction = np.array([[5, 5, 1]], np.uint8)
    result = metrics(truth, prediction, WINDOW_CLASSES)
    assert tuple(result) == WINDOW_CLASSES
    assert result["window"]["truth_pixels"] == 1
    assert result["window"]["precision"] == pytest.approx(.5)
    assert result["window"]["recall"] == 1


def test_trainer_keeps_legacy_default_and_exposes_windows_profile(monkeypatch):
    parsed = []
    monkeypatch.setattr(train_ml, "train", parsed.append)
    monkeypatch.setattr(sys, "argv", ["train_ml.py", "dataset.json", "run"])
    train_ml.main()
    monkeypatch.setattr(sys, "argv", ["train_ml.py", "dataset.json", "run", "--classes", "windows"])
    train_ml.main()
    assert parsed[0].classes == "legacy"
    assert parsed[1].classes == "windows"
    assert parsed[0].crop_aspect_ratio == parsed[1].crop_aspect_ratio == 1
    monkeypatch.setattr(sys, "argv", ["train_ml.py", "dataset.json", "run", "--crop-aspect-ratio", "2"])
    train_ml.main()
    assert parsed[2].crop_aspect_ratio == 2


def test_windows_trainer_requires_window_labels_in_train_and_validation(tmp_path):
    image = np.zeros((32, 32, 3), np.uint8)
    train_mask = np.zeros((32, 32), np.uint8)
    train_mask[8:16, 8:16] = 1
    validation_mask = train_mask.copy()
    validation_mask[20:24, 20:24] = 5
    cv2.imwrite(str(tmp_path / "image.bmp"), image)
    cv2.imwrite(str(tmp_path / "train.bmp"), train_mask)
    cv2.imwrite(str(tmp_path / "validation.bmp"), validation_mask)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "auto-wall-training-v1",
        "classes": WINDOW_CLASSES,
        "samples": [
            {"id": "train", "image": "image.bmp", "mask": "train.bmp", "group": "train-map", "split": "train"},
            {"id": "validation", "image": "image.bmp", "mask": "validation.bmp",
             "group": "validation-map", "split": "validation"},
        ],
    }))
    args = SimpleNamespace(
        steps=1, batch_size=1, validate_every=1, threads=1, learning_rate=.001,
        seed=1, device="cpu", classes="windows", manifest=manifest, selection="pixels",
        geometry_settings=None, output=tmp_path / "output", crop_size=128, init_model=None,
        backbone=None, architecture="compact", width=8, context_dilation=1, backbone_layers=0,
    )
    with pytest.raises(ValueError, match="training split"):
        train_ml.train(args)

    train_mask[20:24, 20:24] = 5
    cv2.imwrite(str(tmp_path / "train.bmp"), train_mask)
    validation_mask[20:24, 20:24] = 0
    cv2.imwrite(str(tmp_path / "validation.bmp"), validation_mask)
    with pytest.raises(ValueError, match="validation for checkpoint selection"):
        train_ml.train(args)
    assert not args.output.exists()

    validation_mask[20:24, 20:24] = 5
    cv2.imwrite(str(tmp_path / "validation.bmp"), validation_mask)
    train_ml.train(args)
    _, details = load_model(args.output / "best.safetensors", "cpu")
    assert "class-first" in details["crop_sampling"]
    assert "renormalized" in details["crop_sampling"]
    assert "windows adds a crop-wide RGB tint" in details["augmentation"]


def test_partial_annotation_crops_include_supervision_for_absent_classes():
    labels = np.full((512, 512), 255, np.uint8)
    labels[20:40, 20:40] = 1
    sample = {"pixels": np.zeros((512, 512, 3), np.uint8), "labels": labels}
    _, crops = CropSampler([sample], seed=8, crop_size=64).batch(30)
    assert np.all(np.any(crops == 1, axis=(1, 2)))


def test_downsampling_cannot_erase_the_only_supervised_pixel():
    labels = np.full((512, 512), 255, np.uint8)
    labels[20, 20] = 1
    sample = {"pixels": np.zeros((512, 512, 3), np.uint8), "labels": labels}
    _, crops = CropSampler([sample], seed=8, crop_size=64).batch(100)
    assert np.all(np.any(crops == 1, axis=(1, 2)))


@pytest.mark.parametrize("ratio", [0, .9, 4.1, float("nan"), float("inf"), True, "2"])
def test_crop_sampler_rejects_invalid_aspect_ratio(ratio):
    with pytest.raises(ValueError, match="aspect ratio"):
        CropSampler([], max_aspect_ratio=ratio)


def test_aspect_augmentation_pairs_image_and_labels_and_preserves_default_rng(monkeypatch):
    labels = np.tile(np.arange(256, dtype=np.uint8) % 6, (192, 1))
    pixels = np.repeat((labels * 40)[:, :, None], 3, axis=2)
    sample = {"pixels": pixels, "labels": labels}
    resize = cv2.resize
    crops = []

    def capture(source, *args, **kwargs):
        crops.append(source.copy())
        return resize(source, *args, **kwargs)

    monkeypatch.setattr(cv2, "resize", capture)
    sampler = CropSampler([sample], seed=19, crop_size=64, classes=WINDOW_CLASSES, max_aspect_ratio=2)
    images, masks, indices = sampler.batch(12, return_indices=True)
    assert images.shape == (12, 3, 64, 64) and masks.shape == (12, 64, 64)
    assert np.all(indices == 0) and set(np.unique(masks)) == set(range(6))
    assert len(crops) == 24
    assert any(image.shape[0] != image.shape[1] for image in crops[::2])
    for image, mask in zip(crops[::2], crops[1::2]):
        assert image.shape[:2] == mask.shape
        np.testing.assert_array_equal(image[:, :, 0], mask * 40)

    crops.clear()
    default = CropSampler([sample], seed=19, crop_size=64, classes=WINDOW_CLASSES)
    explicit = CropSampler([sample], seed=19, crop_size=64, classes=WINDOW_CLASSES, max_aspect_ratio=1)
    for left, right in zip(default.batch(3, return_indices=True), explicit.batch(3, return_indices=True)):
        np.testing.assert_array_equal(left, right)
    assert all(crop.shape[0] == crop.shape[1] for crop in crops)


def test_windows_tint_adds_chroma_without_moving_spatial_labels():
    class NoTintCropSampler(CropSampler):
        def _apply_window_tint(self, image):
            return image

    labels = np.ones((128, 128), np.uint8)
    for y in range(0, 128, 8):
        for x in range(0, 128, 8):
            labels[y, x] = 2
            labels[y, x + 1] = 0
            labels[y + 1, x] = 3
            labels[y + 1, x + 1] = 4
            labels[y + 2, x] = 5
    pixels = np.full((128, 128, 3), 80, np.uint8)
    pixels[labels == 2] = (160, 40, 180)  # BGR purple door pixels.
    sample = {"pixels": pixels, "labels": labels}

    tinted_images, tinted_labels = CropSampler(
        [sample], seed=42, crop_size=64, classes=WINDOW_CLASSES,
    ).batch(1)
    original_images, original_labels = NoTintCropSampler(
        [sample], seed=42, crop_size=64, classes=WINDOW_CLASSES,
    ).batch(1)

    np.testing.assert_array_equal(tinted_labels, original_labels)
    wall_pixels = tinted_labels[0] == 1
    assert wall_pixels.any()
    tinted_rgb = np.moveaxis(tinted_images[0], 0, -1)
    original_rgb = np.moveaxis(original_images[0], 0, -1)
    tinted_chroma = np.ptp(tinted_rgb[wall_pixels], axis=1).mean()
    original_chroma = np.ptp(original_rgb[wall_pixels], axis=1).mean()
    assert tinted_chroma > original_chroma + .03

    door_pixels = tinted_labels[0] == 2
    assert door_pixels.any()
    original_doors = np.clip(original_rgb[door_pixels] * 255, 0, 255).astype(np.uint8)
    tinted_doors = np.clip(tinted_rgb[door_pixels] * 255, 0, 255).astype(np.uint8)
    original_hues = cv2.cvtColor(original_doors[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, 0]
    tinted_hues = cv2.cvtColor(tinted_doors[:, None, :], cv2.COLOR_RGB2HSV)[:, 0, 0]
    hue_delta = np.abs((tinted_hues.astype(np.int16) - original_hues.astype(np.int16) + 90) % 180 - 90)
    assert np.median(hue_delta) >= 2
    assert np.linalg.norm(tinted_doors.astype(float).mean(0) - original_doors.astype(float).mean(0)) > 8


def test_windows_sampler_targets_a_rare_class_across_samples_and_falls_back_globally():
    samples = []
    for index in range(21):
        labels = np.zeros((256, 256), np.uint8)
        labels[:8, :8] = 1
        labels[:8, 16:24] = 2
        labels[16:24, :8] = 3
        labels[16:24, 16:24] = 4
        if index == 0:
            labels[240:248, 240:248] = 5
        samples.append({
            "pixels": np.full((256, 256, 3), 96, np.uint8),
            "labels": labels,
        })

    _, crops = CropSampler(
        samples, seed=108, crop_size=32, classes=WINDOW_CLASSES,
    ).batch(800)
    window_crop_rate = np.any(crops == 5, axis=(1, 2)).mean()
    assert window_crop_rate == pytest.approx(.10, abs=.03)
    assert np.all(np.any(crops != 255, axis=(1, 2)))

    partial = np.full((64, 64), 255, np.uint8)
    partial[24:40, 24:40] = 1
    _, fallback_crops = CropSampler(
        [{"pixels": samples[0]["pixels"], "labels": partial}],
        seed=9, crop_size=32, classes=WINDOW_CLASSES,
    ).batch(32)
    assert np.all(np.any(fallback_crops == 1, axis=(1, 2)))
    assert np.all(np.any(fallback_crops != 255, axis=(1, 2)))


def test_geometry_selection_penalizes_false_instances_and_requires_annotations():
    perfect = {"map": {"geometry": {"rays_passed": 4, "rays_total": 4,
        "doors": {"matched": 2, "prediction_count": 2, "truth_count": 2, "complete_annotation": True}}}}
    assert geometry_score(perfect) == 1
    perfect["map"]["geometry"]["doors"]["prediction_count"] = 3
    assert geometry_score(perfect) == pytest.approx((1 + 10 / 14) / 2)
    with pytest.raises(ValueError, match="complete instance"):
        geometry_score({"map": {"geometry": {"doors": {"complete_annotation": False}}}})


@pytest.mark.parametrize("classes", [LEGACY_CLASSES, WINDOW_CLASSES])
def test_replay_weights_change_map_frequency_without_changing_class_targets(classes):
    first = np.ones((32, 32), np.uint8)
    second = first.copy()
    second[8:24, 8:24] = 0  # identify the second layout after rotations/flips
    samples = [{"pixels": np.full((32, 32, 3), 80, np.uint8), "labels": mask}
               for mask in (first, second)]

    def second_layout_rate(items):
        sampler = CropSampler(items, seed=21, crop_size=32, classes=classes)
        sampler.class_probabilities = (0., 1., 0., 0., 0., 0.)[:len(classes)]
        _, labels = sampler.batch(256)
        assert np.all(np.any(labels == 1, axis=(1, 2)))
        return np.any(labels == 0, axis=(1, 2)).mean()

    assert .35 < second_layout_rate(samples) < .65
    weighted = [samples[0], {**samples[1], "sampling_weight": 16}]
    assert second_layout_rate(weighted) > .88


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True, "5", 10 ** 400])
def test_sampler_rejects_invalid_replay_weights(weight):
    sample = {"pixels": np.zeros((32, 32, 3), np.uint8),
              "labels": np.ones((32, 32), np.uint8), "sampling_weight": weight}
    with pytest.raises(ValueError, match="sampling_weight"):
        CropSampler([sample])


def test_precision_selection_penalizes_empty_map_hallucinations_and_missed_windows():
    def score(truth, prediction):
        return {"confidence_0.9": metrics(np.array([truth], np.uint8),
                                         np.array([prediction], np.uint8), WINDOW_CLASSES)}

    perfect = score([0, 1, 5], [0, 1, 5])
    empty = score([0, 0], [0, 0])
    assert train_ml.precision_score({"structure": perfect, "empty": empty}) == 1
    hallucinated = score([0, 0], [1, 0])
    assert train_ml.precision_score({"structure": perfect, "empty": hallucinated}) == .5
    assert train_ml.precision_score({"structure": score([1, 5], [1, 0])}) == .5
    # Four false pixels count four times as much as four missed pixels.
    assert train_ml.precision_score({"map": score([1, 1, 1, 1, 1], [1, 0, 0, 0, 0])}) > (
        train_ml.precision_score({"map": score([1, 0, 0, 0, 0], [1, 1, 1, 1, 1])}))
    assert train_ml.precision_score({"map": score([1, 255], [1, 5])}) == 1
    with pytest.raises(ValueError, match="validation samples"):
        train_ml.precision_score({})


def test_compact_optimizer_keeps_one_rate_and_rejects_backbone_fine_tuning():
    model = BattlemapNet(8)
    optimizer = training_optimizer(model, .0003)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == group["initial_lr"] == .0003
    assert {id(p) for p in group["params"]} == {id(p) for p in model.parameters()}
    with pytest.raises(ValueError, match="requires a DINO"):
        training_optimizer(model, .0003, backbone_layers=1)
