"""Optional runtime checks; no private maps, pretrained downloads, or GPU required."""
# Optional dependencies must be checked before importing the ML runtime.
# ruff: noqa: E402

import json

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
from src.ml.network import BattlemapNet, CLASSES
from scripts.train_ml import geometry_score, metrics


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
    data["samples"] = [{**item, "id": "../outside"}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Sample IDs"):
        read_dataset(path)
    data["samples"] = [item]
    path.write_text(json.dumps(data))
    cv2.imwrite(str(tmp_path / "mask.bmp"), np.full((64, 64), 7, np.uint8))
    with pytest.raises(ValueError, match="Invalid label"):
        read_dataset(path)
    cv2.imwrite(str(tmp_path / "mask.bmp"), np.full((64, 64), 255, np.uint8))
    with pytest.raises(ValueError, match="No labelled pixels"):
        read_dataset(path)


def test_metrics_distinguish_undefined_classes_from_false_positives():
    truth = np.array([[0, 1, 255]], np.uint8)
    prediction = np.array([[3, 1, 2]], np.uint8)
    result = metrics(truth, prediction)
    assert result["light"]["precision"] == 0 and result["light"]["recall"] is None
    assert result["door"]["iou"] is None and result["door"]["predicted_pixels"] == 0
    assert result["wall"]["iou"] == 1


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


def test_geometry_selection_penalizes_false_instances_and_requires_annotations():
    perfect = {"map": {"geometry": {"rays_passed": 4, "rays_total": 4,
        "doors": {"matched": 2, "prediction_count": 2, "truth_count": 2, "complete_annotation": True}}}}
    assert geometry_score(perfect) == 1
    perfect["map"]["geometry"]["doors"]["prediction_count"] = 3
    assert geometry_score(perfect) == pytest.approx((1 + 10 / 14) / 2)
    with pytest.raises(ValueError, match="complete instance"):
        geometry_score({"map": {"geometry": {"doors": {"complete_annotation": False}}}})
