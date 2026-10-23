"""Optional CPU checks for safe point-model loading and actual light-only training."""
# ruff: noqa: E402

import argparse
import copy
import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from scripts.train_light_centers import center_focal_loss, segmentation_hashes, train
from src.core.light_geometry import light_center_points
from src.ml.dino import DinoBattlemap
from src.ml.light_centers import LightCenterHead
from src.ml.light_data import LightCropSampler, center_heatmap, read_light_dataset, transform_points
from src.ml.model import load_model, predict, save_model


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_safe_point_model_roundtrip_preserves_segmentation_and_handles_odd_edges(tmp_path):
    base = DinoBattlemap().eval()
    model = copy.deepcopy(base)
    model.light_head = LightCenterHead()
    image = np.random.default_rng(12).integers(0, 256, (35, 43, 3), dtype=np.uint8)
    expected = predict(base, image)
    actual = predict(model, image, include_light_centers=True)
    np.testing.assert_array_equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])
    assert actual[2].shape == (35, 43) and np.isfinite(actual[2]).all()
    assert predict(base, image, include_light_centers=True)[2] is None
    assert len(predict(model, image)) == 2
    path = tmp_path / "lights.safetensors"
    save_model(path, model, {"test": True})
    loaded, details = load_model(path, "cpu")
    assert details == {"test": True} and loaded.light_head is not None
    for a, b in zip(actual, predict(loaded, image, include_light_centers=True)):
        np.testing.assert_array_equal(a, b)
    tiny = predict(loaded, image[:1, :1], include_light_centers=True)
    assert all(array.shape == (1, 1) for array in tiny)


def test_point_transform_and_coarse_decoding_preserve_tiny_and_adjacent_sources():
    source = np.zeros((64, 64), np.float32)
    source[19, 31] = 1
    for turns in range(4):
        for flip in (False, True):
            expected = np.rot90(source, turns)
            if flip:
                expected = expected[:, ::-1]
            point = transform_points(np.array([[31., 19.]]), 0, 0, 64, 64, turns, flip)
            actual = center_heatmap(64, point)
            np.testing.assert_array_equal(np.argwhere(actual == 1), np.argwhere(expected == 1))
    points = transform_points(np.array([[127., 99.]]), 40, 30, 256, 64, 0, False)
    assert center_heatmap(16, (points + .5) / 4 - .5).max() == 1
    target = center_heatmap(24, np.array([[8, 10], [13, 10]]))
    smooth = torch.nn.functional.interpolate(torch.from_numpy(target)[None, None], scale_factor=4,
                                             mode="bilinear", align_corners=False)[0, 0].numpy()
    np.testing.assert_allclose(light_center_points(smooth, .35), [[33.5, 41.5], [53.5, 41.5]])
    for value in (-1000., 1000.):
        logits = torch.full((1, 1, 24, 24), value, requires_grad=True)
        loss = center_focal_loss(logits, torch.from_numpy(target)[None], torch.ones((1, 24, 24), dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss) and logits.grad.isfinite().all()


def light_manifest(tmp_path):
    cv2.imwrite(str(tmp_path / "image.bmp"), np.full((64, 64, 3), 90, np.uint8))
    data = {"format": "auto-wall-light-centers-v1", "samples": [
        {"id": name, "group": name, "split": split, "image": "image.bmp", "points": [[30, 30]]}
        for name, split in (("train", "train"), ("validation", "validation"))]}
    path = tmp_path / "points.json"
    path.write_text(json.dumps(data))
    return path, data


def test_sample_loss_weights_sparse_crops_equally_and_ignores_unreviewed_pixels():
    logits = torch.linspace(-2, 2, 32).reshape(2, 1, 4, 4).requires_grad_()
    target = torch.zeros((2, 4, 4))
    target[0, 1, 1] = 1
    target[1, :2, :2] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, 0, :] = False
    actual = center_focal_loss(logits, target, valid, "sample")
    separate = [center_focal_loss(logits[i:i + 1], target[i:i + 1], valid[i:i + 1]) for i in range(2)]
    torch.testing.assert_close(actual, torch.stack(separate).mean())
    assert not torch.isclose(actual, center_focal_loss(logits, target, valid))
    actual.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[0, 0, 0].any()
    with pytest.raises(ValueError, match="normalization"):
        center_focal_loss(logits, target, valid, "unknown")


def test_light_dataset_rejects_leakage_and_invalid_centers_and_keeps_sparse_labels(tmp_path):
    path, data = light_manifest(tmp_path)
    data["samples"][1]["group"] = "train"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly one split"):
        read_light_dataset(path)
    data["samples"] = data["samples"][:1]
    for invalid in ([[64, 20]], [[float("nan"), 20]], [[20, 20], [20, 20]], [[], []]):
        data["samples"][0]["points"] = invalid
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Light (points|centers)"):
            read_light_dataset(path)
    valid = np.zeros((64, 64), np.uint8)
    valid[30, 30] = 255
    cv2.imwrite(str(tmp_path / "valid.bmp"), valid)
    data["samples"][0].update(points=[[30, 30]], valid_mask="valid.bmp")
    path.write_text(json.dumps(data))
    samples = read_light_dataset(path)
    images, targets, masks = LightCropSampler(samples, crop_size=32).batch(10)
    assert images.shape == (10, 3, 32, 32) and targets.shape == masks.shape == (10, 8, 8)
    assert np.all(np.any((targets == 1) & masks, axis=(1, 2)))


def test_negative_crop_anchors_are_reviewed_background_and_never_become_lights(tmp_path):
    path, data = light_manifest(tmp_path)
    data["samples"] = data["samples"][:1]
    sample = data["samples"][0]
    for invalid in ([[64, 20]], [[float("nan"), 20]], [[20, 20], [20, 20]]):
        sample["negative_points"] = invalid
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Negative (points|centers)"):
            read_light_dataset(path)
    sample["negative_points"] = [[30, 30]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cannot also be a negative"):
        read_light_dataset(path)
    valid = np.ones((64, 64), np.uint8)
    valid[20, 20] = 0
    cv2.imwrite(str(tmp_path / "valid.bmp"), valid)
    sample.update(negative_points=[[20, 20]], valid_mask="valid.bmp")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Negative centers must lie in labelled"):
        read_light_dataset(path)

    image = np.zeros((512, 512, 3), np.uint8)
    image[300:421, 300:421] = 255  # Bright, explicitly non-emissive decoration.
    cv2.imwrite(str(tmp_path / "image.bmp"), image)
    sample.update(points=[], negative_points=[[360, 360]])
    del sample["valid_mask"]
    path.write_text(json.dumps(data))
    samples = read_light_dataset(path)
    images, targets, masks = LightCropSampler(samples, crop_size=32).batch(24)
    assert not targets.any() and masks.all()
    assert np.count_nonzero(images.mean(axis=(1, 2, 3)) > .1) >= 16
    assert samples[0]["negative_points_sha256"] != samples[0]["points_sha256"]


def test_actual_light_training_changes_only_head_and_saves_loadable_checkpoints(tmp_path):
    path, _ = light_manifest(tmp_path)
    base = DinoBattlemap().eval()
    before = segmentation_hashes(base)
    base_path = tmp_path / "base.safetensors"
    save_model(base_path, base, {})
    args = argparse.Namespace(manifest=path, output=tmp_path / "run", init_model=base_path,
                              steps=1, batch_size=1, synthetic_per_batch=0, crop_size=32,
                              seed=2, lr=.001, threshold=.35, device="cpu", loss_normalization="sample")
    trained = train(args)
    assert segmentation_hashes(trained) == before
    assert all(not p.requires_grad and p.grad is None for name, p in trained.named_parameters()
               if not name.startswith("light_head."))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained.light_head.parameters())
    assert load_model(base_path, "cpu")[0].light_head is None  # cached base is not mutated
    saved, details = load_model(args.output / "best.safetensors", "cpu")
    assert details["segmentation_tensors_unchanged"] == 313 and saved.light_head is not None
    assert segmentation_hashes(saved) == before
