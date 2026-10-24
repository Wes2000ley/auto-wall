"""Replay consistency must not leak held-out layouts or overwrite reviewed labels."""
# ruff: noqa: E402

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from scripts import train_ml
from src.ml.data import CropSampler
from src.ml.model import load_model, save_model
from src.ml.network import BattlemapNet, LEGACY_CLASSES, WINDOW_CLASSES


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_teacher_mask_respects_known_labels_confidence_and_training_selection():
    teacher = torch.zeros(2, 5, 1, 5)
    teacher[:, 0, :, :3] = 8
    teacher.requires_grad_(True)
    student = torch.zeros(2, 6, 1, 5, requires_grad=True)
    labels = torch.tensor([[[0, 255, 1, 255, 5]], [[0, 0, 0, 0, 0]]])
    loss = train_ml.teacher_consistency_loss(student, teacher, labels, torch.tensor([True, False]))
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[0, :, :, :2]) == 12
    assert student.grad[0, 5, 0, :2].min() > 0  # Penalize new-class competition on replay.
    assert torch.count_nonzero(student.grad[0, :, :, 2:]) == 0
    assert torch.count_nonzero(student.grad[1]) == 0


def test_teacher_equal_predictions_and_empty_mask_have_finite_zero_loss():
    teacher = torch.randn(2, 5, 2, 3, requires_grad=True)
    student = teacher.detach().clone().requires_grad_()
    labels = teacher.detach().argmax(1)
    loss = train_ml.teacher_consistency_loss(student, teacher, labels, torch.tensor([True, True]))
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert teacher.grad is None
    torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=1e-7, rtol=0)
    student.grad = None
    loss = train_ml.teacher_consistency_loss(student, teacher, labels, torch.tensor([False, False]))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("classes", [LEGACY_CLASSES, WINDOW_CLASSES])
def test_sampler_source_indices_preserve_crops_and_random_sequence(classes):
    samples = [{"pixels": np.full((48, 48, 3), 80, np.uint8),
                "labels": np.full((48, 48), index + 1, np.uint8)} for index in range(3)]
    plain = CropSampler(samples, seed=19, crop_size=32, classes=classes)
    indexed = CropSampler(samples, seed=19, crop_size=32, classes=classes)
    for _ in range(2):
        expected_images, expected_labels = plain.batch(12)
        images, labels, indices = indexed.batch(12, return_indices=True)
        np.testing.assert_array_equal(images, expected_images)
        np.testing.assert_array_equal(labels, expected_labels)
        assert indices.shape == (12,) and indices.dtype == np.int64
        np.testing.assert_array_equal(labels[:, 0, 0], indices + 1)


@pytest.fixture
def training_args(tmp_path):
    image = np.full((32, 32, 3), 80, np.uint8)
    labels = np.zeros((32, 32), np.uint8)
    labels[8:16, 8:16] = 1
    labels[20:24, 20:24] = 5
    cv2.imwrite(str(tmp_path / "image.bmp"), image)
    cv2.imwrite(str(tmp_path / "labels.bmp"), labels)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format": "auto-wall-training-v1", "classes": WINDOW_CLASSES,
        "samples": [{"id": split, "image": "image.bmp", "mask": "labels.bmp",
                     "group": split + "-map", "split": split} for split in ("train", "validation", "test")]}))
    legacy = tmp_path / "legacy.safetensors"
    save_model(legacy, BattlemapNet(8, 1).eval(), {})
    return SimpleNamespace(
        steps=1, batch_size=2, validate_every=1, threads=1, learning_rate=.001,
        seed=1, device="cpu", classes="windows", manifest=manifest, selection="precision",
        geometry_settings=None, output=tmp_path / "output", crop_size=128, init_model=legacy,
        backbone=None, architecture="compact", width=8, context_dilation=1, backbone_layers=0,
        teacher_model=legacy, teacher_groups=["train-map"], teacher_weight=.5,
    )


def test_one_step_training_preserves_cached_initializer_and_frozen_teacher(training_args, monkeypatch):
    args = training_args
    cached, _ = load_model(args.init_model, "cpu")
    original = {key: value.clone() for key, value in cached.state_dict().items()}
    observed = []
    consistency_loss = train_ml.teacher_consistency_loss

    def capture(logits, teacher_logits, labels, selected):
        assert not teacher_logits.requires_grad
        observed.append(selected.tolist())
        return consistency_loss(logits, teacher_logits, labels, selected)

    monkeypatch.setattr(train_ml, "teacher_consistency_loss", capture)
    train_ml.train(args)
    assert observed == [[True, True]]
    assert cached.classes == LEGACY_CLASSES and not cached.training
    assert load_model(args.init_model, "cpu")[0] is cached
    for key, value in cached.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=0, rtol=0)
    trained, details = load_model(args.output / "best.safetensors", "cpu")
    assert trained.classes == WINDOW_CLASSES
    assert details["teacher"]["groups"] == ["train-map"]
    assert details["teacher"]["sha256"] == hashlib.sha256(args.teacher_model.read_bytes()).hexdigest()
    assert details["teacher"]["weight"] == .5
    assert details["splits"]["validation"] == ["validation-map"]
    assert details["splits"]["test"] == ["test-map"]


@pytest.mark.parametrize("groups", [[], ["validation-map"], ["test-map"], ["missing"],
                                    ["train-map", "train-map"], ["train-map", "validation-map"]])
def test_teacher_rejects_groups_outside_training_before_creating_output(training_args, groups):
    training_args.teacher_groups = groups
    with pytest.raises(ValueError, match="unique groups in the training split only"):
        train_ml.train(training_args)
    assert not training_args.output.exists()


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True, "5"])
def test_teacher_rejects_invalid_weight(training_args, weight):
    training_args.teacher_weight = weight
    with pytest.raises(ValueError, match="positive and finite"):
        train_ml.train(training_args)
    assert not training_args.output.exists()


def test_teacher_options_require_a_model(training_args):
    training_args.teacher_model = None
    with pytest.raises(ValueError, match="require --teacher-model"):
        train_ml.train(training_args)
    assert not training_args.output.exists()


def test_teacher_rejects_six_class_model(training_args, tmp_path):
    teacher = tmp_path / "six-class.safetensors"
    save_model(teacher, BattlemapNet(8, 1, WINDOW_CLASSES).eval(), {})
    training_args.teacher_model = teacher
    with pytest.raises(ValueError, match="five-class legacy ordering"):
        train_ml.train(training_args)
