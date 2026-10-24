"""Small local segmentation datasets with whole-layout train/validation splits."""

import json
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np

from src.ml.network import CLASSES, LEGACY_CLASSES, WINDOW_CLASSES, validate_classes


def _sampling_weights(samples):
    weights = [item.get("sampling_weight", 1.0) for item in samples]
    if not weights or any(type(value) not in (int, float) for value in weights):
        raise ValueError("Sample sampling_weight must be a finite positive number.")
    try:
        weights = np.asarray(weights, dtype=np.float64)
    except (OverflowError, ValueError) as error:
        raise ValueError("Sample sampling_weight must be a finite positive number.") from error
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Sample sampling_weight must be a finite positive number.")
    return weights / weights.max()


def read_dataset(manifest, classes=CLASSES):
    classes = validate_classes(classes)
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("format") != "auto-wall-training-v1"
            or data.get("classes") != list(classes)):
        profile = "windows" if classes == WINDOW_CLASSES else "legacy"
        raise ValueError(f"Dataset format/classes do not match the selected {profile} class set.")
    if not isinstance(data.get("samples"), list):
        raise ValueError("Dataset manifest needs a samples list.")
    samples, groups, ids = [], {}, set()
    for item in data["samples"]:
        if not isinstance(item, dict):
            raise ValueError("Each dataset sample must be an object.")
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sample_id):
            raise ValueError("Sample IDs must use letters, numbers, underscores, or hyphens.")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in ("image", "mask")):
            raise ValueError("Each sample needs image and mask paths.")
        group, split = item.get("group"), item.get("split")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Each sample needs a nonempty layout group.")
        if split not in ("train", "validation", "test") or groups.setdefault(group, split) != split:
            raise ValueError("Each layout group must belong to exactly one split.")
        if sample_id in ids:
            raise ValueError("Dataset sample IDs must be unique.")
        ids.add(sample_id)
        image_path, mask_path = Path(item["image"]), Path(item["mask"])
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        if not mask_path.is_absolute():
            mask_path = manifest.parent / mask_path
        try:
            image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_UNCHANGED)
        except OSError as error:
            raise ValueError(f"Could not read image/mask files for sample {sample_id}.") from error
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch: {sample_id}")
        if mask.dtype != np.uint8:
            raise ValueError("Labels must be a single-channel uint8 image.")
        if not np.isin(mask, [*range(len(classes)), 255]).all():
            raise ValueError(f"Invalid label IDs: {sample_id}")
        if np.all(mask == 255):
            raise ValueError(f"No labelled pixels: {sample_id}")
        sample = {**item, "pixels": image, "labels": mask,
                  "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                  "labels_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
        if "geometry" in item:
            if not isinstance(item["geometry"], str) or not item["geometry"].strip():
                raise ValueError(f"Geometry annotations need a file path: {sample_id}")
            geometry_path = Path(item["geometry"])
            if not geometry_path.is_absolute():
                geometry_path = manifest.parent / geometry_path
            source = geometry_path.read_bytes()
            sample["geometry_annotations"] = json.loads(source)
            sample["geometry_sha256"] = hashlib.sha256(source).hexdigest()
        samples.append(sample)
    if not samples:
        raise ValueError("Dataset is empty.")
    _sampling_weights(samples)
    return samples


class CropSampler:
    def __init__(self, samples, seed=1729, crop_size=256, classes=CLASSES):
        self.classes = validate_classes(classes)
        self.samples = samples
        self.sample_weights = _sampling_weights(samples)
        self.weighted = not np.all(self.sample_weights == self.sample_weights[0])
        self.rng = np.random.default_rng(seed)
        self.crop_size = crop_size
        self.locations = [[np.argwhere(item["labels"] == cls)[::8] for cls in range(len(self.classes))]
                          for item in samples]
        self.class_probabilities = ((.2, .25, .2, .2, .15) if self.classes == LEGACY_CLASSES
                                    else (.18, .225, .18, .18, .135, .10))
        if self.classes == WINDOW_CLASSES:
            self.sample_indices_by_class = [
                np.flatnonzero([len(sample_locations[cls]) > 0 for sample_locations in self.locations])
                for cls in range(len(self.classes))
            ]
            self.available_classes = np.flatnonzero([len(indices) > 0 for indices in self.sample_indices_by_class])
            if not len(self.available_classes):
                raise ValueError("Windows crop sampling needs at least one supervised class.")
            fallback_weights = np.asarray(self.class_probabilities)[self.available_classes]
            self.fallback_class_probabilities = fallback_weights / fallback_weights.sum()

    def batch(self, batch_size):
        images, labels = [], []
        for _ in range(batch_size):
            if self.classes == LEGACY_CLASSES:
                index = int(self.rng.choice(len(self.samples), p=self.sample_weights / self.sample_weights.sum())
                            if self.weighted else self.rng.integers(len(self.samples)))
                sample = self.samples[index]
                target = int(self.rng.choice(len(self.classes), p=self.class_probabilities))
                points = self.locations[index][target]
                if not len(points):
                    # A partially annotated map may have no lights or doors. Keep
                    # its crops centered on labelled pixels instead of ignored space.
                    available = [locations for locations in self.locations[index] if len(locations)]
                    points = available[int(self.rng.integers(len(available)))]
            else:
                target = int(self.rng.choice(len(self.classes), p=self.class_probabilities))
                candidate_samples = self.sample_indices_by_class[target]
                if not len(candidate_samples):
                    fallback = int(self.rng.choice(len(self.available_classes),
                                                   p=self.fallback_class_probabilities))
                    target = int(self.available_classes[fallback])
                    candidate_samples = self.sample_indices_by_class[target]
                weights = self.sample_weights[candidate_samples]
                selected = (self.rng.choice(len(candidate_samples), p=weights / weights.sum())
                            if self.weighted else self.rng.integers(len(candidate_samples)))
                index = int(candidate_samples[selected])
                sample = self.samples[index]
                points = self.locations[index][target]
            height, width = sample["labels"].shape
            anchor_y, anchor_x = points[int(self.rng.integers(len(points)))]
            size = min(height, width, round(self.crop_size * self.rng.uniform(.75, 1.5)))
            x = int(np.clip(anchor_x - size // 2 + self.rng.integers(-size // 4, size // 4 + 1), 0, width - size))
            y = int(np.clip(anchor_y - size // 2 + self.rng.integers(-size // 4, size // 4 + 1), 0, height - size))
            image = cv2.resize(sample["pixels"][y:y + size, x:x + size], (self.crop_size,) * 2)
            mask = cv2.resize(sample["labels"][y:y + size, x:x + size], (self.crop_size,) * 2, interpolation=cv2.INTER_NEAREST)
            if np.all(mask == 255):
                # Downsampling can erase a one-pixel annotation. Retry around
                # the same supervised point without shrinking source pixels.
                size = min(height, width, self.crop_size)
                x = int(np.clip(anchor_x - size // 2, 0, width - size))
                y = int(np.clip(anchor_y - size // 2, 0, height - size))
                image = cv2.resize(sample["pixels"][y:y + size, x:x + size], (self.crop_size,) * 2)
                mask = cv2.resize(sample["labels"][y:y + size, x:x + size], (self.crop_size,) * 2, interpolation=cv2.INTER_NEAREST)
            turns = int(self.rng.integers(4))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if self.rng.random() < .5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            # Global color changes preserve shape/texture while breaking fixed
            # purple-door and gray-wall shortcuts. Never change labels by color.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + self.rng.integers(-90, 91)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.rng.uniform(.35, 1.4), 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255
            image = np.clip(image ** self.rng.uniform(.75, 1.35) * self.rng.uniform(.7, 1.3, (1, 1, 3)), 0, 1)
            if self.rng.random() < .5:
                # Vary illumination across a surface as well as its global hue.
                lighting = np.linspace(self.rng.uniform(.5, 1.2), self.rng.uniform(.5, 1.2), self.crop_size)
                image = np.clip(image * lighting[None, :, None], 0, 1)
            if self.rng.random() < .15:
                image[:] = image.mean(axis=2, keepdims=True)
            if self.classes == WINDOW_CLASSES:
                image = self._apply_window_tint(image)
            images.append(image.transpose(2, 0, 1))
            labels.append(mask)
        return np.ascontiguousarray(images, dtype=np.float32), np.ascontiguousarray(labels, dtype=np.int64)

    def _apply_window_tint(self, image):
        """Blend one bounded RGB tint over a whole crop, without moving pixels."""
        strength = self.rng.uniform(.15, .38)
        tint = self.rng.uniform(0, 1, (1, 1, 3))
        return np.clip(image * (1 - strength) + tint * strength, 0, 1)
