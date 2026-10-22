"""Small local segmentation datasets with whole-layout train/validation splits."""

import json
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np

from src.ml.network import CLASSES


def read_dataset(manifest):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("format") != "auto-wall-training-v1" or data.get("classes") != list(CLASSES):
        raise ValueError("Dataset format/classes do not match the detector.")
    samples, groups, ids = [], {}, set()
    for item in data["samples"]:
        group, split = item["group"], item["split"]
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Each sample needs a nonempty layout group.")
        if not isinstance(item["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item["id"]):
            raise ValueError("Sample IDs must use letters, numbers, underscores, or hyphens.")
        if split not in ("train", "validation", "test") or groups.setdefault(group, split) != split:
            raise ValueError("Each layout group must belong to exactly one split.")
        if item["id"] in ids:
            raise ValueError("Dataset sample IDs must be unique.")
        ids.add(item["id"])
        image = cv2.imdecode(np.fromfile(manifest.parent / item["image"], np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(manifest.parent / item["mask"], np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch: {item['id']}")
        if mask.dtype != np.uint8:
            raise ValueError("Labels must be a single-channel uint8 image.")
        if not np.isin(mask, [*range(len(CLASSES)), 255]).all():
            raise ValueError(f"Invalid label IDs: {item['id']}")
        if np.all(mask == 255):
            raise ValueError(f"No labelled pixels: {item['id']}")
        sample = {**item, "pixels": image, "labels": mask,
                  "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                  "labels_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
        if "geometry" in item:
            source = (manifest.parent / item["geometry"]).read_bytes()
            sample["geometry_annotations"] = json.loads(source)
            sample["geometry_sha256"] = hashlib.sha256(source).hexdigest()
        samples.append(sample)
    if not samples:
        raise ValueError("Dataset is empty.")
    return samples


class CropSampler:
    def __init__(self, samples, seed=1729, crop_size=256):
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.crop_size = crop_size
        self.locations = [[np.argwhere(item["labels"] == cls)[::8] for cls in range(len(CLASSES))]
                          for item in samples]

    def batch(self, batch_size):
        images, labels = [], []
        for _ in range(batch_size):
            index = int(self.rng.integers(len(self.samples)))
            sample = self.samples[index]
            target = int(self.rng.choice(5, p=(.2, .25, .2, .2, .15)))
            points = self.locations[index][target]
            height, width = sample["labels"].shape
            if not len(points):
                # A partially annotated map may have no lights or doors. Keep
                # its crops centered on labelled pixels instead of ignored space.
                available = [locations for locations in self.locations[index] if len(locations)]
                points = available[int(self.rng.integers(len(available)))]
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
            images.append(image.transpose(2, 0, 1))
            labels.append(mask)
        return np.ascontiguousarray(images, dtype=np.float32), np.ascontiguousarray(labels, dtype=np.int64)
