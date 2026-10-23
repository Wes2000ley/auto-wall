"""Point annotations and augmentation for physical light sources."""

import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np


def _read_points(values, width, height, name, label):
    points = np.asarray(values, dtype=np.float32)
    if points.shape == (0,):
        points = np.empty((0, 2), np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points > [width - 1, height - 1])):
        raise ValueError(f"{label} points must be finite XY coordinates inside the source image: {name}")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{label} centers must be unique within each image: {name}")
    return points


def read_light_dataset(manifest):
    """XY centers use source-image pixels; a zero valid-mask pixel is unlabelled."""
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("format") != "auto-wall-light-centers-v1":
        raise ValueError("Unknown light-center dataset format.")
    samples, groups, ids = [], {}, set()
    for item in data["samples"]:
        name, group, split = item["id"], item["group"], item["split"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in ids:
            raise ValueError("Light sample IDs must be unique letters, numbers, underscores, or hyphens.")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Each light sample needs a nonempty layout group.")
        if split not in ("train", "validation", "test") or groups.setdefault(group, split) != split:
            raise ValueError("Each layout group must belong to exactly one split.")
        ids.add(name)
        image = cv2.imdecode(np.fromfile(manifest.parent / item["image"], np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 8:
            raise ValueError(f"Light training needs an image at least eight pixels wide and high: {name}")
        height, width = image.shape[:2]
        points = _read_points(item["points"], width, height, name, "Light")
        negatives = _read_points(item.get("negative_points", []), width, height, name, "Negative")
        if len(points) and len(negatives) and np.any(np.all(points[:, None] == negatives[None], axis=-1)):
            raise ValueError(f"A light center cannot also be a negative crop anchor: {name}")
        valid = np.ones((height, width), np.uint8)
        if item.get("valid_mask"):
            valid = cv2.imdecode(np.fromfile(manifest.parent / item["valid_mask"], np.uint8), cv2.IMREAD_UNCHANGED)
            if valid is None or valid.shape != (height, width) or valid.dtype != np.uint8:
                raise ValueError(f"Light validity masks must be single-channel uint8 and match the image: {name}")
            valid = (valid != 0).astype(np.uint8)
        if not valid.any():
            raise ValueError(f"No labelled light regions: {name}")
        for centers, label in ((points, "Light"), (negatives, "Negative")):
            coordinates = np.rint(centers).astype(int)
            if len(centers) and not valid[coordinates[:, 1], coordinates[:, 0]].all():
                raise ValueError(f"{label} centers must lie in labelled regions: {name}")
        synthetic = item.get("synthetic", False)
        if type(synthetic) is not bool:
            raise ValueError("The synthetic sample flag must be boolean.")
        samples.append({**item, "pixels": image, "points": points, "negative_points": negatives,
                        "valid": valid, "synthetic": synthetic,
                        "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                        "negative_points_sha256": hashlib.sha256(negatives.tobytes()).hexdigest(),
                        "valid_sha256": hashlib.sha256(valid.tobytes()).hexdigest()})
    if not samples:
        raise ValueError("Light dataset is empty.")
    return samples


def center_heatmap(size, points, sigma=1.25):
    """Render after spatial transforms so resizing cannot erase a tiny source."""
    target = np.zeros((size, size), np.float32)
    radius = math.ceil(3 * sigma)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    for px, py in points:
        x, y = np.rint([px, py]).astype(int)
        if not (0 <= x < size and 0 <= y < size):
            continue
        x0, x1 = max(0, x - radius), min(size, x + radius + 1)
        y0, y1 = max(0, y - radius), min(size, y + radius + 1)
        np.maximum(target[y0:y1, x0:x1],
                   kernel[y0 - y + radius:y1 - y + radius, x0 - x + radius:x1 - x + radius],
                   out=target[y0:y1, x0:x1])
    return target


def transform_points(points, x, y, size, output_size, turns, flip):
    """Match OpenCV pixel-center resizing and numpy's counterclockwise rot90."""
    points = (points - [x, y] + .5) * (output_size / size) - .5
    for _ in range(turns):
        points = np.column_stack((points[:, 1], output_size - 1 - points[:, 0]))
    if flip:
        points[:, 0] = output_size - 1 - points[:, 0]
    return points


def cutmix_light_crops(images, targets, valid, rng, probability=0.):
    """Exchange aligned image/label patches; ignore ambiguous cut boundaries.

    Adapted from CutMix (https://arxiv.org/abs/1905.04899) for quarter-resolution
    point targets, rather than mixing image-level class labels. Inputs are never
    mutated. The disabled path consumes no random numbers.
    """
    if not 0 <= probability <= 1:
        raise ValueError("CutMix probability must be in [0, 1].")
    if probability == 0:
        return images, targets, valid
    count, height, width = targets.shape
    if count < 2 or min(height, width) < 8:
        raise ValueError("CutMix needs at least two crops with eight-by-eight point targets.")
    if images.shape != (count, 3, 4 * height, 4 * width) or valid.shape != targets.shape:
        raise ValueError("CutMix image, quarter-resolution target, and validity shapes must agree.")
    if rng.random() >= probability:
        return images, targets, valid
    patch_height = int(rng.integers(max(1, height // 4), 3 * height // 4 + 1))
    patch_width = int(rng.integers(max(1, width // 4), 3 * width // 4 + 1))
    y0 = int(rng.integers(height - patch_height + 1))
    x0 = int(rng.integers(width - patch_width + 1))
    y1, x1 = y0 + patch_height, x0 + patch_width
    # A nonzero cyclic shift gives each recipient a different, unchanged donor.
    donors = (np.arange(count) + int(rng.integers(1, count))) % count
    mixed_images, mixed_targets, mixed_valid = images.copy(), targets.copy(), valid.copy()
    mixed_images[:, :, 4 * y0:4 * y1, 4 * x0:4 * x1] = images[donors, :, 4 * y0:4 * y1, 4 * x0:4 * x1]
    mixed_targets[:, y0:y1, x0:x1] = targets[donors, y0:y1, x0:x1]
    mixed_valid[:, y0:y1, x0:x1] = valid[donors, y0:y1, x0:x1]
    # 32 image pixels on ordinary crops cover the local head's receptive radius.
    margin = min(8, min(height, width) // 8)
    for y in (y0, y1):
        mixed_valid[:, max(0, y - margin):min(height, y + margin),
                    max(0, x0 - margin):min(width, x1 + margin)] = False
    for x in (x0, x1):
        mixed_valid[:, max(0, y0 - margin):min(height, y1 + margin),
                    max(0, x - margin):min(width, x + margin)] = False
    # Sparse annotations can disappear under the cut; retain that original crop.
    empty = ~mixed_valid.any(axis=(1, 2))
    mixed_images[empty], mixed_targets[empty], mixed_valid[empty] = images[empty], targets[empty], valid[empty]
    return mixed_images, mixed_targets, mixed_valid


class LightCropSampler:
    def __init__(self, samples, seed=73113, crop_size=384):
        if not samples or crop_size < 32 or crop_size % 8:
            raise ValueError("Light crops need samples and a size of at least 32, divisible by eight.")
        self.samples, self.rng, self.size = samples, np.random.default_rng(seed), crop_size
        # Detach the sampled coordinates from argwhere's full-size backing array.
        self.valid_points = [np.argwhere(item["valid"])[::16].copy() for item in samples]

    def batch(self, count):
        images, targets, masks = [], [], []
        for _ in range(count):
            for attempt in range(20):
                index = int(self.rng.integers(len(self.samples)))
                sample = self.samples[index]
                height, width = sample["valid"].shape
                if len(sample["points"]) and self.rng.random() < .65:
                    anchor_x, anchor_y = sample["points"][self.rng.integers(len(sample["points"]))]
                elif len(sample.get("negative_points", ())) and self.rng.random() < .75:
                    negatives = sample["negative_points"]
                    anchor_x, anchor_y = negatives[self.rng.integers(len(negatives))]
                else:
                    anchor_y, anchor_x = self.valid_points[index][self.rng.integers(len(self.valid_points[index]))]
                size = min(height, width, round(self.size * self.rng.uniform(.75, 1.5)))
                x = int(np.clip(anchor_x - size / 2 + self.rng.uniform(-size / 4, size / 4), 0, width - size))
                y = int(np.clip(anchor_y - size / 2 + self.rng.uniform(-size / 4, size / 4), 0, height - size))
                image = cv2.resize(sample["pixels"][y:y + size, x:x + size], (self.size,) * 2)
                mask = cv2.resize(sample["valid"][y:y + size, x:x + size], (self.size,) * 2, interpolation=cv2.INTER_NEAREST)
                turns, flip = int(self.rng.integers(4)), self.rng.random() < .5
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
                if flip:
                    image, mask = image[:, ::-1], mask[:, ::-1]
                points = transform_points(sample["points"], x, y, size, self.size, turns, flip)
                target = center_heatmap(self.size // 4, (points + .5) / 4 - .5)
                mask = mask.reshape(self.size // 4, 4, self.size // 4, 4).all(axis=(1, 3))
                mask[target == 1] = True  # Explicit centers remain supervised beside ignored pixels.
                if mask.any():
                    break
            else:
                raise ValueError("Annotated regions are too narrow for light crops; label wider regions or enlarge the crop size.")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + self.rng.integers(-90, 91)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.rng.uniform(.35, 1.4), 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255
            image = np.clip(image ** self.rng.uniform(.75, 1.35) * self.rng.uniform(.7, 1.3, (1, 1, 3)), 0, 1)
            if self.rng.random() < .5:
                lighting = np.linspace(self.rng.uniform(.5, 1.2), self.rng.uniform(.5, 1.2), self.size)
                image = np.clip(image * lighting[None, :, None], 0, 1)
            if self.rng.random() < .15:
                image[:] = image.mean(axis=2, keepdims=True)
            images.append(image.transpose(2, 0, 1))
            targets.append(target)
            masks.append(mask)
        return (np.ascontiguousarray(images, dtype=np.float32),
                np.asarray(targets, dtype=np.float32), np.asarray(masks, dtype=bool))
