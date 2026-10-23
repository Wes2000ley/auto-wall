"""Generate raw BMP training maps with lights and non-emissive lookalikes.

Procedural fixtures are training aids, not evidence of accuracy on real maps.
Colors, orientation, background, and illumination vary independently of class.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SIZE = 640
LIGHT_KINDS = ("torch", "lantern", "strip", "orb", "campfire")
DECOY_KINDS = ("flower", "painted_star", "jewel", "rivets", "unlit_lantern")


def shifted(color, amount):
    return tuple(int(value) for value in np.clip(np.asarray(color) + amount, 0, 255))


def draw_object(rng, kind, color, scale):
    """Return one RGBA-free BGR patch and its object mask, centered at (40, 40)."""
    patch = np.zeros((81, 81, 3), np.uint8)
    mask = np.zeros((81, 81), np.uint8)
    dark = (35, 40, 43)
    bright = shifted(color, 65)

    def circle(center, radius, value, width=-1):
        cv2.circle(patch, center, radius, value, width, cv2.LINE_AA)
        cv2.circle(mask, center, radius, 255, width, cv2.LINE_AA)

    def line(a, b, value, width=1):
        cv2.line(patch, a, b, value, width, cv2.LINE_AA)
        cv2.line(mask, a, b, 255, width, cv2.LINE_AA)

    def polygon(points, value):
        points = np.asarray(points, np.int32)
        cv2.fillPoly(patch, [points], value, cv2.LINE_AA)
        cv2.fillPoly(mask, [points], 255, cv2.LINE_AA)

    if kind in ("torch", "campfire"):
        if kind == "torch":
            line((40, 44), (40, 60), (58, 80, 107), 5)
            line((35, 49), (45, 49), dark, 3)
        else:
            line((29, 48), (51, 55), (60, 81, 106), 5)
            line((30, 55), (51, 47), (71, 96, 119), 5)
            for angle in np.linspace(0, 2 * np.pi, 9, endpoint=False):
                circle((40 + round(17 * np.cos(angle)), 44 + round(13 * np.sin(angle))), 3, (100, 104, 108))
        polygon([(31, 47), (30, 38), (36, 26), (39, 37), (46, 23), (48, 37), (50, 45), (42, 50)], color)
        polygon([(36, 46), (37, 36), (41, 43), (45, 34), (46, 45), (40, 48)], bright)
        circle((40, 41), 3, (249, 249, 245))
    elif kind in ("lantern", "unlit_lantern"):
        circle((40, 26), 4, dark, 2)
        polygon([(31, 29), (49, 29), (51, 51), (29, 51)], dark)
        polygon([(34, 33), (46, 33), (47, 47), (33, 47)], color)
        if kind == "lantern":
            circle((40, 40), 4, bright)
            line((40, 36), (40, 44), (245, 251, 249), 2)
        else:
            line((34, 35), (46, 45), shifted(color, -55), 2)
        for x in (33, 47):
            line((x, 31), (x, 49), dark, 2)
        line((29, 50), (51, 50), shifted(dark, 25), 3)
    elif kind == "strip":
        polygon([(22, 34), (58, 34), (58, 46), (22, 46)], dark)
        line((26, 40), (54, 40), color, 6)
        line((27, 40), (53, 40), bright, 2)
        circle((23, 35), 1, (170, 175, 170))
        circle((57, 45), 1, (170, 175, 170))
    elif kind == "orb":
        circle((40, 40), 15, dark)
        circle((40, 40), 12, (105, 110, 112), 2)
        circle((40, 40), 9, color)
        circle((40, 40), 5, bright)
        circle((39, 39), 2, (245, 249, 250))
        for offset in (-12, 12):
            line((40 + offset, 36), (40 + offset, 44), (150, 151, 149), 2)
    elif kind == "flower":
        for x, y in ((31, 50), (49, 51), (30, 32), (50, 29)):
            line((40, 41), (x, y), (52, 105, 57), 2)
            circle((x, y), 4, (56, 124, 62))
        for angle in np.linspace(0, 2 * np.pi, int(rng.integers(5, 9)), endpoint=False):
            circle((40 + round(7 * np.cos(angle)), 40 + round(7 * np.sin(angle))), 4, color)
        circle((40, 40), 4, shifted(color, -70))
    elif kind == "painted_star":
        spokes = int(rng.choice((4, 5, 6, 8)))
        points = [(40 + round((14 if i % 2 == 0 else 5) * np.cos(i * np.pi / spokes)),
                   40 + round((14 if i % 2 == 0 else 5) * np.sin(i * np.pi / spokes)))
                  for i in range(2 * spokes)]
        polygon(points, color)
        if rng.random() < .6:
            circle((40, 40), 24, shifted(color, -40), 1)
            line((13, 40), (67, 40), shifted(color, -70))
    elif kind == "jewel":
        polygon([(40, 25), (51, 35), (48, 49), (36, 53), (28, 40)], dark)
        polygon([(40, 28), (48, 36), (46, 47), (36, 49), (31, 40)], color)
        polygon([(40, 28), (40, 40), (31, 40)], bright)
        line((40, 40), (46, 47), shifted(color, -75))
    elif kind == "rivets":
        polygon([(23, 29), (57, 29), (57, 51), (23, 51)], shifted(color, -95))
        for x in (28, 40, 52):
            for y in (34, 46):
                circle((x, y), 3, color)
                circle((x - 1, y - 1), 1, bright)
    else:
        raise ValueError(f"Unknown fixture kind: {kind}")
    transform = cv2.getRotationMatrix2D((40, 40), float(rng.uniform(0, 360)), scale)
    return (cv2.warpAffine(patch, transform, (81, 81)),
            cv2.warpAffine(mask, transform, (81, 81)))


def render(seed):
    rng = np.random.default_rng(seed)
    ground = rng.integers(35, 145, 3)
    noise = rng.normal(0, rng.uniform(2, 7), (SIZE, SIZE, 1))
    image = np.clip(ground + noise, 0, 255).astype(np.uint8)
    spacing = int(rng.integers(27, 58))
    for offset in range(0, SIZE, spacing):
        cv2.line(image, (0, offset), (SIZE - 1, offset), shifted(ground, -17), 1)
        cv2.line(image, (offset, 0), (offset, SIZE - 1), shifted(ground, -12), 1)
    for _ in range(10):
        x, y = (int(v) for v in rng.integers(30, SIZE - 30, 2))
        cv2.rectangle(image, (x - 12, y - 6), (x + 12, y + 6), shifted(ground, 32), -1)
        cv2.rectangle(image, (x - 12, y - 6), (x + 12, y + 6), shifted(ground, -28), 1)
    kinds = list(LIGHT_KINDS + DECOY_KINDS)
    kinds.extend(rng.choice(LIGHT_KINDS + DECOY_KINDS, 6).tolist())
    rng.shuffle(kinds)
    points, objects = [], []
    yy, xx = np.mgrid[-40:41, -40:41]
    for index, kind in enumerate(kinds):
        x = 80 + (index % 4) * 160 + int(rng.integers(-24, 25))
        y = 80 + (index // 4) * 160 + int(rng.integers(-24, 25))
        hue = int(rng.integers(0, 180))
        saturation = int(rng.integers(70, 240))
        color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, saturation, 225]]]), cv2.COLOR_HSV2BGR)[0, 0])
        scale = float(rng.uniform(.55, 1.35))
        patch, mask = draw_object(rng, kind, color, scale)
        region = image[y - 40:y + 41, x - 40:x + 41]
        if kind in LIGHT_KINDS:
            alpha = np.exp(-(xx * xx + yy * yy) / (150 * scale ** 2)) * rng.uniform(.18, .55)
            region[:] = np.clip(region * (1 - alpha[:, :, None]) + np.asarray(color) * alpha[:, :, None], 0, 255)
            points.append([x, y])
        # Patch RGB already includes antialiasing against black, so add it once.
        region[:] = np.clip(region * (1 - mask[:, :, None] / 255.) + patch, 0, 255)
        objects.append({"kind": kind, "position": [x, y], "emits_light": kind in LIGHT_KINDS})
    illumination = np.linspace(rng.uniform(.55, 1.1), rng.uniform(.55, 1.1), SIZE)
    image = np.clip(image * illumination[None, :, None], 0, 255).astype(np.uint8)
    return image, points, objects


def generate(output, count=72, seed=901000):
    if count < 12:
        raise ValueError("Generate at least 12 maps; the last eight seeds are reserved for testing.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for index in range(count):
        image, points, objects = render(seed + index)
        name = f"light-fixtures-{seed + index}"
        image_name = f"{name}.bmp"
        if not cv2.imwrite(str(output / image_name), image):
            raise OSError(f"Could not write {image_name}")
        (output / f"{name}-objects.json").write_text(json.dumps(objects, indent=2) + "\n", encoding="utf-8")
        samples.append({"id": name, "group": name, "split": "test" if index >= count - 8 else "train",
                        "image": image_name, "points": points, "synthetic": True,
                        "negative_points": [item["position"] for item in objects if not item["emits_light"]]})
    manifest = {"format": "auto-wall-light-centers-v1", "samples": samples,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seed": seed}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for uncompressed BMP images and point labels")
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--seed", type=int, default=901000)
    args = parser.parse_args()
    print(generate(args.output, args.count, args.seed))


if __name__ == "__main__":
    main()
