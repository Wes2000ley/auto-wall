"""Place complete reviewed artist sprites in varied scenes for light training.

Only the supplied sprite sheet is read. No maps, predictions, or downloaded code
are used. Output images are uncompressed BMPs; the last eight seeds are reserved
for testing. Shared artwork does not make these independent art-style tests.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_light_maps import SIZE, render_background


def load_sprites(specification, sheet):
    spec = json.loads(specification.read_text(encoding="utf-8"))
    if spec.get("format") != "auto-wall-light-sprites-v1" or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", spec.get("id", "")):
        raise ValueError("Use a light-sprite specification with a lowercase alphanumeric ID.")
    raw = sheet.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sheet_sha256"]:
        raise ValueError("Sprite sheet SHA-256 mismatch.")
    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if (pixels is None or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4
            or list(pixels.shape[1::-1]) != spec["sheet_size"]):
        raise ValueError("Sprite sheet must be uint8 BGRA with the specified dimensions.")
    families = {}
    for family in spec["families"]:
        name = family["name"]
        if not isinstance(name, str) or not name or name in families or type(family["emits_light"]) is not bool:
            raise ValueError("Sprite families need unique names and boolean light labels.")
        variants = []
        for rectangle in family["rectangles"]:
            if len(rectangle) != 4 or any(type(value) is not int for value in rectangle):
                raise ValueError("Sprite rectangles must contain integer x, y, width, height.")
            x, y, width, height = rectangle
            if (min(x, y) < 0 or min(width, height) < 1 or max(width, height) > 64
                    or x + width > pixels.shape[1] or y + height > pixels.shape[0]):
                raise ValueError("Sprite rectangles must fit the sheet and be at most 64 pixels across.")
            center = np.asarray(family["center"], dtype=float)
            if (center.shape != (2,) or not np.isfinite(center).all()
                    or (center < 0).any() or (center > [width - 1, height - 1]).any()):
                raise ValueError("Each sprite center must be a finite XY point inside its rectangle.")
            patch = pixels[y:y + height, x:x + width].copy()
            if not patch[:, :, 3].any():
                raise ValueError("A sprite cannot be entirely transparent.")
            variants.append({"pixels": patch, "center": center, "rectangle": rectangle,
                             "emits_light": family["emits_light"]})
        if not variants:
            raise ValueError("Each sprite family needs at least one reviewed variant.")
        families[name] = variants
    if not 2 <= len(families) <= 16 or {v[0]["emits_light"] for v in families.values()} != {False, True}:
        raise ValueError("Supply two to sixteen families including emitters and non-emissive lookalikes.")
    return spec, families


def place_sprite(image, sprite, x, y, longest_edge, turns, flip):
    """Transform pixels and their source center together, without clipping."""
    patch = sprite["pixels"]
    height, width = patch.shape[:2]
    cx, cy = sprite["center"]
    for _ in range(turns):
        patch = np.rot90(patch)
        cx, cy = cy, width - 1 - cx
        height, width = width, height
    if flip:
        patch = patch[:, ::-1]
        cx = width - 1 - cx
    scale = longest_edge / max(height, width)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    patch = cv2.resize(patch, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    center = [(cx + .5) * new_width / width - .5, (cy + .5) * new_height / height - .5]
    x0, y0 = x - new_width // 2, y - new_height // 2
    if min(x0, y0) < 0 or x0 + new_width > image.shape[1] or y0 + new_height > image.shape[0]:
        raise ValueError("Sprite placement must remain inside the scene.")
    region = image[y0:y0 + new_height, x0:x0 + new_width]
    alpha = patch[:, :, 3:] / 255.
    region[:] = np.rint(region * (1 - alpha) + patch[:, :, :3] * alpha).astype(np.uint8)
    return [center[0] + x0, center[1] + y0], [x0, y0, new_width, new_height]


def render(seed, families):
    rng = np.random.default_rng(seed)
    image = render_background(rng)
    kinds = list(families)
    kinds.extend(rng.choice(list(families), 16 - len(kinds)).tolist())
    rng.shuffle(kinds)
    points, objects = [], []
    for index, kind in enumerate(kinds):
        variants = families[kind]
        sprite = variants[int(rng.integers(len(variants)))]
        x = 80 + (index % 4) * 160 + int(rng.integers(-24, 25))
        y = 80 + (index // 4) * 160 + int(rng.integers(-24, 25))
        turns, flip = int(rng.integers(4)), bool(rng.random() < .5)
        edge = int(rng.integers(24, 81))
        center, rectangle = place_sprite(image, sprite, x, y, edge, turns, flip)
        if sprite["emits_light"]:
            points.append(center)
        objects.append({"kind": kind, "position": center, "emits_light": sprite["emits_light"],
                        "source_rectangle": sprite["rectangle"], "placed_rectangle": rectangle,
                        "quarter_turns": turns, "horizontal_flip": flip})
    illumination = np.linspace(rng.uniform(.55, 1.1), rng.uniform(.55, 1.1), SIZE)
    return np.clip(image * illumination[None, :, None], 0, 255).astype(np.uint8), points, objects


def generate(specification, sheet, output, count=136, seed=921000):
    if count < 12:
        raise ValueError("Generate at least twelve scenes; the last eight seeds are reserved for testing.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a sprite dataset: {output}")
    spec, families = load_sprites(specification, sheet)
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for index in range(count):
        image, points, objects = render(seed + index, families)
        name = f"sprite-lights-{spec['id']}-{seed + index}"
        if not cv2.imwrite(str(output / (name + ".bmp")), image):
            raise OSError(f"Could not write sprite scene: {name}")
        (output / (name + "-objects.json")).write_text(json.dumps(objects, indent=2) + "\n", encoding="utf-8")
        samples.append({"id": name, "group": name, "split": "test" if index >= count - 8 else "train",
                        "image": name + ".bmp", "points": points, "synthetic": True,
                        "negative_points": [item["position"] for item in objects if not item["emits_light"]]})
    sources = (Path(__file__), Path(__file__).with_name("generate_light_maps.py"))
    manifest = {"format": "auto-wall-light-centers-v1", "samples": samples, "seed": seed,
                "sprite_specification": spec, "specification_sha256": hashlib.sha256(specification.read_bytes()).hexdigest(),
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                "split_notes": "Different generated arrangements share one artist's sprites. Reserved seeds test arrangement/scale, not new styles or unseen object designs."}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("sheet", type=Path, help="Local original sprite sheet; bytes must match the specification")
    parser.add_argument("output", type=Path, help="New directory for uncompressed BMP scenes and point labels")
    parser.add_argument("--count", type=int, default=136)
    parser.add_argument("--seed", type=int, default=921000)
    args = parser.parse_args()
    print(generate(args.specification, args.sheet, args.output, args.count, args.seed))


if __name__ == "__main__":
    main()
