"""Prepare reviewed public light annotations from byte-verified local downloads.

Reads existing source archives; writes only uncompressed BMP images and JSON.
Downloads, source licenses, label choices, and split limitations are documented
in docs/OPEN_LIGHT_DATA.md. This does not run a detector or infer any labels.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.light_data import read_light_dataset


SPECIFICATION = Path(__file__).resolve().parents[1] / "benchmarks/data/open_light_maps.json"


def checked_bytes(data, expected, name):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Source SHA-256 mismatch: {name}")
    return data


def prepare_dataset(source_dir, output, specification):
    """Keep original pixels, splits, exclusions, and pixel-center coordinates."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {output}")
    sources = specification["sources"]
    blobs = {key: checked_bytes((source_dir / info["file"]).read_bytes(), info["sha256"], info["file"])
             for key, info in sources.items()}
    prepared = []
    for sample in specification["samples"]:
        data = blobs[sample["source"]]
        if "member" in sample:
            # Read only the named image; never extract paths or execute archive contents.
            with ZipFile(io.BytesIO(data)) as archive:
                data = archive.read(sample["member"])
        checked_bytes(data, sample["image_sha256"], sample["id"])
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or list(image.shape[1::-1]) != sample["source_size"]:
            raise ValueError(f"Source image dimensions changed: {sample['id']}")
        size = tuple(sample["output_size"])
        interpolation = {"area": cv2.INTER_AREA, "nearest": cv2.INTER_NEAREST}[sample["interpolation"]]
        valid = np.full(image.shape[:2], 255, np.uint8)
        for x0, y0, x1, y1 in sample["ignore_rectangles"]:
            valid[y0:y1, x0:x1] = 0
        image = cv2.resize(image, size, interpolation=interpolation)
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
        scale = np.asarray(size) / sample["source_size"]
        item = {key: sample[key] for key in ("id", "group", "split", "annotation")}
        item.update(image=sample["id"] + ".bmp", synthetic=False,
                    provenance={**sources[sample["source"]], "image_sha256": sample["image_sha256"],
                                "source_size": sample["source_size"], "output_size": list(size),
                                "interpolation": sample["interpolation"]})
        for key in ("points", "negative_points"):
            points = np.asarray(sample[key], dtype=np.float64).reshape(-1, 2)
            item[key] = ((points + .5) * scale - .5).tolist()
        if sample["ignore_rectangles"]:
            item["valid_mask"] = sample["id"] + "-valid.bmp"
        prepared.append((item, image, valid))

    output.mkdir(parents=True, exist_ok=False)
    for item, image, valid in prepared:
        for key, pixels in (("image", image), ("valid_mask", valid)):
            if key in item:
                ok, encoded = cv2.imencode(".bmp", pixels)
                if not ok:
                    raise OSError(f"Could not encode {item[key]}")
                encoded.tofile(output / item[key])
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"format": "auto-wall-light-centers-v1",
                                   "annotation_review": specification["annotation_review"],
                                   "split_notes": specification["split_notes"],
                                   "samples": [item for item, _, _ in prepared]}, indent=2) + "\n", encoding="utf-8")
    samples = read_light_dataset(manifest)
    print(json.dumps({"samples": len(samples), "sources": sum(len(s["points"]) for s in samples),
                      "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing the three unchanged public downloads")
    parser.add_argument("output", type=Path, help="New directory for BMP images and annotations")
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.output, json.loads(SPECIFICATION.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
