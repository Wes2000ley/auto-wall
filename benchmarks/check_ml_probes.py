"""Check hand-labelled points and regions against a saved segmentation mask."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def check_probes(labels, specification):
    scale = labels.shape[1] / specification["reference_width"]
    checks = []
    for item in specification["checks"]:
        x, y, radius = (round(item[key] * scale) for key in ("x", "y", "radius"))
        if radius < 1 or not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
            raise ValueError("Probe is outside the image or too small at this resolution.")
        region = labels[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
        checks.append({**item, "matching_fraction": float(np.mean(region == item["class"])),
                       "opaque_wall_fraction": float(np.mean(region == 1)),
                       "door_fraction": float(np.mean(region == 2)),
                       "light_fraction": float(np.mean(region == 3))})
    return {"description": specification["description"], "image_id": specification["image_id"], "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask", type=Path)
    parser.add_argument("probes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    labels = cv2.imdecode(np.fromfile(args.mask, np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.ndim != 2:
        parser.error("Mask must contain single-channel class IDs.")
    result = check_probes(labels, json.loads(args.probes.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in result["checks"]:
        print(f"{item['name']}: expected fraction={item['matching_fraction']:.3f}; opaque={item['opaque_wall_fraction']:.3f}")


if __name__ == "__main__":
    main()
