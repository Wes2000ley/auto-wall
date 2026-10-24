"""Prepare partial, image-verified encounter masks using frozen layout splits."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.encounter_data import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Read-only original encounter map folder")
    parser.add_argument("annotations", type=Path, help="Reviewed per-site polygon/stroke JSON files")
    parser.add_argument("splits", type=Path, help="Frozen site-to-split JSON, created before training")
    parser.add_argument("output", type=Path, help="New directory for uncompressed BMP masks and manifest")
    parser.add_argument("--replay", type=Path, help="Optional legacy real-map manifest; preserve its layout splits")
    parser.add_argument("--replay-weight", type=float, default=1.0,
                        help="Relative crop sampling weight for each replay training map (default: 1)")
    parser.add_argument("--review", type=Path, required=True,
                        help="Separate accepted reviews matching each source, annotation, and raster SHA-256")
    args = parser.parse_args()
    manifest = prepare_dataset(args.source, args.annotations, args.splits, args.output,
                               args.replay, args.review, args.replay_weight)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"manifest": str(manifest), "samples": len(data["samples"]),
                      "splits": {split: sum(item["split"] == split for item in data["samples"])
                                 for split in ("train", "validation", "test")}}))


if __name__ == "__main__":
    main()
