"""Score one fixed light checkpoint against explicitly selected dataset splits."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_light_centers import evaluate
from src.ml.light_data import read_light_dataset
from src.ml.model import load_model


def run(manifest, checkpoint, output, splits, threshold=.35, device="auto"):
    if not .05 <= threshold <= 1:
        raise ValueError("Light threshold must be in [0.05, 1].")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an evaluation: {output}")
    samples = [s for s in read_light_dataset(manifest) if s["split"] in splits]
    if not samples:
        raise ValueError("No samples in the selected splits.")
    model, details = load_model(checkpoint, device)
    if getattr(model, "light_head", None) is None:
        raise ValueError("Evaluation requires a checkpoint with a light-center head.")
    torch.set_num_threads(4)
    started = time.perf_counter()
    metrics = evaluate(model, samples, threshold)
    elapsed = time.perf_counter() - started
    root = Path(__file__).resolve().parents[1]
    source_names = ("benchmarks/evaluate_lights.py", "scripts/train_light_centers.py",
                    "benchmarks/check_geometry.py", "src/core/light_geometry.py", "src/ml/model.py",
                    "src/ml/dino.py", "src/ml/light_centers.py", "src/ml/light_data.py")
    report = {"model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
              "selected_step": details.get("selected_step"), "splits": sorted(set(splits)),
              "threshold": threshold, "device": str(next(model.parameters()).device),
              "evaluation_seconds": elapsed,
              "timing_scope": "Inference, center decoding, and scoring; excludes imports, model loading, image decoding, and rendering.",
              "annotation_review": "Scores measure agreement with the manifest's reviewed points. Unlabelled pixels are ignored.",
              **metrics}
    output.mkdir(parents=True, exist_ok=False)
    for sample in samples:
        result = report["samples"][sample["id"]]
        result.update({key: sample[key] for key in ("group", "split", "synthetic", "pixels_sha256",
                                                   "points_sha256", "negative_points_sha256", "valid_sha256")})
        result["truth_positions"] = sample["points"].tolist()
        result["reviewed_fraction"] = float(sample["valid"].mean())
        result["matching_tolerance_pixels"] = 6 * max(sample["valid"].shape) / 1024
        canvas = sample["pixels"].copy()
        canvas[sample["valid"] == 0] //= 3
        for index, point in enumerate(sample["points"]):
            xy = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, xy, 9, (60, 220, 60), 1)
            if index in result["missed"]:
                cv2.drawMarker(canvas, xy, (60, 220, 60), cv2.MARKER_TILTED_CROSS, 13, 1)
        for index, point in enumerate(result["positions"]):
            xy = tuple(np.rint(point).astype(int))
            color = (40, 40, 240) if index in result["extra"] else (240, 220, 40)
            cv2.circle(canvas, xy, 5, color, 2)
        ok, encoded = cv2.imencode(".bmp", canvas)
        if not ok:
            raise OSError(f"Could not encode preview: {sample['id']}")
        encoded.tofile(output / f"{sample['id']}.bmp")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_sha256", "splits", "score", "matched", "false", "missed")}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="New directory for metrics and raw BMP previews")
    parser.add_argument("--split", nargs="+", choices=("train", "validation", "test"), default=["validation"])
    parser.add_argument("--threshold", type=float, default=.35)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args.manifest, args.checkpoint, args.output, args.split, args.threshold, args.device)


if __name__ == "__main__":
    main()
