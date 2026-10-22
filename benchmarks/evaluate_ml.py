"""Review custom segmentation on whole maps, with raw timings and BMP overlays."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_ml import metrics
from src.ml.data import read_dataset
from src.ml.model import load_model, predict
from src.ml.network import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.runs, args.threads) < 1:
        parser.error("Runs and threads must be positive.")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    model, details = load_model(args.model, args.device)
    load_seconds = time.perf_counter() - tick
    report = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(), "details": details,
              "device": str(next(model.parameters()).device), "load_seconds": load_seconds,
              "metric_caveat": "Agreement with supplied annotations; weak labels are not independent ground truth.",
              "classes": CLASSES, "samples": []}
    for sample in read_dataset(args.manifest):
        if args.split != "all" and sample["split"] != args.split:
            continue
        image = sample["pixels"]
        times = []
        for index in range(args.runs + 1):
            tick = time.perf_counter()
            labels, confidence = predict(model, image)
            duration = time.perf_counter() - tick
            if index:
                times.append(duration)
            else:
                cold = duration
        strict = np.where(confidence >= .9, labels, 0).astype(np.uint8)
        overlay = image.copy()
        for cls, color in ((1, (60,230,60)), (2, (255,70,255)), (3, (40,220,255)), (4, (255,180,40))):
            overlay[strict == cls] = color
        overlay = cv2.addWeighted(image, .35, overlay, .65, 0)
        for suffix, pixels in (("review", overlay), ("classes", labels), ("strict", strict)):
            success, encoded = cv2.imencode(".bmp", pixels)
            if not success:
                raise ValueError("Could not encode review image.")
            (args.output / f"{sample['id']}-{suffix}.bmp").write_bytes(encoded.tobytes())
        result = {"id": sample["id"], "split": sample["split"], "group": sample["group"],
                  "size": list(image.shape[:2][::-1]), "first_inference_seconds": cold,
                  "warm_seconds": times, "median_seconds": statistics.median(times),
                  "argmax": metrics(sample["labels"], labels), "confidence_0.9": metrics(sample["labels"], strict)}
        report["samples"].append(result)
        print(json.dumps(result), flush=True)
    if next(model.parameters()).device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
