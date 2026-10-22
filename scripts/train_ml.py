"""Train a compact battlemap model from local annotated images (no downloads)."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.data import CropSampler, read_dataset
from src.ml.model import predict, save_model
from src.ml.network import BattlemapNet, CLASSES


def metrics(truth, prediction):
    valid = truth != 255
    matrix = np.bincount((truth[valid].astype(int) * len(CLASSES) + prediction[valid]).ravel(),
                         minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))
    result = {}
    for cls, name in enumerate(CLASSES):
        tp, predicted, actual = int(matrix[cls, cls]), int(matrix[:, cls].sum()), int(matrix[cls].sum())
        union = predicted + actual - tp
        result[name] = {"precision": tp / predicted if predicted else None,
                        "recall": tp / actual if actual else None,
                        "iou": tp / union if union else None, "truth_pixels": actual,
                        "predicted_pixels": predicted}
    return result


def evaluate(model, samples):
    model.eval()
    results = {}
    for sample in samples:
        labels, confidence = predict(model, sample["pixels"])
        strict = np.where(confidence >= .9, labels, 0)
        results[sample["id"]] = {"argmax": metrics(sample["labels"], labels),
                                  "confidence_0.9": metrics(sample["labels"], strict)}
    return results


def train(args):
    if min(args.steps, args.batch_size, args.validate_every, args.threads) < 1:
        raise ValueError("Steps, batch size, validation interval, and threads must be positive.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment.")
    samples = read_dataset(args.manifest)
    training = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Training requires separate train and validation layout groups.")
    if not any(np.any((sample["labels"] > 0) & (sample["labels"] < len(CLASSES))) for sample in validation):
        raise ValueError("Validation needs labelled foreground pixels for checkpoint selection.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sampler = CropSampler(training, args.seed, args.crop_size)
    model = BattlemapNet(args.width, args.context_dilation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    class_weights = torch.tensor([1., 2., 4., 4., 2.], device=device)
    provenance = {"dataset_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  "splits": {split: [s["group"] for s in samples if s["split"] == split]
                             for split in ("train", "validation", "test")},
                  "labels": "Local annotations; assess their provenance before interpreting scores.",
                  "seed": args.seed, "training_steps": args.steps, "batch_size": args.batch_size,
                  "crop_size": args.crop_size, "context_dilation": args.context_dilation,
                  "augmentation": "rotations, flips, scale, hue, saturation, RGB gain, gamma, grayscale, spatial lighting",
                  "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
                      hashlib.sha256(path.read_bytes()).hexdigest() for path in
                      [Path(__file__), *(Path(__file__).resolve().parents[1] / "src/ml").glob("*.py")]},
                  "torch": torch.__version__, "device": device, "experimental": True}
    history, best, started = [], -1., time.perf_counter()
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), **provenance}), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        images, labels = sampler.batch(args.batch_size)
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(labels).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(images)
            ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=255)
            valid = labels != 255
            targets = F.one_hot(labels.clamp(max=len(CLASSES) - 1), len(CLASSES)).permute(0, 3, 1, 2)
            probabilities = logits.float().softmax(1) * valid[:, None]
            targets = targets * valid[:, None]
            intersection = (probabilities * targets).sum((0, 2, 3))
            denominator = (probabilities + targets).sum((0, 2, 3))
            dice = 1 - ((2 * intersection + 1) / (denominator + 1))[1:].mean()
            loss = ce + dice
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()
        for group in optimizer.param_groups:
            group["lr"] = .001 * (.1 + .9 * (1 - step / args.steps))
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            result = evaluate(model, validation)
            score = float(np.mean([entry["argmax"][name]["iou"] for entry in result.values()
                                   for name in CLASSES[1:] if entry["argmax"][name]["truth_pixels"]]))
            event = {"step": step, "mean_foreground_iou": score, "validation": result}
            history.append(event)
            print(json.dumps(event), flush=True)
            if score > best:
                best = score
                save_model(output / "best.safetensors", model, {**provenance, "selected_step": step, "validation": result})
            (output / "training.json").write_text(json.dumps({**provenance, "history": history,
                "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n", encoding="utf-8")
    save_model(output / "last.safetensors", model, {**provenance, "selected_step": args.steps})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, choices=(8, 16, 24, 32), default=16)
    parser.add_argument("--context-dilation", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--crop-size", type=int, choices=(128, 256, 384, 512), default=384)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
