"""Train only physical-light centers; freeze every segmentation tensor.

The point heatmap and penalty-reduced focal loss use the method described in
Objects as Points: https://arxiv.org/abs/1904.07850. No remote code is executed.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.check_geometry import match_instances
from src.core.light_geometry import light_center_points
from src.ml.light_centers import LightCenterHead
from src.ml.light_data import LightCropSampler, read_light_dataset
from src.ml.model import load_model, predict, save_model


def center_focal_loss(logits, target, valid, normalization="batch"):
    logits = logits.float().squeeze(1)
    probability = logits.sigmoid()
    positive = (target == 1) & valid
    negative = (target < 1) & valid
    positive_loss = -F.logsigmoid(logits) * (1 - probability).square() * positive
    negative_loss = -F.logsigmoid(-logits) * probability.square() * (1 - target).pow(4) * negative
    if normalization == "batch":
        return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1)
    if normalization == "sample":
        # Give a sparse torch crop the same weight as a crop full of fixtures.
        counts = positive.sum(dim=(1, 2)).clamp_min(1)
        return ((positive_loss + negative_loss).sum(dim=(1, 2)) / counts).mean()
    raise ValueError("Light loss normalization must be batch or sample.")


def segmentation_hashes(model):
    return {key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for key, value in model.state_dict().items() if not key.startswith("light_head.")}


def local_light_hashes(head):
    """Hash the original RGB/decoder inputs and all following light-head weights."""
    result = {}
    for key, value in head.layers.state_dict().items():
        if key == "0.weight":
            value = value[:, :19]
        result[key] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    return result


def light_optimizer(head, learning_rate, context_only=False):
    if context_only and not head.semantic_context:
        raise ValueError("Context-only training needs --semantic-context or an existing semantic light model.")
    head.requires_grad_(not context_only)
    parameters = head.parameters()
    if context_only:
        head.context_projection.requires_grad_(True)
        head.layers[0].weight.requires_grad_(True)
        # The first convolution mixes fixed local inputs with trainable context.
        # Masking its local gradients also requires disabling weight decay here.
        parameters = [{"params": head.context_projection.parameters()},
                      {"params": [head.layers[0].weight], "weight_decay": 0.}]
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.0001)


def evaluate(model, samples, threshold):
    report, matched, false, missed = {}, 0, 0, 0
    for sample in samples:
        _, _, heatmap = predict(model, sample["pixels"], include_light_centers=True)
        points = light_center_points(heatmap, threshold)
        all_count = len(points)
        indices = np.rint(points).astype(int)
        points = points[sample["valid"][indices[:, 1], indices[:, 0]] > 0]
        tolerance = 6 * max(sample["valid"].shape) / 1024
        result = match_instances(np.linalg.norm(sample["points"][:, None] - points[None], axis=-1), tolerance)
        result.update({"positions": points.tolist(), "all_peak_count": all_count,
                       "complete_annotation": bool(sample["valid"].all())})
        matched += result["matched"]
        false += len(result["extra"])
        missed += len(result["missed"])
        report[sample["id"]] = result
    return {"score": 5 * matched / max(1, 5 * matched + 4 * false + missed),
            "matched": matched, "false": false, "missed": missed, "samples": report}


def train(args):
    if (args.steps < 1 or args.batch_size < 1 or not 0 <= args.synthetic_per_batch <= args.batch_size
            or not math.isfinite(args.lr) or args.lr <= 0 or not .05 <= args.threshold <= 1):
        raise ValueError("Use positive steps, batch size and learning rate, a valid synthetic crop count, and a score in [0.05, 1].")
    if args.loss_normalization not in ("batch", "sample"):
        raise ValueError("Light loss normalization must be batch or sample.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    data = read_light_dataset(args.manifest)
    validation = [s for s in data if s["split"] == "validation"]
    if not validation or not any(len(s["points"]) for s in validation):
        raise ValueError("Supply held-out validation layouts with marked light centers.")
    real = [s for s in data if s["split"] == "train" and not s["synthetic"]]
    synthetic = [s for s in data if s["split"] == "train" and s["synthetic"]]
    synthetic_count = args.synthetic_per_batch if real else args.batch_size
    if not synthetic:
        synthetic_count = 0
    counts = (args.batch_size - synthetic_count, synthetic_count)
    samplers = [(LightCropSampler(group, args.seed + index, args.crop_size), count)
                for index, (group, count) in enumerate(zip((real, synthetic), counts)) if count]
    loaded, _ = load_model(args.init_model, args.device)
    model = copy.deepcopy(loaded)
    if model.architecture != "dino":
        raise ValueError("Light-center training requires a local DINO Auto-Wall model.")
    device = next(model.parameters()).device
    model.requires_grad_(False).eval()
    if args.context_only and model.light_head is None:
        raise ValueError("Context-only training requires an existing light-head checkpoint.")
    if model.light_head is None:
        model.light_head = LightCenterHead(semantic_context=args.semantic_context).to(device)
    elif args.semantic_context and not model.light_head.semantic_context:
        model.light_head = model.light_head.with_semantic_context()
    before = segmentation_hashes(model)
    optimizer = light_optimizer(model.light_head, args.lr, args.context_only)
    local_before = local_light_hashes(model.light_head) if args.context_only else {}
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [root / name for name in ("scripts/train_light_centers.py", "src/ml/light_data.py",
               "src/ml/light_centers.py", "src/ml/dino.py", "src/ml/model.py",
               "src/core/light_geometry.py", "benchmarks/check_geometry.py")]
    details = {"method": "frozen segmentation with physical light centers", "seed": args.seed,
               "base_sha256": hashlib.sha256(args.init_model.read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "steps": args.steps, "batch_size": args.batch_size, "real_crops": counts[0],
               "synthetic_crops": counts[1], "crop_size": args.crop_size, "lr": args.lr,
               "threshold": args.threshold, "loss_normalization": args.loss_normalization,
               "semantic_context": model.light_head.semantic_context,
               "context_only": args.context_only,
               "torch": torch.__version__, "device": str(device),
               "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "samples": [{key: s[key] for key in ("id", "group", "split", "synthetic", "pixels_sha256",
                                                    "points_sha256", "negative_points_sha256", "valid_sha256")} for s in data]}
    (args.output / "run.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    started, best = time.perf_counter(), -1.
    with (args.output / "history.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            batches = [sampler.batch(count) for sampler, count in samplers]
            arrays = [np.concatenate([batch[index] for batch in batches]) for index in range(3)]
            image, target, valid = [torch.from_numpy(array).to(device) for array in arrays]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                features, context = model.forward_features(image, include_context=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = center_focal_loss(model.light_head(features, image, context), target, valid, args.loss_normalization)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite light loss at step {step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.context_only:
                model.light_head.layers[0].weight.grad[:, :19].zero_()
            torch.nn.utils.clip_grad_norm_(model.light_head.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (.2 + .8 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            if step % 100 == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach()), "seconds": time.perf_counter() - started}), flush=True)
            if step % 150 == 0 or step == args.steps:
                metrics = evaluate(model, validation, args.threshold)
                item = {"step": step, "loss": float(loss.detach()), "seconds": time.perf_counter() - started, **metrics}
                log.write(json.dumps(item) + "\n")
                log.flush()
                if segmentation_hashes(model) != before:
                    raise RuntimeError("Segmentation tensors changed during light-only training.")
                if args.context_only and local_light_hashes(model.light_head) != local_before:
                    raise RuntimeError("Fixed local light weights changed during context-only training.")
                checkpoint = {**details, "selected_step": step, "validation": metrics,
                              "segmentation_tensors_unchanged": len(before),
                              "local_light_weight_parts_unchanged": len(local_before)}
                save_model(args.output / "last.safetensors", model, checkpoint)
                if metrics["score"] > best:
                    best = metrics["score"]
                    save_model(args.output / "best.safetensors", model, checkpoint)
                    (args.output / "best.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({key: value for key, value in item.items() if key != "samples"}), flush=True)
    report = {"seconds": time.perf_counter() - started, "segmentation_tensors_unchanged": len(before),
              "local_light_weight_parts_unchanged": len(local_before)}
    if device.type == "cuda":
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "completed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="New directory for raw safetensors and reports")
    parser.add_argument("--init-model", type=Path, required=True, help="Local DINO model, optionally with an existing light head")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--synthetic-per-batch", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=73113)
    parser.add_argument("--lr", type=float, default=.0005)
    parser.add_argument("--threshold", type=float, default=.35)
    parser.add_argument("--loss-normalization", choices=("batch", "sample"), default="batch",
                        help="Normalize by all centers in a batch, or weight each crop equally")
    parser.add_argument("--semantic-context", action="store_true",
                        help="Give the light head direct access to shared frozen DINO features")
    parser.add_argument("--context-only", action="store_true",
                        help="Train only the added DINO inputs of an existing light model; freeze its local weights")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
