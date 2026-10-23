"""Train a battlemap decoder from local annotations and optional local weights."""

import argparse
from dataclasses import asdict, replace
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
from src.ml.model import load_model, predict, save_model
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


def evaluate(model, samples, geometry_settings=None):
    model.eval()
    results = {}
    for sample in samples:
        labels, confidence = predict(model, sample["pixels"])
        strict = np.where(confidence >= .9, labels, 0)
        results[sample["id"]] = {"argmax": metrics(sample["labels"], labels),
                                  "confidence_0.9": metrics(sample["labels"], strict)}
        if geometry_settings is not None and "geometry_annotations" in sample:
            from benchmarks.check_geometry import evaluate_geometry
            from src.core.fast_pipeline import detect_map, to_uvtt
            detection = detect_map(sample["pixels"], replace(geometry_settings, max_dimension=0),
                                   prediction=(labels, confidence))
            results[sample["id"]]["geometry"] = evaluate_geometry(
                to_uvtt(detection, geometry_settings.grid_size), sample["geometry_annotations"])
    return results


def geometry_score(result):
    """Mean ray correctness and precision-weighted instance F0.5 on complete labels."""
    scores = []
    for sample in result.values():
        geometry = sample.get("geometry", {})
        if geometry.get("rays_total"):
            scores.append(geometry["rays_passed"] / geometry["rays_total"])
        for name in ("doors", "lights"):
            instances = geometry.get(name, {})
            if instances.get("complete_annotation"):
                true = instances["matched"]
                false = instances["prediction_count"] - true
                missed = instances["truth_count"] - true
                denominator = 5 * true + 4 * false + missed
                if denominator:
                    scores.append(5 * true / denominator)
    if not scores:
        raise ValueError("Geometry selection needs visibility rays or complete instance annotations.")
    return float(np.mean(scores))


def training_optimizer(model, learning_rate, backbone_layers=0):
    """Keep pretrained adaptation ten times slower than decoder learning."""
    if backbone_layers and model.architecture != "dino":
        raise ValueError("Backbone fine-tuning requires a DINO model.")
    if model.architecture == "dino":
        model.train_backbone_layers(backbone_layers)
    decoder, backbone = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("backbone.") else decoder).append(parameter)
    groups = [{"params": decoder, "lr": learning_rate, "initial_lr": learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": learning_rate * .1, "initial_lr": learning_rate * .1})
    return torch.optim.AdamW(groups, weight_decay=.0001)


def train(args):
    if min(args.steps, args.batch_size, args.validate_every, args.threads) < 1:
        raise ValueError("Steps, batch size, validation interval, and threads must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite.")
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
    geometry_settings = None
    if args.selection == "geometry":
        from src.core.detection_profiles import load_settings
        from src.core.fast_pipeline import FastSettings
        if not any("geometry_annotations" in sample for sample in validation):
            raise ValueError("Geometry selection needs a validation sample's geometry annotation file.")
        geometry_settings = load_settings(args.geometry_settings) if args.geometry_settings else FastSettings()
    elif args.geometry_settings:
        raise ValueError("--geometry-settings requires --selection geometry.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sampler = CropSampler(training, args.seed, args.crop_size)
    backbone_hash = None
    initial_hash = None
    if args.init_model:
        if args.backbone:
            raise ValueError("Choose --init-model or --backbone, not both.")
        model, _ = load_model(args.init_model, "cpu")
        if getattr(model, "light_head", None) is not None:
            raise ValueError("Use the segmentation-only base for segmentation training; changing its features would invalidate the trained light head.")
        if (model.architecture, model.width, model.context_dilation) != (args.architecture, args.width, args.context_dilation):
            raise ValueError("Initial model does not match the requested architecture, width, and context dilation.")
        initial_hash = hashlib.sha256(args.init_model.read_bytes()).hexdigest()
    elif args.architecture == "dino":
        if args.width != 16 or args.context_dilation != 4 or args.backbone is None:
            raise ValueError("DINO needs --backbone with local DINOv2-small safetensors, width 16, and dilation 4.")
        from safetensors.torch import load_file
        from src.ml.dino import DinoBattlemap
        model = DinoBattlemap()
        state = load_file(args.backbone)
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("The pretrained backbone contains nonfinite weights.")
        model.backbone.load_state_dict(state, strict=True)
        backbone_hash = hashlib.sha256(args.backbone.read_bytes()).hexdigest()
    else:
        if args.backbone is not None:
            raise ValueError("--backbone is only supported with --architecture dino.")
        model = BattlemapNet(args.width, args.context_dilation)
    model = model.to(device)
    optimizer = training_optimizer(model, args.learning_rate, args.backbone_layers)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    class_weights = torch.tensor([1., 2., 4., 4., 2.], device=device)
    provenance = {"dataset_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  "splits": {split: [s["group"] for s in samples if s["split"] == split]
                             for split in ("train", "validation", "test")},
                  "labels": "Local annotations; assess their provenance before interpreting scores.",
                  "architecture": args.architecture, "backbone_sha256": backbone_hash,
                  "initial_model_sha256": initial_hash, "learning_rate": args.learning_rate,
                  "backbone_training_layers": args.backbone_layers,
                  "backbone_learning_rate": args.learning_rate * .1 if args.backbone_layers else None,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "selection": args.selection,
                  "geometry_settings": asdict(geometry_settings) if geometry_settings else None,
                  "data_hashes": [{key: sample[key] for key in ("id", "pixels_sha256", "labels_sha256", "geometry_sha256")
                                   if key in sample} for sample in samples],
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
            group["lr"] = group["initial_lr"] * (.1 + .9 * (1 - step / args.steps))
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            result = evaluate(model, validation, geometry_settings)
            overlap = float(np.mean([entry["argmax"][name]["iou"] for entry in result.values()
                                     for name in CLASSES[1:] if entry["argmax"][name]["truth_pixels"]]))
            score = .6 * geometry_score(result) + .4 * overlap if geometry_settings else overlap
            event = {"step": step, "mean_foreground_iou": overlap, "selection_score": score, "validation": result}
            history.append(event)
            print(json.dumps(event), flush=True)
            if score > best:
                best = score
                save_model(output / "best.safetensors", model, {**provenance, "selected_step": step,
                           "selection_score": score, "validation": result})
            (output / "training.json").write_text(json.dumps({**provenance, "history": history,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
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
    parser.add_argument("--architecture", choices=("compact", "dino"), default="compact")
    parser.add_argument("--backbone", type=Path, help="Local DINOv2-small safetensors; never downloaded automatically")
    parser.add_argument("--backbone-layers", type=int, choices=range(5), default=0,
                        help="Fine-tune this many final DINO blocks at 0.1x decoder learning rate (default: frozen)")
    parser.add_argument("--init-model", type=Path, help="Start from an Auto-Wall checkpoint with a fresh optimizer")
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--selection", choices=("pixels", "geometry"), default="pixels")
    parser.add_argument("--geometry-settings", type=Path, help="Detector settings used for geometry validation")
    parser.add_argument("--context-dilation", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--crop-size", type=int, choices=(128, 256, 384, 512), default=384)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
