"""Evaluate an external CubiCasa5k model without adding it to Auto-Wall.

Requires PyTorch, an inspected local architecture file defining
hg_furukawa_original, and the official 44-channel checkpoint. No downloads or
training occur here. The external code/weights have their own license.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def prepare(image, max_edge):
    ratio = min(1.0, max_edge / max(image.shape[:2]))
    if ratio < 1:
        image = cv2.resize(image, (max(1, round(image.shape[1] * ratio)),
                                   max(1, round(image.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    return image, tensor * (2 / 255) - 1


def predict(model, tensor, rotations):
    import torch
    import torch.nn.functional as functional

    predictions = []
    for rotation in range(rotations):
        output = model(torch.rot90(tensor, rotation, (-2, -1)))
        # Junction heatmaps have orientation-specific channel permutations; they
        # are not used here. Room/icon logits only require the spatial inverse.
        output = torch.rot90(output[:, 21:], -rotation, (-2, -1))
        predictions.append(functional.interpolate(output, size=tensor.shape[-2:],
                                                  mode="bilinear", align_corners=True))
    return torch.stack(predictions).mean(0)[0]


def masks_and_stats(logits, threshold):
    rooms = logits[:12].softmax(0).cpu().numpy()
    icons = logits[12:].softmax(0).cpu().numpy()
    walls = (rooms.argmax(0) == 2) & (rooms[2] >= threshold)
    doors = (icons.argmax(0) == 2) & (icons[2] >= threshold)
    return walls, doors, {
        "threshold": threshold, "wall_pixels": int(walls.sum()), "door_pixels": int(doors.sum()),
        "wall_fraction": float(walls.mean()), "door_fraction": float(doors.mean()),
        "door_mask_components": cv2.connectedComponents(doors.astype(np.uint8), connectivity=8)[0] - 1,
    }


def render(image, walls, doors):
    colors = image.copy()
    colors[walls] = (60, 230, 60)
    colors[doors] = (255, 70, 255)
    return cv2.addWeighted(image, 0.3, colors, 0.7, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model-source", type=Path, required=True, help="Inspected external Python architecture")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--rotations", type=int, choices=(1, 4), default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.runs < 1 or args.max_edge < 128 or args.threads < 1:
        parser.error("Runs/threads must be positive and max-edge at least 128.")
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this PyTorch environment.")
    torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("external_cubicasa_model", args.model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tick = time.perf_counter()
    model = module.hg_furukawa_original(44)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    model.to(args.device).eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
    report = {"model": "CubiCasa5k hg_furukawa_original, 44 channels", "device": args.device,
              "torch": torch.__version__, "threads": args.threads, "rotations": args.rotations,
              "model_load_seconds": time.perf_counter() - tick,
              "checkpoint_sha256": digest(args.checkpoint), "source_sha256": digest(args.model_source),
              "max_edge": args.max_edge, "images": [],
              "note": "Mask counts are not accuracy metrics; overlays require independent visual review."}
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for number, path in enumerate(args.images):
            tick = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode {path}")
            original_size = [image.shape[1], image.shape[0]]
            image, array = prepare(image, args.max_edge)
            tensor = torch.from_numpy(array).unsqueeze(0).to(args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            prepare_seconds = time.perf_counter() - tick
            durations = []
            for _ in range(args.runs + 1):
                tick = time.perf_counter()
                logits = predict(model, tensor, args.rotations)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite predictions.")
            panels = [image]
            stats = []
            for threshold in (0.5, 0.8):
                walls, doors, counts = masks_and_stats(logits, threshold)
                stats.append(counts)
                panels.append(render(image, walls, doors))
                for label, mask in (("walls", walls), ("doors", doors)):
                    destination = args.output / f"{number:02d}-{label}-{threshold}.bmp"
                    if not cv2.imwrite(str(destination), mask.astype(np.uint8) * 255):
                        raise OSError(f"Could not write {destination}")
            sheet = np.concatenate(panels, axis=1)
            title = np.full((32, sheet.shape[1], 3), 24, np.uint8)
            for index, label in enumerate(("Original", "p >= 0.5", "p >= 0.8")):
                cv2.putText(title, label, (index * image.shape[1] + 8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (240, 240, 240), 1, cv2.LINE_AA)
            preview = args.output / f"{number:02d}-review.bmp"
            if not cv2.imwrite(str(preview), np.concatenate((title, sheet))):
                raise OSError(f"Could not write {preview}")
            row = {"name": path.name, "original_size": original_size,
                   "working_size": [image.shape[1], image.shape[0]], "prepare_seconds": prepare_seconds,
                   "first_inference_seconds": durations[0], "warm_inference_seconds": durations[1:],
                   "warm_median_seconds": statistics.median(durations[1:]), "masks": stats,
                   "preview": preview.name}
            report["images"].append(row)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
