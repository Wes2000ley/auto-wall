"""Optional PyTorch runtime. Models contain tensors and JSON metadata, never pickle."""

from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from src.ml.network import BattlemapNet, validate_classes


FORMAT = "auto-wall-unet-v1"
DINO_FORMAT = "auto-wall-dino-v1"
DINO_LIGHTS_FORMAT = "auto-wall-dino-lights-v1"
DINO_SEMANTIC_LIGHTS_FORMAT = "auto-wall-dino-lights-v2"


def save_model(path, model, details):
    """Atomically save uncompressed, portable CPU tensors with their provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_format = DINO_FORMAT if model.architecture == "dino" else FORMAT
    if model.architecture == "dino" and model.light_head is not None:
        model_format = DINO_SEMANTIC_LIGHTS_FORMAT if model.light_head.semantic_context else DINO_LIGHTS_FORMAT
    classes = validate_classes(model.classes)
    metadata = {"format": model_format, "classes": json.dumps(classes),
                "width": str(model.width), "context_dilation": str(model.context_dilation),
                "details": json.dumps(details, allow_nan=False)}
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
    os.close(descriptor)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_model(path, device="auto"):
    path = Path(path).resolve(strict=True)
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("ML device must be auto, cpu, or cuda.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA build of PyTorch or select CPU.")
    stat = path.stat()
    if stat.st_size > 128 * 1024 * 1024:
        raise ValueError("This detector accepts compact Auto-Wall models up to 128 MiB.")
    try:
        return _cached_model(str(path), stat.st_mtime_ns, stat.st_size, device)
    except (SafetensorError, KeyError) as error:
        raise ValueError("Could not read this Auto-Wall model: invalid or incomplete safetensors file.") from error


@lru_cache(maxsize=1)
def _cached_model(path, modified, size, device):
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        model_format = metadata.get("format")
        try:
            classes = validate_classes(json.loads(metadata.get("classes", "[]")))
        except (TypeError, ValueError):
            classes = None
        if (model_format not in (FORMAT, DINO_FORMAT, DINO_LIGHTS_FORMAT, DINO_SEMANTIC_LIGHTS_FORMAT)
                or classes is None):
            raise ValueError("This is not a compatible Auto-Wall segmentation model.")
        width = int(metadata.get("width", "0"))
        if width not in (8, 16, 24, 32):
            raise ValueError("Unsupported model width.")
        dilation = int(metadata.get("context_dilation", "1"))
        if dilation not in (1, 2, 4):
            raise ValueError("Unsupported model context dilation.")
        if model_format in (DINO_FORMAT, DINO_LIGHTS_FORMAT, DINO_SEMANTIC_LIGHTS_FORMAT):
            if width != 16 or dilation != 4:
                raise ValueError("Unsupported DINO decoder configuration.")
            from src.ml.dino import DinoBattlemap
            model = DinoBattlemap(light_centers=model_format != DINO_FORMAT,
                                 semantic_light_context=model_format == DINO_SEMANTIC_LIGHTS_FORMAT,
                                 classes=classes)
        else:
            model = BattlemapNet(width, dilation, classes)
        state = {key: source.get_tensor(key) for key in source.keys()}
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("The model contains nonfinite weights.")
        model.load_state_dict(state, strict=True)
        details = json.loads(metadata.get("details", "{}"))
    return model.to(device).eval(), details


def predict(model, image, cancelled=None, progress=None, tile_size=384, include_light_centers=False):
    """Predict BGR uint8 pixels with a context halo and bounded GPU memory.

    Output is (class IDs, winning probabilities); the caller decides thresholds.
    With include_light_centers, append a heatmap, or None for older models.
    Segmentation uses the same features and tiles regardless of this option.
    Compact CNN halos exceed its receptive radius and are discarded.
    CUDA queues two single-image tiles before downloading their results; model
    batch shape and tile order are unchanged. CPU execution completes each tile.
    DINO uses bounded attention windows with this overlap; its predictions can vary with
    tile context and are not identical to whole-image transformer inference.
    """
    if tile_size < 8 or tile_size % 8:
        raise ValueError("ML tile size must be a positive multiple of eight.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("ML prediction needs a nonempty uint8 BGR image.")
    height, width = image.shape[:2]
    classes = np.empty((height, width), np.uint8)
    confidence = np.empty((height, width), np.float32)
    light_head = getattr(model, "light_head", None) if include_light_centers else None
    light_heatmap = np.empty((height, width), np.float32) if light_head is not None else None
    device = next(model.parameters()).device
    halo = 64 if model.context_dilation == 1 else 128
    tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    completed = 0
    is_cuda = device.type == "cuda"
    pending = []

    def check_cancelled():
        if cancelled and cancelled():
            from src.core.fast_pipeline import DetectionCancelled
            raise DetectionCancelled()

    def flush():
        nonlocal completed
        for output_region, labels, probabilities, light_scores, finite, light_finite, source in pending:
            check_cancelled()
            # Downloading waits for the queued operations. Keep each pinned
            # input alive until its asynchronous device transfer has completed.
            labels = labels.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            if not bool(finite):
                raise RuntimeError("The model produced nonfinite probabilities.")
            classes[output_region] = labels
            confidence[output_region] = probabilities
            if light_scores is not None:
                light_scores = light_scores.cpu().numpy()
                if not bool(light_finite):
                    raise RuntimeError("The model produced nonfinite light scores.")
                light_heatmap[output_region] = light_scores
            completed += 1
            if progress:
                progress(f"ML segmentation: {completed}/{tiles} tiles")
        pending.clear()

    with torch.inference_mode():
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                check_cancelled()
                y0, x0 = max(0, y - halo), max(0, x - halo)
                y1, x1 = min(height, y + tile_size + halo), min(width, x + tile_size + halo)
                rgb = np.ascontiguousarray(image[y0:y1, x0:x1, ::-1].transpose(2, 0, 1))
                source = torch.from_numpy(rgb)
                if is_cuda:
                    source = source.pin_memory()
                tensor = source.to(device=device, dtype=torch.float32, non_blocking=is_cuda).unsqueeze(0) / 255
                tensor = F.pad(tensor, (0, (-tensor.shape[3]) % 8, 0, (-tensor.shape[2]) % 8), mode="replicate")
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    if light_head is None:
                        logits = model(tensor)
                    else:
                        features, context = model.forward_features(tensor, include_context=True)
                        logits = model.local.head(features)
                        light_scores = light_head(features, tensor, context).float().sigmoid()
                        light_scores = F.interpolate(light_scores, size=tensor.shape[-2:],
                                                     mode="bilinear", align_corners=False)[0, 0]
                probabilities, labels = logits[0].float().softmax(0).max(0)
                finite = torch.isfinite(probabilities).all()
                end_y, end_x = min(height, y + tile_size), min(width, x + tile_size)
                region = np.s_[y - y0:end_y - y0, x - x0:end_x - x0]
                output_light_scores = light_finite = None
                if light_heatmap is not None:
                    light_finite = torch.isfinite(light_scores).all()
                    output_light_scores = light_scores[region]
                pending.append((np.s_[y:end_y, x:end_x], labels[region].to(torch.uint8),
                                probabilities[region], output_light_scores, finite, light_finite, source))
                if len(pending) == (2 if is_cuda else 1):
                    flush()
        flush()
    if include_light_centers:
        return classes, confidence, light_heatmap
    return classes, confidence
