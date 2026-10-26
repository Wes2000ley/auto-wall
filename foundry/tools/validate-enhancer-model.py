"""Development validation against the official six-block PyTorch architecture.

Requires numpy, torch, onnx, onnxruntime and Pillow. None are runtime dependencies
of the Foundry module. Use the official RealESRGAN_x4plus_anime_6B.pth checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(1, 6):
            setattr(self, f"conv{i}", nn.Conv2d(64 + (i - 1) * 32, 64 if i == 5 else 32, 3, 1, 1))

    def forward(self, x):
        features = [x]
        for i in range(1, 5):
            features.append(F.leaky_relu(getattr(self, f"conv{i}")(torch.cat(features, 1)), .2))
        return x + self.conv5(torch.cat(features, 1)) * .2


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = DenseBlock(), DenseBlock(), DenseBlock()

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * .2 + x


class Anime6B(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(6)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = x + self.conv_body(self.body(x))
        x = F.leaky_relu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")), .2)
        x = F.leaky_relu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")), .2)
        return self.conv_last(F.leaky_relu(self.conv_hr(x), .2))


def ncnn_half(values):
    # NCNN's converter truncates toward zero; numpy normally rounds to nearest.
    half = values.astype(np.float16)
    bits = half.view(np.uint16).copy()
    bits[np.abs(half.astype(np.float32)) > np.abs(values)] -= 1
    return bits.view(np.float16).astype(np.float32)


def validate(model_path, checkpoint_path, fixture):
    expected = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != expected:
        raise ValueError("The official reference checkpoint does not match its SHA-256 pin")
    torch.set_num_threads(1)
    reference = torch.load(checkpoint_path, weights_only=True, map_location="cpu")["params_ema"]
    graph = onnx.load(model_path)
    tensors = {v.name: numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
    convolutions = [node for node in graph.graph.node if node.op_type == "Conv"]
    keys = [key[:-7] for key in reference if key.endswith(".weight")]
    assert len(keys) == len(convolutions) == 96
    quantized = {}
    for key, node in zip(keys, convolutions, strict=True):
        weight, bias = tensors[node.input[1]], tensors[node.input[2]]
        np.testing.assert_array_equal(weight, ncnn_half(reference[key + ".weight"].numpy()), err_msg=key)
        np.testing.assert_array_equal(bias, reference[key + ".bias"].numpy(), err_msg=key)
        quantized[key + ".weight"], quantized[key + ".bias"] = torch.from_numpy(weight), torch.from_numpy(bias)
    model = Anime6B().eval()
    model.load_state_dict(quantized, strict=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    x = rng.random((1, 3, 13, 17), dtype=np.float32)
    with torch.no_grad():
        torch_result = model(torch.from_numpy(x)).numpy()
    actual = session.run(None, {"data": x})[0]
    error = float(np.max(np.abs(actual - torch_result)))
    np.testing.assert_allclose(actual, torch_result, rtol=1e-4, atol=1e-4)

    # Independent tiled reference, using the same reflection halo as the browser.
    image = np.asarray(Image.open(fixture).convert("RGB"), dtype=np.float32) / 255
    image = image[:min(72, image.shape[0]), :min(100, image.shape[1])]
    h, w = image.shape[:2]
    padded = np.pad(image, ((48, 48), (48, 48), (0, 0)), mode="reflect")
    evaluate = lambda a: session.run(None, {"data": a.transpose(2, 0, 1)[None].copy()})[0][0].transpose(1, 2, 0)
    whole = evaluate(padded)[192:192 + h * 4, 192:192 + w * 4]
    tiled = np.zeros_like(whole)
    for y in range(0, h, 48):
        for x in range(0, w, 64):
            th, tw = min(48, h - y), min(64, w - x)
            out = evaluate(padded[y:y + th + 96, x:x + tw + 96])
            tiled[y * 4:(y + th) * 4, x * 4:(x + tw) * 4] = out[192:192 + th * 4, 192:192 + tw * 4]
    delta = np.abs(whole - tiled) * 255
    mean, maximum = float(delta.mean()), float(delta.max())
    if mean > .2 or maximum > 4:
        raise AssertionError(f"Visible tile discrepancy: mean={mean}, maximum={maximum}")
    print(json.dumps({"convolutions_exact": 96, "torch_onnx_max_error": error,
                      "tile_error_8bit_mean": mean, "tile_error_8bit_max": maximum,
                      "reference_fixture": str(fixture)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    validate(args.model, args.checkpoint, args.fixture)
