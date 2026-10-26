"""Prepare research models outside the installed module; never promote automatically.

Development dependencies: torch, spandrel==0.4.2, safetensors, onnx, onnxruntime, numpy, Pillow.
Downloads are explicit (--download), pinned author releases or checksum-verified mirrors.
Checkpoints use safetensors or torch.load(weights_only=True). No repository code is run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from types import MethodType

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from PIL import Image
import spandrel
import torch
from safetensors.torch import load_file


def source_file(directory, spec, download):
    path = directory / spec["file"]
    if not path.exists() and download:
        if "archive" in spec:
            archive = source_file(directory, spec["archive"], download)
            # Read a named member into our chosen filename; never extract paths.
            with zipfile.ZipFile(archive) as bundle:
                data = bundle.read(spec["member"])
        else:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"Download hash mismatch: {path.name}")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Source hash mismatch: {path.name}")
    return path


def pytorch_model(path):
    state = load_file(path) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu", weights_only=True)
    state = spandrel.canonicalize_state_dict(state)
    return spandrel.ModelLoader().load_from_state_dict(state).float().eval()


class ImageOnly(torch.nn.Module):
    """Keep optional architecture arguments at their defaults, not ONNX inputs."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)


def dynamic_swin_mask(block, size):
    """The same SwinIR window mask, using runtime shapes instead of a cache."""
    height, width = size
    window, shift = block.window_size, block.shift_size
    if shift == 0:
        return torch.zeros((height // window * (width // window), window * window, window * window))
    y, x = torch.arange(height), torch.arange(width)
    y = torch.where(y < height - window, 0, torch.where(y < height - shift, 1, 2))
    x = torch.where(x < width - window, 0, torch.where(x < width - shift, 1, 2))
    labels = y[:, None] * 3 + x[None, :]
    windows = labels.reshape(height // window, window, width // window, window).permute(0, 2, 1, 3).reshape(-1, window * window)
    return (windows[:, None, :] != windows[:, :, None]).to(torch.float32) * -100


def export_model(reference, entry):
    model = reference.model
    if entry.get("dynamicSwinMask"):
        # Do not modify the independent PyTorch reference. Spandrel's cached
        # training-size mask must not be frozen into a dynamic-size ONNX graph.
        model = copy.deepcopy(model)
        blocks = [block for block in model.modules() if block.__class__.__name__ == "SwinTransformerBlock"]
        if not blocks:
            raise ValueError("Unexpected SwinIR architecture; no window attention blocks")
        for block in blocks:
            block.input_resolution = (-1, -1)
            block.calculate_mask = MethodType(dynamic_swin_mask, block)
    # The exporter restores the wrapper's original training state recursively.
    # An unevaluated wrapper would re-enable dropout/batch norm in the reference.
    return ImageOnly(model).eval()


def expand_float_types(message):
    """Use FP32 operations with the exact half-precision stored weight values."""
    half = (TensorProto.FLOAT16, TensorProto.BFLOAT16)
    if isinstance(message, TensorProto) and message.data_type in half:
        message.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(message).astype(np.float32), message.name))
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            for child in value if field.is_repeated else [value]:
                expand_float_types(child)
    if isinstance(message, onnx.TypeProto.Tensor) and message.elem_type in half:
        message.elem_type = TensorProto.FLOAT
    if isinstance(message, onnx.NodeProto) and message.op_type == "Cast":
        for attribute in message.attribute:
            if attribute.name == "to" and attribute.i in half:
                attribute.i = TensorProto.FLOAT


def dynamic_dysample_normalizer(model):
    """Spandrel 0.4.2's legacy export freezes DySample's [W, H] tensor at 32.

    Derive that same normalizer from the sampled feature's runtime dimensions.
    The unmodified PyTorch model remains the independent numerical reference.
    """
    nodes = model.graph.node
    producers = {output: node for node in nodes for output in node.output}
    grids = [node for node in nodes if node.op_type == "GridSample"]
    constants = [node for node in nodes if node.op_type == "Constant" and any(
        a.name == "value" and list(a.t.dims) == [1, 2, 1, 1, 1] and
        np.array_equal(numpy_helper.to_array(a.t).flatten(), [32, 32]) for a in node.attribute)]
    if len(grids) != 1 or len(constants) != 1:
        raise ValueError("Unexpected DySample graph; recheck the exporter before adapting it")
    sampled = producers[grids[0].input[0]]
    if sampled.op_type != "Reshape":
        raise ValueError("Unexpected DySample sampled feature")
    constant = constants[0]
    prefix = constant.output[0] + "_dynamic"
    replacements = [
        helper.make_node("Shape", [sampled.input[0]], [prefix + "_shape"]),
        helper.make_node("Gather", [prefix + "_shape", prefix + "_indices"], [prefix + "_wh"], axis=0),
        helper.make_node("Cast", [prefix + "_wh"], [prefix + "_float"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", [prefix + "_float", prefix + "_target"], list(constant.output)),
    ]
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([3, 2], dtype=np.int64), prefix + "_indices"),
        numpy_helper.from_array(np.array([1, 2, 1, 1, 1], dtype=np.int64), prefix + "_target"),
    ])
    index = list(nodes).index(constant)
    nodes.remove(constant)
    for offset, node in enumerate(replacements):
        nodes.insert(index + offset, node)


def prepare_graph(path, dynamic_dysample):
    model = onnx.load(path)
    expand_float_types(model)
    if dynamic_dysample:
        dynamic_dysample_normalizer(model)
    # Do not retain exporter shape aliases that incorrectly equate dynamic axes.
    del model.graph.value_info[:]
    for value in model.graph.output:
        for index, dimension in enumerate(value.type.tensor_type.shape.dim):
            dimension.Clear()
            dimension.dim_param = f"{value.name}_dimension_{index}"
    weights = {v.name: v for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = {a.name for a in node.attribute}
        defaults = {"strides": [1, 1], "dilations": [1, 1], "group": 1}
        if "kernel_shape" not in attrs:
            defaults["kernel_shape"] = list(weights[node.input[1]].dims[-2:])
        node.attribute.extend(helper.make_attribute(key, value) for key, value in defaults.items() if key not in attrs)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def padded_size(size, entry):
    multiple, minimum = entry.get("inputMultiple", 1), entry.get("minimumSize", 1)
    remainder = entry.get("inputRemainder", 0)
    return (max(size, minimum) - remainder + multiple - 1) // multiple * multiple + remainder


def padded_input(value, entry):
    height, width = value.shape[-2:]
    target = [padded_size(size, entry) for size in [height, width]]
    return np.pad(value, ((0, 0), (0, 0), (0, target[0] - height), (0, target[1] - width)), mode="reflect")


def prepare(sources, output, download, selected, validate_only=False):
    catalog = json.loads(Path(__file__).with_name("anime-trials.json").read_text(encoding="utf-8"))
    if selected:
        unknown = set(selected) - {entry["id"] for entry in catalog}
        if unknown:
            raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}")
        catalog = [entry for entry in catalog if entry["id"] in selected]
    sources.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    results = []
    for entry in catalog:
        identifier, scale = entry["id"], entry["scale"]
        result = {"id": identifier, "label": entry["label"], "scale": scale, "page": entry["page"],
                  "license": entry["license"], "sourceSHA256": entry["source"]["sha256"],
                  "inputMultiple": entry.get("inputMultiple", 1), "minimumSize": entry.get("minimumSize", 1),
                  "inputRemainder": entry.get("inputRemainder", 0), "outputOffset": entry.get("outputOffset", 0)}
        try:
            path = source_file(sources, entry["source"], download)
            destination = output / f"{identifier}.onnx"
            reference = None
            if path.suffix in (".pth", ".safetensors"):
                reference = pytorch_model(path)
                if reference.scale != scale:
                    raise ValueError("Checkpoint scale does not match the catalog")
                if not validate_only:
                    export_size = padded_size(entry.get("exportSize", 32), entry)
                    torch.onnx.export(export_model(reference, entry), torch.rand(1, 3, export_size, export_size), destination,
                                      input_names=["input"], output_names=["output"], opset_version=17,
                                      dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "out_height", 3: "out_width"}},
                                      dynamo=False)
            elif not validate_only:
                destination.write_bytes(path.read_bytes())
            if validate_only:
                onnx.checker.check_model(onnx.load(destination))
            else:
                prepare_graph(destination, entry.get("dynamicDysample", False))
            if "reference" in entry:
                reference = pytorch_model(source_file(sources, entry["reference"], download))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.log_severity_level = 3
            session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
            author_session = None
            author_dtype = np.float32
            if reference is None:
                original = source_file(sources, entry["referenceONNX"], download) if "referenceONNX" in entry else path
                author_session = ort.InferenceSession(str(original), sess_options=options, providers=["CPUExecutionProvider"])
                author_type = author_session.get_inputs()[0].type
                if author_type not in ("tensor(float)", "tensor(float16)"):
                    raise ValueError(f"Unsupported author reference input: {author_type}")
                author_dtype = np.float16 if author_type == "tensor(float16)" else np.float32
            maximum = 0
            offset = entry.get("outputOffset", 0)
            sizes = [(32, 32), (35, 49), (67, 81)]
            if entry.get("dynamicSwinMask"):
                sizes += [(64, 96), (128, 160)]
            for height, width in sizes:
                value = np.random.default_rng(42).random((1, 3, height, width), dtype=np.float32)
                value = padded_input(value, entry)
                # Feed identical representable values to both graphs when the
                # author graph takes FP16. This isolates arithmetic differences.
                value = value.astype(author_dtype).astype(np.float32)
                actual = session.run(None, {session.get_inputs()[0].name: value})[0]
                if actual.shape != (1, 3, value.shape[2] * scale - 2 * offset, value.shape[3] * scale - 2 * offset) or not np.isfinite(actual).all():
                    raise ValueError(f"Invalid output on {width} x {height}: {actual.shape}")
                if reference is not None:
                    with torch.inference_mode():
                        expected = reference.model(torch.from_numpy(value)).numpy()
                else:
                    expected = author_session.run(None, {author_session.get_inputs()[0].name: value.astype(author_dtype)})[0].astype(np.float32)
                if expected.shape != actual.shape or not np.isfinite(expected).all():
                    raise ValueError(f"Invalid reference output on {width} x {height}")
                error = float(np.abs(actual - expected).max())
                maximum = max(maximum, error)
                if error > 1e-3:
                    raise ValueError(f"Prepared/reference mismatch on {width} x {height}: {error}")
            # Lossless, independent native CPU reference for WebGPU validation.
            value = np.random.default_rng(7).random((16, 20, 3), dtype=np.float32)
            image = Image.fromarray(np.rint(value * 255).astype(np.uint8))
            image.save(output / "reference-input.png")
            height, width = [padded_size(size + 96, entry) for size in [image.height, image.width]]
            padded = np.pad(np.asarray(image, dtype=np.float32) / 255,
                            ((48, height - image.height - 48), (48, width - image.width - 48), (0, 0)), mode="reflect")
            actual = session.run(None, {session.get_inputs()[0].name: padded.transpose(2, 0, 1)[None].copy()})[0][0]
            origin = 48 * scale - offset
            actual = actual[:, origin:origin + image.height * scale, origin:origin + image.width * scale].transpose(1, 2, 0)
            Image.fromarray(np.rint(actual.clip(0, 1) * 255).astype(np.uint8)).save(output / f"{identifier}-reference.png")
            result.update(file=destination.name, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                          bytes=destination.stat().st_size, pytorch_max_error=maximum if reference else None,
                          reference_max_error=maximum,
                          reference="independent PyTorch" if reference else "author ONNX graph; no independent PyTorch comparison")
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        results.append(result)
        print(json.dumps(result), flush=True)
        (output / "models.json").write_text(json.dumps(results, indent=2) + "\n")
    if any("error" in result for result in results):
        raise SystemExit("Some model trials failed; see models.json. No models were installed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path, help="Private harness comparison directory, never map-enhancer/models")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Recheck existing prepared graphs against their sources without replacing graph bytes")
    parser.add_argument("--models", nargs="+", help="Prepare only these catalog IDs (a separate output directory per batch is recommended)")
    args = parser.parse_args()
    target = args.output.resolve()
    if target == args.sources.resolve():
        parser.error("Keep original sources and prepared output in separate directories")
    if any((directory / "module.json").exists() for directory in [target, *target.parents]) or target.is_relative_to(Path(__file__).resolve().parents[1] / "map-enhancer"):
        parser.error("Research models must stay outside the installed module")
    prepare(args.sources, args.output, args.download, args.models, args.validate_only)
