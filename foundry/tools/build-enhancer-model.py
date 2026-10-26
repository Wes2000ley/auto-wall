"""Development-only conversion of Upscayl 2.15 Digital Art NCNN weights to ONNX.

The installed Foundry module needs neither Python nor Upscayl. The conversion
accepts only the two pinned upstream files and the operations in this network.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PINS = {
    "digital-art-4x.bin": "fe01c269cfd10cdef8e018ab66ebe750cf79c7af4d1f9c16c737e1295229bacc",
    "digital-art-4x.param": "2b8fb6e0ae4d2d85704ca08c119a2f5ea40add4f2ecd512eb7f4cd44b6127ed4",
}


def convert(source, destination):
    for name, expected in PINS.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unrecognized model: {name}")
    data = (source / "digital-art-4x.bin").read_bytes()
    offset = 0
    nodes, tensors, aliases, convolutions = [], [], {}, []

    def constant(name, value):
        tensors.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name

    for line in (source / "digital-art-4x.param").read_text().splitlines()[2:]:
        fields = line.split()
        kind, name = fields[:2]
        ni, no = map(int, fields[2:4])
        inputs = [aliases.get(v, v) for v in fields[4:4 + ni]]
        outputs = fields[4 + ni:4 + ni + no]
        params = dict(v.split("=", 1) for v in fields[4 + ni + no:])
        if kind == "Input":
            continue
        if kind == "Split":
            for output in outputs:
                aliases[output] = inputs[0]
        elif kind == "Convolution":
            count, channels, kernel = (int(params[k]) for k in ["6", "0", "1"])
            tag, = struct.unpack_from("<I", data, offset)
            if tag != 0x01306B47:
                raise ValueError(f"Expected NCNN FP16 weights at {name}")
            offset += 4
            weights = np.frombuffer(data, "<f2", count, offset).astype(np.float32)
            offset += ((count * 2 + 3) // 4) * 4
            bias = np.frombuffer(data, "<f4", channels, offset).copy()
            offset += channels * 4
            weights = weights.reshape(channels, -1, kernel, kernel)
            convolutions.append((weights, bias))
            out = outputs[0] if "9" not in params else name + "_linear"
            pad = int(params.get("4", 0))
            nodes.append(helper.make_node("Conv", inputs + [constant(name + "_w", weights), constant(name + "_b", bias)], [out],
                                          name=name, pads=[pad] * 4, kernel_shape=[kernel] * 2,
                                          strides=[1, 1], dilations=[1, 1], group=1))
            if "9" in params:
                if params["9"] != "2":
                    raise ValueError("Unexpected activation")
                nodes.append(helper.make_node("LeakyRelu", [out], outputs, alpha=float(params["-23310"].split(",")[1])))
        elif kind == "Concat":
            nodes.append(helper.make_node("Concat", inputs, outputs, axis=1))
        elif kind == "Eltwise":
            if params["0"] != "1" or ni != 2:
                raise ValueError("Unexpected weighted sum")
            coefficients = list(map(float, params["-23301"].split(",")[1:]))
            terms = []
            for i, (value, coefficient) in enumerate(zip(inputs, coefficients, strict=True)):
                term = name + f"_term{i}"
                nodes.append(helper.make_node("Mul", [value, constant(term + "_scale", [coefficient])], [term]))
                terms.append(term)
            nodes.append(helper.make_node("Add", terms, outputs))
        elif kind == "BinaryOp" and not params:
            nodes.append(helper.make_node("Add", inputs, outputs))
        elif kind == "Interp" and params == {"0": "1", "1": "2.000000e+00", "2": "2.000000e+00"}:
            nodes.append(helper.make_node("Resize", inputs + ["", constant(name + "_scales", [1, 1, 2, 2])], outputs,
                                          mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"))
        else:
            raise ValueError(f"Unsupported NCNN operation: {kind}")
    if offset != len(data) or len(convolutions) != 96:
        raise ValueError(f"Incomplete conversion: {offset}/{len(data)} bytes, {len(convolutions)} convolutions")
    graph = helper.make_graph(nodes, "Upscayl Digital Art 4x (Real-ESRGAN Anime 6B)",
                              [helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 3, "height", "width"])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "out_height", "out_width"])], tensors)
    model = helper.make_model(graph, producer_name="Auto-Wall Map Enhancer", opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    helper.set_model_props(model, {"source": "Upscayl 2.15.0 digital-art-4x", "license": "BSD-3-Clause", "input": "NCHW RGB float32 [0,1]", "scale": "4"})
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    print(json.dumps({"path": str(destination), "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "convolutions": len(convolutions)}))
    return convolutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
