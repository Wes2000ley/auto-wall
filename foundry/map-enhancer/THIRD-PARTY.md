# Bundled third-party assets

All runtime assets are local to this module. The module's own code is MIT;
bundled third-party assets retain their respective licenses.

## Digital Art 4× model

- Source: Upscayl **2.15.0**, `resources/models/digital-art-4x.bin` and `.param`.
- Architecture: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_model.md), six RRDB blocks, 4× output, RGB NCHW input in [0,1].
- Published NCNN assets: [Upscayl models](https://github.com/upscayl/upscayl/tree/main/resources/models).
- Reference: [official PyTorch checkpoint](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth), used only for development validation.
- License: BSD 3-Clause, copyright 2021 Xintao Wang. Full license:
  `licenses/Real-ESRGAN-BSD-3-Clause.txt`.
- Distribution is an ONNX conversion of the installed NCNN values. The module
  does not include or execute Upscayl's application code.

Source SHA-256 pins:

```text
fe01c269cfd10cdef8e018ab66ebe750cf79c7af4d1f9c16c737e1295229bacc  digital-art-4x.bin
2b8fb6e0ae4d2d85704ca08c119a2f5ea40add4f2ecd512eb7f4cd44b6127ed4  digital-art-4x.param
f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da  RealESRGAN_x4plus_anime_6B.pth
```

The build tool rejects different NCNN inputs, checks all model operations,
consumes the entire weight stream, and validates the resulting ONNX graph.
The separate validator checks all 96 convolutions against the official reference,
then compares independent PyTorch inference and tiled native ONNX inference.

## ONNX Runtime Web 1.30.0

- Source: official [`onnxruntime-web@1.30.0`](https://www.npmjs.com/package/onnxruntime-web/v/1.30.0) npm distribution.
- Repository: [Microsoft ONNX Runtime v1.30.0](https://github.com/microsoft/onnxruntime/tree/v1.30.0).
- License: MIT, copyright Microsoft Corporation, in
  `licenses/ONNX-Runtime-MIT.txt`. Bundled third-party notices are in
  `licenses/ONNX-Runtime-ThirdPartyNotices.txt`, from the same upstream tag.
- Selected `dist/` files are preserved byte for byte in `vendor/ort/`:
  `ort.all.min.mjs`, the SIMD threaded standard WASM loader/binary and the
  SIMD threaded JSEP loader/binary. The module sets one WASM thread so ordinary
  Foundry hosts do not need isolation headers.
- Reproduce the vendor source with `npm pack onnxruntime-web@1.30.0`, then copy
  those five `package/dist/` files. Check their hashes against `asset-pins.json`.
  The npm tarball omits license files; obtain LICENSE and ThirdPartyNotices.txt
  from the exact repository tag above, retaining both in distributed modules.

`asset-pins.json` records SHA-256 and byte length for every model/runtime asset.
The repository's Node tests verify those pins. Git attributes disable newline
conversion for the pinned vendor bytes and ONNX binary.
