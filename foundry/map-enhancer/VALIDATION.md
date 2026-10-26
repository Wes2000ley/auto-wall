# Validation record — 2026-10-25

This records observed development checks, including failed trials. Browser
checks use the actual module template, worker, ONNX model, runtime and filters,
with explicit Foundry Application/Level/FilePicker doubles. No campaign was
modified to run them. The installed Foundry 14.369 sources were inspected for
the Level background update, texture placement and upload API contracts.

## Model identity and numerical checks

- Installed Upscayl: 2.15.0. Its Digital Art NCNN files match the published
  Upscayl assets and the source SHA-256 pins in THIRD-PARTY.md.
- All 96 convolution weights match the official
  `RealESRGAN_x4plus_anime_6B.pth` after NCNN FP16 truncation; all 96 biases
  match exactly. The first attempt used NumPy's round-to-nearest conversion
  and failed. Comparing the half-precision bit patterns identified NCNN's
  truncation toward zero; the validator explicitly reproduces that behavior.
- An independent six-block PyTorch implementation loaded with the exact NCNN
  values agrees with native ONNX Runtime: maximum absolute error
  **0.0000021458** on deterministic 13 × 17 RGB input.
- A 24-pixel tile halo failed the seam check (maximum difference about 25
  levels on an 8-bit scale). A 32-pixel trial still reached 4.46. The selected
  **48-pixel halo** reduced maximum discrepancy to **0.452** on the synthetic
  fixture and **0.819** on a stone/vegetation crop from Emberroot Burrow.
  Mean errors were 0.00257 and 0.00623 respectively. A 64-pixel trial was even
  closer but required more redundant inference; 48 passed both image checks.
- The first browser GPU trial fell back to CPU because ONNX Runtime Web's
  WebGPU convolution kernel rejected omitted dilation attributes. The valid
  ONNX defaults worked on CPU. The converter now writes explicit kernel size,
  stride, dilation and group attributes. GPU inference subsequently passed.

Reproduce model validation with the commands in README.md. The checked-in
synthetic input is 96 × 72; its independent native ONNX reference is 384 × 288.
These are software test fixtures, not ML labels or training data.

## Browser observations

Chrome 156 on Windows, localhost, without cross-origin isolation headers:

- Actual **WebGPU** inference and 4× WebP output passed the native-reference
  comparison. Mean 8-bit RGB difference after quality-100 WebP was below 0.001
  in that run. A fresh 96 × 72 inference/encode took about **1.7 seconds**.
- The included single-thread **WASM CPU** backend also passed. That earlier
  run took **25.88 seconds** on the same small fixture; it demonstrates the
  fallback's correctness, not acceptable whole-map CPU speed.
- The complete UI flow produced a GPU preview in **1.08 seconds** and refreshed
  three rapid slider changes in **126 ms**, including the 80 ms debounce.
- GPU bilateral/unsharp filters agree with the independent CPU path within
  two 8-bit levels. A 530-pixel-wide test crosses a filter tile boundary and
  agrees with whole-image filtering at opaque pixels.
- UI checks passed for preview, cached sliders, pre-inference refusal of stale
  placement/invalid folders, one WebP upload, native background-only update,
  retained dimensions/transforms/grid/walls, restore, action locking during
  restore, worker cancellation and consecutive reloads.
- A detail from the **1536 × 1024 Emberroot Burrow** encounter map previewed on
  GPU in about **1.4 seconds**. This is a crop timing, not a whole-map claim.
- The same complete encounter map processed as 96 GPU tiles, bilateral
  smoothing, sharpening and quality-85 WebP encoding, producing **3072 × 2048**
  output. The in-memory upload double received the WebP; the level background
  switched and restored successfully. No precise whole-map duration was
  recorded, so no throughput figure is claimed for that pass.

The Node suite covers filters, output limits, reflection padding, exact tile
coverage, destination derivation, permissions, stale saves, failed uploads,
native update rejection, restore history and every bundled asset hash. It also
runs the existing Auto-Wall Live drawing/document regressions. The PowerShell
installer checks both module selections, copied hashes, obsolete ownership,
unknown-file preservation and refusal to delete a modified obsolete file.

## Practical limits

No live-campaign test of multiplayer rendering, token vision or interactions
with other modules was run. Native scene operations are validated with doubles
and checked against the installed API implementation. No full desktop Python
application suite was run because that application is unchanged.

The Digital Art model can simplify or reinterpret fine textures and lettering;
matching Upscayl's model does not guarantee fidelity for every map. A preview
should be inspected before saving. GPU support, total map size and output limits
affect throughput. The CPU fallback can be slow. External images need CORS.
