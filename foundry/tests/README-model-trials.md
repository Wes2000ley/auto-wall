# Anime model trials

This is a development comparison harness for Map Enhancer. Candidate weights,
private map crops and generated images stay outside the repository and installed
Foundry module. The harness uses the bundled ONNX Runtime and production tile
helpers. It never opens a campaign or changes a Foundry document.

The target is clean enlargement with small details intact. Stronger edges,
additional texture, a newer architecture or a faster crop are not sufficient
reasons to replace Digital Art. See [the research record](model-research.md).

## Prepare models

Use a development Python environment with `torch`, `spandrel==0.4.2`,
`safetensors`, `onnx`, `onnxruntime`, `numpy` and `Pillow`. These are development
dependencies; the installed module needs none of them. The trials were run with
Python 3.13.1, PyTorch 2.11, ONNX 1.21 and ONNX Runtime 1.25.1.

Choose separate private directories for downloaded sources and each trial:

```powershell
python foundry/tools/prepare-anime-trials.py C:/model-research/sources C:/model-research/trial/models --download --models realcugan-noise1 realcugan-noise2 realcugan-noise3
```

Omit `--download` to require already downloaded, checksum-matching sources.
Omit `--models` to prepare the whole catalog. Preparation writes an incremental
`models/models.json`, reports failures, continues with other models and exits
with a nonzero status if any model failed. A failed model stays in the browser
report with its preparation error. It is never silently counted as a success.
Use `--validate-only` to recheck an earlier prepared directory without replacing
its ONNX bytes. This still runs all numerical checks and regenerates reference
PNGs/metadata; it fails if a requested prepared graph is absent. It is useful
when an exporter changes names or metadata while preserving the computation.

[The catalog](../tools/anime-trials.json) pins source hashes, source URLs,
licenses and input/output contracts. Original author releases are preferred;
some older weights use mirrors checked against published hashes. A catalog
entry is not permission to redistribute its weights. Review the original
license before bundling a candidate.

Preparation checks raw, unclipped FP32 outputs against an independent PyTorch
checkpoint where available. Otherwise it compares to the original author ONNX
graph and records that weaker reference explicitly. The absolute maximum
tolerance is `1e-3`, with shape and finite-value checks on three input sizes.
SwinIR's dynamic mask adapter gets two additional rectangular sizes. When the
author graph requires FP16 input, both graphs receive the same representable
input values. The browser reference PNG is then produced by native CPU ONNX
Runtime; it is a numerical test fixture, not training data.

Two narrow export adapters correct size-dependent tracing: SwinIR window masks
and Alchemy's DySample coordinate normalizer. The original PyTorch network stays
separate and unmodified for the numerical comparison. The exporter wrapper and
reference both remain in evaluation mode.

## Prepare private map crops

Put lossless PNG crops and an `inputs.json` beside `models/`. Each crop needs
48 real source pixels of context on all sides of its measured region. For a
160 by 128 region, save the surrounding 256 by 224 source pixels without
resizing. Choose interior regions; a crop touching a map edge needs reflection
padding consistent with the production tile helper.

```json
[
  {
    "id": "map-detail",
    "name": "Map detail",
    "file": "map-detail.png",
    "region": {"x": 48, "y": 48, "width": 160, "height": 128},
    "sourceSHA256": "SHA256_OF_ORIGINAL_IMAGE",
    "sourceCrop": [690, 290, 160, 128]
  }
]
```

`sourceCrop` uses original-image pixels, before adding context. Use a real image
hash in your manifest. Keep the manifest local when it contains private paths or
other identifying information. Inspect source features before judging changes:
colored annotation or architectural marks already painted into a source are
part of the input, not an output overlay from the harness.

## Run and inspect

Supply the installed Handlebars browser distribution and your trial directory:

```powershell
node foundry/tests/serve-harness.mjs "C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/handlebars/dist/handlebars.js" C:/model-research/trial
```

Open `http://127.0.0.1:43125/tests/enhancer-comparison.html`. Select a backend,
then **Compare private map crops**. WebGPU is explicitly requested and the
harness does not retry a failed model on WASM. This provider setting is not
per-operator profiling. For a separate CPU trial, choose CPU and use a separate
directory so its results do not overwrite the GPU measurements.

Each model is loaded once into its own worker. Its bytes are hashed before
inference and checked against the prepared metadata; Digital Art is checked
against the installed asset pin. The first map also checks the candidate's
browser output against the native reference PNG: maximum RGB error must be at
most 3 and mean error at most 0.5 on the 0–255 scale. Later maps reuse the
validated session. Model/session errors remain in `browser-results.json`.

The harness saves lossless 2× and 4× outputs, using one native-scale AI pass
followed by the same browser high-quality resize. A 1× denoiser's 2×/4× output is
resized denoising output; it is not a learned upscale. No extra sharpening,
bilateral filter or WebP encoding is applied. Generated PNGs are posted directly
to the local server, avoiding text/base64 truncation.

**Display** changes between saved 2× and 4× results at one output pixel per
display pixel. **Assembly** switches between tiled and whole-crop inference.
**View saved results** reloads a completed or partial run. **Cancel** terminates
the current worker and keeps completed records. A new run in the same directory
replaces identically named results; use distinct directories to retain trials.

`firstMs` and `warmMs` time crop inference including pixel assembly/readback;
they exclude session creation, reference validation, PNG encoding, whole-crop
comparison and saving. They are not whole-map estimates or a controlled
cross-device benchmark. `warmMs` is one warmed measurement per map, not a
statistically stable timing distribution.

`seamError` is the RGB difference between tiled and whole-crop output across the
entire native-scale crop. It is a compatibility diagnostic, not a perceptual
quality score or a measurement confined to the tile seam. Global pooling and
attention can change output throughout a tile. Good whole-crop quality does not
establish that the current tile strategy is suitable for a new model.

The server binds only to `127.0.0.1`. It accepts same-origin result POSTs with
simple PNG/JSON filenames up to 10 MB. It is a development server serving the
chosen private directory, so stop it when finished. It has no role in the
self-contained Foundry installation.
