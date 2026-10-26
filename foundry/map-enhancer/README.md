# Map Enhancer for Foundry VTT 14

Upscale the current scene level's background using the Digital Art model bundled
with Upscayl 2.15.0. Compare a detail before and after, adjust sharpening and
bilateral smoothing, then save a WebP into Foundry's assets and switch the
background in one action. This is a separate addon from Auto-Wall Live.

The model, ONNX Runtime Web, WebAssembly CPU backend, GPU backend and filters are
included. Processing stays in the GM's browser worker. No Upscayl installation,
Python, native helper, API key, CDN, model download or external service is needed.
The roughly 62 MB module is served by the same Foundry server as the map.

## Install

From this repository:

```powershell
./foundry/install.ps1 -DataPath "$env:LOCALAPPDATA/FoundryVTT/Data" -ModuleId map-enhancer
```

Alternatively copy this entire `map-enhancer` directory into `Data/modules/`.
For a first installation, restart Foundry when convenient, then enable **Map
Enhancer** in **Manage Modules**. Updates need a client reload. The installer
verifies every copied file and preserves unknown user files; it does not edit
world databases or activation settings.

As GM, open a scene and choose **Tile Controls → Enhance current map**. A script
macro can open the same panel:

```js
game.modules.get("map-enhancer").api.open();
```

## Preview and save

1. Click the small overview to select a detail, then **Generate preview**. AI
   processes up to 160 × 128 source pixels for the comparison. The overview box
   shows the exact crop; it is not a whole-map preview.
2. Drag the before/after split. Wheel to zoom the comparison, drag to pan, and
   use **Fit preview** to reset the view.
3. Adjust sharpening amount/radius, edge-preserving smoothing amount/radius,
   color tolerance, output scale and WebP quality. Sliders reuse the cached 4×
   AI result. They resize, filter and encode that crop without another AI pass.
   Lower smoothing tolerance retains stronger color boundaries.
4. **Save WebP & switch scene** processes the whole map, uploads a new unique
   WebP through Foundry's normal FilePicker API, and changes this level's
   background. The default is **2×, WebP quality 85%**, with an 8192-pixel edge
   cap. Select 1× for enhancement at the original size or 4× for a larger output.
5. **Restore previous map** returns to the prior background. The last five
   switches are recorded in this level's module flags. Processed files remain
   available; the original file is never overwritten.

The asset folder defaults to the original map's local Data folder. External or
installed package maps default to `assets/maps/enhanced`. The folder is editable;
missing directories are created through Foundry. Uploaded filenames include the
dimensions and a native Foundry random ID, so existing files are not replaced.

Save changes only `background.src` and restore history on the current Level
document. Scene dimensions, grid, walls, tokens, other levels and texture
rotation, scale and offsets stay in place. Foundry applies the usual background
document update to connected clients. Stale map or placement changes are refused.
If upload succeeds but the scene update fails, the error identifies the saved
file so the work is recoverable.

## Processing and limits

Automatic processing tries WebGPU and falls back to the included CPU backend
when necessary. WebGPU needs a browser with GPU support and a secure context
(HTTPS or localhost); a remote plain-HTTP host normally uses CPU. There is also
an explicit CPU setting. Filters use WebGL2 where available and have the same
CPU fallback. No cross-origin isolation headers are required.

AI runs in tiles with a 48-source-pixel reflection halo, cropped before assembly
to suppress seams. The entire 4× map is not held in memory when a smaller output
is selected. The largest requested edge, the client's texture limit, WebP's
16383-pixel edge limit and a 64-megapixel output budget bound the final size.
The UI reports the actual output dimensions. Large source images still need
time and memory, particularly on CPU; a small preview is not a timing estimate
for the whole map. **Cancel processing** terminates the worker before any upload.
Once the upload/scene update begins, the panel waits for that operation to finish.

Maps served by another origin must allow browser CORS reads. Video backgrounds
are not supported. This module targets the installed Foundry **14.369** Level
API; it does not claim compatibility with Foundry 13.

## Model and reproducibility

The included model is **RealESRGAN_x4plus_anime_6B**, a 4× RRDB network with six
blocks. The installed Upscayl `digital-art-4x.bin` and `.param` match the published
Upscayl assets byte for byte. The ONNX conversion preserves all 96 convolution
weights and biases from those NCNN files. They use FP16 weights truncated toward
zero and FP32 biases; ONNX stores the exact recovered values as FP32. The model
is trained for illustration/anime, not specifically for dungeon maps. Preview
small features and text before accepting an enhancement.

See [THIRD-PARTY.md](THIRD-PARTY.md) for upstream sources, licenses and pinned
assets. `asset-pins.json` records the exact bundled bytes and is verified by the
Node checks. Development conversion requires Python with `onnx==1.21.0` and
`numpy==2.2.2`; these are not needed to install or run the module:

```powershell
python foundry/tools/build-enhancer-model.py "C:/Program Files/Upscayl/resources/models" foundry/map-enhancer/models/digital-art-4x.onnx
```

An independent PyTorch architecture and the official reference checkpoint can
validate both weights and inference. That development check additionally uses
PyTorch, Pillow and ONNX Runtime (checked with `onnxruntime==1.25.1`):

```powershell
python foundry/tools/validate-enhancer-model.py foundry/map-enhancer/models/digital-art-4x.onnx path/to/RealESRGAN_x4plus_anime_6B.pth foundry/tests/fixtures/enhancer-input.png
node --test --test-reporter=spec foundry/tests/*.test.mjs
node foundry/tests/serve-harness.mjs "C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/handlebars/dist/handlebars.js"
```

Open `http://127.0.0.1:43125/tests/enhancer-harness.html` for the actual worker,
model, filter and UI checks against explicit Foundry document/upload doubles.
Those checks do not substitute for a campaign compatibility test. The tiny
fixture and native ONNX reference are test assets, not ML training data.
