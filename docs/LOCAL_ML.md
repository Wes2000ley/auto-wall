# Local battlemap segmentation

Fast Setup can run a compact U-Net or a DINOv2-backed decoder for **solid walls,
doors, lights, and see-through barriers**. These are experimental options. The
palette detector remains the default, and no model is downloaded or executed automatically.

## Use a model

Install the ordinary fast requirements, then install the appropriate official
[PyTorch CPU or CUDA build](https://pytorch.org/get-started/locally/) for your
machine, plus `safetensors`. `requirements-ml.txt` lists the optional dependencies.
The tested GPU environment uses Python 3.13, PyTorch 2.11.0+cu128, and an RTX 3080.
The CPU and GPU runtimes use the same model file.

DINO checkpoints additionally need `requirements-ml-dino.txt`. They combine a
frozen, pretrained DINOv2-small backbone with a trainable local decoder. This adds
visual context while retaining fine boundary features. Both architectures load
one local safetensors file; the app does not fetch pretrained weights or execute
remote model code. DINO is larger and slower than the compact model. Consult the
[model notes](../models/README.md) before choosing a checkpoint.

```shell
python fast_auto_wall.py "map.png" --model "models/battlemap-v1.safetensors" --ml-device auto --resolution 1536
```

In the desktop app, open **ML → Choose model**. This replaces color detection for
all three output layers. Shape/size limits, door-jamb validation, geometry mode,
and light range still apply. **Use palette** restores the ordinary detector.
The three confidence controls select accepted pixels; higher values usually
accept fewer pixels. They are not calibrated accuracy percentages.

`auto` selects CUDA when this Python environment supports it, otherwise CPU.
Explicit `cuda` reports an error when unavailable. CLI reports include the actual
`detection_backend`. Selecting a model does not install dependencies or contact a
cloud service. Keep the original PNG for export: detection resolution never
reduces the embedded image's resolution.

The runtime caches the loaded model and processes tiles with contextual overlap.
The overlap is discarded before joining predictions. The compact network's halo
covers its receptive field. DINO attention uses bounded windows: overlapping
predictions can depend on their window, and are not equivalent to running the
transformer on the entire image. GPU memory is bounded by a tile rather than the
whole upscaled map.
Cancellation is checked between tiles. The first pass includes Python/torch/model
initialization; subsequent passes are much faster. Measure both.

## See-through barriers

The model distinguishes an opaque palisade from a rail fence. Detected transparent
barriers appear as a separate dashed preview layer. They are **omitted from UVTT
walls**, including `objects_line_of_sight`, so they cannot silently become opaque
vision blockers. They also do not become door portals.

This choice is intentional. In the inspected
[Foundry Universal Battlemap Importer source](https://github.com/moo-man/FVTT-DD-Import/blob/3c78eb1b640f00b9bc9b3e30382ebbd03ee374ba/ddimport.js),
object lines are merged with ordinary walls. Portals with `closed: false` are
interpreted as proximity-limited windows, not universally transparent fences.
Other importers differ. A separate movement-only export is not implemented.
Misclassified barriers remain possible; inspect the preview. Palette detection
does not infer transparency from arbitrary artwork.

## Train on your own annotations

`scripts/train_ml.py` trains using local images and label masks, with either a
randomly initialized compact network or a locally supplied DINOv2-small backbone.
It does not use the noncommercial CubiCasa checkpoint or code. Images are BGR/RGB
artwork; masks are single-channel uint8 images with these IDs:

| ID | Meaning |
| --- | --- |
| 0 | Background, floor, and nonstructural objects |
| 1 | Opaque wall material |
| 2 | Entire door leaf, including shading and hinges |
| 3 | Light source |
| 4 | See-through physical barrier |
| 255 | Uncertain / unlabelled; ignored by training |

Keep source images, labels, and checkpoints uncompressed (BMP and safetensors are
supported). A dataset manifest has this structure, with as many examples as needed:

```json
{
  "format": "auto-wall-training-v1",
  "classes": ["background", "wall", "door", "light", "transparent_barrier"],
  "samples": [
    {"id": "map-a", "image": "a.bmp", "mask": "a-labels.bmp", "group": "layout-a", "split": "train"},
    {"id": "map-b", "image": "b.bmp", "mask": "b-labels.bmp", "group": "layout-b", "split": "validation"},
    {"id": "map-c", "image": "c.bmp", "mask": "c-labels.bmp", "group": "layout-c", "split": "test"}
  ]
}
```

Paths are relative to the manifest, or absolute. Originals, upscales, recolors,
and crops of the same source layout must share a group. The loader rejects a group
appearing in multiple splits. Training selects checkpoints using validation only;
the test split is reserved for the final evaluation.

```shell
python scripts/train_ml.py dataset/manifest.json training/run-1 --device cuda --steps 2400 --batch-size 8 --crop-size 384 --context-dilation 4
python benchmarks/evaluate_ml.py training/run-1/best.safetensors dataset/manifest.json review/validation --split validation
python benchmarks/evaluate_ml.py training/run-1/best.safetensors dataset/manifest.json review/test --split test
```

The trainer uses class-balanced crop locations, rotations/flips, scale and color
augmentation, spatial illumination changes, AdamW, cross-entropy plus Dice loss,
and mixed precision on CUDA. It writes `best.safetensors`, `last.safetensors`, and
`training.json`. The compact network has 488,069 parameters at width 16; dilated bottom
convolutions add context without increasing parameter count. Checkpoint metadata
records classes, architecture, layout splits, seed, recipe, and source hashes.
Weights are saved atomically as safetensors; loading never unpickles Python code.

For DINO training, install `requirements-ml-dino.txt` and supply the official
[facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) safetensors
weights locally. The fixed architecture expects that model's 384-channel,
12-layer, 14-pixel-patch backbone. Loading checks every tensor strictly. The
backbone remains frozen; the context projection and local decoder learn the map
classes. The supplied backbone's SHA-256 is recorded in training metadata.

```shell
python scripts/train_ml.py dataset/manifest.json training/dino-1 --architecture dino --backbone pretrained/dinov2-small.safetensors --device cuda
```

Use `--init-model training/dino-1/best.safetensors --architecture dino
--learning-rate 0.0002` to fine-tune an Auto-Wall checkpoint. This starts a **fresh
optimizer and schedule**, not an exact interrupted-run resume. The requested
architecture, width, and context dilation must match the checkpoint. Do not
combine `--init-model` with `--backbone`. Initial weights are hashed and preserved;
write the new run to a separate output directory.

DINOv2 weights are covered by Apache 2.0. Preserve the upstream license and
attribution when distributing derived checkpoints; see
[models/licenses/DINOv2.txt](../models/licenses/DINOv2.txt).

### Select using exported geometry

Pixel agreement alone can favor a model with missing gates or overly thick walls.
Add a `geometry` JSON path to validation samples and use `--selection geometry`.
The trainer runs each prediction through the actual detector and UVTT conversion,
then checks door endpoints, light positions, and open/blocked sight lines. It
reuses the prediction instead of running the model a second time.

```shell
python scripts/train_ml.py dataset/manifest.json training/geometry-1 --selection geometry --geometry-settings profiles/ml-small-lights.json
```

Geometry settings must match the intended preview/export settings. Validation
images are processed at their stored resolution; create bounded-resolution
dataset images before training. The selected score is 60% geometry and 40%
foreground pixel IoU. Geometry averages ray correctness and precision-weighted
instance F0.5; false extra doors/lights cost more than misses. Only complete
instance lists contribute to that score. Partial lists still report matches,
but do not claim false-positive precision. These are tunable research choices,
not proof of general accuracy. See the
[annotation format and evaluation limits](GEOMETRY_EVALUATION.md).

Partially annotated masks use 255 outside reviewed regions. Include explicit
background around traced wall boundaries: marking only the wall stroke leaves
the trainer free to cover neighboring floor. Samples without a requested class
still crop around supervised pixels. Dataset metadata records decoded image and
mask hashes, geometry annotation hashes, and the selected detector settings.

The evaluator writes uncompressed class masks, overlays, per-class pixel overlap,
and first/warm inference times. Green is opaque wall, magenta door, yellow light,
and blue transparent barrier. `benchmarks/check_ml_probes.py` also checks manually
specified reference regions against saved class IDs. These sparse checks do not
replace complete independent annotations.

## Limits of the initial training set

The initial dataset contains three training layouts: a temple, a spaceship, and a
wooden camp. A stone camp is used for validation; an astral cavern is reserved for
testing. The upscaled spaceship and related temple renders are not counted as new
layouts. The source artwork and private generation document are not redistributed.

These are **weak, visually reviewed labels**, seeded partly from color masks and
partly from hand-traced structure. The training labels include manual corrections
for opaque palisades, transparent rail fences, shaded masonry, and complete door
surfaces. Overlap scores against these masks measure agreement with those labels,
not independent map-wide semantic accuracy. Lighting labels are primarily flames;
unreviewed spaceship emissive machinery is ignored. Reliable recognition of all
cool lights, glass, barred doors, and unfamiliar art styles is not established.

Full geometry still needs review. Pixel overlap alone cannot establish continuous
walls, correct door endpoints, or open-passage behavior. A candidate rejected by
the jamb checks remains absent from the export. Do not lower those checks merely
to increase the reported door count. More diverse, fully corrected labels are
needed before making this the default detector.
