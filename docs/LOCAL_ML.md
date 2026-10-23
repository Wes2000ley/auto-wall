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
pretrained DINOv2-small backbone with a trainable local decoder. The backbone is
frozen by default. This adds visual context while retaining fine boundary features.
Both architectures load
one local safetensors file; the app does not fetch pretrained weights or execute
remote model code. DINO is larger and slower than the compact model. Consult the
[model notes](../models/README.md) before choosing a checkpoint.

The included `models/battlemap-dino-v1.safetensors` is an optional comparison
checkpoint. See its [model card](../models/DINO_MODEL_CARD.md) for measured warm
and cold runtimes, strict geometry results, and known errors. It does not replace
the compact baseline or make ML the default.

```shell
python fast_auto_wall.py "map.png" --model "models/battlemap-v1.safetensors" --ml-device auto --resolution 1536
```

In the desktop app, open **ML → Choose model**. This replaces color detection for
all three output layers. Door shape/size limits, door-jamb validation, geometry
mode, and light range still apply. **Use palette** restores the ordinary detector.
The three confidence controls select accepted pixels; higher values usually
accept fewer pixels. They are not calibrated accuracy percentages.

The optional `models/battlemap-dino-lights-v1.safetensors` adds a separate learned
light-center head to exactly the same segmentation tensors. It predicts one
location per physical source, so disconnected flame pixels need not become
separate lights. Set **Light center score** to adjust acceptance (the tested
profile uses 0.35). This score is not a calibrated probability of correctness.
For this model, light pixel confidence, area, fill, and brightness limits are
replaced by the point score; source range and door-highlight rejection still
apply. Wall and door confidence controls keep their existing meaning.

```shell
python fast_auto_wall.py "map.png" --settings profiles/ml-light-centers.json --model models/battlemap-dino-lights-v1.safetensors --ml-device auto
```

This model improves the two checked camps but also mistakes some flowers and
decorative stars for lights and misses cool machinery emitters. It remains an
optional comparison. See the [light-center model card](../models/DINO_LIGHTS_MODEL_CARD.md).

**Recover shaded door jambs** is an optional ML control for missing wall faces
beside otherwise plausible doors. It accepts lower-confidence wall predictions
only when they connect to confident walls and lie near the two ends of a
shape-validated door. It never turns background, door, or transparent-barrier
classes into walls. Door size, shape, and connection checks still apply. Review
the added wall edges; this improves the two evaluated camps but does not solve
unfamiliar door styles or prove accuracy on other layouts.

The option is off in the application defaults. To select it from the CLI:

```shell
python fast_auto_wall.py "map.png" --settings profiles/ml-jamb-recovery.json --model models/battlemap-dino-v1.safetensors --ml-device auto
```

The profile uses a 1536 px detection edge, 0.9 class confidence, and the small-light
area setting. Unchecking the ML control restores strict wall confidence. See
[the experiment notes](ML_EXPERIMENTS.md) for paired geometry checks and limitations.

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
backbone remains frozen by default; the context projection and local decoder
learn the map classes. The supplied backbone's SHA-256 is recorded in training metadata.

```shell
python scripts/train_ml.py dataset/manifest.json training/dino-1 --architecture dino --backbone pretrained/dinov2-small.safetensors --device cuda
```

Use `--init-model training/dino-1/best.safetensors --architecture dino
--learning-rate 0.0002` to fine-tune an Auto-Wall checkpoint. This starts a **fresh
optimizer and schedule**, not an exact interrupted-run resume. The requested
architecture, width, and context dilation must match the checkpoint. Do not
combine `--init-model` with `--backbone`. Initial weights are hashed and preserved;
write the new run to a separate output directory.

For a controlled domain-adaptation experiment, add `--backbone-layers 1` (up to
four final transformer blocks). The selected blocks and final layer normalization
learn at one tenth of the decoder's learning rate; the earlier backbone remains
frozen. Both rates follow the same decay schedule and are recorded in metadata.
This costs additional training memory and can overfit a small dataset. Compare
exported geometry on separate layouts before choosing the resulting checkpoint.
Zero, the default, preserves decoder-only training. The stored model format and
inference path are unchanged; loading for inference freezes all backbone weights.

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

## Train light centers without changing walls or doors

`scripts/train_light_centers.py` adds or updates the small light head on a local
DINO model. It freezes all segmentation parameters and normalization buffers,
and checks their hashes before saving each checkpoint. It uses center heatmaps
and the penalty-reduced focal loss described in [Objects as Points](https://arxiv.org/abs/1904.07850).
Training the segmentation decoder underneath a trained light head would change
its input features, so the segmentation trainer requires the segmentation-only
base checkpoint instead.

Use a separate point manifest. Each `points` entry is one physical source's XY
center in the stored image's pixels. Include all sources in every labelled
region; an empty list marks reviewed regions with no light sources. An optional
single-channel uint8 `valid_mask` uses zero for unreviewed pixels and any nonzero
value for reviewed pixels. Without it, the entire image is treated as reviewed.
Unmarked emitters in reviewed regions would become incorrect negative labels.

```json
{
  "format": "auto-wall-light-centers-v1",
  "samples": [
    {"id": "camp-a", "group": "camp-a", "split": "train", "image": "a.bmp", "points": [[120, 80], [300, 220]]},
    {"id": "ink-a", "group": "ink-a", "split": "train", "image": "ink.bmp", "points": [], "valid_mask": "ink-reviewed.bmp"},
    {"id": "camp-b", "group": "camp-b", "split": "validation", "image": "b.bmp", "points": [[96, 144]]}
  ]
}
```

```shell
python scripts/train_light_centers.py dataset/lights.json training/lights-1 --init-model models/battlemap-dino-v1.safetensors --device cuda --steps 2400
```

The default batch has four 384 px crops. If samples have `synthetic: true`, one
crop per batch comes from that group and three come from real maps; adjust
`--synthetic-per-batch` when needed. If only one group exists, it supplies the
whole batch. The sampler transforms coordinates before rendering targets so tiny
lights survive scaling. Rotations, flips, varied hue/saturation, exposure, and
illumination preserve physical source identities.

Validation uses separate layouts and a fixed score threshold (`--threshold`,
default 0.35). Checkpoint selection maximizes light-instance F0.5, which penalizes
extra detections more than misses. Predictions outside reviewed regions do not
count as false positives; each report also includes their total count. Matching
tolerance is six pixels at a 1024 px longest edge. Use bounded-resolution dataset
images matching intended inference quality. The test split is not used for training
or selection. Review exported geometry and additional layouts before choosing a
new model; this score does not establish general semantic accuracy.

Each new output directory receives raw, uncompressed `best.safetensors` and
`last.safetensors`, recipe/source hashes, validation history, and completion timing.
Both model files load directly in Fast Setup. Supplying a point model as
`--init-model` continues from its light-head weights with a fresh optimizer and
schedule. CPU works as well as CUDA; no cloud service or download is required.

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
