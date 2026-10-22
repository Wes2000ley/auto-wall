# Accuracy across styles and the next useful ML step

Assessment date: 2026-10-22. This release implements palette imports, stricter
light-source grouping, dark/shaded door support, and optional C++ centerlines.
The app does not contain a trained semantic model. An official pretrained CubiCasa
checkpoint was downloaded and evaluated locally on CPU, outside the application.
No training, dataset download, or GPU inference was performed.

## What the supplied schema actually provides

The map-style JSON is a generation specification. Its structural palette is gray
`#A0A0A0` walls, purple `#702090` doors, and black void. Art style and theme are
independent of that palette. The importer reads those colors while ignoring all
generation instructions. The full private generation document and map images are
not included in the repository; `profiles/foundry-v2.json` is the minimal palette.

The images do not consistently obey the specification. Five of the six additional
images contain zero exact `#702090` pixels; the upscaled spaceship contains one.
The wooded camp uses brown palisades rather than reserved gray. Consequently,
strict RGB matching and "all gray pixels are walls" cannot solve this corpus.
Tolerant matching helps shaded schema-aligned maps, while exact mode remains useful
for genuinely exact structural artwork. Merely increasing tolerances on wood would
also detect crates, tents, furniture, and tree trunks.

## What was observed

Five consecutive local detection runs per additional image, capped at a 2048 px
working edge; median timings exclude image decoding and export. The maps themselves
were reviewed locally and are not distributed. These are output counts, not measured
precision/recall against independently annotated ground truth.

| Artwork | Wall segments | Door candidates | Light candidates | Median |
| --- | ---: | ---: | ---: | ---: |
| Wooden camp | 0 | 0 | 9 | 24 ms |
| Spaceship, upscaled | 1615 | 54 | 125 | 105 ms |
| Spaceship, original | 2114 | 55 | 110 | 97 ms |
| Stone camp | 2101 | 4 | 9 | 42 ms |
| Purple temple | 2220 | 68 | 30 | 95 ms |
| Astral cavern | 1837 | 46 | 21 | 109 ms |

The first five use the supplied wall/door palette. Spaceships use cool-source light
matching. The cavern uses auto-gray masonry, orange doors, and cool sources.
The original and upscaled spaceship are the same layout, not independent examples.

The temple's 68 door candidates and 30 torches are retained by centerlines, which
reduce its wall segment count to 1140. The original spaceship's 55 detected door
crops were reviewed: they span visible purple doors, including diagonal ones. This
does not establish that all real doors were found. Every accepted outline portal
endpoint in the six-map run touches a wall within 1e-9 original-image pixels.

The wooden camp's nine light crops correspond to eight gate torches and a central
campfire. Its structural pass returns zero instead of labelling unrelated gray
rocks as walls, and warns that the palette may not match the artwork. Manual wall
editing or semantic segmentation is still needed there. The stone camp detects
four wide gates but follows masonry shading and omits brown interior fences; its
outline needs review. Some small flames are missed by the conservative light limits.

The cavern light pass includes luminous eye motifs as well as braziers. The spaceship
light pass includes displays and machinery and misses some pale strip lights. Neither
should be described as a verified set of physical light fixtures. Color filtering
can identify emissive-looking material but cannot reliably infer its semantic role.

## Failed trials and resulting fixes

- A fixed minimum door brightness excluded the schema's dark purple. Brightness
  now follows the selected sample, and exact RGB uses original pixels rather than
  a lossy HSV round trip.
- The earlier maximum door span and split shaded regions missed the stone camp's
  gates. Wider spans plus small seam repair found all four; wall-jamb checks remain.
- Using bright components alone produced 291 light candidates on the wooded camp.
  Initial warm-source grouping still produced 41 false-heavy candidates and missed
  the large campfire. Raising the same threshold for both source and core then lost
  real torches. Separating colored-source area from bright-core fraction, checking
  fill, and allowing larger sources yielded the reviewed nine. These thresholds
  still need independent maps to establish generalization.
- Connecting centerline doors only to free endpoints found 2 of the temple's 68.
  Doors often meet the middle of a continuing wall. Axis intersections and short
  connectors constrained by the wall mask resolve this topology. Three remaining
  connections failed on one shaded endpoint pixel; a two-reference-pixel raster
  tolerance retained all 68. Open-floor connectors remain rejected and rejected
  door cuts are restored.
- A new geometry test exposed an ambiguous NumPy truth check in the connector API.
  Explicit length checking fixed it. The failed run is retained in the local
  validation history; the corrected focused suite passed.

## Actual pretrained-model trial

The official CubiCasa5k `hg_furukawa_original` checkpoint loads successfully with
PyTorch 2.11.0+cpu, `weights_only=True`, and strict state-dictionary matching. Its
740 state entries match the 44-channel architecture. Checkpoint SHA-256:
`dd20b4e1bf1d670f2125107b079df06958b1ccd36e49a464ab739aeb00b8e7a2`.
The official architecture and weights stay outside the fork because the project's
[published license](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE) is
CC BY-NC 4.0. The evaluation wrapper contains no copied model implementation.

Preprocessing follows the official RGB [-1, 1] normalization. Four rotations are
run and their spatially restored room/icon logits averaged, following the official
sample's semantic pass. Wall is room class 2; door is icon class 2. Junction heatmaps
and the author's polygon postprocessor are not evaluated. The input keeps its aspect
ratio, capped at 1024 px, with no upscaling. Masks require both class argmax and
probability >= 0.5 or >= 0.8; these probabilities are not calibrated guarantees.
[Official sample procedure](https://github.com/CubiCasa/CubiCasa5k/blob/master/samples.ipynb),
[official normalization](https://github.com/CubiCasa/CubiCasa5k/blob/master/floortrans/loaders/svg_loader.py).

A raster crop of the architectural source example embedded in the official notebook
serves as a positive control. It produces recognizable walls and eight disconnected
door-mask regions. This is a visual sanity check, not a reproduction of the paper's
dataset metrics. The same loaded model produces the following on all seven supplied
images (the spaceship pair shares one layout):

| Artwork | Door-mask regions at p >= 0.8 | Warm CPU inference |
| --- | ---: | ---: |
| Wooden camp | 0 | 2.92 s |
| Purple temple | 0 | 3.20 s |
| Spaceship, original | 1 | 3.04 s |
| Astral cavern | 0 | 3.66 s |
| Stone camp | 0 | 3.30 s |
| Demon Temple | 0 | 3.05 s |
| Spaceship, upscaled | 1 | 2.94 s |

Each time is the median of two measured four-rotation passes after one warmup, using
eight CPU threads; it excludes decode, model load, mask conversion, and rendering.
These region counts are not valid-door counts or precision/recall. At p >= 0.5,
the five non-spaceship images still contain zero door pixels. Visual review shows
walls placed on the spaceship's walkable corridors, on black temple backgrounds,
and on camp props, while much actual masonry/palisade is missed. The stricter
threshold reduces coverage without repairing these semantic errors.

To check whether resolution caused the failure, the camp (1536 × 1024) and purple
temple (1448 × 1086) were rerun at their original size, using four rotations. They
still produce zero door pixels at both thresholds and misplaced wall masks. One
warm measured pass took 7.26 s and 7.36 s respectively, after one warmup. Thus the
tested checkpoint is unsuitable for automatic wall/door placement on this corpus;
this does not establish that a model trained on battle maps would fail. The model
also lacks a general torch/light-source class.

The evaluation can be reproduced with the optional PyTorch dependency and inspected
external source/weights:

```shell
python benchmarks/evaluate_cubicasa.py "map.png" --model-source "research/hg_furukawa_original.py" --checkpoint "research/model_best_val_loss_var.pkl" --output "research-results" --max-edge 1024 --rotations 4 --runs 2
```

Obtain the checkpoint from the official README's pretrained-model link. For the
standalone architecture file, remove its unused `from floortrans.models import
model_1427` import; the wrapper constructs the 44-channel class and replaces all
weights directly, so its legacy `init_weights` method is never called. This avoids
loading the unrelated pose-pretraining checkpoint. Inspect the external source before
running it. The wrapper refuses unsafe pickle loading, checks finite predictions,
records hashes/backend/timings, and writes uncompressed BMP masks and comparison
panels (green walls, magenta doors). No images or weights are sent to a service.

CUDA was not benchmarked: the installed PyTorch and ONNX Runtime environments are
CPU builds/providers. GPU inference could change latency, but would not by itself
repair the demonstrated semantic failures. SAM 3 and training a U-Net/SegFormer were
not run; no labeled battlemap training set or SAM checkpoint was available for this
trial. Architectural pretraining alone does not meet the requested precision.

## Next ML design

Keep this fast palette path for structurally colored maps. For artwork such as
wooden palisades, train a small segmentation model on **background/floor, wall,
door, and light-source** labels. Feed its accepted masks into the existing geometry
stage; retain connected jamb validation, open-passage checks, and editable vectors.
Keep confidence thresholds class-specific and prefer abstaining to drawing an
unsupported barrier. This is a proposed next implementation, not a shipped model.

Start with a small U-Net baseline, then compare a compact SegFormer only if it
improves held-out errors enough to justify its latency. Segmentation Models PyTorch
supports both architectures and configurable output classes. Pretrained image
encoders do not supply battlemap-specific labels by themselves.
[Official model documentation](https://smp.readthedocs.io/en/latest/models.html).

SAM 3 is useful to evaluate as an annotation or interactive correction tool: it
accepts textual and visual prompts. Its current prerequisites include Python 3.12+,
PyTorch 2.7+, and CUDA 12.6+, and checkpoint access has a separate setup process.
That capability does not establish reliable wall/door topology on these maps.
Benchmark memory and latency on the target GPU before choosing it for the desktop
loop. [Official SAM 3 repository](https://github.com/facebookresearch/sam3).

CubiCasa5k contains 5000 architectural floor plans with many annotated categories.
It is a possible research pretraining source, subject to its license, rather than
evidence of performance on fantasy or spaceship artwork. The domain difference
must be measured with held-out battle maps.
[Official CubiCasa5k repository](https://github.com/CubiCasa/CubiCasa5k).

## Data and acceptance criteria

1. Annotate walls, actual door spans, intentional open passages, and physical light
   sources. Include confusing negatives: floors, rocks, crates, logs, fences that
   should not block sight, decorative purple objects, screens, symbols, and glints.
   Record whether magical effects and emissive screens should count as lights.
2. Correct any color-generated labels manually. Training against this detector's
   own uncorrected masks would teach its present mistakes. Keep uncertain regions
   ignored during training until reviewed.
3. Split by source layout and artist/style. Keep each original, upscale, recolor,
   crop, and augmentation in the same split. The two spaceship resolutions and
   related temple layouts must not inflate the test set.
4. Evaluate false wall length on walkable floor, falsely sealed passage count,
   doorway precision/recall and endpoint error, closed-door leaks, opened-door
   obstruction, and light precision/recall. Pixel overlap alone misses topology
   failures. A useful release gate should prioritize fewer false barriers/doors;
   choose numeric targets with a sufficiently large annotated evaluation set.
5. Measure complete decode-to-preview latency, warm/cold inference, RAM/VRAM, and
   cancellation on actual large maps. Retain the current CPU baseline for comparison.

## GPU and native-code choices

The evaluated machine has an RTX 3080 with 10 GiB VRAM. A compact tiled model is a
reasonable experiment, but no particular training batch size or model speed has
been established here. Tile overlap must preserve thin walls and door context;
measure seam errors before reducing resolution. GPU acceleration becomes useful
for learned inference, while existing OpenCV kernels already run natively on CPU.

ONNX Runtime offers a CUDA execution provider for Python and C++. Its package,
CUDA, and cuDNN compatibility matrix must be matched before deployment. CPU fallback
should report its actual backend. Test numerical and geometry parity after export;
using a GPU is not evidence of better detection accuracy.
[Official CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

Profiling identified Python centerline graph tracing as a real bottleneck. The
optional C++17 tracer now replaces only that graph walk and preserves exact ordered
paths, junctions, loops, and cancellation. Complete centerline detection is
1.86–2.20× faster on the six nonempty maps at the 2048 px cap, with exact geometry
and light parity across all seven inputs. At original Demon resolution the gain is
1.42× because skeletonization remains expensive. OpenCV outlines retain their
existing native implementation. See [build instructions and timings](FAST_SETUP.md#optional-c-centerline-acceleration).
