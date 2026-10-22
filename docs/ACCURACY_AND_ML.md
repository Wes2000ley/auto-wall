# Accuracy across styles and the next useful ML step

Assessment date: 2026-10-22. This release implements palette imports, stricter
light-source grouping, dark/shaded door support, and optional centerlines. It does
not contain a trained semantic model. No weights or datasets were downloaded and
no training or GPU inference was performed.

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

## Recommended ML design

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

A Rust/C++ rewrite is not currently justified by the measured outline timings.
Profile future learned-mask vectorization and centerline graph tracing first. Move
only a demonstrated bottleneck into a native extension if it materially improves
interactive latency; keep the same tests and geometry contract.
