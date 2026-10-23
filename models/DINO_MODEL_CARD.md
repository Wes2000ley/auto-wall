# DINO battlemap v1 — experimental comparison model

`battlemap-dino-v1.safetensors` combines a frozen DINOv2-small backbone with a
trainable U-Net decoder. It is available in the existing ML model picker and CLI.
It improves some cross-style boundaries and gate detection, but still misses
doors and produces false lights. **It is not the default detector.**

The optional [light-center model](DINO_LIGHTS_MODEL_CARD.md) retains this model's
exact segmentation tensors and adds a separately trained light head. It improves
some source counts/grouping but has its own false-positive limits; choose it
explicitly to compare.

The checkpoint has 22,593,925 parameters, of which 537,349 were trained. The raw,
uncompressed file is 90,429,548 bytes. SHA-256:
`02824e2c6f67a0c800fd027a1c194863d9733e75eb05c1f46b8c64b59a82786c`.
The machine-readable [model record](battlemap-dino-v1.json) contains source hashes,
training metadata, selected validation results, and benchmark settings.

## Run it

Install `requirements-fast.txt`, the appropriate PyTorch build, and
`requirements-ml-dino.txt` in the same Python environment. The model is local;
the application does not download weights or run remote model code.

```shell
python fast_auto_wall.py "map.png" --settings profiles/ml-small-lights.json --model models/battlemap-dino-v1.safetensors --ml-device auto
```

The small-light profile uses a 1536 px detection edge and accepts small emitter
candidates. This exposes some light false positives; inspect the preview. Detection
resolution does not change the embedded export image. Use the ML tab to select a
different local model, change confidence thresholds, or return to palette detection.

See-through barriers have a separate dashed preview and are omitted from opaque
UVTT walls and portals. Classification mistakes are still possible. Existing
door shape, size, and two-jamb checks remain enabled.

## Training and selection

The backbone comes from [Meta's DINOv2-small weights](https://huggingface.co/facebook/dinov2-small)
at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Its original safetensors
SHA-256 is `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
Auto-Wall adds a context projection and local decoder for background, opaque wall,
door, light source, and transparent barrier classes. The backbone stays frozen.

Real-art training groups were the temple family, spaceship, wooden camp, and
Dyson Logos's Index Card Dungeon II map 13. The stone camp and Dyson map 14 were
used for validation. Originals and upscales of one layout are not independent
examples. Labels combine color seeds and manual corrections, with uncertain
pixels ignored; they are not complete, independently verified ground truth.

This is trial 8's step-300 checkpoint, initialized from trial 6's selected
step-2250 model. Selection combined exported geometry and foreground pixel IoU:
60% ray/instance checks, 40% IoU. Its recorded score was 0.834331 and mean
foreground IoU 0.746307. These are protocol-specific label-agreement measures,
not a percentage of correctly understood maps.

Trial 8 planned 6000 steps and was stopped after step 2800 when inspection found
a shifted door annotation in the training map. The earlier best checkpoint was
retained. A subsequent 3000-step run with corrected door labels scored lower on
the same validation protocol and did not replace it. Both runs are preserved.

The selected checkpoint was fixed before inspecting the reserved Dyson map 15.
That map exposed missed doors and some interior false boundaries. The astral
cavern and a newly supplied palisade layout also have unresolved errors. These
observed test maps must not be presented as fresh, unseen evidence after tuning
against their results.

## Geometry checks and limits

The original checks below use confidence 0.9 and the small-light profile, before
the shaded-jamb option and sloped-jamb clipping fix. Door matches require
both endpoints to fall within the fixed annotation tolerance. A detected door
with the wrong span counts as a failed match.

| Map | Use | Door matches / annotated | Exported doors | Light matches / annotated | Exported lights | Sight-line checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wooden camp | Training | 1 / 4 | 1 | 7 / 9 | 8 | 12 / 16 |
| Stone camp | Validation | 3 / 4 | 4 | 11 / 13 | 11 | 20 / 20 |
| Dyson map 14 | Validation, partial instances | 1 / 3 | 1 | Not fully annotated | 3 | 12 / 14 |

These are agent-reviewed annotations and sparse probes, not an independent
whole-map audit. The stone camp's north portal is too long, even though it is
visibly recognized. The wooden camp still misses most gates. The three light
candidates on Dyson map 14 were visually identified as pool strokes. The held-out
Dyson map 15 exported zero doors despite visible door symbols. The new palisade
map has missed open gates and duplicate light fragments near its campfire.

Astral output includes some false interior wall fragments and decorative-symbol
light candidates, and misses cool emitters. Spaceship strip lights are not
reliably recognized. Boundary continuity, excessive vector segments, open door
leaves, and light-instance grouping all need further work. Higher confidence
does not guarantee that an accepted object is real.

### Subsequent geometry improvements

The checkpoint weights are unchanged. The optional **Recover shaded door jambs**
control now restores connected wall predictions near validated door ends. With
`profiles/ml-jamb-recovery.json`, all four annotated gates match in each camp,
including the stone camp's north portal. The paired strict results remain 1/4
wooden gates and 3/4 stone gate spans. Lights remain 7/9 matched with one extra
fragment in the wooden camp and 11/13 matched without extras in the stone camp.

The new evaluation also corrects one faulty wooden-pen probe: its original start
lay inside the solid palisade behind the fence. With the start moved to the visible
gap, strict detection passes 13/16 rays and recovery passes 16/16; the stone camp
passes 20/20 in both modes. Original annotations and reports are retained. These
are still sparse agent-reviewed checks, not full-map or independent ground truth.

Fixing door clipping preserves sloped jamb intersections. This restores three
previously rejected, visually confirmed spaceship doors and prevents four real
doors from disappearing when recovery changes their jamb outlines. The source
spaceship now exports 62 doors with or without recovery. This count is not a
complete door-accuracy measurement. See [experiment notes](../docs/ML_EXPERIMENTS.md)
for current paired timings, failed alternatives, and remaining errors.

## Measured runtime

Original benchmark: Windows, RTX 3080, PyTorch 2.11.0+cu128, four CPU threads; five warm runs per map,
without concurrent model training. These times cover `detect_map`, including
resizing, inference, and geometry, but exclude image decoding and UVTT writing.

| Input | Original size | Working size | Median warm detection |
| --- | --- | --- | ---: |
| Wooden camp | 1536 × 1024 | 1536 × 1024 | 0.208 s |
| Stone camp | 1448 × 1086 | 1448 × 1086 | 0.213 s |
| Temple | 1448 × 1086 | 1448 × 1086 | 0.275 s |
| Spaceship | 1536 × 1024 | 1536 × 1024 | 0.311 s |
| Demon temple | 5792 × 4344 | 1536 × 1152 | 0.295 s |
| Dyson map 14 | 9000 × 5400 | 1536 × 922 | 0.236 s |
| Astral cavern | 5792 × 4344 | 1536 × 1152 | 0.278 s |
| Dyson map 15 | 9000 × 5400 | 1536 × 922 | 0.218 s |
| New palisade map | 1536 × 1024 | 1536 × 1024 | 0.244 s |
| Upscaled spaceship | 6144 × 4096 | 1536 × 1024 | 0.324 s |

Model loading took 11.55 s in the main benchmark, excluding the preceding torch
import. The first actual Qt preview took 22.74 s including startup. Cached runs
avoid that initialization. CPU stone-camp detection took a median 5.96 s across
three warm runs with four threads. The compact model remains a faster option.

PyTorch peak allocated GPU tensors were 384,369,152 bytes. This excludes CUDA
context, reserved allocator memory, and other applications. The runtime uses
bounded tiles with overlap and checks cancellation between tiles. Transformer
predictions depend on tile context; they are not identical to whole-map inference.

All ten benchmark exports retained their original input PNG bytes exactly,
including the 9000 px Dyson images. A live Foundry import, other operating systems,
other GPUs, and release packaging were not tested in this experiment.

## Attribution and license

DINOv2 is by Meta Platforms, Inc. and affiliates, distributed under Apache 2.0.
Preserve [its license](licenses/DINOv2.txt) and [attribution](licenses/README.md)
when redistributing this modified checkpoint. This is an Auto-Wall research
model, not an official Meta battlemap detector.

Cartography by Dyson Logos. The map-13 training and map-14 validation sources are
from the [commercial-use map collection](https://dysonlogos.blog/maps/commercial-maps/).
No original cartography or private user artwork is included in this repository.
