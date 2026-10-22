# Battlemap v1: experimental baseline

`battlemap-v1.safetensors` is a 488,069-parameter U-Net trained from scratch locally
on an RTX 3080. It is 1,969,292 bytes, uncompressed. Its SHA-256 is
`cf7a10498fa9c938ce96f68fded84632c9e70fe4115f7a46f71edd777aca4bbe`.
`battlemap-v1.json` records the recipe, data hashes, and checkpoint selection.
No CubiCasa weights or code are included.

This is a preserved research baseline, **not the default detector**. The ongoing
goal is substantially stronger structural accuracy and cross-style generalization.
See [usage and training instructions](../docs/LOCAL_ML.md).

## Training and selection

Three layouts were used for training: temple, spaceship, and wooden camp. The
stone camp was used for validation and checkpoint selection. The astral cavern
was evaluated only after selecting the final checkpoint. Original/upscaled and
related temple images are grouped together. No private source maps are included.

Labels were partly color-generated and partly hand-traced, then visually reviewed
and corrected. They are weak supervision, not independently complete ground truth.
Validation uses the same stone-camp annotations in each trial.

| Trial | Change | Steps | Selected step | Mean foreground IoU against validation labels |
| --- | --- | ---: | ---: | ---: |
| 1 | Small U-Net, 256 px crops | 1600 | 800 | 0.5350 |
| 2 | Wider context, 384 px crops, spatial lighting augmentation | 2400 | 1200 | 0.6510 |
| 3 | Corrected shaded masonry and complete door-leaf training labels | 2400 | 1500 | 0.7457 |

These overlap values are label agreement, not map-wide accuracy. A separate
normalization-statistics calibration trial reduced wall recall and was not selected.
Broad neutral-color expansion on spaceship annotations incorrectly captured
walkable areas; that proposed annotation change was rejected before training.

## Observed performance

With PyTorch 2.11.0+cu128, four CPU threads, and CUDA on an RTX 3080, median warm
segmentation times over five runs were 57.9–62.6 ms across five layouts at a
1536-pixel maximum edge. PyTorch's peak allocated GPU tensor memory was 276,673,024
bytes; this excludes CUDA context/allocator overhead and other applications.
The same stone-camp model took a median 1.43 s on CPU with four threads.
Timing varies with concurrent machine activity. These are inference measurements,
not image decoding, geometry, preview, or export times.

The conservative full pipeline, with the default light minimum area, produced:

| Map | Split | Wall segments | Doors | Lights | Median warm detection |
| --- | --- | ---: | ---: | ---: | ---: |
| Wooden camp | Train | 1064 | 0 | 9 | 120 ms |
| Stone camp | Validation | 1714 | 3 | 9 | 136 ms |
| Temple | Train | 2183 | 68 | 30 | 212 ms |
| Spaceship | Train | 1782 | 58 | 1 | 183 ms |
| Astral cavern | Test | 2384 | 37 | 2 | 203 ms |
| Demon temple | Related train layout | 2411 | 62 | 28 | 341 ms |

These are candidate counts, not precision/recall. Detection uses a 1536 px edge;
exported map pixels are unchanged. All six UVTT checks preserved the original PNG
bytes exactly. Transparent preview segments were excluded from both LOS arrays.
The palette detector remains substantially faster on maps matching its palette.

## Known failures to address

- Wooden camp gate pixels are recognized, but no gate survives the surrounding
  wall-jamb checks. Platforms and their structural edges need better supervision.
- The stone camp's east gate is incomplete; three of four gates are exported.
  Dark wall faces and some railings remain partly missed.
- At a smaller light-area cutoff, the temple produces 30 real torch candidates
  plus two bright-symbol false positives. The astral map produces 13 visible flame
  candidates plus two symbol fragments. Retaining the conservative cutoff removes
  small false positives but misses many small real lights. This is an unresolved
  semantic/instance-detection problem, not a finished lighting solution.
- Spaceship output misses most strip lights and contains questionable door pixels
  on some fixtures. The generation schema does not provide complete semantic labels.
- Sparse hand-selected checks cover twelve negative regions (tents, wagon, floor,
  crates, a log, and a tree) and four see-through regions in the stone camp. They
  contain no predicted opaque pixels at the 0.9 threshold. This limited spot check
  does not establish that the whole map is free of false walls or closed passages.
- Small seams and jagged boundaries still produce excessive vector segments.
  Correct pixel classes alone do not establish usable topology.

Source code, checkpoints, overlays, raw trial histories, and the exact selected
training source are retained locally. The model should be compared against future
experiments on geometry and unseen layouts before replacing it or changing defaults.
