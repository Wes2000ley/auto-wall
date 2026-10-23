# Cool-light and negative-example experiments

Trials 15–17 extend the optional DINO light-center work. None replaces the
bundled v1 model: learning cool emitters exposed regressions on small lanterns
and non-emissive flowers. The new dataset generator, negative crop anchors,
optional loss normalization, and sampler memory fix are available for further
training. Inference, the GUI, export behavior, and bundled weights are unchanged.

The [machine-readable record](../models/LIGHT_RESEARCH_TRIALS_15_17.json) includes
checkpoint hashes, recipes, source hashes, selection results, and comparisons.
Raw best/final checkpoints, histories, source snapshots, and review images were
retained locally. Private map artwork is not redistributed.

## Data and selection

The expanded dataset contains four real training samples: the wooden camp,
temple, reviewed negative regions of Dyson 13, and newly reviewed spaceship
regions. The ship has 39 marked cool emitters, including corridor strips,
cryopods, and luminous machinery. Only 11.18% of its pixels are reviewed;
remaining screens, machinery, and other areas are ignored. Ship figures below
are partial **training** results, not unseen-map generalization.

There are 64 earlier procedural training maps and 64 new fixture maps. The new
generator draws torches, lanterns, campfires, strips, and orbs, together with
flowers, painted stars, jewels, rivets, and unlit lanterns. Their palettes
overlap. Eight additional fixture seeds remain outside training and checkpoint
selection. Procedural results measure the construction task, not real art.

Checkpoint selection uses only the stone camp and reviewed Dyson 14 regions,
at threshold 0.35 and the existing precision-weighted F0.5 score. Original images,
upscales, and related versions retain their groups. No validation/test images,
point labels, validity masks, or splits changed between these trials.

Trial 17 adds 533 negative crop anchors: 524 procedural decoys, three reviewed
flower errors in the training camp, and six ship bed pillows. It does not train
on the new camp or Astral map. Separate diagnostic annotations were added for
those previously observed test maps: six camp sources and partial Astral regions
covering 20 emitters and 12 painted marks. These are agent-reviewed annotations
after seeing earlier predictions, not independent human ground truth or fresh
unseen validation. Unreviewed Astral portals and glyphs remain ignored.

## Training outcomes

All runs start from the bundled v1 light-center weights, train only the 14,569
light-head parameters, and preserve all 313 segmentation tensors exactly. They
use four 384 px crops, AdamW at initial learning rate 0.0002, cosine decay, seed
73113, mixed precision, and the local RTX 3080.

| Trial | Recipe | Steps | Selected step | Selected stone matches | Validation extras |
| --- | --- | ---: | ---: | ---: | ---: |
| 15 | Expanded data; batch-normalized loss; 3 real + 1 procedural crop | 2400 | 150 | 12/13 | 0 |
| 16 | Same data/mix; normalize each crop's loss separately | 2400 | 300 | 10/13 | 0 |
| 17 | Per-crop loss; negative anchors; 2 real + 2 procedural crops | 4800 | 150 | 11/13 | 0 |

Trial 15's selected early checkpoint learned only 2/39 ship sources and added
six false lights on the new camp. Later checkpoints learned more cool sources
but lost small stone-camp lights; at one checkpoint validation also contained
12 extras. Trial 16 reduced some false positives but did not restore the missing
lights. Trial 17 combined negative anchors, a larger procedural fraction, and
more steps; its gains cannot be attributed to just one of those changes.

Completed training took 260.03, 155.99, and 348.33 seconds respectively. Each
reported 602,834,944 bytes of peak allocated CUDA tensors, excluding reserved
memory, context, and other processes. Other desktop work and validation varied;
these timings are not a controlled speed comparison between loss functions.

## Fixed final-checkpoint diagnostics

Each entry reports matched sources / marked sources, followed by false extras.
These final checkpoints are retained diagnostic results, not substitutes for
the checkpoints selected above. Predictions use the normal model loader and
light decoder before the export pipeline rejects doorway highlights.

| Reviewed region | Bundled v1 | Trial 15 final | Trial 16 final | Trial 17 final |
| --- | --- | --- | --- | --- |
| Wooden camp, training | 9/9 + 0 | 9/9 + 3 | 9/9 + 1 | 9/9 + 0 |
| Stone camp, validation | 12/13 + 0 | 10/13 + 0 | 10/13 + 0 | 10/13 + 0 |
| Ship, partial training | 0/39 + 0 | 36/39 + 9 | 36/39 + 7 | 37/39 + 5 |
| New camp, observed test | 6/6 + 1 | 6/6 + 7 | 6/6 + 4 | 6/6 + 4 |
| Astral, partial observed test | 16/20 + 8 | 17/20 + 7 | 17/20 + 6 | 20/20 + 7 |
| Eight procedural test seeds | 42/63 + 76 | 52/63 + 58 | 54/63 + 47 | 59/63 + 39 |

The temple remained 30/30 with no extras. Reviewed ink regions had no extras.
Trial 17 recognizes all four reviewed cool Astral fixtures without training on
that layout, but it still confuses decorative marks and flowers with emitters.
Partial-region extras exclude predictions elsewhere: trial 17 has 32 total
Astral peaks, of which five lie outside the reviewed regions. Neither these
figures nor the ship scores establish full-map precision.

The data changes help teach new emitter types, while persistent errors show
that the current head needs better semantic discrimination and retention of
dim/small sources. Increasing candidate counts or lowering the acceptance
threshold is not a demonstrated solution.

## Sampler memory and validation

`np.argwhere(valid)[::16]` retained the full coordinate backing array even though
only every sixteenth coordinate was used. Copying the sampled coordinates
releases that buffer. On all 132 training samples, retained coordinate storage
fell from 1,081,996,288 to 67,624,784 bytes, a 93.75% reduction. Every coordinate
is identical. This describes those NumPy buffers, not total process RAM or
inference memory. The change was made after the three runs; it does not explain
their differing runtimes.

Validation covers deterministic generation, uncompressed BMP output, preserved
test seeds, dataset loading, negative-anchor bounds/validity, ignored pixels,
loss normalization, actual optimizer/save/load behavior, and frozen segmentation
tensors. The data audit checked that all 142 samples retained their images,
positive points, masks, and splits, and that all 64 new training images regenerate
exactly. An initial audit assumed slash-separated source keys and failed on
Windows; normalizing separators repaired the audit without changing model/data.

The app model and inference code were not replaced, so this batch did not rerun
the desktop GUI/export benchmark or live Foundry import. It does not claim
improved application latency. See [local training instructions](LOCAL_ML.md) for
the generator, `negative_points`, and `--loss-normalization sample`.
