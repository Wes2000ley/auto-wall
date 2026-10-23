# Semantic light centers: direct DINO context

The optional `battlemap-dino-lights-v2.safetensors` checkpoint recognizes more
cool emitters and rejects more decorative Astral marks than v1 on the checked
maps. It is a research comparison, not the application default. It still mistakes
flowers and some ink detail for lights. Keep v1 for comparison and inspect the
preview before exporting.

```shell
python fast_auto_wall.py "map.png" --settings profiles/ml-light-centers.json --model models/battlemap-dino-lights-v2.safetensors --ml-device auto
```

The existing model picker, light-center score, cancellation, preview, and export
controls work with this model. The tested score is 0.35. Wall, door, and
transparent-barrier predictions use the same frozen segmentation weights as v1.
Detected see-through barriers remain omitted from opaque UVTT walls and portals.
The original image is kept for export.

The uncompressed model is 90,583,912 bytes. Its SHA-256 is
`357df3bfc3c842331e07784e1babbf12122a8c450f68aaa07bcd0fc9c4c1300d`.

## Architecture and training

The previous light head receives RGB pixels and 16 decoder feature channels.
The new head also receives the frozen DINOv2 patch features through a learned
384-to-16 channel projection. These features come from the same encoder pass;
there is no second transformer or external service. Patch features are aligned
to image cell centers before padded edges are discarded.

The head contains 24,185 parameters, an increase of 9,616. When upgrading v1,
the existing head weights are copied and the new channels initially contribute
zero. Training updates the head only. All 313 segmentation tensors remain
identical. The new format is `auto-wall-dino-lights-v2`; the loader continues to
accept all existing formats, using strict local safetensors loading.

Trial 18 starts from the bundled v1 light checkpoint and uses the unchanged
trial-17 dataset: four real training samples, 128 procedural training samples,
533 negative crop anchors, two validation layouts, and eight procedural test
seeds. It uses 4,800 steps, four 384 px crops per batch (two real, two procedural),
per-crop focal-loss normalization, AdamW with initial learning rate 0.0002,
cosine decay, seed 73113, and mixed precision on the RTX 3080.

Only the stone camp and reviewed Dyson 14 regions select checkpoints, using
precision-weighted F0.5 at score 0.35. The selected checkpoint is step 2,250:
12 of 13 stone lights and no validation extras. The final step regresses to
11 matches with one extra. Both files and the failed refinements are retained;
additional training is not assumed to improve a model.

Two follow-up trials both start from that selected step, with fresh AdamW
optimizers, initial learning rate 0.00005, and 2,400 steps. Trial 19 keeps the
384 px crops; trial 20 changes only crop size to 640 px, matching the largest
inference context window. Both retain the same data, crop mixture, augmentation,
seed, loss, and selection rule. The larger crop changes the amount and scale
of scene content as well as transformer context; this is not an isolated test
of attention alone.

| Trial | Selected step | Stone matches | Validation extras | Decision |
| --- | ---: | ---: | ---: | --- |
| 18, direct DINO context | 2,250 | 12/13 | 0 | Published as optional v2 |
| 19, lower-rate refinement | 150 | 10/13 | 0 | Retained locally; regresses |
| 20, larger-window refinement | 300 | 11/13 | 0 | Retained locally; regresses |

Neither refinement exceeds its starting checkpoint. The final trial-19 model
still has 10/13 stone matches; trial 20 ends with 10/13 and one extra. Their
best and final weights, histories, and exact training source snapshots are
preserved. They have not replaced the selected checkpoint.

See [the machine-readable trial record](../models/SEMANTIC_LIGHT_TRIALS_18_20.json)
for exact checkpoint and dataset hashes, source snapshots, training recipes,
validation histories summarized by run, and fixed-checkpoint comparisons.
Private map artwork is not included in this repository.

### Trial 21: keep the original local light weights fixed

This experiment starts again from v1 and trains only the 9,616 added context
parameters with `--semantic-context --context-only`. It uses the same data,
seed, crop mixture, learning rate, loss, and 4,800-step schedule as trial 18.
All 313 segmentation tensors and eight original light-weight parts remain
bit-for-bit identical. The first convolution's fixed columns have zeroed
gradients and no weight decay; its new columns also receive no decay. This
optimizer detail means it is not a pure one-factor comparison with trial 18.

The validation winner occurs at step 150: 13/13 stone sources and no extras.
That early checkpoint has learned too little about cool fixtures. The final
checkpoint learns more of them but drops to 11/13 stone sources. Preserving
local weights does not preserve outputs after the added context learns.

| Reviewed region | Bundled v2 | Trial 21, step 150 | Trial 21, step 4,800 |
| --- | --- | --- | --- |
| Wooden camp, training | 9/9 + 0 | 9/9 + 0 | 9/9 + 0 |
| Stone camp, validation | 12/13 + 0 | 13/13 + 0 | 11/13 + 0 |
| Temple, training | 30/30 + 0 | 30/30 + 0 | 30/30 + 0 |
| Ship, partial training | 38/39 + 1 | 29/39 + 9 | 35/39 + 1 |
| Astral, partial observed test | 20/20 + 2 | 16/20 + 5 | 17/20 + 0 |
| New camp, observed test | 6/6 + 2 | 6/6 + 6 | 6/6 + 2 |
| Eight procedural test seeds | 63/63 + 8 | 49/63 + 34 | 61/63 + 8 |

Neither checkpoint replaces v2. The trial-21 validation set has warm sources
and reviewed ink regions but no marked cool fixtures; its winning score is
insufficient evidence of broad improvement. Test layouts remain excluded from
training and checkpoint selection. More diverse, separately annotated validation
layouts are needed before further selection claims.

The run took 361.31 seconds and peaked at 607,963,136 allocated CUDA bytes.
Both checkpoints, the history, and verified source snapshots are retained locally.
The [trial record](../models/CONTEXT_ONLY_LIGHT_TRIAL_21.json) includes the recipe,
hashes, validation history, fixed-checkpoint comparisons, and evaluation limits.
No runtime model or geometry behavior changed in this experiment.

## Fixed-checkpoint results

Entries show matched sources / marked sources, followed by false extras. These
are raw light-center results before the normal export pipeline rejects doorway
highlights. The threshold is 0.35 for both models.

| Reviewed region | V1 | Semantic v2, trial 18 step 2,250 |
| --- | --- | --- |
| Wooden camp, training | 9/9 + 0 | 9/9 + 0 |
| Stone camp, validation | 12/13 + 0 | 12/13 + 0 |
| Temple, training | 30/30 + 0 | 30/30 + 0 |
| Ship, partial training | 0/39 + 0 | 38/39 + 1 |
| Astral, partial observed test | 16/20 + 8 | 20/20 + 2 |
| New camp, observed test | 6/6 + 1 | 6/6 + 2 |
| Eight procedural test seeds | 42/63 + 76 | 63/63 + 8 |

Reviewed Dyson 13 and 14 regions remain free of light detections. The observed
Dyson 15 map gains one false light near a stream/bridge detail. The new camp's two
false detections are flower patches. These regressions prevent treating this
as a universally more precise model.

The ship labels cover only 11.18% of its pixels. V2 produces 63 raw peaks across
the whole ship, but only 39 fall in reviewed regions: 38 matches and one extra.
Astral has 28 total peaks; six lie outside the reviewed regions. Full-map
precision is therefore unknown. Ship results measure partial training accuracy,
and procedural test results measure a generator task, not general artwork.

All labels are agent-reviewed. The Astral and new-camp diagnostic labels were
created after earlier predictions had been inspected. Those layouts remain out
of training and checkpoint selection, but are already observed test maps, not
fresh independent evaluation. Repeated use of the same small validation set also
limits conclusions about new styles.

A separate operating-threshold check tested 0.05 through 0.95 in increments of
0.05, maximizing validation F0.5 and selecting the highest threshold in a tie.
It selected 0.35 for both v1 and the chosen v2 checkpoint. The decision was saved
before evaluating diagnostic/test images. Raising the threshold is not a free
precision improvement: it drops small real sources. Scores are not calibrated
probabilities of correctness.

## Application checks and performance

A paired check on ten maps used the normal detection pipeline with OpenCV outline
geometry and serialized UVTT exports. Every wall, door, and transparent
segment was identical between v1 and v2; both camp class/confidence arrays were
also bit-for-bit identical. Every export preserved its original PNG bytes.
Scoring the serialized light positions reproduced the reviewed-region figures
above. Transparent barriers were omitted from the opaque export as intended.

Both camps still export all four annotated doors and pass all 36 checked sight
lines with the existing optional jamb recovery enabled. These are sparse
geometry checks, not proof that every map boundary is correct.

With CUDA, four CPU threads, a 1536 px working edge, and five warm measurements
per map, v2's median detection times ranged from 0.224 to 0.337 seconds; v1 ranged
from 0.226 to 0.365 seconds in the same session. Detection includes resize,
inference, and geometry, but excludes model loading, image decode, preview
painting, and export writing. The small timing differences vary with machine
activity and do not establish a speedup. No training was running during this
benchmark. The added head reuses DINO features without a second encoder pass.

Actual Qt worker checks loaded v2, changed its light score from 0.35 to 0.60 and
back, verified that export was invalidated and recomputed, and exported both
camps with the expected four doors and 9/12 lights. Captured previews were
inspected. The first detection in a fresh process took 20.45 seconds; subsequent
checked detections took 0.45–0.62 seconds. Startup remains a material cost.
The later [isolated-runtime comparison](ML_RUNTIME.md) reduces median first Qt
detection from 16.51 to 5.45 seconds in three paired runs, with identical outputs.

Training took 297.90, 172.77, and 382.00 seconds for trials 18–20. Peak allocated
CUDA tensors were 608,052,224, 608,091,136, and 1,348,270,080 bytes respectively.
These are allocator figures, excluding reserved memory, CUDA context, and other
processes. Concurrent desktop work and diagnostic evaluation varied during
training; the run durations are not a controlled performance comparison.

Compatibility checks captured v1 predictions before the implementation change
and compared them afterward on CPU and CUDA at 35×43, 384×384, and 385×511 pixels.
All 18 class/confidence/heatmap arrays were exactly equal. Tests also cover
semantic model save/load, one-pixel and odd image sizes, patch alignment,
zero-influence upgrades, actual optimizer updates with frozen segmentation,
one backbone pass per tile, and cancellation between tiles.
The final full suite passed 157 tests with three dependency deprecation warnings;
Ruff and Python compilation checks passed. The existing native centerline tests
remain in that suite; the ten-map timing run used outline geometry.

Live Foundry import, other GPUs/operating systems, and release packaging were not
run. The check uses existing UVTT behavior, not a new native Foundry wall format.

## Public artwork expansion: trial 22

Trial 22 adds [ten reviewed public maps](OPEN_LIGHT_DATA.md) with 59 marked
sources. Labels and splits were fixed before predicting on those images. Puny
maps sharing the same four-room structure remain together in training. The
expanded validation set contains five maps and 21 sources, including one cool
portal. Sprite families overlap across splits, so this tests new layouts within
those families rather than wholly unfamiliar art.

Starting from bundled semantic v2, the run trained the light head for 4,800 steps
at learning rate 0.0001, with 384 px crops, two real and two procedural crops per
batch, sample-normalized loss, and seed 73113. The original segmentation remained
frozen. Validation F0.5 at the fixed 0.35 threshold selected step 2,850; the earliest
checkpoint wins a tie. Test images did not participate in that selection.

| Reviewed region | Bundled v2 | Trial 22, step 2,850 |
| --- | --- | --- |
| Expanded validation, all five maps | 16/21 + 5 | 17/21 + 1 |
| Stone camp, validation subset | 12/13 + 0 | 11/13 + 1 |
| Castle 3, validation subset | 4/7 + 0 | 6/7 + 0 |
| Puny 5, validation subset | 0/1 + 5 | 0/1 + 0 |
| Castle 4, first-use test | 2/2 + 0 | 2/2 + 0 |
| Puny 3, first-use fire-jet test | 0/7 + 16 | 1/7 + 3 |
| New camp, previously observed test | 6/6 + 2 | 6/6 + 0 |
| Astral, partial previously observed test | 20/20 + 2 | 18/20 + 1 |
| Eight earlier procedural test seeds | 63/63 + 8 | 62/63 + 4 |
| Ship, partial training | 38/39 + 1 | 38/39 + 1 |

Entries are matches / annotated sources, followed by unmatched predictions.
An unmatched prediction can be a displaced or duplicated point on a real emitter.
For example, Puny 5's five baseline extras are three gate-post detections and two
points on the portal rim. Trial 22 removes those points but still misses its
annotated center. It also removes the two flower detections on the newer camp.
Its two new Astral misses are the side fixtures in the upper-left circular room.
The sparse masks do not establish full-map precision.

All 42 sources in the five new training maps are matched, with one extra. Both
training portals are centered correctly, but the blue portal in a different
validation layout is still missed. A diagnostic used 256, 384, 640, and 960 px
inference tiles on training/validation portals only. Trial 22's maximum score
within five pixels of the held-out portal center remained between 0.037 and
0.085, below the 0.35 threshold. Larger tiles alone did not repair this failure;
whole-image context also lost the training portals. No runtime tiling was changed.

The last step scored 17/21 + 2 on validation and was not selected. Training took
314.98 seconds, with 608,091,136 bytes of peak allocated CUDA tensors. This is not
a controlled speed comparison. Independent safetensors inspection found all 323
tensors finite in both checkpoints and all 313 segmentation tensors bit-identical
to v2. Only the ten light-head tensors changed.

The checkpoint remains a research result. Better aggregate validation precision
does not justify the lost stone/Astral sources, and the reserved fire jets are
still mostly missed. Bundled v2 and the application's model choice stay unchanged.
Both raw checkpoints, exact training source snapshots, all 32 validation records,
and per-image evaluations are retained locally. The
[public trial record](../models/OPEN_LIGHT_DATA_TRIAL_22.json) records hashes,
counts, the complete validation history, context probes, and evaluation limits.

The full test suite passed 166 tests with three dependency deprecation warnings;
Ruff, Python compilation, and diff whitespace checks passed. Live Foundry import,
other GPUs/operating systems, and release packaging were not run for this data and
training-tool change. No runtime inference or export implementation changed.

## Training context augmentation: trial 23

Trial 23 changes only training augmentation relative to trial 22. It adapts
[CutMix](https://arxiv.org/abs/1905.04899) by exchanging rectangular image patches
with their point heatmaps and review masks. A separate RNG leaves the original
crop streams unchanged. Mixing has probability 0.5; a 32 px band around cuts is
ignored on these 384 px crops, and crops with no surviving supervision are
restored. This is our dense-target adaptation, not the paper's image-level loss.
It adds no inference work and leaves the model format unchanged.

The initial model, v4 dataset, seed, batch mixture, optimizer, learning rate,
loss, 4,800-step schedule, and fixed 0.35 threshold match trial 22. The first
128 batches contained 55 mixing events. They retained 1,409 of the original
1,638 supervised centers and 3,724,805 of 4,295,328 supervised target pixels.
Input/target alignment, ignored regions, sparse-crop restoration, deterministic
sampling, and disabled behavior were checked before training. The cut margin
does not guarantee that large objects remain whole.

Validation F0.5 selected step 600, with 16/21 sources and no extras (0.94118).
This exceeds trial 22's aggregate selection score (0.91398), but loses small
stone-camp lights. The final checkpoint was evaluated separately as a diagnostic;
it did not replace the validation winner. Test layouts were already observed
in previous trials and remained outside training and checkpoint selection.

| Reviewed region | Bundled v2 | Trial 22 best | Trial 23 best | Trial 23 last |
| --- | --- | --- | --- | --- |
| Expanded validation, five maps | 16/21 + 5 | 17/21 + 1 | 16/21 + 0 | 16/21 + 1 |
| Stone camp, validation subset | 12/13 + 0 | 11/13 + 1 | 10/13 + 0 | 10/13 + 1 |
| Castle 3, validation subset | 4/7 + 0 | 6/7 + 0 | 6/7 + 0 | 6/7 + 0 |
| Puny 5, validation portal | 0/1 + 5 | 0/1 + 0 | 0/1 + 0 | 0/1 + 0 |
| Castle 4, observed test | 2/2 + 0 | 2/2 + 0 | 2/2 + 0 | 2/2 + 0 |
| Puny 3, observed fire-jet test | 0/7 + 16 | 1/7 + 3 | 0/7 + 2 | 1/7 + 3 |
| New camp, observed test | 6/6 + 2 | 6/6 + 0 | 6/6 + 0 | 6/6 + 0 |
| Astral, partial observed test | 20/20 + 2 | 18/20 + 1 | 18/20 + 1 | 18/20 + 1 |
| Eight procedural test seeds | 63/63 + 8 | 62/63 + 4 | 63/63 + 2 | 62/63 + 2 |
| Ship, partial training | 38/39 + 1 | 38/39 + 1 | 37/39 + 0 | 38/39 + 1 |

Entries are matches / marked sources, followed by unmatched predictions.
Partial reviews do not establish whole-map precision. The selected model's
three stone misses are one northwestern watchtower light and both small lamps
beside the eastern paddock. Its two Puny 3 extras lie on a bone and a gate.
The two missed Astral side fixtures persist. These regressions outweigh the
improved aggregate precision, so bundled v2 remains the selected research model.

A separate diagnostic examines the blue training and validation portals. Their
central 32 by 32 pixels are exactly equal. Their 64 by 64 patches differ in 7.81%
of pixels, with mean channel difference only 0.443 on the 0–255 scale. Swapping
those 64 px patches barely changes either model's response. Swapping wider
patches, which include surrounding architecture and nearby sources, has a
larger effect while preserving the central pixels.

For trial 22, inserting the validation portal's 192 px surroundings into the
training map lowers its center score from 0.447 to 0.278, losing the detection.
Inserting the training surroundings into validation raises 0.049 to 0.277,
still below 0.35. Trial 23's final checkpoint retains the altered training
portal (0.489 to 0.373), but the reverse change reaches only 0.251 from 0.052.
The early selected checkpoint misses the blue training portal too. This
confirms sensitivity to surrounding pixels on these two images; it does not
isolate every mechanism or establish broad context robustness. No diagnostic
image, modified label, or runtime tiling change entered training or selection.

Training took 391.86 seconds and peaked at 608,091,136 allocated CUDA bytes.
Part of the CPU test suite ran concurrently, so this is not a controlled training
speed comparison. Independent safetensors inspection found 323 finite tensors
in each checkpoint, all 313 segmentation tensors identical to v2, and only the
ten light-head tensors changed. Both raw checkpoints, exact training sources,
all 32 validation records, per-image overlays, and diagnostic sources are kept
locally. The [trial record](../models/CUTMIX_LIGHT_TRIAL_23.json) contains the
recipe, hashes, counts, history, context probes, and limitations.

The full suite passed 172 tests with three dependency deprecation warnings;
Ruff, compilation, and diff whitespace checks passed. Actual CPU optimization
with CutMix is covered, including frozen segmentation and loadable checkpoints.
Runtime inference, GUI, export, and bundled weights are unchanged. Live Foundry
import, other GPUs/operating systems, and release packaging were not run for
this training-only change.

## Attribution

The inherited DINOv2-small weights are by Meta Platforms, Inc. and affiliates,
under Apache 2.0. Preserve [the license](../models/licenses/DINOv2.txt) and
[attribution](../models/licenses/README.md). The
[base model card](../models/DINO_MODEL_CARD.md) records its upstream revision,
weight hash, segmentation training, and structural limitations. Cartography by
Dyson Logos is used under the recorded commercial-use map terms; source art is
not redistributed.
Trial 22 also uses sample maps by Gary Shaw, Shade, and Ferrin; their source
pages, licenses, and preparation changes are recorded in [the dataset notes](OPEN_LIGHT_DATA.md).
