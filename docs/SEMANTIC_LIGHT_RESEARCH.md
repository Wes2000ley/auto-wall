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

Neither checkpoint replaces v2. The current validation set has warm sources
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

## Attribution

The inherited DINOv2-small weights are by Meta Platforms, Inc. and affiliates,
under Apache 2.0. Preserve [the license](../models/licenses/DINOv2.txt) and
[attribution](../models/licenses/README.md). The
[base model card](../models/DINO_MODEL_CARD.md) records its upstream revision,
weight hash, segmentation training, and structural limitations. Cartography by
Dyson Logos is used under the recorded commercial-use map terms; source art is
not redistributed.
