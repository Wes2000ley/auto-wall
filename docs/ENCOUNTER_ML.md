# Encounter structure ML experiments

The six-class model is **experimental and not promoted**. The installed model
remains unchanged. Early candidates improved some marked doors and walls but
missed glass, produced false geometry, or regressed on older artwork. Passing
the software tests does not establish model accuracy.

## Labels and split discipline

The encounter inventory contains 31 canonical layouts, each with a PNG and a
WebP derivative. Whole-layout membership was fixed before training: 25 train,
three validation and three test. Recolored variants stay with their originals.
Seven older real maps retain their previous splits, producing 29 training,
five validation and four test samples in the expanded dataset.

The initial delegated annotation previews were rejected after coordinate and
semantic errors were found. They were draft training annotations, not model
predictions. No candidate described here trained on that rejected pass.

The admitted [seed annotations](../benchmarks/data/encounters/partial-source-v2/README.md)
are partial source-assisted weak supervision. Artist cap/panel interiors,
selected material traces and explicit negative regions are supervised; uncertain
pixels stay 255. Source markings are useful seed labels, not an independent
measure of physical wall boundaries. The primary checked originals, selected
components and close-ups, removed ambiguous outside cave borders and removed
snow rectangles that overlapped broken fence fragments.

Only one new training layout has reviewed glazing positives. The other reviewed
glazing layout is validation-only. This coverage is insufficient to claim a
general window detector. Sparse fence strokes similarly do not define complete
rail topology. New encounter light positives are not claimed; possible emitters
are conservatively excluded from negative regions, and older light labels are
replayed.

`prepare_encounter_training.py` requires a separate review record matching the
source image bytes, annotation bytes and raster bytes. It writes a fresh dataset
outside the originals. It never uses model predictions to create labels. Review
admission is limited to the scope recorded in `review.json`.

## Training and measurement

Five-class checkpoints retain their class order and default behavior. The
opt-in `--classes windows` adds ID 5. Expanding a segmentation checkpoint copies
channels 0–4 exactly before training. Changed segmentation features require a
fresh light-center head; the old trained head must not be grafted onto them.

Window-mode crops choose a target class first, then a map containing that class.
Optional manifest `sampling_weight` values preserve older map exposure when a
larger collection of weak source-marked maps is added. The default is uniform.
Color, brightness, spatial lighting and tint augmentation affect images only;
they do not move or recolor labels. These augmentations are stress preparation,
not evidence of robustness to an independently drawn art style.

`--selection precision` selects by per-map foreground F0.5 at a 0.9 score cutoff.
It penalizes false-only classes and hallucinations on reviewed empty maps.
Window misses remain in the score. A cutoff is not a calibrated accuracy
probability. Final promotion still requires per-class and geometry review;
an average score can conceal a serious failure in one class.

```powershell
python scripts/train_ml.py DATASET/manifest.json OUTPUT/new-trial `
  --classes windows --architecture dino `
  --init-model models/battlemap-dino-v1.safetensors --device cuda `
  --steps 2400 --batch-size 4 --crop-size 384 --learning-rate .00008 `
  --backbone-layers 1 --context-dilation 4 --selection precision `
  --validate-every 400 --threads 4

python benchmarks/evaluate_structures.py OUTPUT/new-trial/best.safetensors `
  DATASET/manifest.json OUTPUT/validation-original `
  --split validation --device cuda --runs 3 --threads 4

python benchmarks/evaluate_structures.py OUTPUT/new-trial/best.safetensors `
  DATASET/manifest.json OUTPUT/validation-recolored `
  --split validation --device cuda --runs 3 --threads 4 --appearance recolored
```

Compare the candidate and segmentation baseline against the **same manifest**.
The evaluator records hashes, device, settings, class cutoff, first inference,
three warm runs, geometry time, segmentation counts and optional passage/span
checks. Every prediction preview carries a model-prediction banner; draft labels
have a separate draft-training-label banner. Raw labels and confidence arrays
retain source pixel coordinates.

The common `--threshold` applies to all classes in a benchmark run. Production
window/fence defaults are stricter (0.93/0.97), so record and inspect settings
before equating benchmark geometry to a desktop run. Add
`--geometry-thresholds profile` to keep the runtime's per-class cutoffs (including
any `--settings` overrides) while measuring pixel scores at the common cutoff.
The report records both choices. Recoloring is a paired
same-layout stress test, not an independent style test. Partial labels support
precision claims only within explicitly reviewed pixels; unknown areas cannot
be counted as correct. Incomplete span annotations report recall and leave
unmatched predictions unassessed.

## Observed failures

The 600-step seed trial used three new training layouts plus older replay. It
improved door pixels on the two new validation maps, but the actual geometry
missed all Mothveil doors and glass. Rimebell acquired false wall predictions
on snow and cliff terrain. This trial was rejected for promotion.

The expanded, frozen-backbone 2400-step run used all 31 encounter layouts in
their fixed splits, plus older replay. Its selected checkpoint was step 2000.
On the expanded validation labels at a common 0.9 cutoff:

| Check | Baseline | Expanded candidate |
| --- | ---: | ---: |
| Antler reviewed-background false structural pixels | 57 | 0 |
| Mothveil reviewed-background false structural pixels | 7895 | 1366 |
| Mothveil extracted doors | 0 | 1 |
| Mothveil reviewed window-pixel recall | 0 | 0.0421 |
| Mothveil extracted windows | 0 | 0 |
| Rimebell extracted doors | 0 | 2 |
| Rimebell fence movement crossing | missed | missed |
| Older Dyson-14 wall-pixel recall | 0.7971 | 0.4990 |
| Older Dyson-14 door-pixel precision | 0.7804 | 0.4887 |

The candidate remains rejected. Finding more new marked doors does not excuse
missing glass, failing fence movement or losing older-map performance. Its
validation inference medians ranged from 0.217 to 0.390 seconds on the local
RTX 3080 while another workload used the GPU; these are local observations, not
an isolated hardware comparison or an end-to-end export benchmark.

The replay-weight-5 trial adapted one DINO block at one tenth of the decoder
learning rate. Its best checkpoint was step 400 of 2400. It raised older Stone
wall precision to 0.9976, but Mothveil glass recall stayed zero. Rimebell's fence
pixels partly recovered, yet its actual fence crossing still did not block
movement. Dyson-14 wall recall fell to 0.3187. It was rejected.

The next trial added 32 explicitly constructed synthetic rooms with random
stone/wood/glass colors, doors, railings and negative props. They are synthetic
training examples, not corrected real-map annotations. No validation/test map
provided textures or geometry. An early generator version had incorrect wall-band
geometry; it was fixed and all 32 corrected examples were checked before any
training. The unused failed version remains separate in the workspace.

This synthetic trial started from the expanded candidate, used older-map replay
weight 10, and selected step 400 of 2400. On original Mothveil at 0.9, reviewed
glass-pixel precision/recall reached 0.9984/0.9129; the earlier expanded trial's
recall was 0.0421. This is one partially reviewed validation layout, not a broad
accuracy claim. Mothveil still had no extracted door. Rimebell had three extracted
doors but no fence. Dyson-14 wall recall was 0.4357 and door precision 0.6370.
The model remains rejected despite the glass improvement.

A fifth trial continued from the synthetic trial's best checkpoint on real maps
only, preserving the same 38 real samples, files, frozen splits and training
weights. It used a fresh optimizer at learning rate 0.00001 for 1200 steps.
Mothveil door-pixel recall improved from 0.4929 to 0.7210, but glass recall fell
to 0.6684. Dyson-14 wall recall/door precision deteriorated to 0.3389/0.3918.
Fence recall on Rimebell remained zero. This trial was also rejected. Changes to
initialization, curriculum and learning rate prevent attributing these outcomes
to one factor alone.

Full local reports, failed checkpoints, source snapshots and uncompressed BMP
prediction previews remain under `outputs/Encounter ML review` and
`work/encounter-ml` in the working workspace. The compact committed
[trial record](ENCOUNTER_ML_RESULTS.json) records source/checkpoint/report hashes,
observed metrics and promotion decisions. All five runs are segmentation trials;
none received a newly trained production light-center head. The four test
layouts remain reserved until candidate selection is complete.

## Geometry review and measurement corrections

The original Mothveil window movement ray ended inside the outlined window
assembly rather than crossing both faces. It was corrected after inspecting
native source pixels. Its span now covers the full framed opening between real
jambs. This changed passage/instance annotations only: the training raster SHA-256
remains `77b0cc7edd90cd0ecaedf4f1281c27b1248593119b9f94cf87d26b489c3acbdc`.
Historical training datasets were preserved; none of these trials selected
checkpoints using passage rays. Earlier results using the short ray should not be
read as a reliable full-window movement check.

Additional manual floor, snow/cliff and window-frame passage regressions are in
`benchmarks/data/encounters/negative-ray-checks-v1.json`. They were added after
reviewing validation failures, are not an independent test set, and do not create
training labels. Reports keep them separate from the original partial checks.
A blocked closed-door sight ray alone does not prove that a door was detected:
an incorrectly opaque wall also blocks that ray. Door-span matches must be checked.

Geometry tests now cover deep bays, tapered/partly missing panes, frame gaps,
long and cornered glazing, diagonal windows and negative props. Compact windows
use their actual extent rather than a shortened skeleton. Both exported faces
must reach supporting jambs; long runs retain that support when split for editing.
A nearby building-axis fit is considered for a tilted partial prediction only
when the fitted axis fails jamb support. Confidence and semantic class checks
remain required. These improvements cannot recover a window or fence the model
does not recognize.

At the common 0.90 cutoff, the synthetic candidate now produces the reviewed
Mothveil window in both original and recolored images, with the sight and full
movement/frame checks passing. At the actual runtime window cutoff of 0.93,
only the original image retains that window. The recolored runtime case still
fails movement, Mothveil still has zero extracted doors, and Rimebell still has
no movement-blocking fence. These remain promotion blockers. The final cached
prediction recheck matched all 40 geometry/span/passage cases after export and
preview integration. Updated previews show exported window faces, rather than
the candidate editing axis.

## Geometry, export and promotion

The opt-in windows profile extracts supported glazing runs separately from
opaque walls. Confident, long/thin fence components become movement-only
centerlines. Compact regions are rejected. Thinning and tracing cooperate with
cancellation. Legacy transparent-barrier predictions stay review-only and are
not exported as movement barriers.

Native Foundry Scene JSON preserves movement-only fences. ML windows use
movement blocking with Foundry proximity sight/light/sound, with explicit
thresholds; these have finite proximity semantics, not unrestricted vision.
UVTT cannot represent every Foundry barrier setting. Its open portals preserve
the installed importer's window convention, and private editor fields are
removed from public UVTT. The native scene fixture was validated against the
installed Foundry 14.369 data classes without starting a server or modifying
a world. This validates schema compatibility, not model accuracy or live-world
visibility behavior.

Before promotion, inspect actual extracted geometry across the held-out maps,
verify doors attach to real jambs, check movement and sight through real glass
and fences, reject walls on floor/props/terrain, compare older style performance
and preserve lights with a newly trained head when required. Keep failed trial
weights, reports and provenance separate from the installed model. Do not lower
confidence or geometric checks merely to make a preview look complete.

## CUDA transfer queue and replay consistency

CUDA inference now queues two tiles before downloading results, while retaining
single-image model calls, the existing tile order, halos and thresholds. CPU
execution completes each tile. Cancellation stops result delivery, including
already queued tiles, and the complete probability/light output of each tile
is checked for nonfinite values before use.

`ML_CUDA_PIPELINE_BENCHMARK.json` records five alternating warm runs per method
on each of the five validation maps with the installed lights-v2 checkpoint on
an RTX 3080. All class IDs, winning probabilities, light heatmaps and extracted
exports matched the archived runtime exactly. Median inference times fell by
1.8–4.6%, with essentially unchanged peak allocated GPU memory. Another user
GPU workload remained active; these are local measurements, not isolated
hardware benchmarks or whole-application latency claims. A faster two-image
batch experiment was rejected because floating-point changes altered exported
walls on three maps.

The trainer optionally accepts `--teacher-model`, `--teacher-groups` and
`--teacher-weight` (default 0.5 when enabled). The frozen five-class reference
regularizes predictions only on the named training layout groups. Explicit
labels must agree with its argmax; ignored pixels require reference confidence
of at least 0.9. Window labels are never overridden by the legacy reference.
Temperature-2 KL divergence includes all student classes, with zero reference
probability assigned to windows. This is a regularizer, not corrected training
truth or a guarantee of unchanged detections. Validation and test groups cannot
be selected for replay. Initialization uses a separate model object so training
does not change the inference loader's cached model.

The complete local test suite passed: `python -B -m pytest -q tests` reported
326 passed and three dependency deprecation warnings in 49.69 seconds. CUDA
was available, so the new CUDA checks ran rather than being skipped.
