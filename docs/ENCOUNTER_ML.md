# Encounter structure ML experiments

The six-class model is **experimental and not promoted**. The installed model
remains unchanged. Early candidates improved some marked doors and walls but
missed glass, produced false geometry, or regressed on older artwork. Passing
the software tests does not establish model accuracy.

## Labels and split discipline

The initial encounter inventory contains 31 canonical layouts, each with a PNG and a
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

## Follow-up trials and additional reviewed data

The [follow-up trial record](ENCOUNTER_ML_FOLLOWUP_RESULTS.json) preserves
checkpoint, dataset, source and evaluation hashes for v6–v15. None is promoted.
Pixel precision below refers only to reviewed pixels. Runtime results use the
unchanged per-class profile thresholds: wall/door 0.90, window 0.93, fence 0.97.

| Trial | Change | Observed promotion blocker |
| --- | --- | --- |
| v6 | Freeze legacy tensors; train a new window row | Zero held-out glass recall; unchanged weights do not preserve softmax probabilities when a class is added. |
| v7 | Train a residual decoder over frozen features | Learns glass on the training map but misses held-out glass. This experimental wrapper is not runtime compatible. |
| v8 | Vary procedural frame examples | Held-out glass recall remains zero; older-map precision also fails. |
| v9 | Frozen legacy-reference replay | No extracted Mothveil door/window or Rimebell fence in either appearance. |
| v10 | Add 16 constructed replacement windows in training-only building bays | Better glass pixels do not produce a matched runtime window or door in Mothveil. Rimebell still has no exported fence. |
| v11 | Add four reviewed Frostwake rail/post strokes and increase that map's sampling | Some held-out fence pixels return. Original Rimebell exports no fence; recolored fragments fail the fence movement ray. Mothveil has no matched door/window. |
| v12 | Permit aspect-ratio crop augmentation up to 2 | No matched Mothveil door/window; recolored Rimebell fails both negative cliff rays. A new recolored landscape acquires 107 false opaque wall segments. |
| v13 | Add reviewed Sunspoke landscape negatives, retaining square crops | All three annotated Rimebell doors match in both appearances and Cinderglass has zero exported structures in both. Mothveil still has no extracted door/window; Rimebell still has no fence. |
| v14 | Add a second real training railing layout and increase its sampling | Learns some Gullthorn wood pixels but fails its full-span movement check. Rimebell still exports no fence, and original door matches fall to 2/3. Mothveil has no matched door/window. |
| v15 | Add source-reviewed floor/crate negatives on the same training layout | Fewer false fence pixels in the new negative patches, but the training railing movement check still fails. Validation still has no Rimebell fence or matched Mothveil door/window. |

Trials v6–v8 failed pixel checks; their actual exports and recolors were not
evaluated. Trials v9–v11 were evaluated on the same five validation maps in both
appearances. Trial v12 selected checkpoints on those five maps, then also ran
on the newly reserved Cinderglass validation map. These iterative validation
experiments are not an independent final test. Trial v13 also selects on
Cinderglass, making its selection average cover six maps rather than five;
compare shared-map metrics separately. The four final-test layouts
remain reserved. None of these candidates has a new production light-center
head, and the installed five-class light model remains unchanged.

Trials v14 and v15 retain those six validation layouts. Their selected scores
are 0.659960 and 0.661577, both at step 1,200. Those aggregate pixel scores do
not establish usable geometry. Both retain zero exported structures on
Cinderglass in both appearances, but still fail the class-specific checks above.

The [additional Frostwake annotations](../benchmarks/data/encounters/additional-rails-v1/README.md)
add exactly 613 previously unknown pixels as fence labels. All existing
supervision is retained. Increasing its sampling weight from 1 to 10 affects
every represented target class on that map, not just fences. For fence-target
crops its share rises from 1/13 to 10/22, including older and procedural examples.

The optional `--crop-aspect-ratio` parameter accepts values from 1 to 4. Its
default of 1 preserves the old square crops and random sequence. Larger values
sample a log-uniform width/height ratio and resize the same image/label rectangle,
using nearest-neighbor interpolation for labels. Image boundaries and integer
rounding can reduce the realized ratio. Trial v12 is a failed experiment with
this option, not evidence that the augmentation improves production accuracy.

Two newly found layouts bring the canonical encounter inventory to 33. Their
[reviewed landscape negatives](../benchmarks/data/encounters/new-landscapes-v1/README.md)
keep Sunspoke in training and Cinderglass in validation, fixed before prediction
evaluation. The earlier 31 splits are unchanged. Raster supervision remains
partial: possible emitters are unknown, and two flawed earlier Cinderglass
drafts were rejected before admission or training. The expanded manifest
retains all 86 previous samples and adds these two, giving 78 training,
six validation and four test samples, including controlled examples and replay.

These two landscapes have no visible architectural structures anywhere. Their
explicit `structure_free: true` annotation checks every exported wall, door,
window face and movement-only segment across the whole image, including unknown
segmentation pixels. It excludes lights and review-only barrier candidates.
On recolored Cinderglass, both sampled passage rays passed despite the installed
model exporting 23 false walls and v12 exporting 107. Re-extraction of cached
predictions reproduced those counts; v11 exports zero structures in both
appearances. This is a whole-map export check, not new pixel supervision.

Unshipped door-refinement probes inspect the actual failure before changing
geometry checks. Connected weak-pixel growth still fails Mothveil extraction.
Image-guided color growth recovers its original door on v9–v11, but fails the
recolor. An added Stone gate is visibly real yet exceeds the existing endpoint
tolerance, and accepted spans can move. Those probes need further work;
neither their thresholds nor the reference annotations were relaxed to pass.

Later unshipped probes recover the Mothveil door in both appearances on the
installed segmentation model and two candidates. They still fail precision:
one creates a door on a plain gray wall indentation, and real Stone gates can
extend through their solid jambs. Comparing a seed with the median local wall
color does not remove that false door. Comparing against a local wall-color
subset removes it for v13, but it remains on recolored v9. Re-running existing
weak-jamb recovery fixes some attachments, leaving other endpoint failures.
All of these probes remain outside production. The first expanded probe also
had a harness error: it called public UVTT conversion for movement-only walls.
The corrected diagnostic uses internal preview conversion for legacy door
metrics and assesses movement separately; the initial failure log is retained.

The [Gullthorn railing extension](../benchmarks/data/encounters/additional-gullthorn-v1/README.md)
adds eleven narrow wood interiors and two clear floor/crate patches. Exactly
1,622 unknown pixels become fence and 884 become background. Earlier labels
remain unchanged. Its training sampling weight changes from 1 to 10 for its
represented background and fence targets. Fence-target shares become Frostwake
10/32, Gullthorn 10/32, the legacy wood camp 10/32 and procedural examples 2/32.
Source-coordinate crops and raster overlays were reviewed before admission;
these remain partial source-assisted labels, not complete independently
corrected truth. Trees, ladders, lanterns and props are not positive barriers.

Inspection of v14 motivated [additional boundary negatives](../benchmarks/data/encounters/additional-gullthorn-v2/README.md).
Three polygons add 2,697 background pixels on clear platform floor and two
crate interiors. The source artwork determines these labels; predictions only
identified areas needing inspection. All earlier supervised pixels, images,
sampling weights and splits stay unchanged. The v15 trial changes only this
training mask, using the same initializer, optimizer, seed and crop settings.

Training-fit comparisons use identical v2 labels for v13, v14 and v15. At the
working resolution, 2,581 pixels belong to the added negatives. False fence
pixels there at the unchanged 0.97 geometry threshold fall from 1 to 0 in the
original appearance and from 15 to 2 after recoloring, comparing v14 with v15.
Reviewed fence recall also falls: 59.7% to 54.4% originally and 82.0% to 80.1%
after recoloring. All three models fail the full railing movement check.
These are training-layout diagnostics on partial supervision, not held-out
accuracy. They do not justify promotion.

The unchanged component filter also explains part of the loss. In v15 the
front-left Gullthorn component has area 748 originally and 1,551 after
recoloring, with oriented aspect ratios about 1.45 and 2.24. Those broad
predictions fail the existing compact-region/width checks. Some visible wood
faces are real barriers, but this does not certify every rejected pixel as
fence. Rimebell has a separate confidence problem: only 70 and 161 pixels
reach the 0.97 fence cutoff, and no component passes filtering. Removing the
filter would not establish accurate delineation across these cases.

Future benchmark runs save exact working-size class IDs and winning confidence
as separate `*-working-labels.bmp` and `*-working-confidence.npy` files. The
source-size preview rasters remain available for display. A separate
`*-detection.json` stores exact source-coordinate geometry, including window
faces, movement-only barriers and emitted lights. Each sample report records
artifact paths, sizes and hashes. The prediction cache has no light-center
heatmap; it cannot reproduce that head's output by itself.

This corrects an audit limitation: source-size preview caches on downscaled maps
must not be resized back and described as exact working predictions. Original
inference and geometry metrics were computed before display resizing and remain
valid. Earlier six-map cached probes asserted that no input was downscaled.
New real-map replay on all six Gullthorn fit cases reproduces every geometry
coordinate and passage check exactly. The four final-test layouts remain unused.

After the augmentation, data-integrity and whole-map checks, the full CUDA
development environment ran `python -B -m pytest -q tests`: **345 passed**,
three dependency warnings, 48.25 seconds. The portable runtime alone could not
collect two legacy image-utils tests because it lacks `requests`; the complete
development environment supplied that dependency. Software-test success does
not establish model quality.

After adding both Gullthorn data-integrity checks and exact prediction replay,
the workspace's `work/ml-training-venv/Scripts/python.exe` ran with the
`outputs/auto-wall` repository as its working directory.
`-B -m pytest -q tests/test_encounter_training.py tests/test_structure_metrics.py`
passed all 46 tests in 3.69 seconds. The full CUDA suite, `-B -m pytest -q tests`,
passed **348 tests**, with three dependency warnings, in 40.45 seconds. The
machine-readable record includes the full executable path and working directory.
Ruff passed on all three changed Python files. The first
replay fixture failed because it requested an unsupported 48-pixel detector
resolution; the fixture was enlarged to a valid resolution and rerun, preserving
the production resolution check.

The first follow-up commit failed two remote annotation-integrity tests because
Git converted the reviewed JSON line endings. Scoped `.gitattributes` files now
preserve exact annotation bytes, and the staged blob hashes match the reviews.
Corrected commit `e89ac84` passes remote CI: 259 tests and four CUDA-only skips
in 31.66 seconds. Its installed smoke check verified 80 file hashes, preserved
the legacy light head, ran real-map CUDA inference, checked dialog settings and
shutdown, and generated the 11-wall synthetic Foundry fixture. This is functional
validation, not an accuracy or speed benchmark.

The Gullthorn and exact-replay change, commit `65e3079`, also passes the
[targeted Linux CPU workflow](https://github.com/Wes2000ley/auto-wall/actions/runs/40094341112):
262 tests passed and four CUDA-only checks were skipped in 30.99 seconds.
Both installed copies were synchronized with all four model hashes unchanged.
The installed smoke run checked all 90 updated-file hashes, retained the legacy
light head, completed real-map CUDA inference, round-tripped dialog settings,
closed before queued detection and generated the 11-wall synthetic export
fixture. The real source map was unchanged. These checks validate software
integration; both new ML candidates remain rejected for promotion.

## SAM 2.1 door-boundary research

The [SAM 2.1 probe record](ENCOUNTER_ML_SAM2_RESULTS.json) preserves the raw
results, artifact hashes and exact experiment source. These are unshipped
experiments. No installed model, runtime detector or training annotation changed.
SAM proposal masks never supplied training labels. The four reserved final-test
layouts remain unevaluated.

The public Apache-2.0 [tiny checkpoint](https://huggingface.co/facebook/sam2.1-hiera-tiny)
is pinned to revision `de431c4043854a71d8101e17995dfe596bf101a5`.
Only safetensors, processor/config JSON and the model card were downloaded.
The 155,908,064-byte model matches SHA-256
`48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.
The checkpoint config emits a `sam2_video`/`sam2` warning when loading the image
model. A separate loading audit found no missing, unexpected or mismatched keys;
all 31,440,721 image-model parameters loaded.

Prompts use only strong door predictions, with bounded neighborhoods and no
reference-label assistance. Existing confidence, shape, wall-support and jamb
attachment rules remain in place. Full-image point/box prompts fail to recover
Mothveil's door in either appearance. Native crop point prompts recover it, but
also turn a plain gray Stone wall indentation into a door. Native source and
proposal crops were reviewed: a good class-agnostic object mask does not establish
that its object is a door. Real Stone gate endpoints can still cross solid jambs.

A separate linear door verifier uses the existing frozen DINO image features.
It trains on 160 explicit door anchors and 448 explicit negative anchors from
30 real training maps. Six deterministic hue rotations produce 3,648 feature
rows; these are correlated augmentations, not 3,648 independent examples.
Unknown labels never supply a target or enter the class-specific feature pool.
Synthetic examples and validation/test maps are excluded from this training.

At the initial 0.9 cutoff the verifier rejects the gray false door and Mothveil's
real door. Pooling the original semantic seed instead of the whole SAM mask
does not resolve that cutoff's recall loss. Five folds of training-map groups
then set a cutoff strictly above every out-of-fold explicit negative score,
with a minimum of 0.5. Every hue variant of a map stays in the same fold.
This selects `0.7464216947555543`, with 931/960 augmented door rows retained and
zero observed calibration false positives. Neither calibration nor the high
training score establishes precision on unseen maps.

The calibrated seed verifier was checked on all 32 cached cases across the
installed segmentation model, v9 and v13, in original and recolored appearances.
Each unfiltered SAM result was replayed and checked before applying the verifier.

| Segmentation model | Mothveil original | Mothveil recolored |
| --- | --- | --- |
| Installed model | One span, matches the annotated door | Door remains missing |
| v9 | One span, matches the annotated door | One span, fails endpoint tolerance |
| v13 | One span, matches the annotated door | One span, matches the annotated door |

On v13, mean endpoint errors are 1.12 and 1.00 original-image pixels. Its original
door-verifier score is only about 0.00575 above the calibrated cutoff. The gray
wall detail is rejected, but that narrow margin does not establish robustness.
The expanded legacy cases expose remaining misses, misplaced endpoints and
pre-existing false doors that this growth-only verifier leaves unchanged.
Mothveil's window and Rimebell's railing failures also remain unresolved.
The research remains unpromoted. Extra SAM processing adds work; timings collected
alongside another GPU workload are not an end-to-end speed benchmark.

The current native asset labels were also checked. All 30 current
`wall-review.json` files explicitly mark provisional Auto-Wall candidates awaiting
manual review. Ashenwatch's reconciliation export identifies four protected user
wall IDs. Those IDs do not establish that its whole export was hand-corrected.
No provisional export was admitted as exhaustive truth. Existing partial
source-assisted labels retain their recorded weak-supervision limitations.

The first verifier-training invocation stopped before loading the model because
the script used the wrong checkpoint filename. The corrected invocation uses
the existing `battlemap-dino-v1.safetensors`; the failed log and manifest are
preserved. Successful training emitted a scalar-conversion warning while
recording the loss, which does not change the trained tensors. The artifact audit
checks source/model/mask hashes, split membership, calibration folds and counts.
The full application suite was not rerun for this research-only documentation
change; the last functional source validation remains the 348-test CUDA run and
the targeted remote CI results above.

## Door-object semantic follow-up

The [object-verification record](ENCOUNTER_ML_OBJECT_RESULTS.json) preserves four
additional training runs, their group calibration, 152 completed trial cases,
exact experiment source, artifact hashes and rejected results. These are repeated
controls on six validation layouts, across colors and segmentation models. They
are not 152 independent maps. No application model or runtime detector changed.
The four final-test layouts remain unevaluated.

Native review identified two different problems. A false wall patch inside
Mothveil's real door shortened its exported span. Removing that patch only inside
a verified, shape-supported door panel fixes the span. Separately, broad crop
features confused correctly labeled floor and torch anchors with nearby doors.
The high-scoring window negative was actual greenhouse glass. Those source
labels were not changed to accommodate the classifier.

Fragment augmentation retained every original anchor and added partial views,
but lost additional real doors. Tight object crops recovered Mothveil while
still rejecting real gates and drawn doors. Both DINO variants failed the
recall check. Their results remain in the record.

A controlled comparison then used the public Apache-2.0
[SigLIP 2 base vision encoder](https://huggingface.co/google/siglip2-base-patch16-224),
pinned to revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`.
Its native safetensors file is 1,500,800,904 bytes, with SHA-256
`612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`.
Only that file, config/preprocessor JSON and the model card were downloaded,
without authentication, archives or remote Python. All 208 vision tensors,
containing 92,884,224 parameters, loaded strictly. Unused text tensors stayed on
disk. The initial load stopped before training because the standalone vision
class omits the full checkpoint's `vision_model.` key prefix. The corrected
loader removes exactly that prefix and checks complete key identity; the failed
script, log and initial manifest are preserved.

The DINO and SigLIP tight-crop comparisons use exactly the same 3,648 anchor,
crop, target and hue records. They pool only explicitly labeled class pixels;
unknown pixels never supply targets or feature weights. The semantic variant
improves the known door controls, but its first version also accepts a plain
Stone wall detail as a door. This is a real false detection, confirmed against
the native source, and disqualifies that version for release.

Systematic negatives at training-door wall caps then add 294 anchors from the
existing explicit wall labels. Their locations derive from training component
geometry, not validation coordinates or model scores. All original anchors are
retained. No annotation is changed, and no SAM mask becomes a training label.
Primary native-source reviews sampled 45 door anchors and 22 added wall-cap
anchors; these checks do not establish exhaustive or independent human gold.

Every cutoff below comes from five training-map group folds, strictly above the
highest out-of-fold negative score, with a minimum of 0.5. Correlated hue views
stay together within each fold. Zero observed calibration false positives does
not guarantee precision on unseen maps.

| Training variant | Cutoff | Retained augmented door rows |
| --- | --- | --- |
| DINO with added fragments | 0.93847179 | 3,441 / 3,840 |
| DINO with tight object crops | 0.96309769 | 715 / 960 |
| SigLIP with the same tight crops | 0.52797383 | 935 / 960 |
| SigLIP with training wall-cap negatives | 0.66465580 | 929 / 960 |

Wall-cap training rejects the false Stone detail, but whole-seed filtering still
removes a real drawn-map door. That filter also remains rejected. A narrower
v11 experiment keeps all original seeds and uses the verifier only to approve
added SAM panels. It removes wall conflicts inside accepted panels, rebuilds
strong wall candidates and recovers jambs once using the updated door/window
union. If no panel changes a mask, it returns the original masks unchanged.
This isolates verified growth from the failed destructive classification step;
it does not correct pre-existing false or misplaced original doors.

In all 32 cached model/color cases, v11 preserves the existing matched door
spans. The repeated matched-span total rises from 27 to 36, with nine added spans
and no existing matched span lost. Six of those additions are the same known
Mothveil door across model/color combinations; the other three are repeated
Rimebell controls. Mothveil now matches in all six combinations, with mean
endpoint errors between 1.00 and 2.06 source-image pixels. These counts do not
represent nine independent newly validated physical doors.

This is useful evidence for door refinement, but the experiment is unshipped.
Known original Stone errors and noisy masonry contours remain. Mothveil's
window and Rimebell's railing failures remain. Window, movement-barrier and light counts stay unchanged in
these cached cases; the caches omit the light-center heatmap, so this is not
full light-head validation. No end-to-end speed improvement has been established.
The current training distribution also remains uneven: Temple and ship supply
128 of 160 original positive door anchors, while only one real training map
supplies physical glass. More hue views do not add independent style coverage.

The artifact audit checks every recorded experiment script, training/calibration
split, feature/target file, classifier checkpoint and output-mask hash. All 36
original training/validation image-mask pairs and all four production model
hashes remain unchanged. The full application suite was not rerun for this
research-only documentation change; its last functional source validation
remains the 348-test local CUDA run and targeted remote CI described above.

## Two-block adaptation and checkpoint loading

The next controlled segmentation trial, v16, adapts the final two DINO blocks
along with the decoder. It retains v15's labels, initializer, crop sampling,
augmentation, replay teacher, batch size and optimizer settings. The decoder
learning rate is 0.00004 and the backbone rate is 0.000004. There are 4,088,598
trainable parameters. The 2,400-step run took 469.07 seconds locally. Final-test
entries were removed before reading the dataset; the run used 78 training and
six validation samples. No training or validation labels changed.

[The audited record](ENCOUNTER_ML_BACKBONE_RESULTS.json) includes the exact
training sources, invocation, input hashes, checkpoint hashes, six evaluation
reports and selected native failure crops. Weight comparisons confirm changes
in blocks 10 and 11 and the final layer normalization, with the other backbone
tensors preserved exactly. The selected checkpoint is step 1,200; the final
checkpoint is step 2,400. Both were evaluated on original and recolored
validation maps. Training-fit checks use the selected checkpoint separately.

| Validation check | Selected: original / recolored | Final: original / recolored |
| --- | --- | --- |
| Mothveil matched door spans | 0 / 0 | 0 / 0 |
| Mothveil matched window spans | 0 / 0 | 0 / 0 |
| Mothveil unmatched window predictions | 0 / 2 | 0 / 1 |
| Rimebell matched doors out of three | 3 / 2 | 2 / 3 |
| Rimebell movement-only segments | 0 / 0 | 23 / 17 |
| Rimebell fence movement check | fail / fail | fail / fail |

The recolored Mothveil windows lie on a masonry stair landing, rather than the
real glazing. The final checkpoint also emits movement-only geometry on an
interior timber support. Native source crops confirm these failures; they are
model output, not corrected annotations. The extra Rimebell fence fragments
still leave its tested passage open. Selected-checkpoint training fit emits
40/43 movement-only segments on Gullthorn, but its full-span movement check
also fails in both appearances. Legacy geometry checks remain in the record:
Stone has an unmatched original-image door, and the final recolored Dyson
checkpoint loses the selected checkpoint's one matched door.

This trial is not promoted. Backbone adaptation alone has not resolved the
missing glass, broken fence coverage or noisy masonry outlines. Partial labels
and the single real training layout with glass remain limitations. This result
does not establish that backbone adaptation cannot work.

The independent runtime change skips random initialization when loading a DINO
checkpoint, then allocates ordinary CPU storage before the existing strict
state copy. It preserves dtype conversion, independent tensor storage, frozen
backbone parameters, metadata checks and cache behavior. It follows PyTorch's
[documented meta-device loading approach](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html),
while retaining a normal state copy. The small CNN keeps its existing loader.

Ten interleaved, uncached CPU loads per implementation and checkpoint produced
these local medians with four CPU threads:

| Production checkpoint | Previous load | Updated load |
| --- | --- | --- |
| DINO segmentation | 0.354 s | 0.125 s |
| DINO light centers v1 | 0.373 s | 0.141 s |
| DINO semantic light centers v2 | 0.326 s | 0.142 s |

These measurements exclude first-process library imports, disk-cold I/O, CUDA
transfer and inference. They show a faster checkpoint-loading step, not faster
segmentation or improved accuracy. All four production checkpoints and the
six-class v16 checkpoint preserve exact CPU tensors, dtypes and predictions on
an odd-size input, including light heatmaps where present. A fresh process
loads DINO through the new path before constructing a reference model. On the
full Mothveil source, CUDA classes, confidence, light heatmap, UVTT preview and
native Foundry Scene data match exactly; generated IDs are fixed for the Scene
comparison. This is functional parity, not a claim that those predictions are
accurate. All four production model hashes and original image/mask pairs remain
unchanged.

Validation: the focused DINO/buffer checks pass 8 tests; the final local CUDA
suite passes 353 tests with three dependency deprecation warnings in 46.75
seconds. Ruff passes on the three changed Python files. The audit retains an
initial regression-fixture typo and an initial report-reader error with their
causes and corrected reruns. No live Foundry world import was rerun for this
loader change.

## Source-reviewed rope-rail extension

The Whistling Stones training layout contained visible rope handrails but no
fence supervision. Its rope pixels were unknown, not background. The
[training-only extension](../benchmarks/data/encounters/additional-whistling-v1/README.md)
adds seven one-pixel rope cores, ten narrow wood-post interiors and six clear
path/ground patches. All 1,187 new fence pixels and 2,468 new background pixels
were previously unknown; every earlier supervised pixel is preserved. New
positives exclude flag fabric, rock/cliff edges, plants and the tent. Suspended
ropes are not filled down to the ground.

The primary reviewer compared native-coordinate grids and enlarged original
pixels with actual raster overlays. The first draft touched flag fabric or
missed some rope cores. The second placed one post on the wrong X coordinate
and extended another into rock below its base. Both were rejected before
training and remain recorded with hashes and reasons. Only the corrected
third draft was admitted. These selected source-assisted patches are partial
training supervision, not exhaustive independently human-corrected truth.
No model prediction or proposal supplied a label. The source images, all
validation/test annotations and the frozen whole-layout splits are unchanged.

A partial-mask oracle passes all nine added native passage checks: three
sight-transparent ropes block movement, and three reviewed ground passages
stay open. Unknown pixels remain 255 in training and are neutralized only in
this extraction-interface test. This confirms that correct material cores can
reach the native exporter; it does not measure model accuracy.

The v17 experiment retains v16's two-block DINO setup, initializer, replay
teacher, hyperparameters and six validation layouts. Only Whistling's training
mask changes, and its crop sampling weight increases from 1 to 10. The current
loader also consumes a different torch RNG sequence because it skips random
checkpoint initialization. These changes are recorded together; this is not a
single-variable causal ablation. The 2,400-step run took 422.09 seconds locally.
The partial-label selector chooses step 1,200 with score 0.69799, compared with
v16's 0.67594. Neither that increase nor a lower loss establishes correct spans.

[The audited record](ENCOUNTER_ML_RAIL_RESULTS.json) preserves the training
sources, invocation, input/checkpoint hashes, admission scripts, ten evaluation
reports and native source/output crops. Its 36 cases comprise selected/final
validation in both appearances, separate selected-checkpoint training fit, and
Whistling controls using v16 selected and v17 final checkpoints. Final-test
entries are absent before reading the dataset. Recolored views share layouts
with their originals and are not independent examples.

| Check | Selected v17: original / recolored | Final v17: original / recolored |
| --- | --- | --- |
| Rimebell fence movement check | fail / fail | pass / pass |
| Rimebell matched doors out of three | 2 / 2 | 2 / 2 |
| Mothveil matched door spans | 0 / 0 | 0 / 0 |
| Mothveil matched window spans | 0 / 0 | 0 / 0 |
| Mothveil unmatched window predictions | 0 / 0 | 0 / 1 |
| Whistling training rope movement checks passed out of three | 1 / 1 | 1 / 1 |

The final checkpoint gains a real partial fence passage on Rimebell in both
appearances, where both v16 checkpoints failed. Native review shows the tested
geometry following wooden railing. Coverage remains partial and fragmented.
Both v17 checkpoints lose Rimebell's south door. Mothveil still misses its
physical glass and door; its final recolored prediction places a window across
masonry steps, and the final original prediction places movement-only segments
on an interior timber support. Antler and Cinderglass still export zero
structures in all four validation cases. Original Stone retains an unmatched
door; final recolored Dyson loses its remaining matched door.

Whistling's selected model exports 115/84 movement segments and its final model
72/68, but the lower horizontal and curved rope movement checks still fail in
both appearances. The v16 control exports no fence and fails all three rope
checks. Gullthorn's full-span training railing check also still fails on the
selected v17 checkpoint. These training-fit results do not establish held-out
generalization.

Neither checkpoint is promoted. The six-class segmentation candidate also
lacks the installed semantic-light center head. Further source-reviewed rail
boundaries and independent physical-glass styles are needed; confidence cuts
and prop guards are unchanged. All four production model hashes remain intact.
The new source/raster/native-export regression passes; the full local suite
passes 354 tests with three dependency warnings in 34.07 seconds. Ruff and
`git diff --check` pass. No live Foundry world import was rerun for this
training-data and research change.

## Boundary trial rejected; walls and doors take priority

The [v2 source-reviewed extension](../benchmarks/data/encounters/additional-whistling-v2/README.md)
adds 3,212 background pixels beside Whistling's rope cores, preserving every
earlier supervised pixel. A first strip near a post was rejected before
training. These are partial primary source-assisted labels, not independently
human-corrected gold. All other masks, sampling weights and frozen splits stay
unchanged. The v18 trial keeps v17's recipe and runs 2,400 steps in 467.81 seconds.
Changed anchor positions can change crops; this is not an exact paired-minibatch
causal comparison.

The [completed record](ENCOUNTER_ML_BOUNDARY_RESULTS.json) retains hashes,
invocations, 40 native evaluation cases and rejected diagnostics. Background
classification improves on the new strips, but all three Whistling rope checks
fail in both appearances at both checkpoints. Mothveil still misses its actual
door and glass. Rimebell still misses one of three doors, and final recolored
Dyson loses its remaining matched door. A separate weak-confidence recovery
probe is also rejected because it introduces interior-beam false detections.
Neither model nor any threshold change is promoted. Training scores and raw
segment counts do not establish usable wall or door accuracy.

Further fence experiments stop here. Correct wall placement, clean outlines and
doors fitted to real openings take priority. The new label-provenance regression
passes; the full local suite passes 355 tests with three dependency warnings in
58.66 seconds. Ruff passes on the changed test. No live Foundry import was rerun
for this research/data update, and all four production model hashes are intact.
