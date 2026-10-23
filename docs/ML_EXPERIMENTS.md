# ML experiments and geometry checkpoint — 2026-10-22

The bundled DINO model remains the selected comparison checkpoint. Two additional
3000-step training runs improved individual cases but did not consistently improve
structural accuracy. The successful application changes are optional shaded-jamb
recovery, preservation of sloped jamb intersections, and less work in door clipping.
The palette detector and compact model remain available. ML is not the default.

## What changed in the application

`ml_recover_jambs`, exposed as **Recover shaded door jambs**, keeps the normal
0.9 wall cutoff except near validated door ends. A 0.5-confidence wall component
must connect to an accepted wall, support both ends of a shape-valid door, and
fall within a small neighborhood of those endpoints. Other classes never become
walls. This is local recovery of predicted material, not arbitrary gap filling.

The first version exposed a separate geometry defect: clipping the door footprint
could remove a sloped jamb edge. On the spaceship it lost four real doors while
recovering three others, changing 59 exported doors to 58. Stage diagnostics showed
that all 62 candidates still existed before clipping. Preserving actual door-axis
intersections near the jambs fixes the cause. The corrected result exports 62
doors with either confidence mode. The three previously rejected additions were
visually checked against the artwork. Counts alone are not accuracy measurements.

Jamb classification now operates only on wall segments whose bounding boxes
overlap the door. A paired cached-prediction benchmark retained exactly identical
UVTT geometry on seven maps. Median geometry time fell from 94.5 to 64.7 ms on the
temple and 83.3 to 54.9 ms on the spaceship. Maps with few doors showed little
benefit or ordinary timing noise. This comparison excludes model inference.

Use the new profile with an explicitly selected model:

```shell
python fast_auto_wall.py "map.png" --settings profiles/ml-jamb-recovery.json --model models/battlemap-dino-v1.safetensors --ml-device auto
```

Uncheck the recovery control to restore strict wall confidence. Both modes retain
door size, shape, and geometric attachment checks. Transparent barriers stay in
their separate preview layer and are omitted from both UVTT wall arrays and portals.

## Paired geometry checks

These results use the same bundled weights, latest geometry code, and fixed manual
door-span tolerances. The wooden camp was a training layout. The stone camp and
Dyson map 14 are observed validation layouts. Their annotations are agent-reviewed,
not independent or comprehensive semantic ground truth.

| Map | Strict door matches | Recovery door matches | Strict rays | Recovery rays |
| --- | ---: | ---: | ---: | ---: |
| Wooden camp | 1/4 | 4/4 | 13/16 | 16/16 |
| Stone camp | 3/4 | 4/4 | 20/20 | 20/20 |
| Dyson map 14, partial instances | 1/3 | 1/3 | 12/14 | 12/14 |

The strict stone output already contained four portals, but its north span was too
long. Recovery corrects that span; merely counting four detections would miss this
improvement. Both camps' checked open gates remain unobstructed, and their closed
portals block the corresponding test rays.

One wooden-pen ray was corrected before the paired check: its original start was
inside the solid palisade behind the fence. It now starts in the visible gap below
that palisade. This explains the strict result changing from the historical 12/16
to 13/16 without a model improvement. The original annotations and reports remain
in the local experiment record. Training selection did not use this wooden-map ray.

Lights are unchanged by recovery: the wooden camp matches 7/9 annotated sources
with one extra campfire fragment; the stone camp matches 11/13 without extras.
Dyson map 14 still has three false pool-stroke light candidates. The new palisade
layout still misses open gates and splits its campfire into multiple lights.

The corrected clipping and recovery were also run through the normal model loader,
detection pipeline, and UVTT export on ten inputs. All original PNG bytes matched
their exports exactly. The upscale spaceship gains two visually checked doors
with recovery (58 to 60). This is the same layout as the smaller spaceship, not an
independent test. Astral and reserved Dyson artwork still expose false boundaries
and missed doors; no general accuracy claim follows from this batch.

## Runtime of the final paired check

RTX 3080, PyTorch 2.11.0+cu128, Windows, four CPU threads; five warm detections per
mode. Detection uses a maximum 1536 px edge. These medians include resize, model
inference, and geometry, and exclude image decoding, initial model loading, and
UVTT writing. Recovery adds work; it is an accuracy option, not a speed improvement.

| Input | Strict | Recovery |
| --- | ---: | ---: |
| Wooden camp | 0.281 s | 0.331 s |
| Stone camp | 0.293 s | 0.306 s |
| Temple | 0.379 s | 0.396 s |
| Spaceship | 0.356 s | 0.377 s |
| Demon temple, 5792 × 4344 source | 0.404 s | 0.420 s |
| Dyson map 14 | 0.329 s | 0.348 s |
| Astral cavern, 5792 × 4344 source | 0.366 s | 0.447 s |
| Dyson map 15 | 0.303 s | 0.319 s |
| New palisade map | 0.278 s | 0.307 s |
| Upscaled spaceship, 6144 × 4096 source | 0.355 s | 0.381 s |

These runs occurred later than the original model-card benchmark. Desktop/GPU
activity varies; compare the paired modes here rather than treating the difference
from earlier absolute times as an isolated software regression or speedup.

## Training experiments retained

Whole-layout splits stay separate. Original images, upscales, recolors, and crops
are one layout group. No observed test layout was added to these training runs.
The default remains the simpler frozen-backbone training path.

| Trial | Change | Selected step | Result |
| --- | --- | ---: | --- |
| 4 | Compact model with corrected labels | 1500 | Mean foreground IoU 0.5611; did not replace compact v1 |
| 5 | Frozen DINOv2-small context with local decoder | 750 | IoU 0.7823 on that trial's stone-only validation |
| 6 | Add ink-map training and initial partial ink labels | 2250 | Geometry score 0.8649 under the earlier protocol; overly broad ink wall masks |
| 7 | Explicit negative floor bands around ink walls | 2250 | Narrower ink boundaries; score 0.8012 under the earlier protocol |
| 8 | Select using stone and ink geometry | 300 | Score 0.8343; selected DINO v1 checkpoint |
| 9 | Correct misplaced training-map door annotations | 450 | Score 0.8179; did not replace trial 8 |
| 10 | Frozen MobileNetV4-small multiscale features | 1050 | Score 0.6705; fewer useful gates than DINO |
| 11 | Add procedural structures, varied materials/colors and light types | 2250 | Score 0.8420; improved lights but missed a stone gate and wall probes |
| 12 | Same mixture; train the final two DINO blocks and final normalization | 2550 | Score 0.8160; improved ink doors after recovery, with stone-wall regressions |

Scores are **not comparable across changed annotations and selection protocols**.
Trials 6–7 selected geometry using the stone map; trials 8 onward use both observed
validation layouts. They do not measure a percentage of correctly understood maps.
Trial 8 was stopped after logged step 2800 of a planned 6000 when a shifted training
door label was found. Its earlier best checkpoint and interrupted history were
preserved. Correcting the label was tested in trial 9, rather than dismissing the
method based on the flawed annotation.

Trials 11–12 used six real and two synthetic crops per batch. The procedural set
has 64 training layouts and eight separately seeded reserved layouts; selection
still used real validation maps. An initial generator eroded small light labels at
boundaries. That implementation was corrected before training. Door-label oracle
checks on the first eight corrected examples passed 68/68 spans and 136/136
open/closed rays. Detached one-pixel flame tips still produced two extra instances,
so even perfect procedural labels did not prove light-instance correctness.

Both latest runs completed 3000 steps: 464.25 s for trial 11 and 506.60 s for trial
12. Trial 12 trained 4,088,581 parameters, with backbone learning rate one tenth of
the decoder rate. Its peak allocated CUDA tensors were 1,769,472,512 bytes, excluding
the CUDA context and other applications. A tensor comparison confirmed only the
intended backbone blocks and final normalization changed; 185 earlier backbone
tensors stayed identical. The production trainer supports `--backbone-layers 0–4`.

After the clipping fix, all three candidates were reevaluated through identical
geometry code with recovery enabled:

| Candidate | Wood gates | Stone gate spans | Stone rays | Ink door spans | Ink rays |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected trial 8 | 4/4 | 4/4 | 20/20 | 1/3 | 12/14 |
| Synthetic trial 11 | 4/4 | 3/4 | 18/20 | 1/3 | 12/14 |
| Adapted trial 12 | 4/4 | 3/4 | 18/20 | 2/3 | 13/14 |

Trial 11 matches all 13 stone lights; trial 8 matches 11. Trials 11 and 12 also
remove the observed ink pool-light false positives. Those gains are useful evidence
for continued training, but do not outweigh the two missed stone-wall probes and
incorrect/missing gate span for this checkpoint choice. Raw runs, source snapshots,
labels, and failed alternatives remain local and uncompressed; private map artwork
is not redistributed in Git.

## Validation and remaining work

The full suite passed: **138 tests**, with three dependency deprecation warnings.
Focused sloped-jamb, opening-clearance, confidence-anchor, transparency, settings,
and GUI tests passed. Actual offscreen Qt workers detected and exported both camps;
the checkbox invalidated the prior preview and triggered a fresh result. Source
PNG bytes were verified through that export path as well as the headless runs.

The first new clipping test passed a NumPy array where the internal function takes
a list. That test-harness error was corrected and the checks rerun; it was not a
geometry failure. The failed initial recovery trial and light-grouping experiments
are retained. Lowering light confidence alone did not connect the fragmented fires.

Live Foundry import, other operating systems/GPUs, and release packaging were not
tested in this checkpoint. Open door leaves, unfamiliar materials, small cool
emitters, fire-instance grouping, continuous thin ink boundaries, and false cavern
structures still need better data and evaluation. The current result is an ongoing
research checkpoint, not a claim that arbitrary maps are solved.
