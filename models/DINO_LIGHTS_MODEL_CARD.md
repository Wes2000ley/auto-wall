# DINO light centers v1 — optional comparison model

`battlemap-dino-lights-v1.safetensors` adds a learned physical-light center head
to the original DINO segmentation model. Every one of the base model's 313 stored
tensors is identical. Walls, doors, transparent barriers, and their confidence
controls keep the same behavior. The light head shares the same feature pass.

The file is 90,523,296 bytes, uncompressed. SHA-256:
`dcd383eb130e5a973e1e4df5d389fde223f5e92a21e42a6110cd7301bc9600b1`.
The head has 14,569 trained parameters. See the
[training record](DINO_LIGHTS_TRAINING_RECORD.json) for hashes, source identities,
selection, and recipe. This is **not the default detector**.

[Follow-up trials 15–17](../docs/LIGHT_RESEARCH.md) added cool emitters and
negative examples. They learned new source types but regressed on small lanterns
and flowers, so this bundled checkpoint remains unchanged.

```shell
python fast_auto_wall.py "map.png" --settings profiles/ml-light-centers.json --model models/battlemap-dino-lights-v1.safetensors --ml-device auto
```

The ML tab's **Light center score** controls acceptance, with 0.35 used for the
reported checks. Higher values accept fewer sources. Scores are not calibrated
accuracy probabilities. This model replaces light pixel confidence and area/fill
limits with point predictions. Door-highlight rejection and light range still
apply. See-through barriers remain omitted from UVTT vision-blocking walls.

## Training and failed alternative

The method uses point heatmaps inspired by [Objects as Points](https://arxiv.org/abs/1904.07850).
A small convolutional head receives the frozen decoder's 16 features plus RGB,
pooled to quarter resolution. Its local RGB context spans 60 source pixels;
the existing decoder features also carry broader context. Bilinear heatmap
upsampling avoids false local maxima on nearest-neighbor stair steps.

The light-head data contains the wooden camp's nine marked sources, the temple's
30 reviewed flames, reviewed negative regions of Dyson map 13, and 64 separately
seeded procedural maps. Procedural lights include varied colors, shapes, and
shading. The stone camp's 13 sources and reviewed regions of Dyson map 14 are
validation. The spaceship contributes to the inherited segmentation model but
was excluded from light-head training because its cool emitters lack complete
labels. Original images and upscales remain one layout group. Source artwork is
not redistributed.

Trial 13 trained a full-resolution head for 2400 steps. Its selected checkpoint
found only 10/13 stone lights and placed three centers on the wooden camp's single
fire. That run is retained and not shipped. Trial 14 widened the local context
through quarter-resolution pooling, with the same real/procedural mixture.
It completed 2400 steps in 268.42 seconds, including final review. Its best step
was 1350, selected only by precision-weighted light F0.5 at threshold 0.35 on the
validation layouts. The score was 0.983607; it is not whole-map accuracy.

Both runs used four crops per batch (three real, one procedural), AdamW at initial
learning rate 0.0005, mixed precision, seed 73113, and an RTX 3080. Trial 14's peak
allocated CUDA tensors were 512,191,488 bytes, excluding context, reserved memory,
and other applications. All 313 segmentation tensors were checked unchanged.

## Observed lighting results

| Map | Previous lights | Center-model lights | Interpretation |
| --- | --- | --- | --- |
| Wooden camp, training | 7/9 matched; one extra fire fragment | 9/9 matched; no extras | One center for the campfire |
| Stone camp, validation | 11/13 matched; no extras | 12/13 matched; no extras | One small hanging lantern still missed |
| Temple, training | 30 flame candidates | 30 matched flames | No extra candidates |
| Dyson map 14, partial validation | Three pool-stroke candidates | Zero through the normal pipeline | Partial annotations; no full-map precision claim |
| New palisade camp, observed holdout | Five torches plus four campfire fragments | Five torches, one campfire, one false flower | Better grouping, with a new false positive |
| Spaceship and its upscale | Zero | Zero | Cool machinery/strip emitters remain missed |
| Astral cavern, observed test | 19 candidates | 24 candidates | Decorative stars remain false lights; not a general improvement |

These are agent-reviewed labels and visual checks, not independent human ground
truth. Observed test maps are diagnostic evidence, not fresh unseen validation.
The model must remain optional until source coverage and precision improve across
styles. Flowers, painted symbols, cool emitters, and unfamiliar lanterns need
better training data. More candidates alone do not establish better detection.

## Runtime and geometry preservation

The normal loader, detector, and UVTT exporter were compared on ten maps at a
1536 px maximum working edge. Wall segments, door endpoints, and transparent
segments were exactly identical between models on all ten. Raw segmentation
classes and winning probabilities were also bitwise identical on both camps.
Every exported PNG matched the input bytes exactly.

Five warm detections per model, PyTorch 2.11.0+cu128, RTX 3080, four CPU threads:

| Input | Original DINO | DINO with light centers |
| --- | ---: | ---: |
| Wooden camp | 0.284 s | 0.303 s |
| Stone camp | 0.281 s | 0.292 s |
| Temple | 0.372 s | 0.377 s |
| Spaceship | 0.339 s | 0.368 s |
| Demon temple, 5792 × 4344 | 0.398 s | 0.405 s |
| Dyson map 14 | 0.316 s | 0.319 s |
| Astral cavern, 5792 × 4344 | 0.353 s | 0.354 s |
| Dyson map 15 | 0.264 s | 0.287 s |
| New palisade camp | 0.279 s | 0.291 s |
| Upscaled spaceship, 6144 × 4096 | 0.355 s | 0.362 s |

These medians include resize, inference, and geometry, but exclude startup/model
loading, image decoding, preview painting, and export writing. Other desktop/GPU
work varied during the session. The head adds work; this is an accuracy experiment,
not a claim of a speedup. It reuses the expensive feature computation rather than
running a second segmentation model.

Actual Qt-worker checks on both camps completed detection and export, including
score changes and exact source-PNG preservation. Separate desktop runs observed
19–24 s for a cold process and about 0.6–1.0 s for cached detections under concurrent
GPU work. A repeated 8-versus-4 CPU-thread comparison did not show a consistent
thread-count advantage, so the app's thread policy was not changed. Live Foundry
import, other GPUs/operating systems, and release packaging were not tested here.

## Attribution

The inherited DINOv2-small weights are by Meta Platforms, Inc. and affiliates,
under Apache 2.0. Preserve [the license](licenses/DINOv2.txt) and
[attribution](licenses/README.md). Base-weight revision, SHA-256, earlier training
groups, and limitations are in the [base model card](DINO_MODEL_CARD.md).
Cartography by Dyson Logos, from the
[commercial-use map collection](https://dysonlogos.blog/maps/commercial-maps/).
No original cartography or private map artwork is included in the repository.
