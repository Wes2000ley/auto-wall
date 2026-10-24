# Artist sprites for light training

`scripts/generate_sprite_lights.py` places complete reviewed light sprites and
non-emissive lookalikes on simple procedural floors. It supplies exact point
targets for varied positions, rotations, reflections, and sizes. This addresses
the context sensitivity seen in trials 22–23 without cutting through emitters.
It is a training-data experiment, not a runtime detector or proof of general
recognition accuracy.

The approach is informed by [Simple Copy-Paste is a Strong Data Augmentation
Method for Instance Segmentation](https://arxiv.org/abs/2012.07177). That paper
evaluates instance segmentation. Here the objects come from a reviewed sprite
sheet and produce center labels; the procedure is not a reproduction of its
training or benchmark results.

## Prepare scenes

Download the original [Puny dungeon sprite sheet](https://opengameart.org/sites/default/files/punyworld-dungeon-tileset.png)
from [Shade's source page](https://opengameart.org/content/16x16-puny-dungeon-tileset).
The source is CC0; retain the author and source information when sharing derived
data. The generator checks its SHA-256 before creating any output. The supplied
specification expects `3aa73f296b872c930465bc1c03858312b37dd1229b31d4718b888f643aa80de7`.
It does not fetch files or execute remote code.

```shell
python scripts/generate_sprite_lights.py benchmarks/data/puny_light_sprites.json downloaded/punyworld-dungeon-tileset.png dataset/puny-lights --count 136 --seed 921000
```

The original PNG is read without rewriting it. Output consists of raw BMP
scenes, object JSON records, and a point manifest accepted by
`scripts/train_light_centers.py`. An existing output directory is rejected.
The default generates 128 training scenes and eight reserved test scenes.
Use new seeds for additional data and preserve each existing seed's split.
The dataset loader rejects duplicate IDs and layout groups crossing splits.

To combine these with real annotations, resolve image paths against each source
manifest and preserve all IDs, groups, splits, and `synthetic` flags. Keep the
original real-map validation set separate. Synthetic test results measure the
generator task and do not replace evaluation on real artwork.

## Labels and rendering

[The specification](../benchmarks/data/puny_light_sprites.json) contains 59
reviewed 16 px frames: eight torches, twelve portals, twenty fire jets, nine
blades, four bones, and six keys. Active torches, portals, and fire jets emit
light. Blades, bones, and keys are negative examples despite bright highlights.
Partially extinguished fire and closing portal frames are excluded. Each fire
jet has one center even if its animated flame contains disconnected pixels.
Portal centers can lie inside transparent holes.

Each scene has sixteen nonoverlapping objects, including every family. Remaining
families and animation variants are sampled randomly. Objects are resized to
24–80 px on their longest side using nearest-neighbor interpolation, rotated in
quarter turns, and optionally reflected. Their centers follow the same pixel-
center transform. Straight alpha is composited once, preserving background
inside transparent holes. Placement never crops the sprite or covers an earlier
object. A mild illumination gradient changes the whole scene; the existing
training sampler adds its normal color and scale augmentation afterward.

The JSON object records preserve each source rectangle, destination rectangle,
rotation, reflection, center, and emission label. The manifest records the full
sprite specification, source-sheet hash, generator hashes, and seed. Invalid
bounds, empty sprites, malformed centers, or missing positive/negative families
are rejected before any dataset directory is created.

Other local sprite sheets can use the same schema. `rectangles` entries are
`[x, y, width, height]`; `center` is relative to each rectangle. Every variant in
a family uses that center and emission label. Source rectangles must be at most
64 px across; the sheet must decode to uint8 BGRA. Use a new specification ID
for different source artwork so generated sample IDs remain distinct. The
specification is an annotation document, not an automatic label inference step.

## Evaluation limits and provenance

These sprites are shared with the Puny maps already inspected in earlier trials.
Generated layouts are distinct, but the object designs and art style are not
unseen. Adding fire-jet frames is informed by a previous failure on the reserved
Puny map, so subsequent results are adaptation evidence, not a fresh test.
Agent review and construction labels are not independent human ground truth.
Repeated use of the small real validation set also limits selection claims.

The standalone PNG differs from the sprite sheet inside the earlier source ZIP.
All 59 selected rectangles were compared and have identical BGRA pixels in both;
their positions and colors are therefore consistent with the existing samples.
No source map layout, character, or private user artwork is copied into the new
scenes. The sprite sheet itself is not redistributed in this repository.

The initial specification mistakenly included one blank cell as a blade. The
nonempty-alpha check rejected it before creating a dataset or starting training.
That preflight failure and its correction are retained locally. The original
procedural generator still produces byte-identical images, centers, and object
records on the four checked seeds after extracting its shared floor renderer.
