# Reviewed public light maps

This dataset adds ten artist-created sample maps with 59 marked light sources.
It supplements the existing private maps and procedural fixtures; it does not
replace them. The original images and point locations were visually reviewed
before running any model predictions on these ten images. These are agent
annotations, not independently reviewed human ground truth.

## Sources and attribution

| Source | Artist | License | Local download filename |
| --- | --- | --- | --- |
| [Castle / Dungeon](https://opengameart.org/content/castle-dungeon) | Gary Shaw | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `castle-dungeon_tiles.zip` |
| [16x16 Puny Dungeon Tileset](https://opengameart.org/content/16x16-puny-dungeon-tileset) | Shade | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | `puny_dungeon_v1.zip` |
| [Cavern Battlemap](https://opengameart.org/content/cavern-battlemap) | Ferrin | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | `cavern.png` |

The source pages provide the license statements. This repository distributes
annotations and a preparation script, not the source artwork. Preserve this
attribution when distributing prepared Castle / Dungeon images. Changes made
here are resizing, light-center annotation, character exclusion masks, and
labelled preview overlays. Training uses the resized source images, not overlays.

The four Castle / Dungeon samples come from Gary Shaw's original asset archive.
The five Puny samples come from Shade's Tiled sample-image directory. The cave
is Ferrin's published battlemap. No maps are assembled from tiles by this script.
No downloaded code is executed, and no existing archive is modified or recreated.

## Preparation

Download the three original files into one directory:

- [Castle / Dungeon archive](https://opengameart.org/sites/default/files/castle-dungeon_tiles.zip)
- [Puny Dungeon archive](https://opengameart.org/sites/default/files/puny_dungeon_v1.zip)
- [Cavern image](https://opengameart.org/sites/default/files/cavern_0.png), saved as `cavern.png`

Then run from the repository root:

```shell
python scripts/prepare_open_light_maps.py downloaded-art dataset/open-lights
```

The script verifies every complete download and selected image against the
SHA-256 values in [the annotation specification](../benchmarks/data/open_light_maps.json).
Changed files are rejected before creating the output directory. The expected
download hashes are:

```text
castle-dungeon_tiles.zip  710b9236b42b88cf95c39a50a96ea49a36058d805d58d0e57a6e6c4c05e65d60
puny_dungeon_v1.zip       22fd85633ccd3d92ff7f4f036cc3934841ef497637f34e22c90431d479fa7954
cavern.png               90955254e64cac4455044313c2ed00616fe1abd93bff5eacf10c9c868dbf80f0
```

Outputs are raw BMP images, optional raw BMP validity masks, and a manifest
accepted by `scripts/train_light_centers.py`. The destination must not already
exist. Archives are read only for the explicitly named sample images; their
paths are never extracted. Source image bytes are left untouched.

The 1920 px castle samples and 2000 px cavern are reduced to a 1536 px edge with
area interpolation. The 896×856 castle room stays at its original size. The
240×240 Puny samples are enlarged to 960×960 with nearest-neighbor interpolation.
That enlargement preserves their pixel art and adds no independent detail.
Points use pixel-center resizing, `(coordinate + 0.5) * scale - 0.5`.
Character exclusion masks use nearest-neighbor resizing.

## Labels and splits

| Sample | Split | Sources | Interpretation |
| --- | --- | ---: | --- |
| Castle 1 | Train | 13 | Wall torches |
| Castle 2 | Train | 5 | Three torches and two fireplaces |
| Castle 3 | Validation | 7 | Six torches and one fireplace |
| Castle 4 | Test | 2 | Wall torches |
| Puny 1 | Train | 0 | Crates, gates, and other non-emissive details |
| Puny 2 | Train | 6 | Torches; green water is background |
| Puny 4 | Train | 18 | Sixteen torches and blue/green luminous portals |
| Puny 5 | Validation | 1 | Blue portal; gate posts and boulders are background |
| Puny 3 | Test | 7 | One center per continuous fire jet |
| Ferrin cavern | Validation | 0 | Water and shore highlights are background |

Puny 1, 2, and 4 share their four-room structure and remain in the same training
group. Puny 3 and 5 use different layouts. Castle samples have different room
layouts. Validation and test nevertheless share sprite families with training;
success would show transfer between layouts, not wholly unseen artistic styles.
The cavern style appears only in validation.

Each fireplace and fire jet has one center even if its flame has multiple bright
parts. A luminous portal is annotated at its center, including its dark interior.
Gold, keys, treasure, painted shields, shiny weapons, unlit gate posts, water,
and pale rocks are not annotated as lights. Reviewed non-emissive objects also
provide negative crop anchors. Puny character sprites are excluded because their
small carried objects and bright eyes do not give reliable scene-fixture labels.
They are neither positive nor negative training evidence. All remaining image
pixels are reviewed; the castle and cavern maps have complete annotations.

The new validation images contain only eight sources, including just one cool
portal. This is a useful additional failure check, not a large or balanced
benchmark. The combined local v4 manifest retains all earlier splits and adds
these ten samples: 137 training samples, five validation samples, and ten test
samples. The training count includes 128 procedural maps. The combined validation
set has 21 marked sources. Existing partial ship labels and already observed
diagnostic maps keep their earlier limitations.

## Reproducible checkpoint evaluation

Evaluate a fixed checkpoint without training or threshold selection:

```shell
python benchmarks/evaluate_lights.py dataset/open-lights/manifest.json models/battlemap-dino-lights-v2.safetensors checks/open-validation --split validation --device cuda
```

The default split is validation. Pass `--split test` explicitly only after fixing
the checkpoint and threshold. A fresh output directory receives per-image counts,
matched/missed/extra indices, predicted and annotated coordinates, reviewed
coverage, model/manifest/pixel/label hashes, and raw BMP overlays. Green circles
show annotations, green crosses mark misses, cyan circles mark matched predictions,
and red circles mark extras. Excluded regions are darkened. The metric uses the
same point matching and F0.5 score as training; it favors precision. Timing excludes
imports, model loading, image decoding, and preview writing.

These are light-center checks. They do not establish wall continuity, correct
door endpoints, or Foundry export quality. Keep the existing geometry and export
checks when considering a model for application use.

The [trial 22 results](SEMANTIC_LIGHT_RESEARCH.md#public-artwork-expansion-trial-22)
record the first training run and its failures. It improves some counts but does
not replace the bundled model.
