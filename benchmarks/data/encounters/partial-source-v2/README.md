# Partial encounter seed annotations

These are **partial source-assisted weak labels**, admitted for experiments after
a second visual pass by the primary agent. They are not exhaustive, independently
human-corrected ground truth. The original images are not included in this repository.

The 31 layouts have frozen whole-layout splits: 25 train, three validation and
three test. Source PNGs and their WebP derivatives belong to the same layout.
`splits.json` must stay unchanged when adding labels or appearance variants.

`annotations/` stores explicit polygons and narrow strokes in original image
pixels. Missing labels remain **255 (unknown)**. Label 0 is an explicit negative,
1 a wall, 2 a door, 3 a light, 4 a movement-blocking transparent barrier and 5 a
window or glazed barrier. No model prediction created these labels.

Wall-cap and door-panel interiors were selected from the artist's existing
markings. Some unmarked textured masonry and wood rails have narrow hand traces.
Outdoor cutaway borders, cliffs, individual rocks, trees, water, tents and props
are not inferred to be walls. Ambiguous cave-border sections were omitted.
Possible emitters inside negative regions were conservatively ignored rather
than labelled as either lights or background.

Glass positives are limited to Gilded Snare in training and Mothveil Lodge in
validation. Fence material patches are also sparse. A result on this seed set
cannot establish broad window/fence accuracy or new-art-style generalization.
Door/window span annotations and passage rays are partial checks. Unmatched
predictions outside reviewed coverage remain unassessed, not proven correct.

`review.json` records this limited admission scope and exact source, annotation
and raster SHA-256 hashes. Any edit requires another review and fresh hashes.
The local `.gitattributes` preserves exact JSON bytes across checkouts so Git
newline conversion cannot silently invalidate the review hashes.

The Mothveil passage check has a geometry-only erratum: its window span now
covers the full framed opening, and sight/movement rays cross both exported
faces. The original short ray ended inside the assembly. `review.json` records
the revised annotation hash; the training raster is unchanged. Historical
training datasets retain their original checks and were not used for geometry
checkpoint selection. Extra post-review validation regressions live separately
in `../negative-ray-checks-v1.json` and are not training labels.

Prepare masks into a **new directory outside the source maps**:

```powershell
python scripts/prepare_encounter_training.py "PATH/TO/encounters" `
  benchmarks/data/encounters/partial-source-v2/annotations `
  benchmarks/data/encounters/partial-source-v2/splits.json "PATH/TO/new-dataset" `
  --review benchmarks/data/encounters/partial-source-v2/review.json
```

An optional `--replay PATH/TO/legacy-manifest.json` retains older real maps and
their original splits. `--replay-weight 5` gives each older training map five
times the crop sampling weight of a new map. It does not change validation,
test membership or evaluation weighting. Images and masks use uncompressed BMP;
confidence arrays use NPY and checkpoints use safetensors.

See [the experiment notes](../../../../docs/ENCOUNTER_ML.md) for failures and
promotion criteria. The rejected initial annotation previews and later model
predictions are kept separately in the local workspace; neither is ground truth.
