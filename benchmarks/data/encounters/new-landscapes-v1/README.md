# Additional landscape negatives

This extension adds two source maps found after the original 31-layout corpus
was frozen. It does not alter that corpus, its labels or its splits. The new
membership was fixed before evaluating any model predictions: Sunspoke Shoals
is training, Cinderglass Basin is validation. Neither is a reserved final test.

The primary reviewer inspected native artwork and raster overlays. Both maps
contain terrain and loose props, with no visible architectural walls, doors,
windows or fences. These must remain empty of exported structural geometry.
Their explicit `structure_free: true` check counts all exported structures
across the full image, including unlabelled pixels. A few open passage rays
cannot establish that a map contains no false walls. This check does not
assess lights or review-only barrier candidates.
The separate review records that expectation and the source, annotation,
raster and reviewed-preview hashes. Hashes establish identity, not correctness.

Pixel supervision is partial. Sunspoke excludes the ambiguous bright treasure.
Cinderglass labels only nine selected cold-terrain rectangles; other pixels,
including possible geothermal emitters, stay 255. Two earlier Cinderglass
drafts attempted to subtract hot regions from a whole-map background label and
missed small border patches. They were rejected during overlay review and were
never admitted or used for training. The accepted version selects clear cold
terrain directly. None of these labels comes from model predictions.

Use `scripts/prepare_encounter_training.py` with this folder's `annotations`,
`review.json` and `splits.json` to prepare the subset. Merge it into a new
manifest, retaining the existing layout groups. Trials through v12 use earlier
datasets and do not contain these two new maps. Expect zero exported structures
across each full map; known-pixel metrics alone cannot establish that result.
