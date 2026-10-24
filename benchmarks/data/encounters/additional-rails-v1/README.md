# Additional Frostwake railing supervision

This is a versioned addition to the partial `partial-source-v2` Frostwake
annotation. Its full annotation inherits every earlier region, stroke and
instance check, then adds four narrow wood interiors on the south courtyard
railing: two rails and two posts. Coordinates refer to the original 1536 × 1024
source image, in pixels from the top-left corner.

The primary reviewer inspected the native source and a 5× nearest-neighbor
overlay. Exactly 613 previously ignored pixels become transparent-barrier
positives. Earlier supervised pixels, source artwork and frozen layout
membership do not change. The separate review records source, annotation and
raster hashes. These hashes establish identity, not semantic correctness.

This remains partial source-assisted supervision. Unreviewed cliff, snow,
water and ground remain ignored by this addition. It is not a complete fence
outline or exhaustively corrected truth. The subset split is inherited from
the existing training layout; validation and test maps are unaffected.

`scripts/prepare_encounter_training.py` can prepare this one-layout subset with
the included review and split files. Merge the resulting sample into a copy
of an existing training manifest by its ID, preserving all other samples and
splits. Historical datasets and trial records must remain unchanged.
