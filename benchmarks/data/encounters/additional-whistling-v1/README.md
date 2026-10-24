# Whistling Stones rope-rail training patches

This training-only extension adds 1,187 fence-material pixels and 2,468 ground
pixels, all previously unknown. Seven one-pixel rope cores and ten narrow wood
post interiors were traced from the native original. Six adjacent path/ground
patches provide negatives. All earlier supervised pixels and the frozen layout
split remain unchanged. Suspended ropes are not filled down to the ground.

The primary reviewer inspected enlarged original/raster pairs and coordinate
grids. Two drafts were rejected before training: some rope lines touched flag
fabric or missed the cable, and some post traces reached rock or the wrong X
coordinate. Their hashes and reasons remain in review.json. Only the corrected
third draft was admitted. No model prediction or proposal supplied a label.

These are selected partial source-assisted patches, not exhaustive independently
human-corrected truth. Unreviewed pixels remain unknown. Flags, cliff/rock edges,
plants and the tent are not newly labelled as structures. Nine partial passage
rays are training-fit diagnostics, not held-out evaluation. Source, annotation,
raster and reviewed-preview hashes preserve provenance.

Prepare this subset with scripts/prepare_encounter_training.py and its review
record into a new directory, then replace only this training sample in a copied
manifest. Keep validation and final-test inputs and labels unchanged.
