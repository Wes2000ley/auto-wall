# Additional real guardrail supervision

This versioned extension adds eleven narrow visible wood interiors from the
Gullthorn Blind platform guardrail and two clear negative patches: platform
floor and a crate surface. The tree, ladder, lantern and other props are not
barriers. Native source coordinates use the original 1568 by 1003 image.

The primary reviewer inspected the native map, a coordinate-labelled 5x crop,
and the raster overlay. Exactly 1,622 previously unknown pixels become class 4
and 884 become background. All earlier supervised pixels remain unchanged.
The full annotation inherits earlier partial regions; they are not reclassified
as exhaustive corrected truth. No prediction provided training labels.

These selected wood patches do not define complete guardrail geometry. Two
passage checks cross the full selected front-left railing: sight remains open
and movement is blocked. Re-extracting perfect partial labels passes those
checks; that is an extraction check, not a claim of model accuracy.

The layout retains its existing training split. Source, parent annotation,
new annotation, raster and review-preview hashes are recorded separately.
Scoped Git attributes preserve the reviewed JSON bytes across checkouts.
Prepare this subset with `scripts/prepare_encounter_training.py`, then replace
only its matching sample in a new manifest. Preserve historical datasets and
all other samples and layout splits.
