# Whistling Stones rope-side background patches

This training-only extension adds 3,212 background pixels in fourteen narrow
strips beside the seven previously reviewed rope cores. The original material
cores, posts, all earlier supervised pixels, nine passage checks and frozen
layout split remain unchanged. Unreviewed pixels stay unknown (255).

The primary reviewer compared each native source/raster pair at five times
native size. The first draft was rejected because a strip could touch the tall
supporting post beside rope seven. The accepted revision excludes that beginning;
its first six pairs retain the exact reviewed coordinates. Review metadata keeps
both draft hashes and the original/revised close-up hashes. Source images and
model predictions do not get overwritten or relabelled by this experiment.

These are partial source-assisted training patches, not exhaustive independent
human ground truth. Only visible non-barrier ground, vegetation and rock receive
new background labels. Prepare into a new dataset directory with its matching
review record; never mutate an existing training manifest or mask.
