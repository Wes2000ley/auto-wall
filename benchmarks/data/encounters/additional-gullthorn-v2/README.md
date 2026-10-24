# Guardrail boundary negatives

This version extends additional-gullthorn-v1 with a clear platform-floor polygon
and two crate interiors. The primary reviewer inspected the native coordinate
crop and the 5x raster overlay. All 2,697 changed pixels were previously unknown
and become background; every earlier supervised pixel, including all 1,622
guardrail pixels, stays unchanged. Two open-movement checks cover floor and crate.

The v14 training prediction spilled into adjacent floor and props. That failure
motivated the source review; predictions did not supply training labels. These
are selected source-assisted patches, not exhaustive independently corrected
truth. The tree, ladder, lantern and uncertain edges remain unlabelled here.

The source and the training split are unchanged. Parent, annotation, raster,
source-crop and accepted-preview hashes preserve provenance. Earlier versions
remain available to reproduce recorded trials. Prepare this subset separately
and replace only the matching training sample in a new manifest.
