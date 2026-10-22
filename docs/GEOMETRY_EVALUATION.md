# Check exported structures

`benchmarks/check_geometry.py` checks the actual UVTT output against manually
reviewed door spans, physical light-source positions, and selected sight lines.
It requires NumPy and SciPy, both included in `requirements-ml.txt`.

```shell
python benchmarks/check_geometry.py map.dd2vtt annotations.json geometry-results.json
```

Annotations use pixels at a stated reference size. The evaluator scales exported
grid coordinates into those pixels, so checks remain valid across export grid
sizes and proportional image upscales. Use the same aspect ratio and origin as
the map. This example describes a 1024 by 768 image:

```json
{
  "format": "auto-wall-geometry-checks-v1",
  "description": "Reviewed door, torch, and visibility checks",
  "reference_size": [1024, 768],
  "door_tolerance": 8,
  "light_tolerance": 6,
  "complete_instances": {"doors": false, "lights": false},
  "doors": [{"bounds": [[100, 200], [100, 240]]}],
  "lights": [{"position": [300, 400]}],
  "rays": [
    {"name": "open doorway", "from": [80, 220], "to": [120, 220], "blocked": false},
    {"name": "closed doorway", "from": [80, 220], "to": [120, 220], "blocked": true, "closed_doors": true},
    {"name": "solid wall", "from": [80, 180], "to": [120, 180], "blocked": true}
  ]
}
```

Door matching compares both endpoints, allowing their order to reverse. Both
must fall within the tolerance: a short fragment near a door's center does not
count as the whole door. Light matching uses center distance. Matching is
one-to-one, maximizing the valid match count before minimizing position error.
Duplicate detections cannot each match the same object.

Set `complete_instances.doors` or `.lights` to `true` only after annotating **every
instance of that class on the map**. Otherwise precision is `null`; unmatched
detections may be real objects outside the partial list. Complete lists report
precision, recall, misses, and extras. An absent denominator is `null`, not zero.
Matching tolerance is part of the evaluation protocol; do not increase it after
seeing a failure merely to improve the score.

Rays intersect both UVTT wall arrays. `closed_doors: true` adds all exported
portals as blockers; otherwise portals are open for that check. Endpoint touches
and collinear overlaps count as blocked. This is a straight-segment geometry
check, not a simulation of every Foundry importer, window type, or lighting rule.
Include both open and closed checks through each doorway, wall-crossing checks,
walkable-floor checks, and checks through transparent railings.

Add `"geometry": "map-geometry.json"` to a validation sample in the training
manifest to use these annotations for checkpoint selection. Paths are relative
to that manifest, or absolute. The loader reads and hashes the annotations at
the beginning of the run. With `--selection geometry`, the trainer averages:

- The fraction of correctly blocked/open rays for each map with rays.
- Door F0.5 for each complete door list with any annotated or predicted objects.
- Light F0.5 for each complete light list with any annotated or predicted objects.

F0.5 is `5 TP / (5 TP + 4 FP + FN)`. The final selection score is 60% of that
mean plus 40% mean foreground pixel IoU. Raw metrics remain in `training.json`;
inspect them individually because one aggregate can conceal a regression.

Sparse probes establish only the cases they cover. They do not establish global
wall continuity, perfect curves, complete instance accuracy, or successful live
Foundry import. Review full overlays and exported geometry, keep related map
layouts in one split, and reserve new layouts for evaluation after selection.
