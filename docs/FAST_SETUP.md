# Fast Setup

Fast Setup adds automatic walls, doors, and torch lights to the existing PyQt desktop
application. It runs local OpenCV kernels in a background worker and preserves the
original map resolution when exporting. No model download, cloud service, CUDA
installation, or additional native compiler is needed.

## Run it

Use Python 3.10 or newer. The measured environment used Python 3.13, OpenCV 4.11.0,
NumPy 2.2.2, and Qt 6.11.0 on Windows.

```shell
python -m pip install -r requirements-fast.txt
python fast_auto_wall.py "path/to/map.png" --schema "profiles/foundry-v2.json"
```

Omit the image argument to open a file picker through **Open map**. On Windows,
`run-fast.bat` launches this workspace and accepts a dropped image path.

The full editor uses the original `requirements.txt`. Launch `python auto_wall.py`,
choose **Auto setup**, and use **Continue in wall editor** for individual wall and
door edits. The standalone Fast Setup launcher needs only `requirements-fast.txt`;
its review tools toggle layers and remove/undo individual door or light candidates.

For optional single centerlines, install `requirements-centerline.txt`. This adds
scikit-image's skeletonization; outlines need only the fast requirements. A missing
optional dependency produces an actionable error without changing existing results.

## Palettes and saved settings

The default **Foundry schema · tolerant** preset uses wall `#A0A0A0`, door `#702090`,
and void `#000000`. **Import JSON** accepts a map-generation schema containing
`reserved_colors` or `palette`, or saved detector settings. Only color data is read
from generation schemas. Prose, prompts, paths, and URLs are never executed.
Conflicting or identical structural colors and malformed settings are rejected.

**Foundry schema · exact** uses zero RGB tolerance for both walls and doors, with
nearest-neighbor downsampling to preserve palette values. Use it only when the image
actually contains the reserved RGB values. The reviewed generated maps mostly had
no exact purple pixels despite requesting that color; tolerant matching is the
practical default. Exact mode can still miss features smaller than a working pixel.

**Gray walls · orange doors** (`--preset demon`) retains the original Demon Temple
workflow. Other palettes can be imported or sampled directly from the map. The
schema is a starting palette, not evidence that every similarly colored pixel is
structural. **Save settings** retains all detection controls for another session.

## Use the controls

1. Choose **Original pixels** for the most detailed outlines. The 2048 px setting
   reduces detection work; every setting still exports the original bitmap.
2. Set **Pixels / grid** to the original map's grid spacing. The default is 70; an
   upscaled map may need a larger value. This changes VTT units, not detection pixels.
3. **Walls:** sampled color is the default. Auto stone/gray estimates the dominant
   neutral masonry tone. Edge mode is a fallback for line art.
   Increase Minimum area to exclude small objects; reduce Simplify for finer corners.
   Gap repair defaults to zero to preserve intentional openings. Increase it only
   when inspecting and repairing artwork seams. Exact void pixels remain excluded.
   **Centerlines** trace connected skeleton branches and loops through wall masses;
   **Outlines** follow their boundaries. Centerlines suit thick schematic masonry,
   but cracks and textured materials can create unwanted branches.
4. **Doors:** pick the door-marker color, then adjust length, hue, and rectangularity.
   Keep **Require nearby wall jambs** enabled for conservative detection. Candidates
   must have wall pixels along their axis at both ends and connect to actual wall
   geometry. Unsupported candidates are skipped without cutting holes in the walls.
   Hue matching derives brightness from the chosen sample, including dark purple
   and brown. RGB matching offers strict palette bounds. Seam repair groups shaded
   gate leaves without overwriting wall pixels; set it to zero for exact artwork.
5. **Lights:** choose warm, cool, sampled-hue, or all-bright-region sources. Colored
   sources are grouped before their bright cores are checked, so separate highlights
   on one flame do not become multiple lights. Source area, bright fraction, and
   bounding-box fill reject small glints, sparse markings, and large emissive regions.
   Door footprints are excluded. These remain candidates: a screen, magical eye,
   or bright prop can still look like a light. Range is in grid squares. Toggle layers
   to inspect the output, and remove false candidates.
6. Scroll to zoom, drag to pan, and choose **Export UVTT**. The `.dd2vtt` file includes
   walls, closed door portals, lights, and the original map image.

Lengths and areas use a 2048-pixel reference edge: lengths scale linearly with
detection resolution and areas quadratically. Simplification uses the same reference.
This makes controls behave consistently when changing quality. Small features can
still disappear when downsampling, so use original pixels for a final detailed pass.

Settings changes debounce for 200 ms. One detection job runs at a time; a newer
request supersedes the old job. Cancellation happens between native processing
stages. Export runs in another worker and writes atomically. Closing waits for active
workers to finish safely. Panning and zooming use cached pixels and vector overlays.

## Headless export and timing

```shell
python fast_auto_wall.py "map.png" --schema "profiles/foundry-v2.json" --resolution 0 --grid-size 70 --export "map.dd2vtt" --preview "preview.bmp" --benchmark 5 --report "timings.json"
```

`--settings settings.json` accepts overrides named after `FastSettings` fields.
For example, `{"wall_mode":"color","wall_color":[135,120,100],"door_rectangularity":0.85}`.
Precedence is preset → schema palette → settings JSON → explicit CLI resolution,
grid, and geometry arguments. `--geometry centerline` selects skeleton geometry.
Preview output is an
uncompressed BMP. Existing PNG source bytes are embedded unchanged; other image
formats are converted to PNG with compression disabled.

## Measured performance

Measured on 2026-10-22 with a Ryzen 7 7800X3D, OpenCV 4.11.0, and 16 OpenCV threads.
The input was a 5792 × 4344 (25.16 MP) Demon Temple map. The map is not included in
the repository. Disk decoding, preview rendering, and export are excluded from
detection timings and reported separately by the CLI.

The existing edge detector was compared against upstream commit
`c11b8fa7ad5923d742722a063b3d4385f68527d8`, with default parameters and zero edge margin.
Each cell is the median of three runs on the same pixels and machine:

| Longest edge | Upstream | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| 1500 px | 176.3 ms | 28.4 ms | 6.21× |
| 2048 px | 296.2 ms | 46.6 ms | 6.35× |
| 5792 px | 10,424.6 ms | 223.3 ms | 46.68× |

Contour point sets match at all three resolutions (including all 88,124 points at
full resolution). The old implementation scanned the whole image for each connected
component. A single label lookup and contour pass remove that multiplicative cost.
The nonzero edge-margin branch retains its previous algorithm.

The new stone/color pipeline is a separate algorithm, so its times are not an
equivalent-output comparison with the edge detector. The full-resolution combined
wall/door/light pass originally measured **0.46 seconds**. The current five-run
median is **0.347 seconds**, with **1508 wall segments, 68 doors, and 30 lights**,
using `--preset demon --resolution 0`. The earlier 2048 px pass measured **0.15
seconds**. These runs were made separately and are not an isolated speedup study
of the new profile changes. Exact timings
vary with the map, settings, machine load, and resolution; CLI reports retain each run.

On a 1448 × 1086 purple-door temple, outlines use 2220 segments while centerlines
use 1140, with the same 68 portals and 30 torch candidates. Centerline detection's
five-run median was **0.174 seconds**; its first run was **0.685 seconds**, including
the lazy scikit-image import. Additional maps at a maximum 2048 px working edge
took **24–109 ms** median for outlines, excluding decode/export. Counts and timings
measure produced geometry, not ground-truth accuracy. See the accuracy assessment.

To reproduce the old/new comparison (requires the full editor dependencies and Git
history containing the upstream commit):

```shell
python benchmarks/compare_detector.py "map.png" --runs 3 --report "comparison.json"
```

The new pipeline and GUI do not use GPU compute. OpenCV already supplies native C++
image kernels; eliminating repeated scans gave a large gain without a second
compiler toolchain, GPU transfers, or hardware-specific runtime requirements.

## Quality and validation

For the supplied map, all 68 doorway crops and 30 torch crops were visually reviewed.
The whole-map wall overlay was checked for furniture highlights, floor markings,
blood, and lava outlines. Door spans remain horizontal/vertical and all 136 endpoints
touch wall geometry to floating-point precision. Doorway clipping preserves jamb
caps, which avoids gaps around closed doors and leaves the center clear when opened.
The embedded PNG was verified byte-for-byte against the source.

Automated tests cover false door candidates, two-sided jamb support, rotated door
clipping, scale conversion, source immutability, cancellation, stale worker results,
removal/undo, asynchronous export, editor handoff, and map-cache collisions. They
also cover the existing light-position drift and WebP re-encoding regressions.
Additional coverage checks palette imports, alternate colors and dark doors, exact
RGB behavior, preserved openings/voids, wide shaded gates, source grouping, connected
centerline branches/loops, rejected portal restoration, and GUI profile round trips.

```shell
python -m pytest -q
python -m ruff check fast_auto_wall.py src/core/fast_pipeline.py src/core/detection_profiles.py src/core/wall_geometry.py src/gui/fast_setup.py benchmarks/compare_detector.py tests/conftest.py tests/test_fast_pipeline.py tests/test_fast_setup.py tests/test_detection_profiles.py tests/test_wall_geometry.py tests/test_editor_integration.py
python -m compileall -q src fast_auto_wall.py auto_wall.py benchmarks/compare_detector.py
git diff --check
```

This is conservative visual detection, not a general semantic model. The stone
and orange-door preset matches the original artwork; other palettes, textured walls, hidden
doors, and unmarked openings need different settings or manual editing. Same-colored
rectangular decorations with two supporting walls can still resemble doors. The
default minimum wall area may omit small isolated pillars. A wooden palisade sharing
colors with furniture cannot be reliably separated by color alone. Light detection
does not infer invisible light sources or automatically light lava.

UVTT structure, geometry, image preservation, and local round trips were validated.
A live Foundry/Roll20 import was not run because a VTT installation was not available.
Confirm grid spacing and review the overlay before using the file in a campaign.
