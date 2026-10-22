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
python fast_auto_wall.py "path/to/map.png" --resolution 0
```

Omit the image argument to open a file picker through **Open map**. On Windows,
`run-fast.bat` launches this workspace and accepts a dropped image path.

The full editor uses the original `requirements.txt`. Launch `python auto_wall.py`,
choose **Auto setup**, and use **Continue in wall editor** for individual wall and
door edits. The standalone Fast Setup launcher needs only `requirements-fast.txt`;
its review tools toggle layers and remove/undo individual door or light candidates.

## Use the controls

1. Choose **Original pixels** for the most detailed outlines. The 2048 px setting
   reduces detection work; every setting still exports the original bitmap.
2. Set **Pixels / grid** to the original map's grid spacing. The default is 70; an
   upscaled map may need a larger value. This changes VTT units, not detection pixels.
3. **Walls:** Auto stone/gray estimates the dominant neutral masonry tone. Sampled
   color lets you pick different wall materials. Edge mode is a fallback for line art.
   Increase Minimum area to exclude small objects; reduce Simplify for finer corners.
4. **Doors:** pick the door-marker color, then adjust length, hue, and rectangularity.
   Keep **Require nearby wall jambs** enabled for conservative detection. Candidates
   must have wall pixels along their axis at both ends and connect to actual wall
   geometry. Unsupported candidates are skipped without cutting holes in the walls.
5. **Lights:** brightness, saturation, and area identify compact bright regions.
   Door footprints are excluded so their highlights do not become torches. Range is
   in grid squares. Toggle layers to inspect the output, and remove false candidates.
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
python fast_auto_wall.py "map.png" --resolution 0 --grid-size 70 --export "map.dd2vtt" --preview "preview.bmp" --benchmark 5 --report "timings.json"
```

`--settings settings.json` accepts overrides named after `FastSettings` fields.
For example, `{"wall_mode":"color","wall_color":[135,120,100],"door_rectangularity":0.85}`.
CLI resolution/grid arguments override the JSON values. Preview output is an
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
wall/door/light pass is about **0.46 seconds**, with **1508 wall segments, 68 doors,
and 30 lights** on this map. A 2048 px pass is about **0.15 seconds**. Exact timings
vary with the map, settings, machine load, and resolution; CLI reports retain each run.

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

```shell
python -m pytest -q
python -m ruff check fast_auto_wall.py src/core/fast_pipeline.py src/gui/fast_setup.py benchmarks/compare_detector.py tests/conftest.py tests/test_fast_pipeline.py tests/test_fast_setup.py tests/test_editor_integration.py
python -m compileall -q src fast_auto_wall.py auto_wall.py benchmarks/compare_detector.py
git diff --check
```

This is conservative visual detection, not a general semantic model. Default stone
and orange-door settings match this artwork; other palettes, textured walls, hidden
doors, and unmarked openings need different settings or manual editing. Same-colored
rectangular decorations with two supporting walls can still resemble doors. The
default minimum wall area may omit small isolated pillars. Light detection finds
bright spots; it does not infer invisible light sources or automatically light lava.

UVTT structure, geometry, image preservation, and local round trips were validated.
A live Foundry/Roll20 import was not run because a VTT installation was not available.
Confirm grid spacing and review the overlay before using the file in a campaign.
