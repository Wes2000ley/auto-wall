# Faster first load with an isolated ML environment

Use a dedicated virtual environment for Fast Setup and its optional local ML
models. In the measured Windows installation, sharing all system packages made
DINO initialization substantially slower. An isolated environment kept the same
model predictions and avoided unrelated package discovery and imports.

This is an environment change. It does not replace the model, simplify its
architecture, patch Transformers, or reduce the map's detection resolution.

## Windows setup

From the repository directory, create a new environment without
`--system-site-packages`. The following CUDA installation matches the tested
Python 3.13 / RTX 3080 runtime. For other hardware, use the appropriate official
[PyTorch installation command](https://pytorch.org/get-started/locally/).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: -r requirements-centerline.txt -r requirements-ml-dino.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe fast_auto_wall.py "map.png" --settings profiles/ml-light-centers.json --model models/battlemap-dino-lights-v2.safetensors --ml-device auto
```

`requirements-centerline.txt` includes the ordinary Fast Setup dependencies and
its optional skeletonization library. Both the C++ centerline tracer and Python
fallback remain available; build the tracer with `scripts/build_native.py` if
needed. The full legacy editor has its own `requirements.txt`; retain its
existing environment when using the smaller standalone Fast Setup environment.

The app reads the selected local safetensors file. It does not download model
weights during startup. The semantic v2 model remains experimental; see its
[accuracy and training notes](SEMANTIC_LIGHT_RESEARCH.md).

## What the startup profile found

The original training environment includes system packages. A fresh-process
profile spent most of its time importing Transformers' DINO implementation.
The installed Transformers 5.8.0 eagerly calls
`importlib.metadata.packages_distributions()` during import; that call enumerates
installed distribution metadata. Its import module accounted for about 6.36
seconds of self time in the diagnostic trace. Optional packages also enlarged
the downstream import tree. Model construction itself took about 0.17 seconds,
and reading/validating/assigning the weights took well under 0.1 seconds.

Three alternating fresh-process stage profiles used the same Python version,
PyTorch 2.11.0+cu128, Transformers 5.8.0, model, and eight CPU threads:

| Stage | Shared environment, median | Isolated environment, median |
| --- | ---: | ---: |
| Import PyTorch | 1.659 s | 1.507 s |
| Import DINO through Transformers | 11.672 s | 2.778 s |
| Entire CPU model-load profile | 13.569 s | 4.513 s |

These stages exclude CUDA transfer, inference, map decoding, and Qt. The first
isolated pass immediately after installation took 11.403 seconds; the next two
took 4.513 and 4.371 seconds. All three are retained. A fresh Python process does
not imply a cold operating-system file cache, and these measurements are local
observations rather than a universal startup guarantee. The package counts were
546 shared and 42 isolated at this profiling stage; centerline and test packages
were added to the isolated environment afterward.

## Output compatibility

Both environments loaded all four bundled model formats on CPU and CUDA, then
predicted 35×43, 384×384, and 385×511 images. All 60 class, confidence, and light
heatmap arrays were bit-for-bit identical across environments.

The normal pipeline and UVTT serialization were also compared on the wooden
camp, stone camp, temple, ship, Demon Temple, and Astral maps. Walls, doors,
lights, transparent-barrier previews, and serialized geometry were identical.
Every export preserved the original PNG bytes. This establishes compatibility
on those checks; it does not establish accuracy on unseen artwork.

## Actual Qt worker measurements

After installing centerline support and matching the existing Qt versions, the
isolated environment contained 49 packages. Three alternating fresh-process
runs per environment opened the real Qt dialog, detected both camps, changed
the light score from 0.35 to 0.60 and back, and exported both maps.

| Detection worker | Shared environment | Isolated environment |
| --- | ---: | ---: |
| Median first detection | 16.508 s | 5.452 s |
| First-detection range | 16.334–16.557 s | 5.447–5.900 s |
| Warm detection range | 0.429–0.524 s | 0.429–0.549 s |

First detection was 3.03 times faster, a 66.97% reduction, in this comparison.
These worker timings include ML imports, model initialization, CUDA transfer,
inference, and geometry. They exclude Qt startup, map decoding, painting, and
export writing. No training was running during the measurement. Warm detection
did not show a meaningful speed improvement.

All twelve serialized Qt exports matched their paired environment's output
exactly. Both maps kept four doors and the expected 9/12 lights. Changing the
score invalidated export until recomputation completed; walls and transparent
barriers stayed unchanged. Both captured previews were inspected. The final
environment also repeated all 60 raw-array and six-map compatibility checks.

The full existing environment passed `python -m pytest -q`: 161 tests with
three dependency deprecation warnings. The isolated environment passed all
151 Fast Setup, ML, geometry, and native-tracer tests without warnings. It does
not contain the full legacy editor's optional dependencies; that editor was
tested in the retained existing environment. `pip check` reported no broken
requirements. Ruff and compilation checks passed for the changed training code.
Live Foundry import, other operating systems/GPUs, and release packaging were
not run.

[The machine-readable report](ML_RUNTIME_RESULTS.json) preserves every timing,
the final installed versions, model/source hashes, output checks, and limits.
To measure first and warm headless detections in your environment, run this
from a fresh shell process, using your own map path:

```powershell
.\.venv\Scripts\python.exe fast_auto_wall.py "map.png" --settings profiles/ml-light-centers.json --model models/battlemap-dino-lights-v2.safetensors --ml-device auto --benchmark 6 --report ml-timing.json
```

The first entry includes model initialization; later entries use its in-process
cache. This headless measurement has a different boundary from the Qt check.
