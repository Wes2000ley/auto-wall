"""Compare optional C++ centerlines with Python, including exact output parity.

Image decoding and one warmup per backend are excluded. Measurements alternate
backend order each round. Source images stay local and are never copied.
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare(image, settings, runs):
    from src.core.fast_pipeline import detect_map

    previous = os.environ.get("AUTO_WALL_GEOMETRY_BACKEND")
    timings = {"python": [], "cpp": []}
    reference = None
    try:
        for iteration in range(runs + 1):
            order = ("python", "cpp") if iteration % 2 == 0 else ("cpp", "python")
            for backend in order:
                os.environ["AUTO_WALL_GEOMETRY_BACKEND"] = backend
                result = detect_map(image, settings)
                assert result.geometry_backend == backend
                if reference is None:
                    reference = result
                else:
                    np.testing.assert_array_equal(result.walls, reference.walls)
                    np.testing.assert_array_equal(result.doors, reference.doors)
                    assert result.lights == reference.lights
                    assert result.warnings == reference.warnings
                    assert result.working_size == reference.working_size
                if iteration:
                    timings[backend].append(result.timings)
    finally:
        if previous is None:
            os.environ.pop("AUTO_WALL_GEOMETRY_BACKEND", None)
        else:
            os.environ["AUTO_WALL_GEOMETRY_BACKEND"] = previous
    medians = {backend: {stage: statistics.median(run[stage] for run in values)
                         for stage in values[0]} for backend, values in timings.items()}
    return {"image_size": [reference.width, reference.height], "working_size": reference.working_size,
            "walls": len(reference.walls), "doors": len(reference.doors), "lights": len(reference.lights),
            "identical_geometry_and_lights": True, "runs": timings, "median_seconds": medians,
            "total_speedup": medians["python"]["total"] / medians["cpp"]["total"],
            "wall_stage_speedup": medians["python"]["walls"] / medians["cpp"]["walls"]}


def main():
    from src.core.detection_profiles import PRESETS, load_settings, preset_settings
    from src.core.fast_pipeline import load_map
    from src.core.native_geometry import native_available

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--preset", choices=PRESETS, default="foundry")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.resolution < 0:
        parser.error("Runs must be positive and resolution nonnegative.")
    if not native_available():
        parser.error("Build the C++ backend first: python scripts/build_native.py")
    settings = preset_settings(args.preset)
    if args.settings:
        settings = load_settings(args.settings, settings)
    settings = replace(settings, wall_geometry="centerline", max_dimension=args.resolution)
    image, _ = load_map(args.image)
    report = compare(image, settings, args.runs)
    report.update(settings=asdict(settings), opencv=cv2.__version__, threads=cv2.getNumThreads())
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.report:
        args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
