"""Compare the optimized edge detector with an immutable upstream revision.

Run from a clone containing the upstream commit. The input map stays local.
Only timing, contour counts, and point-set equality are written to the report.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical(contours):
    return sorted(tuple(sorted(map(tuple, contour.reshape(-1, 2).tolist()))) for contour in contours)


def main():
    from src.core.fast_pipeline import load_map
    from src.wall_detection.detector import detect_walls

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--revision", default="c11b8fa7ad5923d742722a063b3d4385f68527d8")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1500, 2048, 0], help="Longest edges; 0 is original")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or any(size < 0 for size in args.sizes):
        parser.error("Runs must be positive and sizes nonnegative.")
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.revision + "^{commit}"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(
        ["git", "show", revision + ":src/wall_detection/detector.py"], cwd=ROOT, text=True, encoding="utf-8")
    legacy = types.ModuleType("src.wall_detection._benchmark_upstream")
    legacy.__package__ = "src.wall_detection"
    exec(compile(source, "<upstream detector>", "exec"), legacy.__dict__)
    image, _ = load_map(args.image)
    report = {"upstream_sha": revision, "opencv": cv2.__version__, "threads": cv2.getNumThreads(), "comparisons": []}
    for size in args.sizes:
        ratio = min(1.0, size / max(image.shape[:2])) if size else 1.0
        working = cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
                             interpolation=cv2.INTER_AREA) if ratio < 1 else image
        row = {"width": working.shape[1], "height": working.shape[0]}
        results = []
        for name, detector in (("upstream", legacy.detect_walls), ("optimized", detect_walls)):
            durations = []
            for _ in range(args.runs):
                started = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    contours = detector(working)
                durations.append(time.perf_counter() - started)
            row[name + "_seconds"] = durations
            results.append(contours)
        row["identical_contour_points"] = canonical(results[0]) == canonical(results[1])
        if not row["identical_contour_points"]:
            raise AssertionError("Optimized detector changed the contour point sets.")
        row["contours"] = len(results[0])
        row["points"] = sum(len(contour) for contour in results[0])
        row["median_speedup"] = statistics.median(row["upstream_seconds"]) / statistics.median(row["optimized_seconds"])
        report["comparisons"].append(row)
        print(json.dumps(row), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
