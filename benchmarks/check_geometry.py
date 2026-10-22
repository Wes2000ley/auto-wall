"""Measure exported UVTT doors, lights, and visibility against manual annotations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_instances(distances, tolerance):
    """One-to-one matching: maximize valid matches, then minimize position error."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or not np.isfinite(distances).all():
        raise ValueError("Instance distances must be a finite matrix.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Instance tolerance must be positive.")
    valid = distances <= tolerance
    penalty = 2 * (max(distances.shape) + 1)
    rows, columns = linear_sum_assignment(np.where(valid, distances / tolerance, penalty))
    matches = [(int(row), int(column), float(distances[row, column]))
               for row, column in zip(rows, columns) if valid[row, column]]
    count = len(matches)
    truth_count, prediction_count = distances.shape
    return {"matched": count, "truth_count": truth_count, "prediction_count": prediction_count,
            "precision": count / prediction_count if prediction_count else None,
            "recall": count / truth_count if truth_count else None,
            "mean_error": float(np.mean([error for _, _, error in matches])) if count else None,
            "matches": matches,
            "missed": sorted(set(range(truth_count)) - {row for row, _, _ in matches}),
            "extra": sorted(set(range(prediction_count)) - {column for _, column, _ in matches})}


def ray_blocked(start, end, segments):
    """Include endpoint touches and collinear overlaps, as a vision wall does."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all()
            or not np.isfinite(segments).all() or np.linalg.norm(end - start) == 0):
        raise ValueError("A sight line needs two distinct finite points and finite wall segments.")
    if not len(segments):
        return False
    a, b = segments[:, 0], segments[:, 1]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    first = cross(end - start, a - start) * cross(end - start, b - start)
    second = cross(b - a, start - a) * cross(b - a, end - a)
    overlap = ((np.minimum(a, b) <= np.maximum(start, end) + 1e-8)
               & (np.maximum(a, b) >= np.minimum(start, end) - 1e-8)).all(axis=1)
    return bool(np.any(overlap & (first <= 1e-8) & (second <= 1e-8)))


def evaluate_geometry(data, specification):
    if specification.get("format") != "auto-wall-geometry-checks-v1":
        raise ValueError("Unknown geometry annotation format.")
    reference = np.asarray(specification["reference_size"], dtype=float)
    size = data["resolution"]["map_size"]
    size = np.array([size["x"], size["y"]], dtype=float)
    if reference.shape != (2,) or not np.isfinite([reference, size]).all() or min(*reference, *size) <= 0:
        raise ValueError("Map and reference sizes must be positive and finite.")
    scale = reference / size

    def point(p):
        return np.array([p["x"], p["y"]]) * scale

    walls = [[point(a), point(b)] for a, b in data["line_of_sight"] + data.get("objects_line_of_sight", [])]
    doors = np.asarray([[point(a), point(b)] for a, b in
                        [portal["bounds"] for portal in data["portals"]]], dtype=float).reshape(-1, 2, 2)
    lights = np.asarray([point(light["position"]) for light in data["lights"]], dtype=float).reshape(-1, 2)
    expected_doors = np.asarray([item["bounds"] for item in specification["doors"]], dtype=float).reshape(-1, 2, 2)
    expected_lights = np.asarray([item["position"] for item in specification["lights"]], dtype=float).reshape(-1, 2)
    # Endpoint distance checks orientation and span, not just a nearby center.
    direct = np.linalg.norm(expected_doors[:, None] - doors[None], axis=-1).max(axis=-1)
    reverse = np.linalg.norm(expected_doors[:, None] - doors[None, :, ::-1], axis=-1).max(axis=-1)
    door_metrics = match_instances(np.minimum(direct, reverse), specification.get("door_tolerance", 8))
    light_metrics = match_instances(np.linalg.norm(expected_lights[:, None] - lights[None], axis=-1),
                                   specification.get("light_tolerance", 6))
    for name, metrics in (("doors", door_metrics), ("lights", light_metrics)):
        complete = specification.get("complete_instances", {}).get(name, False)
        if type(complete) is not bool:
            raise ValueError("Instance completeness must be boolean.")
        metrics["complete_annotation"] = complete
        if not complete:
            # Unmatched detections may be real objects outside a partial label set.
            metrics["precision"] = None
    rays = []
    for item in specification["rays"]:
        if type(item["blocked"]) is not bool or type(item.get("closed_doors", False)) is not bool:
            raise ValueError("Ray states must be boolean.")
        blockers = walls + doors.tolist() if item.get("closed_doors", False) else walls
        blocked = ray_blocked(item["from"], item["to"], blockers)
        rays.append({**item, "actual_blocked": blocked, "passed": blocked == item["blocked"]})
    return {"annotation": specification.get("description", ""), "units": "annotation reference pixels",
            "doors": door_metrics, "lights": light_metrics, "rays": rays,
            "rays_passed": sum(ray["passed"] for ray in rays), "rays_total": len(rays),
            "wall_segments": len(walls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uvtt", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate_geometry(json.loads(args.uvtt.read_text(encoding="utf-8")),
                               json.loads(args.annotations.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: result[name] for name in ("doors", "lights", "rays_passed", "rays_total")}))


if __name__ == "__main__":
    main()
