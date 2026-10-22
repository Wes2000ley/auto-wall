"""Launch Fast Setup, or detect and export a map without starting the GUI."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from src.core.fast_pipeline import detect_map, export_image_bytes, load_map, to_uvtt, write_uvtt
from src.core.detection_profiles import PRESETS, load_settings, preset_settings, settings_from_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Local map image")
    parser.add_argument("--export", type=Path, help="Write a .dd2vtt or .uvtt file and exit")
    parser.add_argument("--preview", type=Path, help="Save an uncompressed BMP overlay (headless only)")
    parser.add_argument("--settings", type=Path, help="JSON object containing FastSettings overrides")
    parser.add_argument("--preset", choices=PRESETS, default="foundry", help="Built-in palette (default: foundry)")
    parser.add_argument("--schema", type=Path, help="Map-style JSON containing wall, door, and void colors")
    parser.add_argument("--geometry", choices=("outline", "centerline"), help="Wall representation")
    parser.add_argument("--resolution", type=int, help="Longest detection edge; 0 uses original pixels")
    parser.add_argument("--grid-size", type=float, help="Original-image pixels per grid square")
    parser.add_argument("--benchmark", type=int, metavar="RUNS", help="Run repeated headless detections")
    parser.add_argument("--report", type=Path, help="Save timing and object counts as JSON")
    args = parser.parse_args(argv)
    headless = args.export is not None or args.benchmark is not None
    if (headless or args.preview or args.report) and not args.image:
        parser.error("An image path is required for headless processing.")
    if (args.preview or args.report) and not headless:
        parser.error("--preview and --report require --export or --benchmark.")
    if args.preview and args.preview.suffix.lower() != ".bmp":
        parser.error("Use a .bmp preview path (uncompressed).")
    if args.benchmark is not None and args.benchmark < 1:
        parser.error("Benchmark runs must be positive.")
    try:
        settings = preset_settings(args.preset)
        if args.schema:
            settings = load_settings(args.schema, settings)
        if args.settings:
            settings = load_settings(args.settings, settings)
        values = {}
        if args.resolution is not None:
            values["max_dimension"] = args.resolution
        if args.grid_size is not None:
            values["grid_size"] = args.grid_size
        if args.geometry:
            values["wall_geometry"] = args.geometry
        settings = settings_from_data(values, settings)
        if headless:
            return process(args, settings)
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QApplication
        from src.gui.fast_setup import FastSetupDialog

        app = QApplication([sys.argv[0]])
        app.setStyle("Fusion")
        palette = QPalette()
        for role, color in ((QPalette.ColorRole.Window, "#23272e"), (QPalette.ColorRole.WindowText, "#eef1f5"),
                            (QPalette.ColorRole.Base, "#181c22"), (QPalette.ColorRole.Text, "#eef1f5"),
                            (QPalette.ColorRole.Button, "#363d48"), (QPalette.ColorRole.ButtonText, "#eef1f5"),
                            (QPalette.ColorRole.Highlight, "#287ab5"), (QPalette.ColorRole.HighlightedText, "#ffffff")):
            palette.setColor(role, QColor(color))
        app.setPalette(palette)
        window = FastSetupDialog(path=args.image)
        window.apply_settings(settings)
        window.show()
        return app.exec()
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"Auto-Wall: {error}", file=sys.stderr)
        return 1


def process(args, settings):
    import cv2
    import numpy as np

    tick = time.perf_counter()
    image, source = load_map(args.image)
    decode_seconds = time.perf_counter() - tick
    runs = []
    for _ in range(args.benchmark or 1):
        result = detect_map(image, settings)
        runs.append(result.timings)
    export_seconds = None
    if args.export:
        tick = time.perf_counter()
        write_uvtt(args.export, to_uvtt(result, settings.grid_size, export_image_bytes(image, source)))
        export_seconds = time.perf_counter() - tick
    if args.preview:
        ratio = min(1.0, 2048 / max(result.width, result.height))
        canvas = cv2.resize(image, (round(result.width * ratio), round(result.height * ratio)))
        def point(value):
            return tuple(np.rint(np.asarray(value) * ratio).astype(int))
        for start, end in result.walls:
            cv2.line(canvas, point(start), point(end), (110, 225, 70), 1, cv2.LINE_AA)
        for start, end in result.doors:
            cv2.line(canvas, point(start), point(end), (255, 215, 53), 3, cv2.LINE_AA)
        for light in result.lights:
            cv2.circle(canvas, point((light["position"]["x"], light["position"]["y"])), 7, (94, 215, 255), 2)
        success, buffer = cv2.imencode(".bmp", canvas)
        if not success:
            raise ValueError("Could not render preview.")
        args.preview.write_bytes(buffer.tobytes())
    report = {"image_size": [result.width, result.height], "working_size": result.working_size,
              "walls": len(result.walls), "doors": len(result.doors), "lights": len(result.lights),
              "decode_seconds": decode_seconds, "export_seconds": export_seconds,
              "runs": runs, "settings": asdict(settings), "opencv": cv2.__version__,
              "geometry_backend": result.geometry_backend, "warnings": result.warnings}
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
