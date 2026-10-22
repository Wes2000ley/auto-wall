# Auto-Wall

## Fast Setup fork

This fork adds a responsive **Walls / Doors / Lights** workspace, conservative doorway
checks, original-resolution UVTT export, and a much faster native OpenCV detection path.
It imports map-style JSON palettes, defaults to gray `#A0A0A0` walls and purple
`#702090` doors, and supports other sampled colors, shaded gates, and warm/cool lights.
Optional **centerlines** reduce double outlines while retaining connected door jambs.
An optional C++ tracer makes the complete centerline pass **1.86–2.20× faster** on
six tested maps at a 2048 px cap, with exactly matching geometry. It builds locally
with `python scripts/build_native.py`; the Python fallback remains available.
On the 5792 × 4344 Demon Temple test map, the current complete pass takes about **0.35 s**.
The existing edge detector is **46.7× faster** at full resolution with identical contour
point sets. These are separate measurements; see [the benchmark notes](docs/FAST_SETUP.md#measured-performance).

```shell
python -m pip install -r requirements-fast.txt
python fast_auto_wall.py "path/to/map.png" --schema "profiles/foundry-v2.json"
```

Windows users can also drop a map onto `run-fast.bat`. Use **Auto setup** in the
existing editor to generate walls, doors, and lights, then continue editing them.
See [Fast Setup instructions and limitations](docs/FAST_SETUP.md).
For centerlines, install `requirements-centerline.txt` and choose **Walls → Geometry →
Centerlines**. Use `--preset demon` for the earlier gray/orange-door artwork.
The optional **ML** tab runs a compact custom segmentation model locally on CPU
or CUDA. It distinguishes opaque walls from see-through barriers and keeps those
barriers out of vision-blocking UVTT walls. Models remain experimental and need
preview review. [Local ML setup and training](docs/LOCAL_ML.md) covers the optional
dependencies, model selection, and whole-layout validation. The
[accuracy assessment](docs/ACCURACY_AND_ML.md) also records the pretrained CubiCasa
model's failure on these artworks; that unrelated model is not used by the app.
The upstream downloads linked below do not include these fork changes.

---

![Auto-Wall](resources/icon.ico)

**Automatically detect walls in TTRPG battle maps and export them to your virtual tabletop.**

Auto-Wall transforms battle map images into ready-to-use wall data for FoundryVTT, Roll20, Fantasy Grounds, and other VTTs supporting the Universal VTT format. No more tedious manual wall tracing—load your map, tweak detection settings, and export in minutes.

[![GitHub Release](https://img.shields.io/github/v/release/ThreeHats/auto-wall?style=flat&label=Latest)](https://github.com/ThreeHats/auto-wall/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[📺 Watch the Demo](https://youtu.be/gqkEIWwuJX4) · [💬 Discord](https://discord.gg/HUzEnZy8uJ) · [🌐 Website](https://autowallvtt.com)

---

## Why Auto-Wall?

Setting up dynamic lighting for battle maps is time-consuming. Auto-Wall solves this by using computer vision to automatically detect walls, doors, and light sources, then exporting everything in the industry-standard UVTT format.

**The Problem:** Manually tracing walls on a detailed dungeon map can take 20–30 minutes or more.

**The Solution:** Auto-Wall detects walls automatically using edge detection or color-based algorithms, letting you export a fully traced map in just a few minutes.

---

## Key Features

- **Smart Wall Detection** — Edge detection (Canny algorithm) finds walls from outlines; color detection extracts walls based on specific colors
- **Light Source Detection** — Automatically identifies torches, lamps, and bright areas for dynamic lighting
- **Manual Refinement Tools** — Brush, line, rectangle, and fill tools to add or remove walls; door placement for dynamic entry points
- **Universal VTT Export** — One-click export to UVTT format, compatible with FoundryVTT, Roll20, Fantasy Grounds, and Arkenforge
- **Cross-Platform** — Native builds for Windows (.exe), Linux (.deb, AppImage), and macOS (.dmg)

---

## Quick Start

1. **Open an image:** Click "Open Image" or "Load from URL" to import your battle map
2. **Navigate the view:** Use scroll wheel to zoom, right-click and drag to pan
3. **Choose your tool mode:** Use the left sidebar to select between:
   - **Detect Mode:** For wall and light detection
   - **Paint Mode:** For manual drawing and editing
   - **Walls Mode:** For final wall editing and export
4. **Detection workflow:**
   - Select detection mode (Edge Detection or Color Detection)
   - Use detection presets or adjust parameters in the right panel
   - Fine-tune with sliders and settings
5. **Manual editing:** Switch to Paint mode for manual mask editing with various tools
6. **Wall editing:** Switch to Walls mode to:
   - Generate walls from your detection/drawing
   - Edit, move, or delete individual walls
   - Draw doors
7. **Export:** Save as UVTT file by opening the file menu and clicking "Save File" or using "ctrl + s"

---

## How It Works

Auto-Wall is built on **OpenCV** for image processing, **scikit-learn** for color clustering, and **PyQt6** for the desktop interface.

### Architecture Overview

```
auto_wall.py          → Application entry point with splash screen
src/
├── gui/              → PyQt6 interface (app.py, image_viewer, panels)
├── core/             → Image and contour processing logic
├── wall_detection/   → Detection algorithms (edge, color, light)
└── utils/            → Performance optimizations, update checker
```

### Detection Pipeline

1. **Preprocessing** — Image is optionally scaled for performance, then blurred to reduce noise
2. **Edge Detection** — Canny edge detection finds contours from brightness gradients
3. **Color Detection** (alternate) — User-selected colors are matched within a configurable threshold
4. **Contour Processing** — Small artifacts filtered, contours merged and simplified
5. **Wall Generation** — Contours converted to wall segments with configurable simplification

---

## Installation

### Windows
1. Download the latest `.exe` from [autowallvtt.com](https://autowallvtt.com) or the [Releases page](https://github.com/ThreeHats/auto-wall/releases)
2. Run `Auto-Wall.exe`

### Linux
#### Option 1: Debian Package (.deb)
1. Download the latest `.deb` file from [autowallvtt.com](https://autowallvtt.com) or the [Releases page](https://github.com/ThreeHats/auto-wall/releases)
2. Install: `sudo dpkg -i auto-wall_{version}_amd64.deb`
3. Run from applications menu or terminal: `auto-wall`

#### Option 2: AppImage
1. Download the latest `.AppImage` file from [autowallvtt.com](https://autowallvtt.com) or the [Releases page](https://github.com/ThreeHats/auto-wall/releases)
2. Make executable: `chmod +x Auto-Wall-{version}-x86_64.AppImage`
3. Run: `./Auto-Wall-{version}-x86_64.AppImage`

*Replace `{version}` with the actual version number from the release (e.g., `1.3.1`).*

### macOS
1. Download the latest `.dmg` bundle from [autowallvtt.com](https://autowallvtt.com) or the [Releases page](https://github.com/ThreeHats/auto-wall/releases)
2. Open the `.dmg` file
3. Drag Auto-Wall to Applications folder
4. **Important:** The app is not notarized due to the $99/year cost. You will see a security warning when trying to open the app.

   To open the application, click "OK" on the warning and go to Settings → Privacy & Security. Scroll down to the bottom, and click "Open Anyway".

### From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/ThreeHats/auto-wall.git
   ```

2. Install dependencies:
   ```bash
   cd auto-wall
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python auto_wall.py
   ```

---

## Usage Guide

### Application Layout

Auto-Wall features a three-panel interface:
- **Left Sidebar:** Mode selection (Detect, Paint, Walls)
- **Right Panel:** Mode-specific settings and controls
- **Center:** Image display with interactive tools

**Navigation:**
- Use scroll wheel to zoom
- Right-click and drag to pan
- View controls available in the View menu

### Mode Overview

#### Detect Mode
Used for automatic wall and light detection:
- **Edge Detection:** Best for maps with clear wall lines
- **Color Detection:** Best for maps with distinct wall/background colors
- **Detection Presets:** Save and load detection configurations
- **Light Detection:** Automatically find light sources

#### Paint Mode  
Used for manual drawing and mask editing:
- **Drawing Tools:** Brush, line, rectangle, circle, ellipse, fill
- **Draw/Erase:** Toggle between adding and removing walls
- **Brush Size:** Adjustable brush size for detailed work

#### Walls Mode
Used for final wall editing and export:
- **Wall Editing:** Move, delete, or modify detected walls
- **Door Drawing:** Add doors and portals manually
- **Export Settings:** Configure UVTT export parameters

### Detection Modes

#### Edge Detection (Default)
Best for maps with clear wall lines:
- **Edge Sensitivity & Edge Threshold:** Controls edge detection precision - higher sensitivity finds more edges, higher threshold filters out noise
- **Min Area:** Filter out small artifacts
- **Smoothing:** Reduce noise (higher values = more smoothing)
- **Edge Margin:** Exclude detection near image edges
- **High-Resolution Processing:** Option to process at full resolution for more accurate results

#### Color Detection
Best for maps with distinct colors for the walls or background:
1. Enable "Color Detection"
2. Add colors by:
   - Clicking "Add Color" and selecting from color picker
   - Using "Color Pick" mode to select colors directly from the image
3. Adjust threshold for each color to control matching sensitivity

#### Light Detection
- **Enable Light Detection:** Toggle light detection on/off
- **Brightness Threshold:** Control sensitivity to bright areas
- **Size Filters:** Set minimum and maximum light source sizes
- **Light Colors:** Add specific colors to detect as light sources
- **Merge Distance:** Combine nearby light sources

### Detection Presets

Save and load detection configurations:
- **Save Preset:** Store current detection settings
- **Load Preset:** Apply previously saved settings
- **Manage Presets:** Delete user-created presets
- **Default Presets:** Built-in configurations for common scenarios

### Editing Tools

#### Paint Mode Tools
Multiple drawing tools for precise mask editing:
- **Brush Tool:** Freehand drawing with adjustable size
- **Line Tool:** Draw straight lines
- **Rectangle Tool:** Draw rectangular areas
- **Circle Tool:** Draw circular areas  
- **Ellipse Tool:** Draw elliptical areas
- **Fill Tool:** Fill enclosed areas

#### Walls Mode Tools
- **Draw Walls:** Click and drag to draw new wall segments
- **Draw Doors:** Create doors and portals
- **Edit Mode:** Move wall endpoints by dragging
- **Delete Mode:** Click walls to remove them
- **Multi-selection:** Drag to select multiple walls for batch operations

#### Detection Mode Tools
- **Deletion Tool:** Click to remove unwanted detected areas
- **Thinning Tool:** Reduce thickness of detected walls
- **Color Pick Tool:** Select colors directly from the image (Color Detection mode)

### Export Settings

- **Simplification Tolerance:** Controls how much wall details are simplified (0 = full detail)
- **Maximum Wall Segment Length:** Limits how long each wall segment can be
- **Maximum Number of Generation Points:** Caps the total number of generated points for performance
- **Point Merge Distance:** Connects nearby wall endpoints to fix gaps and lessen the number of walls
- **Angle Tolerance:** Determines when walls at different angles should merge
- **Maximum Straight Gap to Connect:** Maximum distance to bridge between straight nearby walls
- **Grid Snapping:** Optional alignment to a grid for precise positioning
- **Grid Overlay:** Enable grid overlay to visualize alignment. The overlay grid size (in pixels) determines the `pixels_per_grid` value in the exported UVTT file, ensuring the grid scale matches your VTT's expectations

---

## VTT Integration

Export your map as UVTT from Auto-Wall, then import using:

- **FoundryVTT:** Use the [Universal Battlemap Importer](https://foundryvtt.com/packages/dd-import/) module by Moo Man.
- **Roll20:** Use the [UniversalVTTImporter](https://wiki.roll20.net/Script:UniversalVTTImporter) API script by The Aaron.
- **Arkenforge & Fantasy Grounds:** Both platforms support UVTT files natively—simply import your exported file.

---

## Development

### Setup

```bash
git clone https://github.com/ThreeHats/auto-wall.git
cd auto-wall
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python auto_wall.py --debug
```

### Building Releases

```bash
python build.py --platform [linux|windows|macos]
```

**Options:**
- `--clean` — Clean build artifacts first
- `--skip-appimage` — Skip AppImage (Linux)
- `--skip-deb` — Skip Debian package (Linux)

### Tech Stack

- **Python 3.x** — Core language
- **OpenCV** — Edge detection, contour processing, image manipulation
- **scikit-learn** — Color clustering for color-based detection
- **PyQt6** — Cross-platform desktop GUI
- **PyInstaller** — Executable bundling

---

## What's New in v1.3.1

- **Revamped UI:** 
  - New centralized mode management system with cleaner, more intuitive interface
  - Left sidebar with mode selection (Detect, Paint, Walls) and right panel with mode-specific settings
  - Improved visual hierarchy and reduced clutter
  - Better organization of detection, drawing, and export controls

- **Cross-Platform Support:**
  - **Windows:** Portable `.exe` application
  - **Linux:** `.deb` for Ubuntu/Debian based distros and AppImage for all others
  - **macOS:** `.dmg` bundle (unsigned - see installation notes below)

---

## Contributing

Contributions welcome! Feel free to:
- Report bugs or request features via [Issues](https://github.com/ThreeHats/auto-wall/issues)
- Submit pull requests with fixes or improvements
- Suggest new detection algorithms

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [OpenCV](https://opencv.org/) for image processing
- [scikit-learn](https://scikit-learn.org/) for color clustering
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the interface
- [PyInstaller](https://www.pyinstaller.org/) for executable creation
- The TTRPG community for feedback and testing

---

*Made with ❤️ for the TTRPG community*
