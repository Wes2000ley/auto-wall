# Auto-Wall Live for Foundry VTT

A manual GM editor on the Foundry scene canvas. Draw rectangles, paint merged
outlines, or click connected straight lines. Walls, doors, invisible walls,
windows and lights save directly to native Wall and AmbientLight documents.
Foundry broadcasts completed edits through its normal scene document system.
There are no export controls, automatic detectors, pixel sampling, model weights,
image workers, Python processes, or GPU requirements. The desktop app is separate.

This module targets **Foundry 14**, checked against installed **14.369** source
and the official v14 API. Automated tests use explicit document API doubles;
the browser harness runs the actual editor and template in an isolated scene.
Those checks do not certify token vision or compatibility with other modules
inside a live campaign.

## Installation and upgrade

```powershell
./foundry/install.ps1 -DataPath "$env:LOCALAPPDATA/FoundryVTT/Data"
```

The installer verifies copied files by SHA-256 and records a manifest. On an
upgrade it removes obsolete files only when they belong to the previous
installation and still match their recorded hashes. This retires the previous
OpenCV detector, its worker and settings code; unknown user files are preserved.
No world databases or activation settings are changed.

For a first installation, restart Foundry when your current sessions are finished,
enable **Auto-Wall Live** in **Manage Modules**, then open
**Wall Controls â†’ Auto-Wall Live**. For an update, reload the Foundry client.
The GM console/macro `game.modules.get("auto-wall-live").api.open()` opens the
same panel. Players receive native scene changes and cannot open the editor.

## Drawing

| Action | Control |
| --- | --- |
| Rectangle | **R**, then left drag the opposite corners |
| Paint | **B**, then left drag; overlapping strokes share an outline |
| Connected lines | **L**, click the start, then click every corner |
| Finish a line | **Enter**, **Esc**, or Finish line; clicked segments stay saved |
| Close a line loop | Click the start again |
| Wall / door / invisible / window / light | **1 / 2 / 3 / 4 / 5** or panel buttons |
| Pan | Right drag or Space+left drag; a pending line remains anchored |
| Erase a rectangle | Middle drag; hidden materials are protected |
| Resize Paint | Wheel, or Brush / door width |
| Zoom | Ctrl+wheel; ordinary wheel also zooms in Rectangle and Line modes |
| Undo | Ctrl+Z or Undo edit, up to 30 completed edits in the open editor |
| Cancel an unfinished drag | Esc |
| Play with the panel open | Pause editing |

Rectangles use the corners you drag, in scene coordinates. They ignore snapping,
map colors, image size, image rotation, and older brush coverage. Their four
straight edges are not automatically reshaped or cut around earlier doors. Draw
a door afterward to cut its opening. The same exact-corner rule applies to erase
selections. Coordinates are rounded to native Foundry wall pixels on saving.

Line mode adds a straight segment only when you click another point. Mouse
movement is preview only; it never creates extra vertices. Fast consecutive
clicks queue their saves so corners are not dropped while the previous segment
is saving. Collinear overlapping direct lines merge. There is no grid or endpoint
snapping. Line vertices and rectangle corners follow the pointer, rounded only
to the integer pixels required by native Foundry walls.

Paint uses an eight-sided brush cap and a coarser freehand simplification, with
simpler vector outlines instead of many tiny rounded segments. Holes remain
open. A later brush stroke can replace explicit lines where you paint; changing
material respects hidden layers. Doors/windows use a straight jamb-to-jamb
opening, with their width controlling how much painted wall is removed. In Line
mode, each pair of clicks places an opening. Invisible walls block movement and
allow sight, light and sound. Windows use Foundry proximity restrictions, which
can be changed through the native wall configuration. Light color and radius
are manual; radius is measured in grid squares.

Typing in a text field does not trigger shortcuts. Closing or pausing removes
the input overlay. Accepted edits stay saved; unfinished previews are discarded.
Native walls created by an earlier automatic pass are ordinary editable walls.
An upgrade does not rerun, delete or reinterpret them from the map image.

## Persistence and conflicts

Each edit changes only affected native documents. Unchanged IDs, open/secret
doors, other module flags and custom restrictions are retained. Split native
fragments inherit their original restrictions. Editing a shared wall on one
level preserves its other memberships. Vector ink is cached in a scene flag;
reopening restores it when it matches the native documents. External changes
invalidate the cache. Native scene geometry remains authoritative.

Saves are single-flight. A conflicting edit stops a save and rolls back its own
successful batches while preserving unrelated newer changes. Foundry does not
provide a cross-document atomic transaction, so clients may briefly see
intermediate batches. A rollback conflict is reported. Undo refuses to overwrite
a later change to its touched documents, including an opened door.

## Validation and development

No npm install or build is required:

```text
node --test --test-reporter=spec foundry/tests/*.test.mjs
node foundry/tests/benchmark.mjs
```

Tests cover exact rectangle coordinates over earlier paint, drawing without an
image, pan/zoom transforms, clicked-only vertices, queued saves, coarse paint,
open doorway rays, holes, levels, metadata, concurrency, rollback and undo. The
benchmark measures manual vector operations, not image detection or real-world
Foundry latency.

For the isolated browser harness, supply a local Handlebars browser distribution:

```text
node foundry/tests/serve-harness.mjs "C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/handlebars/dist/handlebars.js"
```

Open `http://127.0.0.1:43125/` and run the interaction checks. This uses the real
module scripts/template with explicit Foundry API doubles, outside your campaign.
A live Foundry smoke test should additionally verify token movement, vision,
multiple levels, and a second connected client in a disposable scene.

References: [v14 Canvas coordinates](https://foundryvtt.com/api/v14/classes/foundry.canvas.Canvas.html#canvasCoordinatesFromClient),
[v14 Scene documents](https://foundryvtt.com/api/v14/classes/foundry.documents.Scene.html),
[bundled dependencies and licenses](auto-wall-live/THIRD-PARTY.md).
