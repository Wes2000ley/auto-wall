# Auto-Wall Live for Foundry VTT

A GM editor on the Foundry scene canvas: paint walls, doors, invisible walls,
windows and lights, or run the existing palette/color detector. A completed edit
goes straight to native Wall and AmbientLight documents. Foundry broadcasts the
change and updates its vision, light and movement boundaries. There are no export
buttons, file conversions, model weights, PyTorch/CUDA requirements or companion
processes. The desktop application and its optional ML models are unchanged.

This module targets **Foundry 14**. Its API integration was checked against the
installed **14.369** source and official v14 documentation. Automated tests use
document API doubles; the browser harness runs the actual editor and WASM worker
against an isolated scene. Activation in a real running world requires a further
smoke test; those isolated tests do not certify compatibility with other modules.

## Install

Copy `auto-wall-live` into your Foundry `Data/modules/` directory. On Windows:

```powershell
./foundry/install.ps1 -DataPath "$env:LOCALAPPDATA/FoundryVTT/Data"
```

The installer verifies every copied file by SHA-256 and records an installation
manifest. It never edits world databases or activation settings. For a newly
installed module, restart Foundry when your current sessions are finished, enable
**Auto-Wall Live** in **Manage Modules**, and accept Foundry's normal reload.
Then open **Wall Controls → Auto-Wall Live** (paintbrush icon).

The GM console/macro API `game.modules.get("auto-wall-live").api.open()` also opens
the same panel. Players cannot open the editor. They receive the resulting native
scene changes through Foundry normally.

## Controls

| Action | Control |
| --- | --- |
| Wall / door / invisible wall / window / light | **1 / 2 / 3 / 4 / 5** or panel buttons |
| Paint walls or invisible walls | Left drag with **Brush** enabled |
| Draw a wall/invisible perimeter rectangle | **B** to turn Brush off, then left drag |
| Cut and place one door/window | Select its material, then left drag jamb to jamb |
| Place a light | Select Light and click; radius comes from Light radius |
| Resize brush | Wheel; panel diameter is in scene pixels |
| Zoom | Ctrl+wheel, or wheel with Brush off |
| Pan | Right drag or Space+left drag |
| Erase a rectangle | Middle drag; hidden materials are protected |
| Undo a complete edit or detection pass | Ctrl+Z or Undo edit, up to 30 edits in the open editor |
| Cancel gesture/detection or pause editing | Esc |
| Play normally while leaving the panel open | Pause editing |

Typing in a text field does not trigger tool shortcuts. Closing the panel removes
the input overlay and cancels unfinished detection. Completed geometry stays in
the scene. Vector ink is stored in a module scene flag, so reopening can continue
merging strokes rather than adding overlapping outlines. External native geometry
changes invalidate that cache and reload current documents.

Overlapping brush strokes share an outline, with holes preserved. Painting one
visible wall material replaces the other in that area. Doors/windows cut both
faces of painted or detected wall outlines and connect their jambs to one native
portal. Collinear touching portal strokes merge; perpendicular portals remain
separate. Invisible walls block movement but allow sight, light and sound. Windows
use the same Foundry proximity restrictions as the desktop Foundry JSON output.
Native wall configuration remains available for changing those restrictions.

## Automatic detection

Choose a preset, import a palette/settings JSON, or pick structural colors directly
from the map. Schema prose and generation instructions are inert: only explicit
palette colors and known detector settings are read. Desktop ML options are
discarded during import. Selected-color, neutral-stone and edge wall modes;
outline/centerline geometry; door shape/jamb checks; and warm/cool/color/brightness
light modes are available. Windows and invisible walls are manual in this addon.

**Detect map** processes the viewed level's loaded background in a worker, then
saves the result. It replaces documents still tagged as the prior automatic pass
and retains other native/manual documents. Brush and rectangle correction areas
are excluded from future passes, including deleted openings. A rerun does not
interpret an arbitrary native Foundry deletion as a brush correction: use this
editor for persistent exclusions, or undo a detection pass before starting over.
Detection never starts merely because a scene or the panel opens.

Quality bounds the longest edge to 512/1024/2048/4096 pixels; output coordinates
follow the background sprite's actual transform, including scene padding, fit,
offset, scaling, rotation and mirroring. The background file and scene image
settings are not rewritten. Pure palette detection is not semantic recognition:
artwork that shares wall/door colors with scenery can produce mistakes. Require
door jambs by default and review the native result. Automatic detection is not a
reason to assume walls and doors are perfect on a new art style.

## Persistence and conflict behavior

Each completed gesture creates/updates/deletes only the affected embedded
documents. Unchanged IDs, open/secret doors, other module flags and configuration
are retained. Split native wall fragments inherit their source restrictions.
Editing a wall shared by several levels removes it only from the current level;
the other memberships remain. Lights use the scene's grid distance.

Saves are single-flight. A scene/level/background change discards stale detection.
A concurrent geometry edit stops the pending save and rolls back its successful
batches, while leaving unrelated newer native changes intact. Foundry does not
provide a cross-document atomic transaction, so connected clients may briefly see
intermediate wall batches. A rollback conflict is reported explicitly. Undo also
refuses to overwrite documents changed afterward, including a subsequently opened
door. Native scene data remains authoritative over cached brush ink.

## Validation and development

No npm install or build is needed:

```text
node --test --test-reporter=spec foundry/tests/*.test.mjs
node foundry/tests/benchmark.mjs
```

The tests run the bundled OpenCV WASM and check clear doorway rays, repeated paint,
holes, shared levels, metadata preservation, concurrent edits, rollback and undo.
The 4096px test covers WASM memory growth, which invalidates old typed-array views.
The benchmark reports synthetic processing time separately from worker startup and
native scene persistence; it is not an art-style accuracy benchmark.

For an isolated browser test, supply a local Handlebars browser distribution:

```text
node foundry/tests/serve-harness.mjs "C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/handlebars/dist/handlebars.js"
```

Open `http://127.0.0.1:43125/` and run the interaction checks. The harness uses the
actual module scripts, template and worker, with explicit Foundry API doubles.
It is separate from your campaign. Test real Foundry using a disposable scene:
paint overlapping walls, cut/open a door, check token movement and vision, place
an invisible wall, switch levels, erase/undo, then verify a second connected client.

References: [Foundry module development](https://foundryvtt.com/article/module-development/),
[v14 scene controls](https://foundryvtt.com/api/v14/functions/hookEvents.getSceneControlButtons.html),
[v14 Scene document API](https://foundryvtt.com/api/v14/classes/foundry.documents.Scene.html),
[bundled dependency versions and licenses](auto-wall-live/THIRD-PARTY.md).
