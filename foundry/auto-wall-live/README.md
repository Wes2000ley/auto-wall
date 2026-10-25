# Auto-Wall Live

Enable this Foundry 14 module in **Manage Modules**, then open
**Wall Controls → Auto-Wall Live**. Reload the Foundry client after installing an
update. A first installation may require restarting Foundry to discover it.

**R** Rectangle: drag four straight edges at your exact corners, without snapping
or image-based changes. **B** Paint: drag a merged outline with a coarse edge.
**L** Line: click the start, then click each corner. Moving the pointer adds no
vertices. **Enter** or **Esc** finishes; click the start to close a loop.

**1** wall, **2** door, **3** invisible wall, **4** window, **5** light.
Doors and windows create straight openings; lights use a click. Right drag pans,
middle drag erases a rectangle, Paint's wheel changes brush size, Ctrl+wheel
zooms, and Ctrl+Z undoes a completed edit. Pause editing to play normally.

Every completed rectangle, paint stroke, or clicked line segment saves directly
to native scene documents. The module is entirely manual. There is no automatic
detection, image sampling, ML engine, or export step. Existing scene geometry
stays in place when upgrading.

See the [full controls, installation, conflict behavior and validation guide](https://github.com/Wes2000ley/auto-wall/blob/main/foundry/README.md).
