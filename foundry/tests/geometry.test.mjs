import test from "node:test";
import assert from "node:assert/strict";
import {crosses} from "./helpers.mjs";
import {PaintGeometry, brushArea, contains, areaSegments, splitSegments, rectangle, segmentKey, mergeCollinear} from "../auto-wall-live/scripts/geometry.mjs";

const empty = () => new PaintGeometry([500, 500]);
const keys = segments => segments.map(s => segmentKey(s)).sort();
const blocks = (walls, a, b) => walls.some(s => crosses(s, a, b));

test("overlapping strokes share one outline and repeated paint is idempotent", () => {
  const a = empty().paint("walls", [[50, 100], [220, 100]], 30);
  const repeated = a.paint("walls", [[50, 100], [220, 100]], 30);
  assert.deepEqual(keys(a.compose().walls), keys(repeated.compose().walls));
  const joined = a.paint("walls", [[180, 100], [350, 100]], 30).compose();
  assert.equal(blocks(joined.walls, [150, 100], [250, 100]), false);
  assert.ok(joined.walls.length < 40);
});

test("brush ring retains its hole without bridging interior edges", () => {
  const geometry = empty().paint("walls", [[80, 80], [300, 80], [300, 300], [80, 300], [80, 80]], 20);
  assert.equal(contains(geometry.areas.walls, [180, 180]), false);
  assert.equal(blocks(geometry.compose().walls, [120, 180], [240, 180]), false);
  assert.equal(geometry.areas.walls[0].length, 2);
});

test("one door cuts both wall faces; only the portal blocks the doorway", () => {
  const geometry = empty().paint("walls", [[30, 150], [400, 150]], 40)
    .paint("doors", [[180, 150], [240, 150]], 60);
  const result = geometry.compose();
  assert.equal(result.doors.length, 1);
  assert.equal(blocks(result.walls, [210, 90], [210, 210]), false);
  assert.equal(blocks(result.doors, [210, 90], [210, 210]), true);
  assert.equal(blocks(result.walls, [100, 90], [100, 210]), true);
  assert.ok(result.walls.some(s => s.some(p => Math.hypot(p[0] - 180, p[1] - 150) < 1e-5)));
});

test("door painting cuts baseline wall faces and adds connected jambs", () => {
  const geometry = new PaintGeometry([500, 500], {walls: [[[30, 130], [400, 130]], [[30, 170], [400, 170]]]})
    .paint("doors", [[180, 150], [240, 150]], 60);
  const result = geometry.compose();
  assert.equal(blocks(result.walls, [210, 90], [210, 210]), false);
  for (const portal of result.doors[0]) assert.ok(result.walls.some(s => s.some(p => Math.hypot(p[0] - portal[0], p[1] - portal[1]) < 1e-6)));
});

test("later thick wall paint preserves an existing door opening", () => {
  const geometry = empty().paint("doors", [[180, 150], [240, 150]], 40)
    .paint("walls", [[30, 150], [400, 150]], 80);
  assert.equal(blocks(geometry.compose().walls, [210, 70], [210, 230]), false);
});

test("collinear touching door strokes merge; perpendicular doors stay separate", () => {
  const geometry = empty().paint("doors", [[100, 100], [170, 100]], 30)
    .paint("doors", [[160, 100], [240, 100]], 30)
    .paint("doors", [[170, 80], [170, 140]], 30);
  assert.equal(geometry.compose().doors.length, 2);
  assert.ok(geometry.compose().doors.some(s => segmentKey(s) === segmentKey([[100, 100], [240, 100]])));
});

test("invisible material replaces overlapping opaque ink; hidden material stays", () => {
  const a = empty().paint("walls", [[50, 100], [350, 100]], 30);
  const replaced = a.paint("invisible", [[140, 100], [220, 100]], 50);
  assert.equal(contains(replaced.areas.walls, [180, 100]), false);
  assert.equal(contains(replaced.areas.invisible, [180, 100]), true);
  const hidden = a.paint("invisible", [[140, 100], [220, 100]], 50, false);
  assert.equal(contains(hidden.areas.walls, [180, 100]), true);
});

test("rectangle erasing removes crossing baselines, preserves hidden layers and trims ink", () => {
  const a = new PaintGeometry([500, 500], {walls: [[[10, 100], [400, 100]]], invisible: [[[10, 110], [400, 110]]]})
    .paint("walls", [[50, 200], [350, 200]], 30);
  const b = a.erase([170, 80], [220, 240], {walls: true, invisible: false});
  assert.equal(b.base.walls.length, 0); assert.equal(b.base.invisible.length, 1);
  assert.equal(contains(b.areas.walls, [190, 200]), false);
  assert.equal(contains(a.areas.walls, [190, 200]), true);
});

test("rectangles add four edges; serialized ink survives reopening", () => {
  const a = empty().addRectangle("walls", [40, 40], [200, 200]).paint("invisible", [[250, 200], [350, 200]], 20);
  assert.equal(a.lines.walls.length, 4);
  assert.deepEqual(PaintGeometry.restore(a.serialize()).compose(), a.compose());
});

test("exact clipping handles diagonal and collinear edges", () => {
  const [outside, inside] = splitSegments([[[0, 0], [100, 100]], [[0, 40], [100, 40]]], rectangle([40, 40], [60, 60]));
  assert.equal(outside.length, 4); assert.equal(inside.length, 2);
  assert.ok(inside.some(s => segmentKey(s) === segmentKey([[40, 40], [60, 60]])));
});

test("brush clips to bounds, accepts dots, rejects nonfinite input and flat doors", () => {
  const area = brushArea([[0, 0]], 40, [100, 100]);
  assert.ok(areaSegments(area).flat().every(p => p.every(v => v >= 0 && v <= 99)));
  assert.equal(brushArea([[10, 10], [10, 10]], 20, [100, 100], true).length, 0);
  assert.throws(() => brushArea([[NaN, 10]], 20, [100, 100]), /coordinates/);
  assert.equal(mergeCollinear([[[0, 0], [50, 0]], [[20, 0], [80, 0]]]).length, 1);
});

test("manual windows cut an opening and remain a single native window span", () => {
  const result = empty().paint("walls", [[30, 150], [400, 150]], 40)
    .paint("windows", [[180, 150], [240, 150]], 60).compose();
  assert.equal(result.windows.length, 1); assert.equal(result.doors.length, 0);
  assert.equal(blocks(result.walls, [210, 90], [210, 210]), false);
});

test("loading existing native geometry does not silently split or remove its walls", () => {
  const base = {walls: [[[10, 100], [400, 100]]], doors: [[[180, 100], [240, 100]]]};
  assert.deepEqual(new PaintGeometry([500, 500], base).compose().walls, base.walls);
});

test("painting a window over a door changes the material instead of stacking portals", () => {
  const geometry = empty().paint("doors", [[100, 100], [170, 100]], 30)
    .paint("windows", [[100, 100], [170, 100]], 30);
  assert.equal(geometry.compose().doors.length, 0); assert.equal(geometry.compose().windows.length, 1);
  const adjacent = geometry.paint("doors", [[170, 100], [220, 100]], 30);
  assert.equal(adjacent.compose().doors.length, 1); assert.equal(adjacent.compose().windows.length, 1);
});

test("rectangles keep all four exact edges over older brush ink and portal cuts", () => {
  const a = empty().paint("walls", [[30, 100], [400, 100]], 40)
    .paint("doors", [[180, 100], [240, 100]], 80);
  const b = a.addRectangle("walls", [100, 80], [300, 200]);
  const wanted = areaSegments(rectangle([100, 80], [300, 200]));
  for (const edge of wanted) assert.ok(keys(b.compose().walls).includes(segmentKey(edge)));
  assert.deepEqual(keys(b.lines.walls), keys(wanted));
  assert.deepEqual(PaintGeometry.restore(b.serialize()).compose(), b.compose());
});

test("line mode keeps clicked corners, merges collinear repeats, and adds no curved outline", () => {
  const a = empty().addLines("walls", [[[40, 40], [200, 40]], [[200, 40], [200, 200]]])
    .addLines("walls", [[[40, 40], [100, 40]]]);
  assert.deepEqual(keys(a.compose().walls), keys([[[40, 40], [200, 40]], [[200, 40], [200, 200]]]));
  assert.equal(a.areas.walls.length, 0);
});

test("a subsequently drawn door cuts an explicit rectangle at its exact jambs", () => {
  const result = empty().addRectangle("walls", [40, 40], [300, 300])
    .paint("doors", [[140, 40], [200, 40]], 40).compose();
  assert.equal(result.doors.length, 1);
  assert.equal(blocks(result.walls, [170, 10], [170, 70]), false);
  assert.ok(keys(result.walls).includes(segmentKey([[40, 40], [140, 40]])));
  assert.ok(keys(result.walls).includes(segmentKey([[200, 40], [300, 40]])));
});

test("subsequent painting and erasing affect explicit lines, with hidden materials protected", () => {
  const a = empty().addRectangle("walls", [40, 40], [300, 300])
    .addLines("invisible", [[[40, 50], [300, 50]]]);
  const b = a.paint("walls", [[100, 40], [200, 40]], 30, false);
  assert.equal(b.lines.invisible.length, 1);
  assert.ok(b.lines.walls.some(s => segmentKey(s) === segmentKey([[40, 40], [85, 40]])));
  const c = a.erase([130, 35], [150, 55], {walls: true, invisible: false});
  assert.equal(c.lines.walls.length, 3); assert.equal(c.lines.invisible.length, 1);
});

test("small freehand wiggles use a coarse outline with few vertices", () => {
  const points = Array.from({length: 300}, (_, i) => [50 + i, 150 + Math.sin(i) * 0.6]);
  const result = empty().paint("walls", points, 24).compose();
  assert.ok(result.walls.length <= 12, `Unexpectedly dense outline: ${result.walls.length} segments`);
  assert.equal(blocks(result.walls, [100, 100], [100, 200]), true);
});
