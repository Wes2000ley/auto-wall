import test, {beforeEach} from "node:test";
import assert from "node:assert/strict";
import {Scene, wall} from "./helpers.mjs";
import {CanvasEditor} from "../auto-wall-live/scripts/editor.mjs";
import {areaSegments, rectangle, segmentKey} from "../auto-wall-live/scripts/geometry.mjs";
import {DEFAULT_TOOLS, toolPreferences} from "../auto-wall-live/scripts/settings.mjs";

let editor, scene, messages;
const keys = segments => segments.map(s => segmentKey(s)).sort();
const event = (point, button = 0) => ({clientX: point[0], clientY: point[1], button, pointerId: 1, preventDefault() {}});
const key = code => editor.key({code, preventDefault() {}, stopImmediatePropagation() {}});
const drag = async (a, b, button = 0) => { editor.down(event(a, button)); editor.move(event(b, button)); await editor.up(event(b, button)); };
const click = async p => { editor.down(event(p)); await editor.up(event(p)); };

beforeEach(() => {
  scene = new Scene(); scene.id = "scene"; messages = [];
  const overlay = {style: {}, setAttribute() {}, addEventListener() {}, focus() {}, remove() {},
    setPointerCapture() {}, hasPointerCapture() { return false; }, releasePointerCapture() {}};
  globalThis.document = {createElement: () => overlay, body: {append() {}}};
  globalThis.window = {addEventListener() {}};
  globalThis.requestAnimationFrame = () => 1; globalThis.cancelAnimationFrame = () => {};
  globalThis.game = {user: {isGM: true}};
  globalThis.canvas = {ready: true, scene, level: {id: "ground"}, dimensions: {width: 500, height: 500},
    stage: {scale: {x: 1, y: 1}, pivot: {x: 0, y: 0}},
    canvasCoordinatesFromClient({x, y}) {
      return {x: x / this.stage.scale.x + this.stage.pivot.x, y: y / this.stage.scale.y + this.stage.pivot.y};
    },
    clientCoordinatesFromCanvas({x, y}) {
      return {x: (x - this.stage.pivot.x) * this.stage.scale.x, y: (y - this.stage.pivot.y) * this.stage.scale.y};
    },
    pan({x, y, scale}) {
      if (x !== undefined) this.stage.pivot.x = x;
      if (y !== undefined) this.stage.pivot.y = y;
      if (scale !== undefined) this.stage.scale.x = this.stage.scale.y = scale;
    }};
  Object.defineProperty(canvas, "primary", {get() { throw new Error("Manual drawing must not read the background image."); }});
  editor = new CanvasEditor((message, error) => { if (message) messages.push({message, error}); });
});

test("rectangles save exactly four pointer-defined edges near existing wall endpoints", async () => {
  scene.walls.set("nearby", {id: "nearby", toObject: () => wall("nearby", [40, 40, 80, 40])});
  editor.reload();
  await drag([43, 44], [205, 208]);
  const wanted = keys(areaSegments(rectangle([43, 44], [205, 208])));
  for (const edge of wanted) assert.ok(keys(editor.composed.walls).includes(edge));
  assert.equal(editor.geometry.lines.walls.length, 4);
  assert.equal(scene.walls.size, 5);
});

test("rectangles use the native canvas transform and allow padding beyond the image", async () => {
  canvas.stage.scale.x = canvas.stage.scale.y = 2;
  canvas.stage.pivot = {x: -100, y: 200};
  await drag([20, 40], [120, 140]);
  assert.deepEqual(keys(editor.composed.walls), keys(areaSegments(rectangle([-90, 220], [-40, 270]))));
});

test("old brush ink cannot trim the exact edges of a newly dragged rectangle", async () => {
  editor.setMode("brush"); editor.diameter = 40;
  await drag([30, 100], [400, 100]);
  editor.setMode("rectangle"); await drag([100, 80], [300, 200]);
  for (const edge of areaSegments(rectangle([100, 80], [300, 200]))) {
    assert.ok(keys(editor.composed.walls).includes(segmentKey(edge)));
  }
  await editor.undo(); assert.equal(editor.geometry.lines.walls.length, 0);
});

test("line mode commits only clicked vertices; movement and Enter add no extra segment", async () => {
  editor.setMode("line"); await click([40, 40]);
  for (let x = 45; x <= 195; x += 5) editor.move(event([x, 40 + Math.sin(x)]));
  assert.equal(scene.walls.size, 0);
  await click([200, 40]); await click([200, 200]);
  assert.deepEqual(keys(editor.composed.walls), keys([[[40, 40], [200, 40]], [[200, 40], [200, 200]]]));
  editor.move(event([300, 350])); key("Enter");
  assert.equal(editor.line, null); assert.equal(scene.walls.size, 2);
  await click([300, 300]); assert.equal(scene.walls.size, 2);
});

test("fast consecutive clicks queue their native saves without dropping corners", async () => {
  editor.setMode("line"); await click([30, 30]);
  const create = scene.createEmbeddedDocuments.bind(scene);
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  scene.createEmbeddedDocuments = async (...args) => { await gate; return create(...args); };
  const pending = [click([200, 30]), click([200, 200]), click([350, 250])];
  assert.equal(editor.pendingLines.length, 2);
  key("Enter"); release(); await Promise.all(pending);
  assert.deepEqual(keys(editor.composed.walls), keys([[[30, 30], [200, 30]], [[200, 30], [200, 200]], [[200, 200], [350, 250]]]));
  assert.equal(editor.sync.undoStack.length, 3); assert.equal(editor.lineSave, null);
});

test("right drag pans during a line; Esc keeps completed segments and ends its preview", async () => {
  editor.setMode("line"); await click([40, 40]); await click([200, 40]);
  await drag([300, 100], [320, 100], 2);
  assert.ok(editor.line); assert.equal(canvas.stage.pivot.x, -20);
  const client = canvas.clientCoordinatesFromCanvas({x: 200, y: 200});
  await click([client.x, client.y]); key("Escape");
  assert.equal(editor.line, null); assert.equal(editor.enabled, true); assert.equal(scene.walls.size, 2);
});

test("a drag in line mode does not create a stream of vertices", async () => {
  editor.setMode("line"); await click([40, 40]); await drag([100, 100], [250, 250]);
  assert.deepEqual(editor.line.last, [40, 40]); assert.equal(scene.walls.size, 0);
});

test("middle erase uses exact pointer coordinates and undo restores the exact rectangle", async () => {
  await drag([40, 40], [200, 200]);
  await drag([195, 35], [205, 205], 1);
  assert.equal(scene.walls.size, 1);
  await editor.undo(); assert.equal(scene.walls.size, 4);
});

test("manual light placement reads only its selected color and range", async () => {
  editor.layer = "lights"; editor.lightColor = "#abcdef"; editor.lightRange = 8;
  await click([100, 100]);
  const light = [...scene.lights][0].toObject();
  assert.equal(light.config.color, "#abcdef"); assert.equal(light.config.dim, 40);
});

test("old brush preferences migrate without importing detector settings", () => {
  assert.equal(toolPreferences({brush: true}).mode, "brush");
  assert.equal(toolPreferences({brush: false}).mode, "rectangle");
  assert.deepEqual(toolPreferences({mode: "invalid", diameter: NaN, lightColor: "bad", wall_mode: "edges"}), DEFAULT_TOOLS);
  assert.equal(toolPreferences({mode: "line", diameter: 9000}).diameter, 2048);
});

test("line vertices stay at the clicked pixels near native endpoints and a large grid", async () => {
  canvas.grid = {getSnappedPoint() { throw new Error("Grid snapping is not allowed."); }};
  scene.walls.set("nearby", {id: "nearby", toObject: () => wall("nearby", [40, 40, 200, 40])});
  editor.reload(); editor.setMode("line"); await click([43, 44]); await click([198, 43]);
  assert.ok(keys(editor.composed.walls).includes(segmentKey([[43, 44], [198, 43]])));
  assert.equal(Object.hasOwn(toolPreferences({snap: true}), "snap"), false);
});
