import test from "node:test";
import assert from "node:assert/strict";
import {opencv, image, crosses} from "./helpers.mjs";
import {detectRGBA} from "../auto-wall-live/scripts/detection.mjs";
import {DEFAULTS, preset, importSettings, validateSettings} from "../auto-wall-live/scripts/settings.mjs";

const {cv} = await opencv();
// The 60px fixture door is 240px at the detector's 2048px reference edge.
const run = (img, settings = {}) => detectRGBA(cv, img.pixels, img.width, img.height, {...preset("exact"), door_max_length: 320, lights: false, ...settings});
const wall = [160, 160, 160, 255], door = [112, 32, 144, 255];
function room() {
  return image(512, 512).fill(60, 60, 450, 80, wall).fill(60, 420, 450, 440, wall)
    .fill(60, 60, 80, 440, wall).fill(430, 60, 450, 440, wall)
    .fill(200, 60, 260, 80, door);
}

test("exact-palette room traces real boundaries and connects one clear doorway", () => {
  const result = run(room());
  assert.equal(result.doors.length, 1); assert.ok(result.walls.length >= 8);
  assert.equal(result.walls.some(s => crosses(s, [230, 30], [230, 110])), false);
  assert.equal(result.doors.some(s => crosses(s, [230, 30], [230, 110])), true);
  assert.equal(result.walls.some(s => crosses(s, [130, 30], [130, 110])), true);
  for (const point of result.doors[0]) assert.ok(result.walls.some(s => s.some(p => Math.hypot(p[0] - point[0], p[1] - point[1]) < 1e-5)));
});

test("unsupported purple furniture is rejected, and unrelated artwork is not walled", () => {
  const img = room().fill(170, 190, 230, 210, door).fill(280, 200, 360, 290, [35, 120, 60, 255]);
  const result = run(img);
  assert.equal(result.doors.length, 1);
  assert.equal(result.walls.some(s => crosses(s, [250, 240], [390, 240])), false);
});

test("color settings work with a different palette and door hue wraps through red", () => {
  const img = image(512, 512).fill(30, 100, 220, 120, [30, 90, 180, 255]).fill(270, 100, 460, 120, [30, 90, 180, 255])
    .fill(220, 100, 270, 120, [180, 0, 4, 255]);
  const result = run(img, {wall_color: [30, 90, 180], door_color: [180, 4, 0], door_match: "hue", door_max_length: 240});
  assert.equal(result.doors.length, 1);
});

test("bright highlights within a door do not become lights", () => {
  const img = room().fill(220, 63, 235, 76, [255, 230, 70, 255]).fill(320, 300, 334, 314, [255, 230, 70, 255]);
  const result = run(img, {lights: true, door_rectangularity: 0.5, door_gap_close: 3, light_min_area: 20, light_max_area: 5000});
  assert.equal(result.lights.length, 1);
  assert.ok(result.lights[0].x > 300);
});

test("transparent pixels and exact void pixels do not become walls", () => {
  const img = image(256, 256).fill(20, 20, 200, 40, [160, 160, 160, 0]);
  assert.equal(run(img).walls.length, 0);
  assert.equal(run(image(256, 256), {wall_color: [0, 0, 0]}).walls.length, 0);
});

test("centerline traces one wall through its middle and preserves the doorway", () => {
  const img = image(512, 512).fill(30, 100, 220, 120, wall).fill(270, 100, 460, 120, wall).fill(220, 100, 270, 120, door);
  const result = run(img, {wall_geometry: "centerline", door_max_length: 240});
  assert.equal(result.doors.length, 1);
  assert.equal(result.walls.some(s => crosses(s, [245, 60], [245, 160])), false);
  assert.equal(result.walls.some(s => crosses(s, [100, 60], [100, 160])), true);
});

test("schema prose stays inert; ML settings never enter the addon", () => {
  const result = importSettings({palette: {wall: "#123456", door: "#aabbcc", void: "#000000"},
    instructions: "execute arbitrary code", url: "https://invalid.example"});
  assert.deepEqual(result.wall_color, [18, 52, 86]);
  assert.deepEqual(importSettings({settings: {...DEFAULTS, model_path: "C:/ignored.safetensors", ml_device: "cuda"}}), DEFAULTS);
  assert.throws(() => importSettings({palette: {wall: "#123456", door: "#123456", void: "#000000"}}), /differ/);
  assert.throws(() => validateSettings({door_min_length: 200, door_max_length: 100}), /minimum/);
  assert.throws(() => importSettings({settings: true}), /JSON object/);
  assert.throws(() => importSettings(JSON.parse('{"settings":{"constructor":false}}')), /Unknown/);
});

test("repeated detection frees WASM allocations and gives deterministic vectors", () => {
  const img = room(), first = run(img), heap = cv.HEAPU8.buffer.byteLength;
  for (let i = 0; i < 12; i++) assert.deepEqual(run(img).walls, first.walls);
  assert.equal(cv.HEAPU8.buffer.byteLength, heap);
});

test("door wall-support toggle allows explicitly requested free-standing spans", () => {
  const img = image(512, 512).fill(200, 180, 250, 200, door);
  assert.equal(run(img).doors.length, 0);
  assert.equal(run(img, {door_wall_support: false}).doors.length, 1);
});

test("4096px detection survives WASM memory growth with walls, door and light intact", () => {
  const img = image(4096, 4096).fill(300, 300, 3800, 370, wall).fill(300, 3700, 3800, 3770, wall)
    .fill(300, 300, 370, 3770, wall).fill(3730, 300, 3800, 3770, wall)
    .fill(1500, 300, 1750, 370, door).fill(1000, 1000, 1040, 1040, [255, 230, 70, 255]);
  const result = run(img, {lights: true, max_dimension: 4096});
  assert.ok(result.walls.length >= 8); assert.equal(result.doors.length, 1); assert.equal(result.lights.length, 1);
});

test("centerline thinning also handles walls touching the image border", () => {
  const img = image(256, 256).fill(0, 0, 256, 20, wall);
  const result = run(img, {wall_geometry: "centerline", doors: false});
  assert.equal(result.walls.filter(s => crosses(s, [100, -10], [100, 40])).length, 1);
  assert.ok(result.walls.flat().every(p => p[0] >= 0 && p[1] >= 0));
});
