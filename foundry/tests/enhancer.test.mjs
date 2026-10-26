import test from "node:test";
import assert from "node:assert/strict";
import {readFile, access} from "node:fs/promises";
import {createHash} from "node:crypto";
import {DEFAULTS, MODULE_ID, settings, outputSize, previewRegion} from "../map-enhancer/scripts/settings.mjs";
import {dataFolder, destination, targetSnapshot, saveMap, restoreMap} from "../map-enhancer/scripts/assets.mjs";
import {filterCPU} from "../map-enhancer/scripts/filters.mjs";
import {reflect, tiles, scaledTile} from "../map-enhancer/scripts/tiles.mjs";

test("enhancement options sanitize saved values and bound output memory and WebP dimensions", () => {
  assert.deepEqual(settings(null), DEFAULTS);
  assert.deepEqual(settings({backend: "service", scale: 3, quality: NaN}), DEFAULTS);
  assert.equal(settings({sharpen: 9, smooth: -1, sharpRadius: 2.7}).sharpen, 2);
  assert.equal(settings({sharpRadius: 2.7}).sharpRadius, 3);
  assert.deepEqual(outputSize(100, 70, {scale: 4}), {width: 400, height: 280, limited: false});
  assert.deepEqual(outputSize(5000, 3000, {scale: 4}), {width: 8192, height: 4915, limited: true});
  assert.equal(outputSize(10000, 10000, {scale: 4, maxEdge: 16383}).width, 8192);
  assert.equal(outputSize(5000, 2000, {scale: 4, maxEdge: 16383}, 4096).width, 4096);
  assert.equal(outputSize(10000, 2, {maxEdge: 16383, scale: 4}).width, 16383);
  assert.deepEqual(outputSize(100, 70, {scale: 4}, NaN), outputSize(100, 70, {scale: 4}));
  assert.throws(() => outputSize(0, 70, {}), /dimensions/);
});

test("preview bounds include the chosen detail at all corners and on tiny maps", () => {
  assert.deepEqual(previewRegion(1000, 800, [0, 0]), {x: 0, y: 0, width: 160, height: 128});
  assert.deepEqual(previewRegion(1000, 800, [1, 1]), {x: 840, y: 672, width: 160, height: 128});
  assert.deepEqual(previewRegion(20, 10, [1, 1]), {x: 0, y: 0, width: 20, height: 10});
});

test("reflection padding repeats correctly for arbitrary offsets and one-pixel images", () => {
  assert.deepEqual(Array.from({length: 13}, (_, i) => reflect(i - 4, 4)), [2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2]);
  assert.equal(reflect(-1000, 1), 0);
});

test("odd-sized and heavily downsampled tiles cover every output pixel exactly once", () => {
  for (const [w, h, ow, oh] of [[131, 99, 524, 396], [131, 99, 273, 207], [131, 99, 7, 3], [1, 1, 4, 4]]) {
    const coverage = new Uint8Array(ow * oh);
    for (const tile of tiles(w, h, 64)) {
      const target = scaledTile(tile, w, h, ow, oh);
      for (let y = target.y; y < target.y + target.height; y++) for (let x = target.x; x < target.x + target.width; x++) coverage[y * ow + x]++;
    }
    assert.ok(coverage.every(n => n === 1), `${w}x${h} to ${ow}x${oh}`);
  }
});

function image(w, h, sample) {
  return Uint8ClampedArray.from(Array.from({length: w * h}, (_, i) => sample(i % w, Math.floor(i / w))).flat());
}

test("filters are exact when disabled, preserve alpha, and leave uniform colors unchanged", () => {
  const uniform = image(9, 7, () => [81, 122, 174, 255]);
  assert.deepEqual(filterCPU(uniform, 9, 7, {smooth: 1, sharpen: 2}), uniform);
  const alpha = image(9, 7, (x, y) => [x * 20, y * 20, 100, (x * 61 + y * 11) % 256]);
  assert.deepEqual(filterCPU(alpha, 9, 7, {smooth: 0, sharpen: 0}), alpha);
  const filtered = filterCPU(alpha, 9, 7, {smooth: 1, sharpen: 1});
  assert.deepEqual(filtered.filter((_, i) => i % 4 === 3), alpha.filter((_, i) => i % 4 === 3));
  assert.throws(() => filterCPU(alpha, 8, 7, {}), /dimensions/);
});

test("bilateral filtering reduces interior noise while retaining the wall edge", () => {
  const source = image(21, 11, (x, y) => { const v = (x < 10 ? 40 : 210) + ((x + y) % 2 ? 7 : -7); return [v, v, v, 255]; });
  const result = filterCPU(source, 21, 11, {smooth: 1, smoothRadius: 2, edgeTolerance: .08, sharpen: 0});
  const at = (x, y) => result[(y * 21 + x) * 4];
  assert.ok(Math.abs(at(4, 5) - at(5, 5)) < 4);
  assert.ok(at(9, 5) < 50 && at(10, 5) > 200);
});

test("sharpening increases local contrast and transparent neighbors do not darken opaque edges", () => {
  const soft = image(9, 3, x => { const v = [60, 60, 65, 90, 130, 170, 195, 200, 200][x]; return [v, v, v, 255]; });
  const result = filterCPU(soft, 9, 3, {sharpen: 1, sharpRadius: 2});
  assert.ok(result[3 * 4] < soft[3 * 4]); assert.ok(result[5 * 4] > soft[5 * 4]);
  const transparent = image(9, 3, x => x < 4 ? [0, 0, 0, 0] : [100, 100, 100, 255]);
  const alphaResult = filterCPU(transparent, 9, 3, {smooth: 1, sharpen: 1});
  assert.deepEqual([...alphaResult.slice(4 * 4, 5 * 4)], [100, 100, 100, 255]);
});

test("destination follows encoded local maps and route prefixes, with safe package and external fallbacks", () => {
  assert.deepEqual(destination("assets/maps/Snow%20Temple.png", "https://example.com/game"), {folder: "assets/maps", name: "Snow Temple"});
  assert.deepEqual(destination("/table/assets/maps/Map.png", "https://example.com/table/game", "table"), {folder: "assets/maps", name: "Map"});
  for (const source of ["https://other.example/maps/test.png", "modules/package/maps/test.png", "systems/game/maps/test.png"]) {
    assert.equal(destination(source, "https://example.com/game").folder, "assets/maps/enhanced");
  }
  for (const bad of ["../outside", "C:/maps", "assets/../../elsewhere", "assets//maps", "assets/maps.", null]) assert.throws(() => dataFolder(bad));
  assert.equal(dataFolder("/assets/maps with spaces/"), "assets/maps with spaces");
});

function fixture() {
  globalThis.game = {user: {isGM: true}};
  const calls = [], files = [], folders = new Set(["assets", "assets/maps"]);
  const scene = {id: "scene", width: 4000, height: 3000, grid: {size: 100}, walls: [{c: [4, 6, 40, 60]}], allowed: true,
    canUserModify() { return this.allowed; }, levels: new Map()};
  const level = {id: "ground", background: {src: "assets/maps/original.png"}, textures: {offsetX: 41, rotation: 12, fit: "contain", scaleX: .7}, history: [],
    getFlag(id, key) { assert.equal(id, MODULE_ID); assert.equal(key, "history"); return this.history; },
    async update(patch) { calls.push(patch); if (this.fail) throw new Error("Update failed");
      if (this.reject) return this;
      this.background.src = patch["background.src"]; this.history = patch[`flags.${MODULE_ID}.history`]; return this; }};
  scene.levels.set(level.id, level);
  const picker = {async browse(source, folder) { assert.equal(source, "data"); if (!folders.has(folder)) throw new Error("Missing"); },
    async createDirectory(source, folder) { folders.add(folder); },
    async upload(source, folder, file, body, options) { assert.equal(source, "data"); assert.equal(options.notify, false); files.push(file); return {path: `${folder}/${file.name}`}; }};
  const args = {scene, level, expected: targetSnapshot(scene, level), blob: new Blob(["RIFFtest"], {type: "image/webp"}),
    folder: "assets/maps/enhanced", name: "Temple", options: {...DEFAULTS}, dimensions: {width: 8000, height: 6000}, picker, unique: "unique"};
  return {scene, level, calls, files, folders, picker, args};
}

test("saving creates a unique WebP then switches only the level background; restore preserves geometry", async () => {
  const f = fixture(), before = targetSnapshot(f.scene, f.level);
  const path = await saveMap(f.args);
  assert.equal(path, "assets/maps/enhanced/Temple-enhanced-8000x6000-unique.webp");
  assert.equal(f.files[0].type, "image/webp"); assert.ok(f.folders.has("assets/maps/enhanced"));
  assert.deepEqual(Object.keys(f.calls[0]).sort(), ["background.src", "flags.map-enhancer.history"]);
  assert.equal(targetSnapshot(f.scene, f.level).footprint, before.footprint);
  assert.equal(f.scene.grid.size, 100); assert.deepEqual(f.scene.walls, [{c: [4, 6, 40, 60]}]);
  await restoreMap(f.scene, f.level); assert.equal(f.level.background.src, before.source); assert.equal(f.level.history.length, 0);
  assert.equal(f.files.length, 1, "Restore retains the saved image");
});

test("stale backgrounds, geometry, levels and permissions cause no upload or update", async () => {
  for (const change of [f => { f.level.background.src = "changed"; }, f => { f.level.textures.rotation++; },
    f => { f.scene.width++; }, f => { f.scene.levels.delete("ground"); }, f => { f.scene.allowed = false; }, () => { game.user.isGM = false; }]) {
    const f = fixture(); change(f);
    await assert.rejects(saveMap(f.args), /changed|GM/); assert.equal(f.calls.length, 0); assert.equal(f.files.length, 0);
  }
});

test("a scene change during upload does not overwrite the new background and reports the saved path", async () => {
  const f = fixture(); f.picker.upload = async () => { f.level.background.src = "another.png"; return {path: "assets/maps/processed.webp"}; };
  await assert.rejects(saveMap(f.args), /changed.*saved at assets\/maps\/processed.webp/);
  assert.equal(f.level.background.src, "another.png"); assert.equal(f.calls.length, 0);
});

test("upload failures leave the scene unchanged; update failures retain and identify the processed file", async () => {
  for (const result of [false, undefined, {error: "Upload denied"}, {}]) {
    const f = fixture(); f.picker.upload = async () => result;
    await assert.rejects(saveMap(f.args)); assert.equal(f.level.background.src, f.args.expected.source); assert.equal(f.calls.length, 0);
  }
  for (const failure of ["fail", "reject"]) {
    const f = fixture(); f.level[failure] = true;
    await assert.rejects(saveMap(f.args), /processed file is saved at/); assert.equal(f.files.length, 1);
    assert.equal(f.level.background.src, f.args.expected.source);
  }
});

test("invalid outputs are refused and restore never overwrites a later background change", async () => {
  const f = fixture();
  await assert.rejects(saveMap({...f.args, blob: new Blob()}), /valid WebP/);
  await assert.rejects(saveMap({...f.args, name: "../bad"}), /filename/);
  await assert.rejects(saveMap({...f.args, unique: undefined}), /filename/);
  assert.equal(f.files.length, 0);
  await saveMap(f.args); f.level.background.src = "later.png";
  await assert.rejects(restoreMap(f.scene, f.level), /matching/); assert.equal(f.level.background.src, "later.png");
});

test("history retains five saves and each subsequent restore returns to the preceding source", async () => {
  const f = fixture();
  for (let i = 0; i < 7; i++) await saveMap({...f.args, unique: `save-${i}`, expected: targetSnapshot(f.scene, f.level)});
  assert.equal(f.level.history.length, 5);
  for (let i = 0; i < 5; i++) await restoreMap(f.scene, f.level);
  assert.match(f.level.background.src, /save-1.webp$/); await assert.rejects(restoreMap(f.scene, f.level), /matching/);
});

test("a rejected native restore is reported without claiming the background changed", async () => {
  const f = fixture(); await saveMap(f.args); const path = f.level.background.src; f.level.reject = true;
  await assert.rejects(restoreMap(f.scene, f.level), /hook rejected/); assert.equal(f.level.background.src, path);
  assert.equal(f.level.history.length, 1);
});

test("the standalone module includes the pinned model, both local inference backends and licenses", async () => {
  const root = new URL("../map-enhancer/", import.meta.url);
  const manifest = JSON.parse(await readFile(new URL("module.json", root), "utf8"));
  for (const file of [...manifest.esmodules, ...manifest.styles, manifest.license, "templates/enhancer.hbs"]) await access(new URL(file, root));
  const pins = JSON.parse(await readFile(new URL("asset-pins.json", root), "utf8"));
  for (const [file, expected] of Object.entries(pins.files)) {
    const bytes = await readFile(new URL(file, root));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expected.sha256, file);
    assert.equal(bytes.length, expected.bytes, file);
  }
  assert.ok(pins.files["models/digital-art-4x.onnx"]);
  for (const name of ["ort.all.min.mjs", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) assert.ok(pins.files[`vendor/ort/${name}`]);
  for (const name of ["ONNX-Runtime-MIT.txt", "ONNX-Runtime-ThirdPartyNotices.txt", "Real-ESRGAN-BSD-3-Clause.txt"]) await access(new URL(`licenses/${name}`, root));
});
