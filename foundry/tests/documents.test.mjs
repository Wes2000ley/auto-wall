import test from "node:test";
import assert from "node:assert/strict";
import {Scene, wall} from "./helpers.mjs";
import {MODULE_ID} from "../auto-wall-live/scripts/settings.mjs";
import {LiveSync, fingerprint, geometryFromDocuments, levelSnapshot, planWalls, snapshot, lightData, wallData} from "../auto-wall-live/scripts/documents.mjs";

const read = (scene, level = "ground") => levelSnapshot(scene, level);
const commit = (sync, geometry, extra = {}) => sync.commit({geometry, ...extra}, fingerprint(sync.state()));

test("native document fields distinguish walls, doors, invisible walls and windows", () => {
  const segment = [[10.2, 20.4], [30.6, 40.8]];
  assert.deepEqual(wallData("walls", segment, "ground").c, [10, 20, 31, 41]);
  assert.equal(wallData("doors", segment, "ground").door, 1);
  assert.equal(wallData("invisible", segment, "ground").move, 20);
  assert.equal(wallData("invisible", segment, "ground").sight, 0);
  assert.equal(wallData("windows", segment, "ground").sight, 30);
  assert.equal(lightData({x: 10, y: 20, range: 6, color: "#ffa000"}, "ground", 10).config.dim, 60);
});

test("untouched native IDs, open doors, flags and other levels survive an edit", async () => {
  const original = wall("native", [10, 10, 100, 10], {door: 2, ds: 1, doorSound: "magic", animation: {duration: 200}});
  const upper = wall("upperWall", [20, 20, 200, 20], {levels: ["upper"]});
  const scene = new Scene([original, upper]), sync = new LiveSync(scene, "ground");
  const geometry = geometryFromDocuments(read(scene).walls); geometry.walls.push([[20, 100], [200, 100]]);
  await commit(sync, geometry);
  assert.deepEqual(scene.walls.get("native").toObject(), original);
  assert.deepEqual(scene.walls.get("upperWall").toObject(), upper);
  assert.equal(scene.walls.size, 3);
  await sync.undo(); assert.equal(scene.walls.size, 2);
});

test("shared and global walls are removed only from the edited level", async () => {
  const scene = new Scene([wall("shared", [10, 10, 100, 10], {levels: ["ground", "upper"]}), wall("global", [10, 20, 100, 20], {levels: []})]);
  const sync = new LiveSync(scene, "ground");
  await commit(sync, {walls: []});
  assert.deepEqual(snapshot(scene.walls.get("shared")).levels, ["upper"]);
  assert.deepEqual(snapshot(scene.walls.get("global")).levels, ["upper"]);
  await sync.undo();
  assert.deepEqual(snapshot(scene.walls.get("global")).levels, []);
});

test("split wall fragments inherit native direction, restrictions and flags", () => {
  const original = wall("custom", [10, 10, 200, 10], {dir: 1, move: 0, flags: {important: {value: "keep"}}});
  const scene = new Scene([original]);
  const plan = planWalls(scene, "ground", read(scene).walls, {walls: [[[10, 10], [60, 10]], [[90, 10], [200, 10]]]});
  assert.equal(plan.create.length, 2);
  assert.equal(plan.create[0].dir, 1); assert.equal(plan.create[0].move, 0);
  assert.equal(plan.create[0].flags.important.value, "keep");
  assert.equal(plan.create[0].flags[MODULE_ID].origin, "manual");
});

test("stale edits and insufficient permission cause no native writes", async () => {
  const scene = new Scene([wall("a", [0, 0, 100, 0])]), sync = new LiveSync(scene, "ground");
  const expected = fingerprint(sync.state());
  scene.walls.get("a").data.c[2] = 120;
  await assert.rejects(sync.commit({geometry: {}}, expected), /changed/);
  scene.allowed = false;
  await assert.rejects(commit(sync, {}), /GM/);
  assert.equal(scene.calls.length, 0);
});

test("a failed delete rolls back successful creates and keeps the old walls", async () => {
  const original = wall("a", [0, 0, 100, 0]);
  const scene = new Scene([original]), sync = new LiveSync(scene, "ground"); scene.failure = "delete";
  await assert.rejects(commit(sync, {walls: [[[10, 20], [80, 20]]]}), /Delete failed/);
  assert.deepEqual(read(scene).walls, [original]); assert.equal(sync.undoStack.length, 0); assert.equal(sync.busy, false);
});

test("undo refuses to overwrite a subsequent native door edit", async () => {
  const scene = new Scene(), sync = new LiveSync(scene, "ground");
  await commit(sync, {doors: [[[10, 10], [50, 10]]]});
  const created = [...scene.walls][0]; created.data.ds = 1;
  await assert.rejects(sync.undo(), /changed elsewhere/);
  assert.equal(scene.walls.get(created.id).data.ds, 1);
});

test("undo preserves unrelated native documents created afterward", async () => {
  const scene = new Scene(), sync = new LiveSync(scene, "ground");
  await commit(sync, {walls: [[[0, 0], [100, 0]]]});
  await scene.createEmbeddedDocuments("Wall", [wall("extra", [200, 200, 300, 200])], {keepId: true});
  await sync.undo(); assert.deepEqual([...scene.walls].map(d => d.id), ["extra"]);
});

test("manual light edits retain existing lights and undo restores a removed legacy light", async () => {
  const native = {_id: "nativeLight", x: 10, y: 20, levels: ["ground"], config: {dim: 30}};
  const auto = {...native, _id: "autoLight", flags: {[MODULE_ID]: {origin: "detection"}}};
  const scene = new Scene([], [native, auto]), sync = new LiveSync(scene, "ground");
  await commit(sync, {}, {lights: [native, {x: 40, y: 60, range: 4, color: "#ff9900"}]});
  assert.ok(scene.lights.has("nativeLight")); assert.equal(scene.lights.has("autoLight"), false);
  await sync.undo(); assert.deepEqual(read(scene).lights.sort((a, b) => a._id.localeCompare(b._id)), [auto, native]);
});

test("duplicate desired segments never create duplicate native walls", async () => {
  const scene = new Scene(), sync = new LiveSync(scene, "ground"), segment = [[0, 0], [100, 0]];
  await commit(sync, {walls: [segment, segment.toReversed()]}); assert.equal(scene.walls.size, 1);
});

test("native geometry created during a save survives; stale brush state is not stored", async () => {
  const scene = new Scene(), sync = new LiveSync(scene, "ground");
  scene.afterCreate = () => {
    scene.afterCreate = null;
    scene.walls.set("concurrent", {id: "concurrent", toObject: () => wall("concurrent", [20, 20, 80, 20])});
  };
  await assert.rejects(commit(sync, {walls: [[[0, 0], [100, 0]]]}), /changed while saving/);
  assert.deepEqual([...scene.walls].map(d => d.id), ["concurrent"]);
  assert.equal(sync.saved(), null);
});

test("a native change to a deletion target during saving is preserved", async () => {
  const scene = new Scene([wall("a", [0, 0, 100, 0])]), sync = new LiveSync(scene, "ground");
  scene.afterCreate = () => { scene.afterCreate = null; scene.walls.get("a").data.flags.newFlag = "preserve"; };
  await assert.rejects(commit(sync, {walls: [[[0, 30], [100, 30]]]}), /targeted document changed/);
  assert.equal(scene.walls.size, 1); assert.equal(scene.walls.get("a").data.flags.newFlag, "preserve");
});

test("a cancelled partial batch is reported and its successful creates roll back", async () => {
  const scene = new Scene(), sync = new LiveSync(scene, "ground");
  const create = scene.createEmbeddedDocuments.bind(scene);
  scene.createEmbeddedDocuments = (type, values, options) => create(type, values.slice(0, 1), options);
  await assert.rejects(commit(sync, {walls: [[[0, 0], [100, 0]], [[0, 20], [100, 20]]]}), /complete edit/);
  assert.equal(scene.walls.size, 0); assert.equal(sync.busy, false);
});
