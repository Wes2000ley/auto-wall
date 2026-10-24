// Explicit test doubles for Foundry; this file is never installed in the module.
import {Scene, wall} from "./mock-scene.mjs";
import {crosses} from "./browser-ray.mjs";
const callbacks = new Map();
globalThis.Hooks = {
  on(name, callback) { const list = callbacks.get(name) ?? []; list.push({callback}); callbacks.set(name, list); },
  once(name, callback) { const list = callbacks.get(name) ?? []; list.push({callback, once: true}); callbacks.set(name, list); },
  callAll(name, ...args) {
    const list = [...(callbacks.get(name) ?? [])]; callbacks.set(name, list.filter(c => !c.once));
    for (const {callback} of list) callback(...args);
  }
};
const settings = new Map(); let application;
globalThis.game = {user: {isGM: true}, modules: new Map([["auto-wall-live", {}]]),
  settings: {register(id, key, config) { if (!settings.has(key)) settings.set(key, structuredClone(config.default)); },
    get(id, key) { return structuredClone(settings.get(key)); }, async set(id, key, value) { settings.set(key, structuredClone(value)); }}};
class ApplicationV2 {
  constructor() { this.options = this.constructor.DEFAULT_OPTIONS; this.element = null; this.rendered = false; application = this; }
  async _prepareContext() { return {}; }
  _onRender() {}
  async render() {
    this.element?.remove(); const root = document.createElement("section"); root.id = this.options.id; root.className = "auto-wall-live";
    root.innerHTML = `<header class="window-header"><strong>Auto-Wall Live</strong><button data-close aria-label="Close editor">×</button></header>`;
    const content = document.createElement("div"); content.className = "window-content";
    content.innerHTML = Handlebars.compile(await (await fetch("/modules/auto-wall-live/templates/editor.hbs")).text())(await this._prepareContext());
    root.append(content); document.body.append(root); this.element = root; this.rendered = true;
    root.querySelector("[data-close]").onclick = () => this.close();
    root.addEventListener("click", event => {
      const target = event.target.closest("[data-action]");
      if (target) this.options.actions[target.dataset.action]?.call(this, event, target);
    });
    this._onRender({}, {});
  }
  async close() { this.element?.remove(); this.element = null; this.rendered = false; }
}
globalThis.foundry = {applications: {api: {ApplicationV2, HandlebarsApplicationMixin: base => base}}};
class Point { constructor(x = 0, y = 0) { this.x = x; this.y = y; } }
globalThis.PIXI = {Point};
const scene = new Scene([wall("nativeUpper", [40, 20, 180, 20], {levels: ["upper"]})]);
Object.assign(scene, {id: "testScene", name: "Palette room", width: 1024, height: 768});
const level = {id: "ground", name: "Ground", background: {src: "fixture"}, textures: {rotation: 0}};
const source = document.createElement("canvas"); source.width = source.height = 512;
const image = source.getContext("2d");
image.fillStyle = "#000"; image.fillRect(0, 0, 512, 512);
image.fillStyle = "#2d2522"; image.fillRect(60, 60, 390, 380);
image.fillStyle = "#a0a0a0";
for (const rect of [[60, 60, 390, 20], [60, 420, 390, 20], [60, 60, 20, 380], [430, 60, 20, 380]]) image.fillRect(...rect);
image.fillStyle = "#702090"; image.fillRect(220, 60, 30, 20);
image.fillStyle = "#ffe646"; image.fillRect(320, 300, 14, 14);
let angle = 0, offset = [75, 80];
const stage = {pivot: {x: innerWidth / 2, y: innerHeight / 2}, scale: {x: 1, y: 1},
  toGlobal(p) { return new Point((p.x - this.pivot.x) * this.scale.x + innerWidth / 2, (p.y - this.pivot.y) * this.scale.y + innerHeight / 2); },
  toLocal(p) { return new Point((p.x - innerWidth / 2) / this.scale.x + this.pivot.x, (p.y - innerHeight / 2) / this.scale.y + this.pivot.y); }};
const mesh = {anchor: {x: 0, y: 0}, texture: {valid: true, orig: {width: 512, height: 512}},
  toGlobal(p) { return stage.toGlobal(new Point(offset[0] + p.x * Math.cos(angle) - p.y * Math.sin(angle), offset[1] + p.x * Math.sin(angle) + p.y * Math.cos(angle))); },
  toLocal(p) { const s = stage.toLocal(p), x = s.x - offset[0], y = s.y - offset[1]; return new Point(x * Math.cos(angle) + y * Math.sin(angle), -x * Math.sin(angle) + y * Math.cos(angle)); }};
const board = document.getElementById("board");
globalThis.canvas = {ready: true, scene, level, stage, dimensions: {width: 1024, height: 768},
  primary: {background: mesh, backgroundSource: source}, app: {view: board, screen: {width: innerWidth, height: innerHeight}},
  pan({x, y, scale}) { if (x !== undefined) stage.pivot.x = x; if (y !== undefined) stage.pivot.y = y;
    if (scale !== undefined) stage.scale.x = stage.scale.y = Math.max(0.2, Math.min(4, scale)); draw(); Hooks.callAll("canvasPan", canvas); }};
function draw() {
  board.width = innerWidth; board.height = innerHeight; canvas.app.screen = {width: innerWidth, height: innerHeight};
  const ctx = board.getContext("2d"); ctx.fillStyle = "#1c2027"; ctx.fillRect(0, 0, board.width, board.height);
  ctx.translate(innerWidth / 2, innerHeight / 2); ctx.scale(stage.scale.x, stage.scale.y); ctx.translate(-stage.pivot.x, -stage.pivot.y);
  ctx.translate(...offset); ctx.rotate(angle); ctx.drawImage(source, 0, 0);
}
function readback() {
  document.getElementById("counts").textContent = `${scene.walls.size} native walls · ${scene.lights.size} native lights · ${scene.calls.length} document API calls`;
  document.getElementById("documents").textContent = JSON.stringify([...scene.walls].map(d => d.toObject()).map(({_id, c, door, sight, move, levels}) => ({_id, c, door, sight, move, levels})), null, 1);
}
for (const method of ["createEmbeddedDocuments", "updateEmbeddedDocuments", "deleteEmbeddedDocuments", "update"]) {
  const original = scene[method].bind(scene);
  scene[method] = async (...args) => { const result = await original(...args); readback(); return result; };
}
window.addEventListener("resize", draw); draw(); readback();
await import("../auto-wall-live/scripts/main.mjs"); Hooks.callAll("init"); Hooks.callAll("ready");
document.getElementById("open").onclick = () => game.modules.get("auto-wall-live").api.open();
document.getElementById("transform").onclick = () => { angle = angle ? 0 : 0.2; offset = angle ? [170, 50] : [75, 80]; level.textures.rotation = angle * 180 / Math.PI; draw(); application?.editor?.draw(); };
const waitIdle = async editor => { while (editor.sync.busy || editor.worker) await new Promise(resolve => setTimeout(resolve, 20)); };
document.getElementById("checks").onclick = async () => {
  const output = document.getElementById("test-results"); output.textContent = "Running real event-handler and worker checks…";
  const passed = [];
  const check = (value, message) => { if (!value) throw new Error(message); passed.push(message); };
  try {
    if (!application?.rendered) { game.modules.get("auto-wall-live").api.open(); while (!application?.editor) await new Promise(resolve => setTimeout(resolve, 10)); }
    const editor = application.editor; editor.snap = false; editor.setEnabled(true);
    const pointer = (type, x, y, button = 0) => editor.overlay.dispatchEvent(new PointerEvent(type, {clientX: x, clientY: y, pointerId: 1, button, bubbles: true}));
    const drag = async (a, b, button = 0) => { pointer("pointerdown", ...a, button); pointer("pointermove", ...b, button); pointer("pointerup", ...b, button); await waitIdle(editor); };
    editor.layer = "walls"; editor.brush = true; editor.diameter = 40;
    await drag([120, 300], [480, 300]); check(editor.composed.walls.length > 0, "Brush stroke saved native walls");
    const before = editor.composed.walls.length; await drag([120, 300], [480, 300]);
    check(editor.composed.walls.length === before, "Repeated paint merged without duplicate walls");
    editor.layer = "doors"; editor.diameter = 60; await drag([260, 300], [320, 300]);
    check(editor.composed.doors.length === 1, "Door gesture saved one portal");
    check(!editor.composed.walls.some(s => crosses(s, [290, 240], [290, 360])), "Opened doorway has no blocking wall faces");
    editor.layer = "invisible"; await drag([120, 450], [480, 450]);
    check([...scene.walls].some(d => d.data.sight === 0 && d.data.move === 20), "Invisible walls allow sight and block movement");
    const pivot = stage.pivot.x; await drag([500, 200], [530, 200], 2);
    check(stage.pivot.x !== pivot, "Right drag pans the map"); canvas.pan({x: innerWidth / 2});
    editor.layer = "walls"; editor.brush = false; await drag([110, 500], [220, 590]);
    check(editor.geometry.base.walls.length >= 4, "Rectangle gesture adds four perimeter walls");
    const size = editor.diameter; editor.brush = true;
    editor.overlay.dispatchEvent(new WheelEvent("wheel", {deltaY: -100, cancelable: true}));
    check(editor.diameter > size, "Wheel increases brush diameter");
    await drag([105, 495], [225, 595], 1);
    check(!editor.geometry.base.walls.some(s => s.every(p => p[1] >= 500)), "Middle drag erases the rectangle");
    await editor.undo(); check(editor.geometry.base.walls.length >= 4, "Undo restores erased walls");
    await editor.detect(); await waitIdle(editor);
    check(editor.composed.doors.length >= 2, "Bundled WASM worker adds a detected door beside manual geometry");
    check(scene.walls.has("nativeUpper"), "Detection preserves walls on the other level");
    check(!application.error, "Automatic detection completed without an error");
    const nativeCount = () => [...scene.walls].filter(d => !d.data.levels.length || d.data.levels.includes("ground")).length;
    const overlayCount = () => Object.values(editor.composed).reduce((sum, values) => sum + values.length, 0);
    check(nativeCount() === overlayCount(), "Overlay geometry matches the persisted native wall count");
    angle = 0.2; offset = [170, 50]; level.textures.rotation = angle * 180 / Math.PI; draw();
    await editor.detect(); await waitIdle(editor);
    const expectedCenter = [170 + 234.5 * Math.cos(angle) - 69.5 * Math.sin(angle), 50 + 234.5 * Math.sin(angle) + 69.5 * Math.cos(angle)];
    const detectedDoors = [...scene.walls].filter(d => d.data.door && d.data.flags?.["auto-wall-live"]?.origin === "detection");
    check(detectedDoors.some(d => Math.hypot((d.data.c[0] + d.data.c[2]) / 2 - expectedCenter[0], (d.data.c[1] + d.data.c[3]) / 2 - expectedCenter[1]) < 2), "Rotated and offset background detection uses correct scene coordinates");
    check(nativeCount() === overlayCount(), "Transformed detection keeps overlay and native geometry consistent");
    const saved = editor.geometry.serialize(); await application.close();
    game.modules.get("auto-wall-live").api.open(); while (!application?.editor) await new Promise(resolve => setTimeout(resolve, 10));
    check(JSON.stringify(application.editor.geometry.serialize()) === JSON.stringify(saved), "Reopening restores vector ink from scene flags");
    output.textContent = `PASS ${passed.length} checks\n${passed.join("\n")}`;
  } catch (error) { output.textContent = `FAIL after ${passed.length} checks: ${error.message}\n${passed.join("\n")}`; console.error(error); }
};
