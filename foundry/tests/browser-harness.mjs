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
    root.innerHTML = `<header class="window-header"><strong>Auto-Wall Live</strong><button data-close aria-label="Close editor">X</button></header>`;
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
  canvasCoordinatesFromClient({x, y}) { return stage.toLocal(new Point(x, y)); },
  clientCoordinatesFromCanvas({x, y}) { return stage.toGlobal(new Point(x, y)); },
  pan({x, y, scale}) { if (x !== undefined) stage.pivot.x = x; if (y !== undefined) stage.pivot.y = y;
    if (scale !== undefined) stage.scale.x = stage.scale.y = Math.max(0.2, Math.min(4, scale)); draw(); Hooks.callAll("canvasPan", canvas); }};
function draw() {
  board.width = innerWidth; board.height = innerHeight; canvas.app.screen = {width: innerWidth, height: innerHeight};
  const ctx = board.getContext("2d"); ctx.fillStyle = "#1c2027"; ctx.fillRect(0, 0, board.width, board.height);
  ctx.translate(innerWidth / 2, innerHeight / 2); ctx.scale(stage.scale.x, stage.scale.y); ctx.translate(-stage.pivot.x, -stage.pivot.y);
  ctx.translate(...offset); ctx.rotate(angle); ctx.drawImage(source, 0, 0);
}
function readback() {
  document.getElementById("counts").textContent = `${scene.walls.size} native walls | ${scene.lights.size} native lights | ${scene.calls.length} document API calls`;
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
const waitIdle = async editor => { while (editor.sync.busy || editor.lineSave) await new Promise(resolve => setTimeout(resolve, 10)); };
const sameSegment = (a, b) => a.map(p => p.map(Math.round).join(",")).sort().join("|") === b.map(p => p.map(Math.round).join(",")).sort().join("|");
const rectangleEdges = (a, b) => [[a, [b[0], a[1]]], [[b[0], a[1]], b], [b, [a[0], b[1]]], [[a[0], b[1]], a]];

document.getElementById("checks").onclick = async () => {
  const output = document.getElementById("test-results"); output.textContent = "Running manual drawing interaction checks...";
  const passed = [];
  const check = (value, message) => { if (!value) throw new Error(message); passed.push(message); };
  try {
    if (!application?.rendered) {
      game.modules.get("auto-wall-live").api.open();
      while (!application?.editor) await new Promise(resolve => setTimeout(resolve, 10));
    }
    const editor = application.editor; editor.setEnabled(true);
    const client = p => { const c = canvas.clientCoordinatesFromCanvas({x: p[0], y: p[1]}); return [c.x, c.y]; };
    const pointer = (type, p, button = 0) => {
      const [clientX, clientY] = client(p);
      editor.overlay.dispatchEvent(new PointerEvent(type, {clientX, clientY, pointerId: 1, button, bubbles: true}));
    };
    const drag = async (a, b, button = 0) => {
      pointer("pointerdown", a, button); pointer("pointermove", b, button); pointer("pointerup", b, button); await waitIdle(editor);
    };
    const click = p => drag(p, p);
    const key = code => window.dispatchEvent(new KeyboardEvent("keydown", {code, bubbles: true}));
    // All manual tools must work even if the background API is unavailable.
    Object.defineProperty(canvas, "primary", {configurable: true, get() { throw new Error("Manual drawing read the image"); }});
    editor.layer = "walls"; editor.setMode("brush"); editor.diameter = 40;
    await drag([120, 300], [480, 300]);
    check(editor.composed.walls.length > 0 && editor.composed.walls.length <= 12, "Paint saves a coarse native outline with few vertices");
    const before = editor.composed.walls.length; await drag([120, 300], [480, 300]);
    check(editor.composed.walls.length === before, "Repeated paint merges without duplicate walls");

    editor.setMode("rectangle");
    await drag([160, 280], [440, 440]);
    check(rectangleEdges([160, 280], [440, 440]).every(edge => editor.composed.walls.some(s => sameSegment(s, edge))), "Rectangle retains all four exact edges over older brush ink");
    await editor.undo();
    check(editor.geometry.lines.walls.length === 0, "Undo removes the complete rectangle and retains earlier paint");
    await drag([121, 281], [452, 430]);
    check(rectangleEdges([121, 281], [452, 430]).every(edge => editor.composed.walls.some(s => sameSegment(s, edge))), "Rectangle corners remain exact near existing wall endpoints");

    editor.layer = "doors"; editor.diameter = 60; await drag([260, 300], [320, 300]);
    check(editor.composed.doors.length === 1, "A manual door gesture saves one portal");
    check(!editor.composed.walls.some(s => crosses(s, [290, 240], [290, 360])), "The manually opened doorway has no blocking wall faces");
    editor.layer = "invisible"; await drag([120, 460], [480, 490]);
    check([...scene.walls].some(d => d.data.sight === 0 && d.data.move === 20), "Invisible rectangles allow sight and block movement");

    editor.layer = "walls"; editor.setMode("line");
    const count = scene.walls.size; await click([120, 550]);
    for (let x = 125; x < 300; x += 4) pointer("pointermove", [x, 550 + Math.sin(x)]);
    check(scene.walls.size === count, "Line preview never creates vertices from mouse movement");
    await click([300, 550]); await click([300, 620]);
    check(scene.walls.size === count + 2, "Three clicks save exactly two connected straight line segments");
    key("Enter"); check(!editor.line && scene.walls.size === count + 2, "Enter finishes the line without adding its preview");
    await click([400, 550]); await click([500, 550]);
    const pivot = stage.pivot.x; await drag([500, 200], [530, 200], 2);
    check(stage.pivot.x !== pivot && !!editor.line, "Right drag pans without losing the clicked line anchor");
    key("Escape"); check(!editor.line && editor.enabled, "Esc ends the line while keeping editing enabled");
    canvas.pan({x: innerWidth / 2, y: innerHeight / 2});

    editor.setMode("brush"); const size = editor.diameter;
    editor.overlay.dispatchEvent(new WheelEvent("wheel", {deltaY: -100, clientX: 400, clientY: 300, cancelable: true}));
    check(editor.diameter > size, "The paint wheel increases brush diameter");
    const wallCount = scene.walls.size; await drag([295, 545], [305, 625], 1);
    check(scene.walls.size < wallCount, "Middle drag erases selected native line segments");
    await editor.undo(); check(scene.walls.size === wallCount, "Undo restores erased line segments");

    editor.setMode("rectangle"); angle = 0.8; offset = [170, 50]; source.width = 200;
    level.textures.rotation = angle * 180 / Math.PI; draw();
    canvas.pan({x: 420, y: 330, scale: 1.6});
    canvas.app.screen = {width: innerWidth * 2, height: innerHeight * 2};
    await drag([620, 380], [530, 310]);
    check(rectangleEdges([620, 380], [530, 310]).every(edge => editor.composed.walls.some(s => sameSegment(s, edge))), "Reverse rectangle follows native pan/zoom coordinates despite changed image and renderer size");
    canvas.pan({x: innerWidth / 2, y: innerHeight / 2, scale: 1});
    editor.layer = "lights"; editor.lightColor = "#abcdef"; editor.lightRange = 8; await click([500, 620]);
    check([...scene.lights].some(d => d.data.config.color === "#abcdef" && d.data.config.dim === 40), "Manual light placement saves the selected color and radius");
    check(scene.walls.has("nativeUpper"), "Manual editing preserves walls on another level");
    const nativeCount = [...scene.walls].filter(d => !d.data.levels.length || d.data.levels.includes("ground")).length;
    check(nativeCount === Object.values(editor.composed).reduce((sum, values) => sum + values.length, 0), "Overlay geometry matches persisted native walls");
    check(!document.querySelector('[data-action="detect"]') && !document.querySelector('[data-profile]') && !document.querySelector('[data-snap]'), "The panel has no automatic detection, palette, or snapping controls");
    const saved = editor.geometry.serialize(); await application.close();
    game.modules.get("auto-wall-live").api.open(); while (!application?.editor) await new Promise(resolve => setTimeout(resolve, 10));
    check(JSON.stringify(application.editor.geometry.serialize()) === JSON.stringify(saved), "Reopening restores painted ink and explicit lines from scene flags");
    application.editor.layer = "walls"; application.editor.setMode("rectangle"); application.refresh();
    output.textContent = `PASS ${passed.length} checks\n${passed.join("\n")}`;
  } catch (error) { output.textContent = `FAIL after ${passed.length} checks: ${error.message}\n${passed.join("\n")}`; console.error(error); }
};
