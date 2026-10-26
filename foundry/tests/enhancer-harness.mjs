// Foundry API doubles only. The real module, worker, model and filters run below.
import {filterCPU, FilterEngine, filterCanvas} from "../map-enhancer/scripts/filters.mjs";
import {Processor} from "../map-enhancer/scripts/client.mjs";
import {DEFAULTS} from "../map-enhancer/scripts/settings.mjs";
const callbacks = new Map(), stored = new Map();
let application;
globalThis.Hooks = {on(name, fn) { const list = callbacks.get(name) ?? []; list.push(fn); callbacks.set(name, list); },
  once(name, fn) { this.on(name, fn); }, callAll(name, ...args) { for (const fn of callbacks.get(name) ?? []) fn(...args); }};
globalThis.game = {user: {isGM: true}, data: {options: {routePrefix: ""}}, modules: new Map([["map-enhancer", {}]]),
  settings: {register(id, key, config) { stored.set(key, config.default); }, get(id, key) { return stored.get(key); }, async set(id, key, value) { stored.set(key, value); }}};
globalThis.ui = {notifications: {info(message) { document.getElementById("test-results").textContent = message; }}};
class ApplicationV2 {
  constructor() { this.options = this.constructor.DEFAULT_OPTIONS; this.element = null; application = this; }
  async _onRender() {}
  async render() {
    const root = document.createElement("section"); root.id = this.options.id; root.className = this.options.classes.join(" ");
    root.innerHTML = `<header class="window-header"><strong>Map Enhancer</strong><button data-close>Close</button></header>`;
    const content = document.createElement("div"); content.className = "window-content";
    content.innerHTML = Handlebars.compile(await (await fetch("/modules/map-enhancer/templates/enhancer.hbs")).text())({});
    root.append(content); document.body.append(root); this.element = root;
    root.querySelector("[data-close]").onclick = () => this.close();
    root.addEventListener("click", event => { const target = event.target.closest("[data-action]"); if (target && !target.disabled) this.options.actions[target.dataset.action]?.call(this, event, target); });
    await this._onRender({}, {});
  }
  async close() { this.element?.remove(); this.element = null; }
}
const files = new Map(), calls = [];
const scene = {id: "test", name: "Test map", width: 5000, height: 3000, grid: {size: 100}, walls: [[10, 20, 40, 20]], canUserModify() { return true; }, levels: new Map()};
const level = {id: "ground", name: "Ground", background: {src: "/tests/fixtures/enhancer-input.png"}, textures: {rotation: 12, offsetX: 34, fit: "cover", scaleX: .75}, history: [],
  getFlag() { return this.history; }, async update(patch) {
    if (this.delay) await new Promise(resolve => setTimeout(resolve, this.delay));
    calls.push(structuredClone(patch)); this.background.src = patch["background.src"]; this.history = patch["flags.map-enhancer.history"]; readback(); }};
scene.levels.set(level.id, level);
const picker = {async browse() { return {}; }, async createDirectory() { return {}; }, async upload(source, folder, file) {
  const url = URL.createObjectURL(file); files.set(url, {folder, file}); return {path: url}; }};
globalThis.foundry = {utils: {getRoute() { return "/"; }, randomID(length) { return crypto.randomUUID().replaceAll("-", "").slice(0, length); }},
  applications: {api: {ApplicationV2, HandlebarsApplicationMixin: base => base}, apps: {FilePicker: {implementation: picker}}}};
globalThis.canvas = {ready: true, scene, level, app: {renderer: {}}};
function readback() { document.getElementById("documents").textContent = JSON.stringify({background: level.background.src, textures: level.textures, sceneSize: [scene.width, scene.height], grid: scene.grid, walls: scene.walls, updates: calls.length, uploads: files.size}, null, 2); }
readback();
document.getElementById("capabilities").textContent = `Secure context: ${isSecureContext} · WebGPU API: ${!!navigator.gpu} · ${navigator.userAgent}`;
await import("../map-enhancer/scripts/main.mjs"); Hooks.callAll("init"); Hooks.callAll("ready");
const report = document.getElementById("test-results");
const check = (value, message) => { if (!value) throw new Error(message); report.textContent += `\nPASS ${message}`; };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function idle() { const start = performance.now(); while (!application?.element || application.busy || application.committing || application.pendingFilter) { if (performance.now() - start > 240000) throw new Error("Timed out waiting for worker"); await delay(25); } }
const click = action => application.element.querySelector(`[data-action=${action}]`).click();
function option(key, value) { const field = application.element.querySelector(`[data-option=${key}]`); field.value = value; field.dispatchEvent(new Event("input", {bubbles: true})); }
async function run(button, task) {
  button.disabled = true; report.textContent = "Running…";
  try { await task(); report.textContent += "\nALL CHECKS PASSED"; }
  catch (error) { report.textContent += `\nFAIL ${error.stack}`; }
  finally { button.disabled = false; }
}
document.getElementById("open").onclick = () => game.modules.get("map-enhancer").api.open();
document.getElementById("local-map").onchange = async event => {
  if (!event.target.files[0]) return;
  level.background.src = URL.createObjectURL(event.target.files[0]); level.history = [];
  scene.name = event.target.files[0].name; readback();
  await game.modules.get("map-enhancer").api.open();
};
document.getElementById("filters").onclick = event => run(event.target, async () => {
  const engine = new FilterEngine(); check(!!engine.gl, "WebGL2 filter backend initialized");
  for (const [w, h] of [[19, 11], [530, 12]]) {
    const input = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < input.length; i += 4) { input[i] = (i * 17) % 256; input[i + 1] = (i * 29) % 256; input[i + 2] = (i * 41) % 256; input[i + 3] = i % 16 ? 255 : 128; }
    const options = {...DEFAULTS, smooth: .65, smoothRadius: 4, sharpen: .7, sharpRadius: 3, edgeTolerance: .25};
    const cpu = filterCPU(input, w, h, options), gpu = engine.process(input, w, h, options);
    const difference = gpu.reduce((max, n, i) => Math.max(max, Math.abs(n - cpu[i])), 0);
    check(difference <= 2, `CPU / GPU filters agree within 2 levels (${w}x${h}; actual ${difference})`);
    const image = new OffscreenCanvas(w, h); image.getContext("2d").putImageData(new ImageData(input, w, h), 0, 0);
    // Canvas premultiplication can round semitransparent RGB; compare its readback.
    const read = image.getContext("2d").getImageData(0, 0, w, h).data;
    const full = engine.process(read, w, h, options);
    const tiled = filterCanvas(image, options, engine).getContext("2d").getImageData(0, 0, w, h).data;
    const opaqueIndices = [...read.keys()].filter(i => read[i - i % 4 + 3] === 255);
    check(opaqueIndices.every(i => Math.abs(full[i] - tiled[i]) <= 2), "Tiled filter halo matches whole-image output at opaque pixels");
  }
});
const modelCheck = event => run(event.target, async () => {
  const backend = event.target.id === "gpu" ? "auto" : "cpu";
  const messages = [];
  const processor = new Processor(data => { messages.push(data.message); report.textContent = data.message; });
  try {
    const loaded = await processor.request("load", {source: new URL("fixtures/enhancer-input.png", location.href).href}); loaded.overview.close();
    const start = performance.now();
    const result = await processor.request("export", {options: {...DEFAULTS, scale: 4, sharpen: 0, quality: 1, backend}, textureLimit: 16383});
    check(result.provider === (backend === "cpu" ? "wasm" : "webgpu"), `${backend} backend performs real ONNX inference (${result.provider}) · ${messages.filter(m => m.includes("failed")).join("; ")}`);
    check(result.dimensions.width === loaded.width * 4 && result.dimensions.height === loaded.height * 4, "Model output dimensions are exactly 4x");
    const image = await createImageBitmap(result.blob), expected = await createImageBitmap(await (await fetch("fixtures/enhancer-reference.png")).blob());
    const c = new OffscreenCanvas(image.width, image.height), ctx = c.getContext("2d"); ctx.drawImage(image, 0, 0);
    const actual = ctx.getImageData(0, 0, image.width, image.height).data; ctx.clearRect(0, 0, c.width, c.height); ctx.drawImage(expected, 0, 0);
    const reference = ctx.getImageData(0, 0, c.width, c.height).data;
    let total = 0, count = 0; for (let i = 0; i < actual.length; i++) if (i % 4 !== 3) { total += Math.abs(actual[i] - reference[i]); count++; }
    check(total / count < 5, `Browser output agrees with native ONNX reference after WebP encoding (mean error ${(total / count).toFixed(3)})`);
    check(result.blob.type === "image/webp", `Encoded WebP in ${((performance.now() - start) / 1000).toFixed(2)}s (${result.blob.size} bytes)`);
    image.close(); expected.close();
  } finally { processor.stop(); }
});
document.getElementById("worker").onclick = modelCheck;
document.getElementById("gpu").onclick = modelCheck;
document.getElementById("ui-checks").onclick = event => run(event.target, async () => {
  await game.modules.get("map-enhancer").api.open(); await idle();
  const original = level.background.src, before = JSON.stringify({width: scene.width, height: scene.height, textures: level.textures, walls: scene.walls, grid: scene.grid});
  check(!application.sourceSize ? false : !application.hasPreview, "Map loads without altering the scene");
  const start = performance.now(); click("preview"); await idle();
  check(application.hasPreview, `Real model preview ready in ${((performance.now() - start) / 1000).toFixed(2)}s · ${application.message}`);
  const filterStart = performance.now();
  option("smooth", .3); option("sharpen", .65); option("smoothRadius", 3); await delay(120); await idle();
  check(application.hasPreview && !application.element.querySelector("[data-action=save]").disabled, `Rapid slider changes refresh the cached preview (${Math.round(performance.now() - filterStart)}ms including debounce)`);
  const savedCount = files.size, previousCallCount = calls.length, rotation = level.textures.rotation;
  level.textures.rotation++;
  click("save");
  check(application.error && !application.busy && files.size === savedCount, "A stale placement is refused before full-map inference or upload");
  level.textures.rotation = rotation;
  application.element.querySelector("[data-folder]").value = "../outside";
  click("save");
  check(application.error && !application.busy && files.size === savedCount, "An invalid asset folder is refused before full-map inference");
  application.element.querySelector("[data-folder]").value = "assets/maps/encounters";
  click("save"); await idle();
  check(files.size === savedCount + 1 && level.background.src !== original, "Save uploads one WebP and switches the current level");
  check(files.get(level.background.src).folder === "assets/maps/encounters", "The chosen Foundry Data asset folder receives the WebP");
  check(files.get(level.background.src).file.type === "image/webp", "Saved file is WebP");
  check(JSON.stringify({width: scene.width, height: scene.height, textures: level.textures, walls: scene.walls, grid: scene.grid}) === before, "Scene dimensions, texture transforms, walls and grid remain identical");
  check(Object.keys(calls[previousCallCount]).sort().join() === "background.src,flags.map-enhancer.history", "Native update writes only background path and restore history");
  level.delay = 100; click("restore");
  check(["preview", "save", "restore", "reload"].every(action => application.element.querySelector(`[data-action=${action}]`).disabled), "Restoring blocks overlapping actions during the native update");
  await idle(); level.delay = 0; check(level.background.src === original, "Restore returns to the original background");
  click("preview"); await delay(10); click("cancel"); await idle();
  check(level.background.src === original && files.size === savedCount + 1 && !application.hasPreview, "Cancel terminates processing without a scene change or upload");
  click("reload"); click("reload"); await idle();
  check(!!application.sourceSize && !application.error, "Consecutive reloads do not leave a stale error or loading state");
});
