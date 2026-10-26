import {MODULE_ID, DEFAULTS, settings, outputSize, previewRegion} from "./settings.mjs";
import {dataFolder, destination, targetSnapshot, checkTarget, saveMap, restoreMap} from "./assets.mjs";
import {Processor} from "./client.mjs";
import {Comparison} from "./preview.mjs";

const {ApplicationV2, HandlebarsApplicationMixin} = foundry.applications.api;
let panel, opening = false;

class MapEnhancer extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    id: "map-enhancer", classes: ["map-enhancer"],
    window: {title: "Map Enhancer", icon: "fa-solid fa-up-right-and-down-left-from-center", resizable: true},
    position: {width: 1000, height: 790},
    actions: {
      preview() { this.generate(); }, save() { this.save(); },
      cancel() { this.load("Processing cancelled. The scene was not changed."); },
      reload() { this.load(); }, restore() { this.restore(); }, fit() { this.viewer?.reset(); }
    }
  };
  static PARTS = {content: {template: "modules/map-enhancer/templates/enhancer.hbs", scrollable: [""]}};

  constructor() {
    super(); this.optionsValue = settings(game.settings.get(MODULE_ID, "tools"));
    this.busy = "load"; this.hasPreview = false; this.center = [0.5, 0.5]; this.closed = false;
    this.message = "Loading the current map…"; this.error = false;
  }

  async _onRender(context, options) {
    await super._onRender(context, options);
    this.listeners?.abort(); this.viewer?.dispose();
    this.listeners = new AbortController(); const signal = this.listeners.signal;
    this.viewer = new Comparison(this.element.querySelector("[data-comparison]"));
    this.element.addEventListener("input", event => this.input(event), {signal});
    this.element.querySelector("[data-overview]").addEventListener("click", event => {
      if (this.busy || this.committing) return;
      const bounds = event.target.getBoundingClientRect();
      this.center = [(event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height];
      this.hasPreview = false; this.viewer.reset(); this.drawOverview();
      this.status("Detail selected. Generate a preview for this area."); this.refresh();
    }, {signal});
    for (const field of this.element.querySelectorAll("[data-option]")) field.value = this.optionsValue[field.dataset.option];
    this.load();
  }

  status(message, error = false) {
    if (this.closed) return;
    this.message = message; this.error = error;
    const field = this.element?.querySelector("[data-status]");
    if (field) { field.textContent = message; field.classList.toggle("error", error); }
  }

  isCurrent() { return canvas.ready && canvas.scene === this.scene && canvas.level === this.level; }

  refresh() {
    if (this.closed || !this.element) return;
    const ready = !!this.sourceSize && this.isCurrent(), busy = !!this.busy || !!this.committing;
    for (const [action, disabled] of [["preview", !ready || busy], ["save", !ready || busy || !this.hasPreview],
      ["reload", this.committing], ["restore", !ready || busy || this.level?.background.src !== this.level?.getFlag(MODULE_ID, "history")?.at(-1)?.after]]) {
      this.element.querySelector(`[data-action=${action}]`).disabled = !!disabled;
    }
    this.element.querySelector("[data-action=cancel]").hidden = !busy || !!this.committing;
    for (const field of this.element.querySelectorAll("[data-option],[data-folder]")) field.disabled = this.busy === "export" || this.committing;
    for (const output of this.element.querySelectorAll("[data-value]")) {
      const name = output.dataset.value, value = this.optionsValue[name];
      output.value = ["smooth", "quality"].includes(name) ? `${Math.round(value * 100)}%` : Number(value.toFixed(2)).toString();
    }
    if (this.sourceSize) {
      const size = outputSize(this.sourceSize.width, this.sourceSize.height, this.optionsValue, this.textureLimit);
      this.element.querySelector("[data-output-size]").textContent = `${size.width} × ${size.height} px${size.limited ? " · capped to the image and texture limits" : ""}`;
    }
    this.status(this.message, this.error);
  }

  input(event) {
    const field = event.target;
    if (field.matches("[data-split]")) { this.viewer.split = Number(field.value) / 100; this.viewer.draw(); return; }
    if (!field.matches("[data-option]")) return;
    const name = field.dataset.option;
    this.optionsValue = settings({...this.optionsValue, [name]: name === "backend" ? field.value : Number(field.value)});
    clearTimeout(this.preferenceTimer);
    this.preferenceTimer = setTimeout(() => game.settings.set(MODULE_ID, "tools", this.optionsValue).catch(error => this.status(error.message, true)), 200);
    this.pendingFilter ||= this.hasPreview || this.busy === "preview" || this.busy === "filter";
    this.hasPreview = false; this.refresh();
    clearTimeout(this.filterTimer); this.filterTimer = setTimeout(() => this.refilter(), 80);
  }

  async load(message = "Choose a detail on the overview, then generate a preview.") {
    if (this.committing || this.closed) return;
    const generation = this.generation = (this.generation ?? 0) + 1;
    this.client?.stop(); this.client = null;
    this.busy = "load"; this.hasPreview = false; this.pendingFilter = false; this.sourceSize = null;
    this.viewer?.clear(); this.overview?.close(); this.overview = null;
    this.element.querySelector("[data-progress]").value = 0;
    clearTimeout(this.filterTimer);
    this.scene = canvas.scene; this.level = canvas.level;
    try {
      if (!game.user.isGM || !canvas.ready || !this.scene || !this.level?.background.src) throw new Error("Open a scene level with a background image as GM first.");
      this.expected = targetSnapshot(this.scene, this.level);
      const info = destination(this.expected.source, location.href, foundry.utils.getRoute(""));
      this.filename = info.name; this.element.querySelector("[data-folder]").value = info.folder;
      this.element.querySelector("[data-map]").textContent = `${this.scene.name} · ${this.level.name}`;
      const gl = canvas.app?.renderer?.gl;
      this.textureLimit = gl ? gl.getParameter(gl.MAX_TEXTURE_SIZE) : 16383;
      const client = this.client = new Processor(data => {
        if (this.client !== client) return;
        this.status(data.message);
        const meter = this.element.querySelector("[data-progress]");
        if (data.total) { meter.max = data.total; meter.value = data.completed; } else meter.removeAttribute("value");
      });
      this.status("Reading the current map…"); this.refresh();
      const data = await client.request("load", {source: new URL(this.expected.source, location.href).href});
      if (this.generation !== generation || this.closed) { data.overview.close(); return; }
      this.sourceSize = {width: data.width, height: data.height};
      this.overview?.close(); this.overview = data.overview; this.center = [0.5, 0.5]; this.drawOverview();
      this.element.querySelector("[data-source-size]").textContent = `${data.width} × ${data.height} px · Digital Art 4× model`;
      this.busy = null; this.status(message); this.refresh();
    } catch (error) { if (!this.closed && this.generation === generation) { this.busy = null; this.status(error.message, true); this.refresh(); } }
  }

  drawOverview() {
    if (!this.overview || !this.sourceSize) return;
    const canvas = this.element.querySelector("[data-overview]"); canvas.width = this.overview.width; canvas.height = this.overview.height;
    const ctx = canvas.getContext("2d"); ctx.drawImage(this.overview, 0, 0);
    const r = previewRegion(this.sourceSize.width, this.sourceSize.height, this.center);
    ctx.strokeStyle = "#f4c86a"; ctx.lineWidth = 2;
    ctx.strokeRect(r.x / this.sourceSize.width * canvas.width, r.y / this.sourceSize.height * canvas.height,
      r.width / this.sourceSize.width * canvas.width, r.height / this.sourceSize.height * canvas.height);
  }

  async generate() {
    if (this.busy || this.committing || !this.sourceSize || !this.isCurrent()) return;
    const client = this.client, options = {...this.optionsValue}; this.busy = "preview"; this.hasPreview = false; this.refresh();
    try {
      const region = previewRegion(this.sourceSize.width, this.sourceSize.height, this.center);
      const result = await client.request("preview", {region, options, textureLimit: this.textureLimit});
      if (this.client !== client || this.closed) { result.before.close(); result.after.close(); return; }
      this.pendingFilter ||= JSON.stringify(options) !== JSON.stringify(this.optionsValue);
      this.viewer.set(result.before, result.after); this.hasPreview = !this.pendingFilter;
      this.status(`Preview ready · ${result.provider === "webgpu" ? "GPU" : "CPU"}. Adjust the sliders, then save the whole map.`);
    } catch (error) { if (this.client === client) this.status(error.message, true); }
    finally { if (this.client === client && !this.closed) { this.busy = null; this.refresh(); this.refilter(); } }
  }

  async refilter() {
    if (this.busy || this.committing || !this.pendingFilter || !this.client || this.closed) return;
    this.pendingFilter = false; this.busy = "filter";
    const client = this.client; this.refresh();
    try {
      const result = await client.request("filter", {options: {...this.optionsValue}});
      if (this.client !== client || this.closed) { result.before.close(); result.after.close(); return; }
      this.viewer.set(result.before, result.after); this.hasPreview = !this.pendingFilter;
      this.status("Preview updated. Save processes the entire map with these settings.");
    } catch (error) { if (this.client === client) this.status(error.message, true); }
    finally { if (this.client === client && !this.closed) { this.busy = null; this.refresh(); this.refilter(); } }
  }

  async save() {
    if (this.busy || this.committing || !this.hasPreview || !this.isCurrent()) return;
    let folder;
    try { checkTarget(this.scene, this.level, this.expected); folder = dataFolder(this.element.querySelector("[data-folder]").value); }
    catch (error) { this.status(error.message, true); return; }
    const client = this.client, options = {...this.optionsValue};
    this.busy = "export"; this.refresh();
    try {
      const result = await client.request("export", {options, textureLimit: this.textureLimit});
      if (this.client !== client || this.closed) return;
      this.committing = true; this.status("Saving the WebP and updating this level’s background…"); this.refresh();
      if (!this.isCurrent()) throw new Error("The viewed scene changed. Return to the original scene before saving.");
      const path = await saveMap({scene: this.scene, level: this.level, expected: this.expected, blob: result.blob,
        folder, name: this.filename, options, dimensions: result.dimensions, unique: foundry.utils.randomID(24),
        picker: foundry.applications.apps.FilePicker.implementation});
      this.committing = false; await this.load(`Saved and switched to ${path}`);
    } catch (error) { if (this.client === client) this.status(error.message, true); }
    finally { this.committing = false; if (this.client === client && !this.closed) { this.busy = null; this.refresh(); } }
  }

  async restore() {
    if (this.busy || this.committing || !this.isCurrent()) return;
    this.committing = true; this.status("Restoring the previous background…"); this.refresh();
    try { await restoreMap(this.scene, this.level); this.committing = false; await this.load("The previous background is restored. Processed files remain available in the asset folder."); }
    catch (error) { this.status(error.message, true); }
    finally { this.committing = false; this.refresh(); }
  }

  async close(options) {
    if (this.committing) { ui.notifications.info("The map is being saved. Close the panel when saving finishes."); return this; }
    this.closed = true; this.client?.stop(); this.viewer?.dispose(); this.overview?.close(); this.listeners?.abort();
    clearTimeout(this.filterTimer); clearTimeout(this.preferenceTimer);
    return super.close(options);
  }
}

async function open() {
  if (!game.user.isGM || opening || panel?.committing) return;
  opening = true;
  try { await panel?.close(); panel = new MapEnhancer(); await panel.render({force: true}); }
  finally { opening = false; }
}
Hooks.once("init", () => game.settings.register(MODULE_ID, "tools", {scope: "client", config: false, type: Object, default: DEFAULTS}));
Hooks.once("ready", () => { game.modules.get(MODULE_ID).api = Object.freeze({open}); });
Hooks.on("getSceneControlButtons", controls => {
  if (!game.user.isGM || !controls.tiles) return;
  controls.tiles.tools.mapEnhancer = {name: "mapEnhancer", title: "Enhance current map", icon: "fa-solid fa-up-right-and-down-left-from-center",
    order: Object.keys(controls.tiles.tools).length, button: true, visible: true, onChange: open};
});
Hooks.on("canvasReady", () => panel?.refresh());
