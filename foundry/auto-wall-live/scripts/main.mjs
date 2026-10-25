import {MODULE_ID, DEFAULT_TOOLS, toolPreferences} from "./settings.mjs";
import {CanvasEditor} from "./editor.mjs";

let panel;
const {ApplicationV2, HandlebarsApplicationMixin} = foundry.applications.api;

class AutoWallPanel extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    id: "auto-wall-live", classes: ["auto-wall-live"],
    window: {title: "Auto-Wall Live", icon: "fa-solid fa-paintbrush", resizable: true},
    position: {width: 390, height: 560},
    actions: {
      async tool(event, target) {
        if (this.editor) {
          this.editor.cancel(); this.editor.layer = target.dataset.layer;
          this.editor.visible[this.editor.layer] = true;
        }
        this.refresh(); await this.saveTools();
      },
      async mode(event, target) { this.editor?.setMode(target.dataset.mode); await this.saveTools(); },
      finishLine() { this.editor?.cancel(); this.status = "Line finished. Click to start another."; this.refresh(); },
      editing() { if (this.editor) this.editor.setEnabled(!this.editor.enabled); },
      undo() { this.editor?.undo(); }
    }
  };
  static PARTS = {content: {template: "modules/auto-wall-live/templates/editor.hbs", scrollable: [""]}};

  constructor() {
    super();
    this.status = "Draw directly on the scene. Every completed edit saves automatically.";
    this.error = false; this.editor = null;
  }

  async _prepareContext(options) {
    return {...await super._prepareContext(options), tools: [
      {layer: "walls", label: "1 Wall"}, {layer: "doors", label: "2 Door"},
      {layer: "invisible", label: "3 Invisible"}, {layer: "windows", label: "4 Window"}, {layer: "lights", label: "5 Light"}
    ], modes: [
      {mode: "rectangle", label: "Rectangle (R)"}, {mode: "brush", label: "Paint (B)"}, {mode: "line", label: "Line (L)"}
    ]};
  }

  _onRender(context, options) {
    super._onRender(context, options);
    this.listeners?.abort(); this.listeners = new AbortController();
    this.element.addEventListener("change", event => this.change(event), {signal: this.listeners.signal});
    this.attach(); this.refresh();
  }

  attach() {
    if (this.editor?.current()) return;
    this.editor?.dispose(); this.editor = null;
    if (!canvas.ready || !canvas.scene || !canvas.level || !game.user.isGM) { this.refresh(); return; }
    try {
      let editor;
      editor = new CanvasEditor((message, error) => {
        if (this.editor !== editor) return;
        if (message) { this.status = message; this.error = !!error; }
        this.refresh();
      });
      this.editor = editor;
      Object.assign(editor, toolPreferences(game.settings.get(MODULE_ID, "tools")));
      this.refresh();
    } catch (error) { this.status = error.message; this.error = true; this.refresh(); }
  }

  refresh() {
    const element = this.element; if (!element) return;
    const editor = this.editor, active = editor?.current(), busy = !!(editor?.sync.busy || editor?.lineSave);
    const status = element.querySelector("[data-status]"); if (!status) return;
    status.textContent = active ? this.status : "Open a scene and level as GM to start.";
    status.classList.toggle("error", this.error);
    element.querySelector("[data-scene]").textContent = active ? `${editor.scene.name} · ${editor.level.name}` : "No scene";
    for (const [attribute, value] of [["layer", editor?.layer], ["mode", editor?.mode]]) {
      for (const button of element.querySelectorAll(`[data-${attribute}]`)) {
        button.classList.toggle("active", value === button.dataset[attribute]);
        button.setAttribute("aria-pressed", String(value === button.dataset[attribute]));
        button.disabled = !active || busy;
      }
    }
    const editing = element.querySelector("[data-action=editing]");
    editing.textContent = editor?.enabled ? "Editing on · click to pause" : "Editing paused · click to resume";
    editing.disabled = !active; editing.setAttribute("aria-pressed", String(!!editor?.enabled));
    element.querySelector("[data-action=undo]").disabled = !active || busy || !editor.sync.undoStack.length;
    element.querySelector("[data-action=finishLine]").hidden = !editor?.line;
    for (const [selector, value] of [["[data-diameter]", editor?.diameter ?? 24],
      ["[data-light-color]", editor?.lightColor ?? DEFAULT_TOOLS.lightColor],
      ["[data-light-range]", editor?.lightRange ?? DEFAULT_TOOLS.lightRange]]) {
      const field = element.querySelector(selector);
      if (document.activeElement !== field) field.value = value;
      field.disabled = !active || busy;
    }
    element.querySelector("[data-size-controls]").hidden = editor?.mode !== "brush" && !["doors", "windows"].includes(editor?.layer);
    element.querySelector("[data-light-controls]").hidden = editor?.layer !== "lights";
    for (const field of element.querySelectorAll("[data-visible]")) field.checked = editor?.visible[field.dataset.visible] ?? true;
    element.querySelector("[data-count]").textContent = active ?
      `${editor.composed.walls.length + editor.composed.invisible.length} walls · ${editor.composed.doors.length} doors · ${editor.composed.windows.length} windows · ${editor.lights.length} lights` : "";
    editor?.draw();
  }

  async change(event) {
    const field = event.target, editor = this.editor; if (!editor) return;
    try {
      if (field.matches("[data-diameter]")) {
        const value = Number(field.value);
        if (!Number.isFinite(value) || value < 2 || value > 2048) throw new Error("Brush size must be 2–2048 scene pixels.");
        editor.diameter = Math.round(value);
      }
      if (field.matches("[data-light-color]")) editor.lightColor = field.value;
      if (field.matches("[data-light-range]")) {
        const value = Number(field.value);
        if (!Number.isFinite(value) || value <= 0 || value > 1000) throw new Error("Light radius must be greater than 0 and at most 1000 grid squares.");
        editor.lightRange = value;
      }
      if (field.matches("[data-visible]")) { editor.cancel(); editor.visible[field.dataset.visible] = field.checked; }
      await this.saveTools();
    } catch (error) { this.status = error.message; this.error = true; }
    this.refresh();
  }

  async saveTools() {
    if (this.editor) await game.settings.set(MODULE_ID, "tools", toolPreferences(this.editor));
  }

  async close(options) {
    const saving = this.saveTools();
    this.listeners?.abort(); this.editor?.dispose(); this.editor = null;
    try { await saving; } catch (error) { console.error("Auto-Wall: could not save tool preferences", error); }
    return super.close(options);
  }
}

function openPanel() {
  if (!game.user.isGM) return;
  panel ??= new AutoWallPanel(); panel.render({force: true});
}
Hooks.once("init", () => {
  game.settings.register(MODULE_ID, "tools", {scope: "client", config: false, type: Object, default: DEFAULT_TOOLS});
});
Hooks.on("getSceneControlButtons", controls => {
  if (!game.user.isGM || !controls.walls) return;
  controls.walls.tools.autoWallLive = {name: "autoWallLive", title: "Auto-Wall Live", icon: "fa-solid fa-paintbrush",
    order: Object.keys(controls.walls.tools).length, button: true, visible: true, onChange: openPanel};
});
Hooks.once("ready", () => { game.modules.get(MODULE_ID).api = Object.freeze({open: openPanel}); });
Hooks.on("canvasTearDown", () => { if (panel?.editor) { panel.editor.dispose(); panel.editor = null; panel.refresh(); } });
Hooks.on("canvasReady", () => { if (panel?.rendered) panel.attach(); });
Hooks.on("canvasPan", () => panel?.editor?.draw());
for (const event of ["createWall", "updateWall", "deleteWall", "createAmbientLight", "updateAmbientLight", "deleteAmbientLight"]) {
  Hooks.on(event, document => {
    if (panel?.editor?.scene.id === document.parent?.id) panel.editor.externalChange();
  });
}
