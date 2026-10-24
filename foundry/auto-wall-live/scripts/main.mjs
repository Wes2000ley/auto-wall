import {MODULE_ID, DEFAULTS, validateSettings, preset, importSettings, hex, rgb} from "./settings.mjs";
import {CanvasEditor} from "./editor.mjs";

let panel;
const {ApplicationV2, HandlebarsApplicationMixin} = foundry.applications.api;
const FIELD_GROUPS = [
  {title: "Walls", fields: [
    ["wall_tolerance", "Color tolerance", 0, 255, 1], ["min_wall_area", "Minimum area", 0, 1e8, 1],
    ["simplify", "Outline simplification", 0, 100, 0.25], ["wall_gap_close", "Close small gaps (0 preserves gaps)", 0, 100, 0.5]
  ]},
  {title: "Doors", fields: [
    ["door_color_tolerance", "RGB tolerance", 0, 255, 1], ["door_hue_tolerance", "Hue tolerance", 0, 90, 1],
    ["door_min_length", "Minimum length", 0.1, 4096, 1], ["door_max_length", "Maximum length", 0.1, 4096, 1],
    ["door_gap_close", "Close shading seams", 0, 100, 0.5], ["door_rectangularity", "Minimum rectangularity", 0.01, 1, 0.01]
  ]},
  {title: "Lights", fields: [
    ["light_threshold", "Bright-core threshold", 0, 255, 1], ["light_saturation", "Bright-core saturation", 0, 255, 1],
    ["light_min_area", "Minimum source area", 0.01, 1e8, 1], ["light_max_area", "Maximum source area", 0.01, 1e8, 1],
    ["light_core_fraction", "Minimum bright-core fraction", 0, 1, 0.01], ["light_min_fill", "Minimum source fill", 0, 1, 0.01],
    ["light_range", "Light radius (grid squares)", 0.1, 1000, 0.5]
  ]}
];

class AutoWallPanel extends HandlebarsApplicationMixin(ApplicationV2) {
  static DEFAULT_OPTIONS = {
    id: "auto-wall-live", classes: ["auto-wall-live"],
    window: {title: "Auto-Wall Live", icon: "fa-solid fa-paintbrush", resizable: true},
    position: {width: 390, height: 740},
    actions: {
      async tool(event, target) {
        this.editor?.cancel();
        if (this.editor) { this.editor.layer = target.dataset.layer; this.editor.visible[this.editor.layer] = true; }
        this.refresh(); await this.saveTools();
      },
      editing() { if (this.editor) this.editor.setEnabled(!this.editor.enabled); },
      detect() { this.editor?.detect(); },
      cancelDetection() { this.editor?.cancelDetection(); },
      undo() { this.editor?.undo(); },
      sample(event, target) {
        if (!this.editor) return;
        this.editor.sample = target.dataset.color; this.editor.setEnabled(true);
        this.status = "Click the map to sample a color. Esc cancels."; this.refresh();
      },
      importProfile() { this.element.querySelector("input[data-profile]").click(); }
    }
  };
  static PARTS = {content: {template: "modules/auto-wall-live/templates/editor.hbs", scrollable: [""]}};

  constructor() {
    super();
    this.status = "Every completed edit is saved directly to this scene."; this.error = false;
    this.settings = validateSettings(game.settings.get(MODULE_ID, "detector"));
    this.editor = null;
  }

  async _prepareContext(options) {
    const context = await super._prepareContext(options);
    return {...context, tools: [
      {layer: "walls", label: "1 Wall"}, {layer: "doors", label: "2 Door"},
      {layer: "invisible", label: "3 Invisible"}, {layer: "windows", label: "4 Window"}, {layer: "lights", label: "5 Light"}
    ], colors: [
      {key: "wall_color", label: "Wall"}, {key: "door_color", label: "Door"},
      {key: "light_color", label: "Light"}, {key: "void_color", label: "Void"}
    ], groups: FIELD_GROUPS.map(group => ({title: group.title,
      fields: group.fields.map(([key, label, min, max, step]) => ({key, label, min, max, step}))}))};
  }

  _onRender(context, options) {
    super._onRender(context, options);
    this.listeners?.abort(); this.listeners = new AbortController();
    const signal = this.listeners.signal;
    this.element.addEventListener("change", event => this.change(event), {signal});
    this.attach(); this.fillSettings(); this.refresh();
  }

  attach() {
    if (this.editor?.current()) return;
    this.editor?.dispose(); this.editor = null;
    if (!canvas.ready || !canvas.scene || !canvas.level || !game.user.isGM) { this.refresh(); return; }
    try {
      let editor;
      editor = new CanvasEditor((message, error, settingsChanged) => {
        if (this.editor !== editor) return;
        if (message) { this.status = message; this.error = !!error; }
        if (settingsChanged) { this.settings = editor.settings; this.fillSettings(); this.saveSettings(); }
        this.refresh();
      });
      this.editor = editor; editor.setSettings(this.settings);
      const tools = game.settings.get(MODULE_ID, "tools");
      if (tools && ["walls", "doors", "invisible", "windows", "lights"].includes(tools.layer)) editor.layer = tools.layer;
      if (typeof tools?.brush === "boolean") editor.brush = tools.brush;
      if (typeof tools?.snap === "boolean") editor.snap = tools.snap;
      if (Number.isFinite(tools?.diameter)) editor.diameter = Math.max(2, Math.min(2048, tools.diameter));
      this.refresh();
    } catch (error) { this.status = error.message; this.error = true; this.refresh(); }
  }

  fillSettings() {
    if (!this.element) return;
    for (const field of this.element.querySelectorAll("[data-setting]")) {
      const value = this.settings[field.dataset.setting];
      if (field.type === "checkbox") field.checked = value;
      else field.value = field.type === "color" ? hex(value) : value;
    }
  }

  refresh() {
    const element = this.element; if (!element) return;
    const editor = this.editor, active = editor?.current(), busy = !!(editor?.sync.busy || editor?.worker);
    const status = element.querySelector("[data-status]");
    if (!status) return;
    status.textContent = active ? this.status : "Open a scene and level as GM to start.";
    status.classList.toggle("error", this.error);
    const sceneName = element.querySelector("[data-scene]");
    sceneName.textContent = active ? `${editor.scene.name} · ${editor.level.name}` : "No scene";
    for (const button of element.querySelectorAll("[data-layer]")) {
      button.classList.toggle("active", editor?.layer === button.dataset.layer);
      button.setAttribute("aria-pressed", String(editor?.layer === button.dataset.layer)); button.disabled = !active || busy;
    }
    const editing = element.querySelector("[data-action=editing]");
    editing.textContent = editor?.enabled ? "Editing on · click to pause" : "Editing paused · click to resume";
    editing.disabled = !active; editing.setAttribute("aria-pressed", String(!!editor?.enabled));
    element.querySelector("[data-action=detect]").disabled = !active || busy;
    element.querySelector("[data-action=cancelDetection]").hidden = !editor?.worker;
    element.querySelector("[data-action=undo]").disabled = !active || busy || !editor.sync.undoStack.length;
    const brush = element.querySelector("[data-brush]"); brush.checked = editor?.brush ?? true; brush.disabled = !active;
    const snap = element.querySelector("[data-snap]"); snap.checked = editor?.snap ?? true; snap.disabled = !active;
    const diameter = element.querySelector("[data-diameter]");
    if (document.activeElement !== diameter) diameter.value = editor?.diameter ?? 24;
    for (const field of element.querySelectorAll("[data-visible]")) field.checked = editor?.visible[field.dataset.visible] ?? true;
    element.querySelector("[data-count]").textContent = active ?
      `${editor.composed.walls.length + editor.composed.invisible.length} walls · ${editor.composed.doors.length} doors · ${editor.composed.windows.length} windows · ${editor.lights.length} lights` : "";
    editor?.draw();
  }

  async change(event) {
    const field = event.target;
    try {
      if (field.matches("[data-setting]")) {
        const value = field.type === "checkbox" ? field.checked : typeof DEFAULTS[field.dataset.setting] === "number" ? Number(field.value) : field.type === "color" ? rgb(field.value) : field.value;
        this.settings = validateSettings({...this.settings, [field.dataset.setting]: value});
        this.editor?.setSettings(this.settings); await this.saveSettings();
      } else if (field.matches("[data-preset]") && field.value) {
        this.settings = preset(field.value); this.editor?.setSettings(this.settings); this.fillSettings(); await this.saveSettings(); field.value = "";
      } else if (field.matches("[data-profile]")) {
        const file = field.files[0]; if (!file) return;
        if (file.size > 1024 * 1024) throw new Error("The palette JSON must be at most 1 MiB.");
        this.settings = importSettings(JSON.parse((await file.text()).replace(/^\uFEFF/, "")), this.settings);
        this.editor?.setSettings(this.settings); this.fillSettings(); await this.saveSettings(); field.value = "";
        this.status = "Palette imported. Only colors and detector settings are read."; this.error = false;
      } else if (this.editor) {
        if (field.matches("[data-brush]")) { this.editor.cancel(); this.editor.brush = field.checked; }
        if (field.matches("[data-snap]")) this.editor.snap = field.checked;
        if (field.matches("[data-diameter]")) {
          const value = Number(field.value);
          if (!Number.isFinite(value) || value < 2 || value > 2048) throw new Error("Brush size must be 2–2048 scene pixels.");
          this.editor.diameter = value;
        }
        if (field.matches("[data-visible]")) this.editor.visible[field.dataset.visible] = field.checked;
        await this.saveTools();
      }
    } catch (error) { this.status = error.message; this.error = true; this.fillSettings(); }
    this.refresh();
  }

  async saveSettings() { await game.settings.set(MODULE_ID, "detector", this.settings); }
  async saveTools() {
    if (this.editor) {
      const {layer, brush, snap, diameter} = this.editor;
      await game.settings.set(MODULE_ID, "tools", {layer, brush, snap, diameter});
    }
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
  panel ??= new AutoWallPanel();
  panel.render({force: true});
}

Hooks.once("init", () => {
  game.settings.register(MODULE_ID, "detector", {scope: "client", config: false, type: Object, default: DEFAULTS});
  game.settings.register(MODULE_ID, "tools", {scope: "client", config: false, type: Object, default: {layer: "walls", brush: true, snap: true, diameter: 24}});
});
Hooks.on("getSceneControlButtons", controls => {
  if (!game.user.isGM || !controls.walls) return;
  controls.walls.tools.autoWallLive = {name: "autoWallLive", title: "Auto-Wall Live",
    icon: "fa-solid fa-paintbrush", order: Object.keys(controls.walls.tools).length,
    button: true, visible: true, onChange: openPanel};
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
