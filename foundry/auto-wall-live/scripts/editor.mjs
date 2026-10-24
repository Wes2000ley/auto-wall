import {MODULE_ID, hex, validateSettings} from "./settings.mjs";
import {PaintGeometry, nearest, distance, rectangle, brushArea, union, contains, splitSegments} from "./geometry.mjs";
import {LiveSync, geometryFromDocuments, fingerprint, stable, wallKind, snapshot} from "./documents.mjs";

const COLORS = {walls: "#50db92", doors: "#df79ff", invisible: "#49d7ee", windows: "#80aaff", lights: "#ffce72"};
const inputTarget = target => target?.closest?.("input,select,textarea,[contenteditable=true],.ProseMirror");

export class CanvasEditor {
  constructor(notify) {
    this.scene = canvas.scene; this.level = canvas.level;
    if (!game.user.isGM || !this.scene || !this.level) throw new Error("Open a scene as GM first.");
    this.notify = notify; this.sync = new LiveSync(this.scene, this.level.id);
    this.layer = "walls"; this.brush = true; this.diameter = 24; this.snap = true;
    this.visible = {walls: true, doors: true, invisible: true, windows: true, lights: true};
    this.enabled = false; this.closed = false; this.gesture = null; this.worker = null;
    this.space = false; this.sample = null; this.frame = null;
    this.reload();
    this.abort = new AbortController();
    const signal = this.abort.signal;
    this.overlay = document.createElement("canvas");
    this.overlay.className = "auto-wall-live-overlay"; this.overlay.tabIndex = -1;
    this.overlay.setAttribute("aria-label", "Auto-Wall map editor");
    document.body.append(this.overlay);
    this.overlay.addEventListener("pointerdown", event => this.down(event), {signal});
    this.overlay.addEventListener("pointermove", event => this.move(event), {signal});
    this.overlay.addEventListener("pointerup", event => this.up(event), {signal});
    this.overlay.addEventListener("pointercancel", () => this.cancel(), {signal});
    this.overlay.addEventListener("lostpointercapture", () => this.cancel(), {signal});
    this.overlay.addEventListener("contextmenu", event => event.preventDefault(), {signal});
    this.overlay.addEventListener("wheel", event => this.wheel(event), {signal, passive: false});
    window.addEventListener("keydown", event => this.key(event), {signal, capture: true});
    window.addEventListener("keyup", event => { if (event.code === "Space") this.space = false; }, {signal});
    window.addEventListener("blur", () => { this.space = false; this.cancel(); }, {signal});
    window.addEventListener("resize", () => this.draw(), {signal});
    this.setEnabled(true);
  }

  current() { return !this.closed && canvas.ready && canvas.scene === this.scene && canvas.level?.id === this.level.id && game.user.isGM; }

  reload() {
    const state = this.sync.state(), saved = this.sync.saved();
    this.expected = fingerprint(state);
    this.geometry = null;
    if (saved?.fingerprint === this.expected && saved.paint &&
        saved.paint.size?.[0] === canvas.dimensions.width && saved.paint.size?.[1] === canvas.dimensions.height) {
      try { this.geometry = PaintGeometry.restore(saved.paint); this.geometry.compose(); }
      catch { /* Native documents remain authoritative if a saved cache is obsolete. */ }
    }
    this.geometry ??= new PaintGeometry([canvas.dimensions.width, canvas.dimensions.height], geometryFromDocuments(state.walls));
    this.exclusions = saved?.exclusions ?? [];
    this.composed = geometryFromDocuments(state.walls); this.lights = state.lights;
    this.draw();
  }

  externalChange() {
    if (!this.current() || this.sync.busy || this.gesture) return;
    if (fingerprint(this.sync.state()) !== this.expected) this.reload();
    else { this.lights = this.sync.state().lights; this.draw(); }
  }

  setEnabled(value) {
    this.enabled = value; this.cancel();
    this.overlay.style.pointerEvents = value ? "auto" : "none";
    this.overlay.style.display = value ? "block" : "none";
    if (value) this.overlay.focus({preventScroll: true});
    this.draw(); this.notify();
  }

  cancel() { this.gesture = null; this.draw(); }

  dispose() {
    this.closed = true; this.cancelDetection(); this.abort.abort();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.overlay.remove();
  }

  scenePoint(event) {
    const bounds = canvas.app.view.getBoundingClientRect();
    const p = new PIXI.Point((event.clientX - bounds.left) * canvas.app.screen.width / bounds.width,
      (event.clientY - bounds.top) * canvas.app.screen.height / bounds.height);
    const local = canvas.stage.toLocal(p);
    return [Math.max(0, Math.min(canvas.dimensions.width - 1, local.x)), Math.max(0, Math.min(canvas.dimensions.height - 1, local.y))];
  }

  snapped(point) {
    if (!this.snap) return point;
    const radius = 10 / canvas.stage.scale.x;
    const portal = nearest([...this.composed.doors, ...this.composed.windows], point, radius);
    if (portal) return portal;
    if (this.brush && (contains(this.geometry.areas.walls, point) || contains(this.geometry.areas.invisible, point))) return point;
    const targets = Object.entries(this.composed).filter(([key]) => this.visible[key]).flatMap(([, values]) => values);
    return nearest(targets, point, radius) ?? point;
  }

  down(event) {
    if (!this.current() || !this.enabled) return;
    event.preventDefault();
    if (this.gesture) return;
    this.overlay.setPointerCapture(event.pointerId);
    if (event.button === 2 || (event.button === 0 && this.space)) {
      this.gesture = {type: "pan", pointer: event.pointerId, client: [event.clientX, event.clientY],
        pivot: [canvas.stage.pivot.x, canvas.stage.pivot.y], scale: canvas.stage.scale.x}; return;
    }
    if (this.sync.busy || this.worker) return;
    if (fingerprint(this.sync.state()) !== this.expected) this.reload();
    const raw = this.scenePoint(event), point = this.snapped(raw);
    if (this.sample && event.button === 0) { this.pickColor(raw); return; }
    if (event.button !== 0 && event.button !== 1) return;
    this.gesture = {type: event.button === 1 ? "erase" : "draw", pointer: event.pointerId,
      first: point, last: point, points: [point], expected: this.expected, layer: this.layer,
      brush: this.brush, diameter: this.diameter};
    this.cursor = raw; this.draw();
  }

  move(event) {
    if (!this.current()) return;
    this.cursor = this.scenePoint(event);
    const gesture = this.gesture;
    if (gesture?.pointer === event.pointerId) {
      if (gesture.type === "pan") {
        canvas.pan({x: gesture.pivot[0] - (event.clientX - gesture.client[0]) / gesture.scale,
          y: gesture.pivot[1] - (event.clientY - gesture.client[1]) / gesture.scale});
      } else {
        gesture.last = gesture.type === "erase" ? this.cursor : this.snapped(this.cursor);
        if (gesture.brush && gesture.type === "draw" && ["walls", "invisible"].includes(gesture.layer) &&
            distance(gesture.points.at(-1), gesture.last) >= Math.max(1, gesture.diameter * 0.06)) gesture.points.push(gesture.last);
      }
    }
    this.draw();
  }

  async up(event) {
    const gesture = this.gesture;
    if (!gesture || gesture.pointer !== event.pointerId) return;
    this.move(event); this.gesture = null;
    if (this.overlay.hasPointerCapture(event.pointerId)) this.overlay.releasePointerCapture(event.pointerId);
    if (gesture.type === "pan" || !this.current()) { this.draw(); return; }
    try {
      let geometry = this.geometry, lights = this.lights, exclusion = [];
      if (gesture.type === "erase") {
        geometry = geometry.erase(gesture.first, gesture.last, this.visible);
        const low = [0, 1].map(i => Math.min(gesture.first[i], gesture.last[i]));
        const high = [0, 1].map(i => Math.max(gesture.first[i], gesture.last[i]));
        if (this.visible.lights) lights = lights.filter(l => l.x < low[0] || l.x > high[0] || l.y < low[1] || l.y > high[1]);
        exclusion = rectangle(gesture.first, gesture.last);
      } else if (gesture.layer === "lights") {
        lights = [...lights, {x: gesture.last[0], y: gesture.last[1], color: hex(this.settings.light_color), range: this.settings.light_range}];
      } else if (["doors", "windows"].includes(gesture.layer)) {
        if (distance(gesture.first, gesture.last) < 1) return;
        geometry = geometry.paint(gesture.layer, [gesture.first, gesture.last], gesture.diameter);
        exclusion = brushArea([gesture.first, gesture.last], gesture.diameter, geometry.size, true);
      } else if (gesture.brush) {
        const points = [...gesture.points, gesture.last];
        geometry = geometry.paint(gesture.layer, points, gesture.diameter, this.visible[gesture.layer === "walls" ? "invisible" : "walls"]);
        exclusion = brushArea(points, gesture.diameter, geometry.size);
      } else {
        geometry = geometry.addRectangle(gesture.layer, gesture.first, gesture.last);
        exclusion = rectangle(gesture.first, gesture.last);
      }
      const saving = this.sync.commit({geometry: geometry.compose(), lights, paint: geometry.serialize(),
        exclusions: union(this.exclusions, exclusion)}, gesture.expected);
      this.notify("Saving to scene…"); await saving;
      this.reload(); this.notify("Saved to scene");
    } catch (error) { this.reload(); this.notify(error.message, true); }
    finally { this.draw(); this.notify(); }
  }

  wheel(event) {
    if (!this.current() || !this.enabled) return;
    event.preventDefault();
    if (this.gesture && this.gesture.type !== "pan") return;
    if (event.ctrlKey || !this.brush) {
      const before = this.scenePoint(event), scale = canvas.stage.scale.x;
      const next = scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12);
      canvas.pan({scale: next});
      const after = this.scenePoint(event);
      canvas.pan({x: canvas.stage.pivot.x + before[0] - after[0], y: canvas.stage.pivot.y + before[1] - after[1]});
    } else this.diameter = Math.max(2, Math.min(2048, Math.round(this.diameter * (event.deltaY < 0 ? 1.12 : 1 / 1.12))));
    this.draw(); this.notify();
  }

  key(event) {
    if (!this.current() || !this.enabled || inputTarget(event.target)) return;
    if (event.code === "Space") { event.preventDefault(); this.space = true; return; }
    if ((event.ctrlKey || event.metaKey) && event.code === "KeyZ") {
      if (this.sync.undoStack.length) { event.preventDefault(); event.stopImmediatePropagation(); this.undo(); }
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const layers = {Digit1: "walls", Digit2: "doors", Digit3: "invisible", Digit4: "windows", Digit5: "lights"};
    if (event.code in layers || event.code === "KeyB" || event.code === "Escape") {
      event.preventDefault(); event.stopImmediatePropagation();
      if (event.code in layers) { this.cancel(); this.layer = layers[event.code]; }
      if (event.code === "KeyB") { this.cancel(); this.brush = !this.brush; }
      if (event.code === "Escape") {
        if (this.worker) this.cancelDetection();
        else if (this.gesture || this.sample) { this.cancel(); this.sample = null; }
        else this.setEnabled(false);
      }
      this.draw(); this.notify();
    }
  }

  async undo() {
    if (this.worker || this.sync.busy) return;
    this.cancel();
    try { if (await this.sync.undo()) { this.reload(); this.notify("Undid last edit"); } }
    catch (error) { this.notify(error.message, true); }
    this.notify();
  }

  setSettings(settings) { this.settings = validateSettings(settings); }

  pickColor(point) {
    try {
      const mesh = canvas.primary.background, source = canvas.primary.backgroundSource;
      if (!mesh || !source) throw new Error("This level has no background image.");
      const local = mesh.toLocal(canvas.stage.toGlobal(new PIXI.Point(...point)));
      const x = local.x + mesh.anchor.x * mesh.texture.orig.width, y = local.y + mesh.anchor.y * mesh.texture.orig.height;
      const sampler = new OffscreenCanvas(1, 1), context = sampler.getContext("2d", {willReadFrequently: true});
      context.drawImage(source, Math.floor(x), Math.floor(y), 1, 1, 0, 0, 1, 1);
      const color = [...context.getImageData(0, 0, 1, 1).data].slice(0, 3);
      this.settings = validateSettings({...this.settings, [this.sample]: color});
      this.notify(`Sampled ${hex(color)}`, false, true);
    } catch (error) { this.notify(error.message, true); }
    finally { this.sample = null; this.draw(); }
  }

  imageKey() {
    return stable({background: snapshot(this.level.background), textures: snapshot(this.level.textures),
      width: this.scene.width, height: this.scene.height, grid: snapshot(this.scene.grid)});
  }

  async detect() {
    if (this.worker || this.sync.busy || !this.current()) return;
    this.cancel();
    const expected = fingerprint(this.sync.state()), imageKey = this.imageKey();
    try {
      const settings = validateSettings(this.settings), mesh = canvas.primary.background, source = canvas.primary.backgroundSource;
      if (!mesh?.texture?.valid || !source || !this.level.background.src) throw new Error("This level has no loaded background image.");
      const original = mesh.texture.orig;
      const scale = Math.min(1, settings.max_dimension / Math.max(original.width, original.height));
      const width = Math.max(1, Math.round(original.width * scale)), height = Math.max(1, Math.round(original.height * scale));
      const imageToScene = point => {
        const local = new PIXI.Point(point[0] - mesh.anchor.x * original.width, point[1] - mesh.anchor.y * original.height);
        const scene = canvas.stage.toLocal(mesh.toGlobal(local)); return [scene.x, scene.y];
      };
      // Capture the affine transform now; later panning cannot change detection placement.
      const zero = imageToScene([0, 0]), xAxis = imageToScene([original.width, 0]), yAxis = imageToScene([0, original.height]);
      const transform = ([x, y]) => [0, 1].map(i => zero[i] + (xAxis[i] - zero[i]) * x / width + (yAxis[i] - zero[i]) * y / height);
      const input = new OffscreenCanvas(width, height), context = input.getContext("2d");
      context.imageSmoothingEnabled = !(settings.wall_mode === "color" && settings.wall_tolerance === 0 || settings.door_match === "rgb" && settings.door_color_tolerance === 0);
      context.drawImage(source, 0, 0, width, height);
      const bitmap = input.transferToImageBitmap();
      const worker = new Worker(new URL("./detector-worker.js", import.meta.url)); this.worker = worker;
      this.notify("Loading local detector…");
      const result = await new Promise((resolve, reject) => {
        this.rejectDetection = reject;
        worker.onmessage = ({data}) => {
          if (data.stage) this.notify(`${data.stage}…`);
          else if (data.error) reject(new Error(data.error));
          else if (data.result) resolve(data.result);
        };
        worker.onerror = event => reject(new Error(event.message || "The local detector could not start."));
        worker.postMessage({bitmap, settings}, [bitmap]);
      });
      worker.terminate(); this.worker = null; this.rejectDetection = null;
      if (!this.current() || this.imageKey() !== imageKey) throw new Error("The scene or background changed during detection. Results were discarded.");
      if (stable(this.settings) !== stable(settings)) throw new Error("Detection settings changed while processing. Run Detect map again with the new settings.");
      if (fingerprint(this.sync.state()) !== expected) throw new Error("Scene geometry changed during detection. Results were discarded to preserve the newer edit.");
      const state = this.sync.state(), retained = state.walls.filter(d => d.flags?.[MODULE_ID]?.origin !== "detection");
      const base = geometryFromDocuments(retained);
      const rounded = segment => segment.map(point => point.map(Math.round));
      base.walls.push(...splitSegments(result.walls.map(s => s.map(transform)), this.exclusions)[0].map(rounded).filter(s => distance(...s) >= 1));
      base.doors.push(...result.doors.map(s => s.map(transform)).filter(s => !splitSegments([s], this.exclusions)[1].length).map(rounded).filter(s => distance(...s) >= 1));
      const geometry = new PaintGeometry(this.geometry.size, base);
      const nativeLights = state.lights.filter(l => l.flags?.[MODULE_ID]?.origin !== "detection");
      const lights = [...state.lights.filter(l => l.flags?.[MODULE_ID]?.origin !== "detection"),
        ...result.lights.map(l => { const [x, y] = transform([l.x, l.y]); return {...l, x, y}; })
          .filter(l => !contains(this.exclusions, [l.x, l.y]) && !nativeLights.some(n => Math.hypot(n.x - l.x, n.y - l.y) <= 5))];
      await this.sync.commit({geometry: base, lights, paint: geometry.serialize(), exclusions: this.exclusions,
        origin: "detection", replaceDetectedLights: true}, expected);
      this.reload();
      const detail = `${result.walls.length} wall segments, ${result.doors.length} doors, ${result.lights.length} lights · ${(result.elapsed / 1000).toFixed(2)}s processing`;
      this.notify(result.warnings.length ? `${detail}. ${result.warnings.join(" ")}` : `${detail} · saved to scene`);
    } catch (error) { if (!this.closed) { this.reload(); this.notify(error.message, true); } }
    finally { if (this.worker) this.worker.terminate(); this.worker = null; this.rejectDetection = null; this.notify(); }
  }

  cancelDetection() {
    if (!this.worker) return;
    this.worker.terminate(); this.worker = null;
    this.rejectDetection?.(new Error("Detection cancelled; the scene was not changed."));
    this.rejectDetection = null;
  }

  draw() {
    if (this.closed || !this.overlay || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => { this.frame = null; this.renderOverlay(); });
  }

  renderOverlay() {
    if (!this.current() || !this.enabled) return;
    const bounds = canvas.app.view.getBoundingClientRect(), ratio = devicePixelRatio || 1;
    Object.assign(this.overlay.style, {left: `${bounds.left}px`, top: `${bounds.top}px`, width: `${bounds.width}px`, height: `${bounds.height}px`});
    const width = Math.round(bounds.width * ratio), height = Math.round(bounds.height * ratio);
    if (this.overlay.width !== width || this.overlay.height !== height) { this.overlay.width = width; this.overlay.height = height; }
    const context = this.overlay.getContext("2d"); context.clearRect(0, 0, width, height); context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const screen = p => {
      const global = canvas.stage.toGlobal(new PIXI.Point(...p));
      return [global.x * bounds.width / canvas.app.screen.width, global.y * bounds.height / canvas.app.screen.height];
    };
    const line = segment => { const a = screen(segment[0]), b = screen(segment[1]); context.moveTo(...a); context.lineTo(...b); };
    context.lineWidth = 2;
    for (const [layer, values] of Object.entries(this.composed)) {
      if (!this.visible[layer]) continue;
      context.strokeStyle = COLORS[layer]; context.setLineDash(layer === "invisible" ? [6, 4] : []);
      context.beginPath(); for (const segment of values) line(segment); context.stroke();
    }
    context.setLineDash([]);
    if (this.visible.lights) for (const light of this.lights) {
      context.fillStyle = light.config?.color ?? light.color ?? COLORS.lights;
      context.beginPath(); context.arc(...screen([light.x, light.y]), 5, 0, 2 * Math.PI); context.fill();
    }
    const gesture = this.gesture;
    if (gesture && gesture.type !== "pan") {
      context.strokeStyle = gesture.type === "erase" ? "#ff7474" : COLORS[gesture.layer];
      context.globalAlpha = 0.6;
      if (gesture.type === "erase" || (!gesture.brush && ["walls", "invisible"].includes(gesture.layer))) {
        const a = screen(gesture.first), b = screen(gesture.last);
        context.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      } else {
        context.lineWidth = gesture.diameter * canvas.stage.scale.x * bounds.width / canvas.app.screen.width;
        context.lineCap = ["doors", "windows"].includes(gesture.layer) ? "butt" : "round"; context.lineJoin = "round";
        context.beginPath();
        const points = ["doors", "windows"].includes(gesture.layer) ? [gesture.first, gesture.last] : [...gesture.points, gesture.last];
        context.moveTo(...screen(points[0])); for (const p of points.slice(1)) context.lineTo(...screen(p)); context.stroke();
      }
      context.globalAlpha = 1;
    }
    if (this.cursor && !gesture) {
      context.lineWidth = 1; context.strokeStyle = this.sample ? "#ffffff" : COLORS[this.layer];
      context.beginPath(); context.arc(...screen(this.cursor), this.sample ? 8 : this.brush ? this.diameter / 2 * canvas.stage.scale.x * bounds.width / canvas.app.screen.width : 4, 0, 2 * Math.PI); context.stroke();
    }
    this.overlay.style.cursor = this.sample ? "crosshair" : gesture?.type === "pan" ? "grabbing" : "crosshair";
  }
}
