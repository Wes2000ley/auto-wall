import {DEFAULT_TOOLS, MODES} from "./settings.mjs";
import {PaintGeometry, distance, rectangle, areaSegments} from "./geometry.mjs";
import {LiveSync, geometryFromDocuments, fingerprint} from "./documents.mjs";

const COLORS = {walls: "#50db92", doors: "#df79ff", invisible: "#49d7ee", windows: "#80aaff", lights: "#ffce72"};
const inputTarget = target => target?.closest?.("input,select,textarea,[contenteditable=true],.ProseMirror");

export class CanvasEditor {
  constructor(notify) {
    this.scene = canvas.scene; this.level = canvas.level;
    if (!game.user.isGM || !this.scene || !this.level) throw new Error("Open a scene as GM first.");
    this.notify = notify; this.sync = new LiveSync(this.scene, this.level.id);
    Object.assign(this, DEFAULT_TOOLS);
    this.visible = {walls: true, doors: true, invisible: true, windows: true, lights: true};
    this.enabled = false; this.closed = false; this.gesture = null; this.line = null;
    this.pendingLines = []; this.lineSave = null; this.space = false; this.frame = null;
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
    this.overlay.addEventListener("lostpointercapture", event => {
      if (this.gesture?.pointer === event.pointerId) this.cancel();
    }, {signal});
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
    this.composed = geometryFromDocuments(state.walls); this.lights = state.lights;
    this.draw();
  }

  externalChange() {
    if (!this.current() || this.sync.busy || this.lineSave || this.gesture) return;
    if (fingerprint(this.sync.state()) !== this.expected) {
      this.cancel(); this.reload(); this.notify("Scene changed; the editor refreshed.");
    } else { this.lights = this.sync.state().lights; this.draw(); }
  }

  setEnabled(value) {
    this.enabled = value; this.cancel();
    this.overlay.style.pointerEvents = value ? "auto" : "none";
    this.overlay.style.display = value ? "block" : "none";
    if (value) this.overlay.focus({preventScroll: true});
    this.draw(); this.notify();
  }

  setMode(mode) {
    if (!MODES.includes(mode)) return;
    this.cancel(); this.mode = mode; this.draw(); this.notify();
  }

  cancel() {
    const gesture = this.gesture;
    this.gesture = null; this.line = null;
    if (gesture && this.overlay?.hasPointerCapture(gesture.pointer)) this.overlay.releasePointerCapture(gesture.pointer);
    this.draw();
  }

  dispose() {
    this.closed = true; this.pendingLines.length = 0; this.cancel(); this.abort.abort();
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.overlay.remove();
  }

  scenePoint(event) {
    const point = canvas.canvasCoordinatesFromClient({x: event.clientX, y: event.clientY});
    return [point.x, point.y];
  }

  down(event) {
    if (!this.current() || !this.enabled || this.gesture || ![0, 1, 2].includes(event.button)) return;
    event.preventDefault();
    if (event.button === 2 || (event.button === 0 && this.space)) {
      this.gesture = {type: "pan", pointer: event.pointerId, client: [event.clientX, event.clientY],
        pivot: [canvas.stage.pivot.x, canvas.stage.pivot.y], scale: canvas.stage.scale.x};
    } else {
      const vertex = event.button === 0 && this.mode === "line" && this.layer !== "lights";
      if ((this.sync.busy || this.lineSave) && !(vertex && this.lineSave && this.line)) return;
      if (!this.lineSave && fingerprint(this.sync.state()) !== this.expected) { this.cancel(); this.reload(); }
      const point = this.scenePoint(event);
      if (event.button === 1) this.line = null;
      this.gesture = {type: event.button === 1 ? "erase" : vertex ? "vertex" : "draw", pointer: event.pointerId,
        first: point, last: point, points: [point], expected: this.expected, layer: this.layer,
        mode: this.mode, diameter: this.diameter, client: [event.clientX, event.clientY]};
      this.cursor = point;
    }
    try { this.overlay.setPointerCapture(event.pointerId); } catch { /* The pointer may already have ended. */ }
    this.draw();
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
        gesture.last = this.cursor;
        if (gesture.mode === "brush" && gesture.type === "draw" && ["walls", "invisible"].includes(gesture.layer) &&
            distance(gesture.points.at(-1), gesture.last) >= Math.max(2, gesture.diameter * 0.15)) gesture.points.push(gesture.last);
      }
    }
    this.draw();
  }

  async save(geometry, lights = this.lights, expected = this.expected) {
    const saving = this.sync.commit({geometry: geometry.compose(), lights, paint: geometry.serialize()}, expected);
    this.notify("Saving to scene…"); await saving;
    this.reload(); this.notify("Saved to scene");
  }

  async up(event) {
    const gesture = this.gesture;
    if (!gesture || gesture.pointer !== event.pointerId) return;
    this.move(event); this.gesture = null;
    if (this.overlay.hasPointerCapture(event.pointerId)) this.overlay.releasePointerCapture(event.pointerId);
    if (gesture.type === "pan" || !this.current()) { this.draw(); return; }
    try {
      if (gesture.type === "vertex") {
        if (Math.hypot(event.clientX - gesture.client[0], event.clientY - gesture.client[1]) <= 5) await this.vertex(gesture.last);
        return;
      }
      let geometry = this.geometry, lights = this.lights;
      if (gesture.type === "erase") {
        geometry = geometry.erase(gesture.first, gesture.last, this.visible);
        const low = [0, 1].map(i => Math.min(gesture.first[i], gesture.last[i]));
        const high = [0, 1].map(i => Math.max(gesture.first[i], gesture.last[i]));
        if (this.visible.lights) lights = lights.filter(l => l.x < low[0] || l.x > high[0] || l.y < low[1] || l.y > high[1]);
      } else if (gesture.layer === "lights") {
        lights = [...lights, {x: gesture.last[0], y: gesture.last[1], color: this.lightColor, range: this.lightRange}];
      } else if (["doors", "windows"].includes(gesture.layer)) {
        if (distance(gesture.first, gesture.last) < 1) return;
        geometry = geometry.paint(gesture.layer, [gesture.first, gesture.last], gesture.diameter);
      } else if (gesture.mode === "brush") {
        geometry = geometry.paint(gesture.layer, [...gesture.points, gesture.last], gesture.diameter,
          this.visible[gesture.layer === "walls" ? "invisible" : "walls"]);
      } else {
        if (Math.abs(gesture.first[0] - gesture.last[0]) < 1 || Math.abs(gesture.first[1] - gesture.last[1]) < 1) return;
        geometry = geometry.addRectangle(gesture.layer, gesture.first, gesture.last);
      }
      await this.save(geometry, lights, gesture.expected);
    } catch (error) { this.cancel(); this.reload(); this.notify(error.message, true); }
    finally { this.draw(); this.notify(); }
  }

  vertex(point) {
    point = point.map(Math.round);
    if (!this.line) {
      this.line = {first: point, last: point, layer: this.layer, diameter: this.diameter};
      this.draw(); this.notify("Click to add a vertex. Enter or Esc finishes the line."); return;
    }
    if (distance(this.line.last, point) < 1) return;
    this.pendingLines.push({first: this.line.last, last: point, layer: this.line.layer, diameter: this.line.diameter});
    this.line.last = point;
    if (distance(point, this.line.first) < 1) this.line = null;
    if (!this.lineSave) {
      this.lineSave = this.flushLines().finally(() => { this.lineSave = null; this.draw(); this.notify(); });
    }
    this.draw(); return this.lineSave;
  }

  async flushLines() {
    try {
      while (this.pendingLines.length && this.current()) {
        const segment = this.pendingLines.shift(), span = [segment.first, segment.last];
        const geometry = ["doors", "windows"].includes(segment.layer) ?
          this.geometry.paint(segment.layer, span, segment.diameter) : this.geometry.addLines(segment.layer, [span]);
        await this.save(geometry);
      }
    } catch (error) {
      this.line = null; this.reload(); this.notify(error.message, true);
    } finally { this.pendingLines.length = 0; }
  }

  wheel(event) {
    if (!this.current() || !this.enabled) return;
    event.preventDefault();
    if (this.gesture && this.gesture.type !== "pan") return;
    if (event.ctrlKey || this.mode !== "brush") {
      const before = this.scenePoint(event), scale = canvas.stage.scale.x;
      canvas.pan({scale: scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12)});
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
    const modes = {KeyR: "rectangle", KeyB: "brush", KeyL: "line"};
    if (event.code in layers || event.code in modes || ["Escape", "Enter"].includes(event.code)) {
      event.preventDefault(); event.stopImmediatePropagation();
      if (event.code in layers) { this.cancel(); this.layer = layers[event.code]; this.visible[this.layer] = true; }
      if (event.code in modes) this.setMode(modes[event.code]);
      if (event.code === "Enter") { this.cancel(); this.notify("Line finished. Click to start another."); }
      if (event.code === "Escape") {
        if (this.gesture || this.line) { this.cancel(); this.notify("Gesture finished; completed edits remain saved."); }
        else this.setEnabled(false);
      }
      this.draw(); this.notify();
    }
  }

  async undo() {
    if (this.sync.busy || this.lineSave) return;
    this.cancel();
    try { if (await this.sync.undo()) { this.reload(); this.notify("Undid last edit"); } }
    catch (error) { this.notify(error.message, true); }
    this.notify();
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
      const client = canvas.clientCoordinatesFromCanvas({x: p[0], y: p[1]});
      return [client.x - bounds.left, client.y - bounds.top];
    };
    const line = segment => { context.moveTo(...screen(segment[0])); context.lineTo(...screen(segment[1])); };
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
    if (gesture && !["pan", "vertex"].includes(gesture.type)) {
      context.strokeStyle = gesture.type === "erase" ? "#ff7474" : COLORS[gesture.layer];
      context.globalAlpha = 0.6;
      context.beginPath();
      if (gesture.type === "erase" || gesture.mode === "rectangle" && ["walls", "invisible"].includes(gesture.layer)) {
        for (const edge of areaSegments(rectangle(gesture.first, gesture.last))) line(edge);
      } else {
        context.lineWidth = gesture.diameter * canvas.stage.scale.x;
        context.lineCap = ["doors", "windows"].includes(gesture.layer) ? "butt" : "round"; context.lineJoin = "round";
        const points = ["doors", "windows"].includes(gesture.layer) ? [gesture.first, gesture.last] : [...gesture.points, gesture.last];
        context.moveTo(...screen(points[0])); for (const p of points.slice(1)) context.lineTo(...screen(p));
      }
      context.stroke(); context.globalAlpha = 1;
    }
    if (this.line && this.cursor && gesture?.type !== "pan") {
      context.lineWidth = 2; context.strokeStyle = COLORS[this.line.layer]; context.setLineDash([5, 4]);
      context.beginPath(); line([this.line.last, this.cursor]); context.stroke(); context.setLineDash([]);
      context.beginPath(); context.arc(...screen(this.line.last), 4, 0, 2 * Math.PI); context.stroke();
    } else if (this.cursor && !gesture) {
      context.lineWidth = 1; context.strokeStyle = COLORS[this.layer];
      const radius = this.mode === "brush" ? this.diameter / 2 * canvas.stage.scale.x : 4;
      context.beginPath(); context.arc(...screen(this.cursor), radius, 0, 2 * Math.PI); context.stroke();
    }
    this.overlay.style.cursor = gesture?.type === "pan" ? "grabbing" : "crosshair";
  }
}
