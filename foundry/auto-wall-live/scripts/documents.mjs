import {MODULE_ID} from "./settings.mjs";
import {distance, segmentKey} from "./geometry.mjs";

export const snapshot = document => document?.toObject ? document.toObject() : structuredClone(document);
export function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stable(value[k])}`).join(",")}}`;
  return JSON.stringify(value);
}
export const onLevel = (document, level) => !document.levels?.length || document.levels.includes(level);
export function wallKind(document) {
  if (document.door) return "doors";
  if (document.sight === 0 && document.light === 0 && document.sound === 0) return "invisible";
  if (document.sight === 30 && document.light === 30) return "windows";
  return "walls";
}
export const wallSegment = wall => [[wall.c[0], wall.c[1]], [wall.c[2], wall.c[3]]];
const key = wall => `${wallKind(wall)}:${segmentKey(wallSegment(wall), true)}`;

export function levelSnapshot(scene, level) {
  return {
    walls: [...scene.walls].map(snapshot).filter(d => onLevel(d, level)),
    lights: [...scene.lights].map(snapshot).filter(d => onLevel(d, level))
  };
}

export function fingerprint(state) {
  // Door opening/closing and light animations do not invalidate vector ink.
  // Geometry, restrictions, memberships and native creation/deletion do.
  return stable({walls: state.walls.map(d => ({id: d._id, c: d.c, levels: d.levels,
    door: d.door, move: d.move, sight: d.sight, light: d.light, sound: d.sound,
    dir: d.dir, threshold: d.threshold})).sort((a, b) => a.id.localeCompare(b.id)),
  lights: state.lights.map(d => ({id: d._id, x: d.x, y: d.y, levels: d.levels})).sort((a, b) => a.id.localeCompare(b.id))});
}

export function geometryFromDocuments(walls) {
  const result = {walls: [], invisible: [], doors: [], windows: []};
  for (const wall of walls) result[wallKind(wall)].push(wallSegment(wall));
  return result;
}

function insideOriginal(segment, original) {
  const [a, b] = wallSegment(original), delta = [b[0] - a[0], b[1] - a[1]], length = distance(a, b);
  if (length < 1e-6) return false;
  return segment.every(p => {
    const offset = [p[0] - a[0], p[1] - a[1]], along = (offset[0] * delta[0] + offset[1] * delta[1]) / length;
    return Math.abs(offset[0] * delta[1] - offset[1] * delta[0]) / length < 0.8 && along >= -0.8 && along <= length + 0.8;
  });
}

export function wallData(layer, segment, level, origin = "manual", template) {
  const c = segment.flat().map(Math.round);
  if (c[0] === c[2] && c[1] === c[3]) return null;
  const defaults = {c, levels: [level], move: 20, sight: 20, light: 20, sound: 20, dir: 0, door: 0, ds: 0};
  if (layer === "doors") defaults.door = 1;
  if (layer === "invisible") Object.assign(defaults, {sight: 0, light: 0, sound: 0});
  if (layer === "windows") Object.assign(defaults, {sight: 30, light: 30,
    threshold: {attenuation: true, sight: 10, light: 10}});
  const data = template ? structuredClone(template) : defaults;
  delete data._id; delete data._stats;
  data.c = c; data.levels = [level];
  data.flags = {...data.flags, [MODULE_ID]: {...data.flags?.[MODULE_ID], origin}};
  return data;
}

export function lightData(light, level, gridDistance, origin = "manual") {
  if (![light.x, light.y, light.range, gridDistance].every(Number.isFinite) || light.range <= 0 || gridDistance <= 0) throw new Error("Invalid light position or range.");
  return {x: Math.round(light.x), y: Math.round(light.y), elevation: 0, levels: [level], walls: true,
    config: {angle: 360, color: light.color, dim: light.range * gridDistance,
      bright: light.range * gridDistance / 2, alpha: 0.05},
    flags: {[MODULE_ID]: {origin}}};
}

function remainingLevels(scene, document, level) {
  const levels = document.levels?.length ? document.levels : [...scene.levels].map(l => l.id);
  return levels.filter(id => id !== level);
}

/** Plan only embedded-document changes. No scene replacement or export/import. */
export function planWalls(scene, level, existing, desired, origin = "manual") {
  const wanted = new Map(), existingByKey = new Map(existing.map(d => [key(d), d]));
  for (const layer of ["walls", "invisible", "doors", "windows"]) for (const segment of desired[layer] ?? []) {
    const plain = wallData(layer, segment, level, origin);
    if (!plain) continue;
    const identifier = key(plain);
    if (existingByKey.has(identifier)) { wanted.set(identifier, plain); continue; }
    const template = existing.find(d => wallKind(d) === layer && insideOriginal(segment, d));
    const source = origin === "detection" && template && template.flags?.[MODULE_ID]?.origin !== "detection" ? "manual" : origin;
    const data = wallData(layer, segment, level, source, template);
    if (data) wanted.set(identifier, data);
  }
  const retained = new Set(existing.filter(d => wanted.has(key(d))).map(d => key(d)));
  const create = [...wanted].filter(([k]) => !retained.has(k)).map(([, d]) => d);
  const update = [], remove = [];
  for (const document of existing) {
    if (wanted.has(key(document))) continue;
    const levels = remainingLevels(scene, document, level);
    if (levels.length) update.push({_id: document._id, levels});
    else remove.push(document._id);
  }
  return {create, update, remove};
}

function makeReceipt(type, before, after) {
  const previous = new Map(before.map(d => [d._id, d])), current = new Map(after.map(d => [d._id, d]));
  return [...new Set([...previous.keys(), ...current.keys()])].flatMap(id => {
    const a = previous.get(id), b = current.get(id);
    return stable(a) === stable(b) ? [] : [{type, id, before: a, after: b}];
  });
}

export class LiveSync {
  constructor(scene, level) {
    this.scene = scene; this.level = level; this.busy = false; this.undoStack = [];
    this.operation = 0;
  }

  state() { return levelSnapshot(this.scene, this.level); }
  saved() { return this.scene.getFlag(MODULE_ID, "levels")?.[this.level] ?? null; }

  async #call(type, method, values, options = {}) {
    if (!values.length) return [];
    return this.scene[method](type, values, {...options, autoWallLive: this.operation});
  }

  async #restore(receipt) {
    // Never undo somebody else's subsequent native edits, including door state.
    for (const item of receipt) {
      const collection = item.type === "Wall" ? this.scene.walls : this.scene.lights;
      if (stable(snapshot(collection.get(item.id))) !== stable(item.after)) throw new Error("This edit changed elsewhere. Undo stopped to preserve the newer change.");
    }
    for (const type of ["Wall", "AmbientLight"]) {
      const items = receipt.filter(r => r.type === type);
      await this.#call(type, "createEmbeddedDocuments", items.filter(r => r.before && !r.after).map(r => r.before), {keepId: true});
      await this.#call(type, "updateEmbeddedDocuments", items.filter(r => r.before && r.after).map(r => r.before), {diff: false, recursive: false});
      await this.#call(type, "deleteEmbeddedDocuments", items.filter(r => !r.before && r.after).map(r => r.id));
    }
  }

  async commit({geometry, lights, paint, exclusions = [], origin = "manual", replaceDetectedLights = false}, expected) {
    if (this.busy) throw new Error("Wait for the current scene edit to finish.");
    if (!this.scene.canUserModify(globalThis.game?.user, "update")) throw new Error("Only a GM with scene access can edit this scene.");
    if (fingerprint(this.state()) !== expected) throw new Error("Scene geometry changed. The editor has refreshed; repeat the gesture.");
    this.busy = true; this.operation++;
    const before = this.state(), previousFlag = structuredClone(this.saved()), receipt = [];
    try {
      const plan = planWalls(this.scene, this.level, before.walls, geometry, origin);
      // Record each successful batch separately so a later error can roll it back.
      const apply = async (type, method, values, options) => {
        if (!values.length) return;
        const collection = type === "Wall" ? this.scene.walls : this.scene.lights;
        const old = [...collection].map(snapshot);
        if (method !== "createEmbeddedDocuments") {
          const baseline = type === "Wall" ? before.walls : before.lights;
          for (const value of values) {
            const id = typeof value === "string" ? value : value._id;
            const original = baseline.find(d => d._id === id);
            if (original && stable(original) !== stable(snapshot(collection.get(id)))) {
              throw new Error("A targeted document changed while saving. The edit was stopped to preserve it.");
            }
          }
        }
        const result = await this.#call(type, method, values, options);
        const ids = new Set(method === "deleteEmbeddedDocuments" ? values : result.map(d => d.id ?? d._id));
        receipt.push(...makeReceipt(type, old.filter(d => ids.has(d._id)), [...collection].map(snapshot).filter(d => ids.has(d._id))));
        if (result.length !== values.length) throw new Error("Foundry did not apply the complete edit.");
      };
      await apply("Wall", "createEmbeddedDocuments", plan.create);
      await apply("Wall", "updateEmbeddedDocuments", plan.update);
      await apply("Wall", "deleteEmbeddedDocuments", plan.remove);
      if (lights) {
        const retained = new Set(lights.filter(l => l._id).map(l => l._id));
        const removed = before.lights.filter(l => !retained.has(l._id) &&
          (!replaceDetectedLights || l.flags?.[MODULE_ID]?.origin === "detection"));
        const updates = [], deletes = [];
        for (const light of removed) {
          const levels = remainingLevels(this.scene, light, this.level);
          if (levels.length) updates.push({_id: light._id, levels}); else deletes.push(light._id);
        }
        await apply("AmbientLight", "createEmbeddedDocuments", lights.filter(l => !l._id).map(l => lightData(l, this.level, this.scene.grid.distance, origin)));
        await apply("AmbientLight", "updateEmbeddedDocuments", updates);
        await apply("AmbientLight", "deleteEmbeddedDocuments", deletes);
      }
      const planned = {walls: new Map(before.walls.map(d => [d._id, d])), lights: new Map(before.lights.map(d => [d._id, d]))};
      for (const item of receipt) {
        const collection = item.type === "Wall" ? planned.walls : planned.lights;
        if (item.after && onLevel(item.after, this.level)) collection.set(item.id, item.after);
        else collection.delete(item.id);
      }
      const plannedFingerprint = fingerprint({walls: [...planned.walls.values()], lights: [...planned.lights.values()]});
      if (fingerprint(this.state()) !== plannedFingerprint) throw new Error("Scene geometry changed while saving. Your edit was rolled back; repeat it on the refreshed scene.");
      const saved = {version: 1, fingerprint: plannedFingerprint, paint, exclusions};
      await this.scene.update({[`flags.${MODULE_ID}.levels.${this.level}`]: saved}, {autoWallLive: this.operation});
      if (receipt.length) {
        this.undoStack.push({receipt, previousFlag});
        if (this.undoStack.length > 30) this.undoStack.shift();
      }
      return saved;
    } catch (error) {
      try { await this.#restore(receipt); }
      catch (rollbackError) { throw new Error(`${error.message} Some changes could not be rolled back: ${rollbackError.message}`); }
      throw error;
    } finally { this.busy = false; }
  }

  async undo() {
    if (this.busy) throw new Error("Wait for the current scene edit to finish.");
    const item = this.undoStack.at(-1);
    if (!item) return false;
    this.busy = true; this.operation++;
    try {
      await this.#restore(item.receipt);
      await this.scene.update({[`flags.${MODULE_ID}.levels.${this.level}`]: item.previousFlag}, {autoWallLive: this.operation});
      this.undoStack.pop(); return true;
    } finally { this.busy = false; }
  }
}
