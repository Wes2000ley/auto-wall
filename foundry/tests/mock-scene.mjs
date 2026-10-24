class Document {
  constructor(data) { this.data = structuredClone(data); this.id = data._id; }
  toObject() { return structuredClone(this.data); }
}
class Collection extends Map {
  [Symbol.iterator]() { return this.values(); }
}
export class Scene {
  constructor(walls = [], lights = []) {
    this.walls = new Collection(walls.map(d => [d._id, new Document(d)]));
    this.lights = new Collection(lights.map(d => [d._id, new Document(d)]));
    this.grid = {distance: 5}; this.levels = [{id: "ground"}, {id: "upper"}];
    this.flags = {}; this.calls = []; this.sequence = 0; this.allowed = true;
  }
  canUserModify() { return this.allowed; }
  getFlag(module, field) { return structuredClone(this.flags[module]?.[field]); }
  async update(data) {
    this.calls.push(["Scene", "update"]);
    for (const [path, value] of Object.entries(data)) {
      const parts = path.split("."); let target = this;
      for (const part of parts.slice(0, -1)) target = target[part] ??= {};
      target[parts.at(-1)] = structuredClone(value);
    }
  }
  async createEmbeddedDocuments(type, values, options = {}) {
    this.calls.push([type, "create", values.length]);
    if (this.failure === "create") throw new Error("Create failed");
    const collection = type === "Wall" ? this.walls : this.lights;
    const created = values.map(value => {
      const data = structuredClone(value); data._id = options.keepId ? data._id : `new${++this.sequence}`;
      if (collection.has(data._id)) throw new Error("Duplicate ID");
      const document = new Document(data); collection.set(document.id, document); return document;
    });
    this.afterCreate?.(); return created;
  }
  async updateEmbeddedDocuments(type, values, options = {}) {
    this.calls.push([type, "update", values.length]);
    if (this.failure === "update") throw new Error("Update failed");
    const collection = type === "Wall" ? this.walls : this.lights;
    return values.map(value => {
      const old = collection.get(value._id);
      if (!old) throw new Error("Missing document");
      const document = new Document({...old.toObject(), ...structuredClone(value)}); collection.set(document.id, document); return document;
    });
  }
  async deleteEmbeddedDocuments(type, ids) {
    this.calls.push([type, "delete", ids.length]);
    if (this.failure === "delete") { this.failure = null; throw new Error("Delete failed"); }
    const collection = type === "Wall" ? this.walls : this.lights;
    return ids.map(id => { const document = collection.get(id); collection.delete(id); return document; });
  }
}

export function wall(id, c, extra = {}) {
  return {_id: id, c, move: 20, sight: 20, light: 20, sound: 20, door: 0, ds: 0,
    dir: 0, levels: ["ground"], flags: {otherModule: {keep: true}}, ...extra};
}
