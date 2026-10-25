// Vector areas are retained between strokes. Their shared boundary becomes the
// walls, so overlapping brush passes do not produce stacks of little walls.
const EPS = 1e-6;
export const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const mul = (a, n) => [a[0] * n, a[1] * n];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
const clamp = n => Math.max(0, Math.min(1, n));
const clip = () => globalThis.polygonClipping;
export const union = (...areas) => {
  const nonempty = areas.filter(a => a?.length);
  return nonempty.length ? clip().union(...nonempty) : [];
};
export const difference = (area, cut) => area.length && cut.length ? clip().difference(area, cut) : area;
const polygon = ring => [[ring]];

export function rectangle(first, last) {
  const [x0, x1] = [first[0], last[0]].sort((a, b) => a - b);
  const [y0, y1] = [first[1], last[1]].sort((a, b) => a - b);
  return x1 - x0 > EPS && y1 - y0 > EPS ? polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]) : [];
}

export function areaSegments(area) {
  const result = [];
  for (const poly of area) for (const ring of poly) {
    for (let i = 0; i < ring.length - 1; i++) if (distance(ring[i], ring[i + 1]) > EPS) result.push([ring[i], ring[i + 1]]);
    if (ring.length > 2 && distance(ring[0], ring.at(-1)) > EPS) result.push([ring.at(-1), ring[0]]);
  }
  return result;
}

export function contains(area, point) {
  for (const poly of area) {
    let inside = false;
    for (const ring of poly) {
      let inRing = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const a = ring[j], b = ring[i];
        const delta = sub(b, a), offset = sub(point, a);
        const squared = dot(delta, delta);
        if (squared > EPS * EPS && Math.abs(cross(delta, offset)) < EPS &&
            dot(offset, delta) >= -EPS && dot(offset, delta) <= squared + EPS) return true;
        if ((a[1] > point[1]) !== (b[1] > point[1]) &&
            point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inRing = !inRing;
      }
      if (inRing) inside = !inside;
    }
    if (inside) return true;
  }
  return false;
}

export function simplify(points, tolerance) {
  if (points.length <= 2 || tolerance <= 0) return points;
  // Iterative Douglas–Peucker avoids recursion limits on long freehand strokes.
  const keep = new Uint8Array(points.length); keep[0] = keep[points.length - 1] = 1;
  const pending = [[0, points.length - 1]];
  while (pending.length) {
    const [first, last] = pending.pop(), a = points[first], delta = sub(points[last], a);
    const squared = dot(delta, delta);
    let greatest = tolerance, selected = -1;
    for (let i = first + 1; i < last; i++) {
      const projection = add(a, mul(delta, squared ? clamp(dot(sub(points[i], a), delta) / squared) : 0));
      const d = distance(points[i], projection);
      if (d > greatest) { greatest = d; selected = i; }
    }
    if (selected >= 0) { keep[selected] = 1; pending.push([first, selected], [selected, last]); }
  }
  return points.filter((_, i) => keep[i]);
}

export function brushArea(points, diameter, size, flat = false) {
  if (!Number.isFinite(diameter) || diameter <= 0) throw new Error("Brush size must be positive.");
  if (!points.length) return [];
  if (!points.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) throw new Error("Invalid brush coordinates.");
  const radius = diameter / 2, shapes = [];
  const path = flat ? [points[0], points.at(-1)] : simplify(points, Math.max(1, Math.min(8, diameter * 0.12)));
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1], length = distance(a, b);
    if (length < EPS) continue;
    const n = mul([a[1] - b[1], b[0] - a[0]], radius / length);
    const ring = [add(a, n), add(b, n), sub(b, n), sub(a, n)]; ring.push(ring[0]);
    shapes.push(polygon(ring));
  }
  if (!flat) {
    const count = 8;
    for (const p of path) {
      const ring = Array.from({length: count}, (_, i) => add(p, [radius * Math.cos(i * 2 * Math.PI / count), radius * Math.sin(i * 2 * Math.PI / count)]));
      ring.push(ring[0]); shapes.push(polygon(ring));
    }
  }
  if (!shapes.length) return [];
  // Bound the argument count for very long strokes.
  let area = [];
  for (let i = 0; i < shapes.length; i += 128) area = union(area, ...shapes.slice(i, i + 128));
  return clip().intersection(area, rectangle([0, 0], [size[0] - 1, size[1] - 1]));
}

export function splitSegments(segments, area) {
  if (!area.length || !segments.length) return [segments, []];
  const edges = areaSegments(area), outside = [], inside = [];
  const low = [Infinity, Infinity], high = [-Infinity, -Infinity];
  for (const edge of edges) for (const p of edge) for (let i = 0; i < 2; i++) {
    low[i] = Math.min(low[i], p[i]); high[i] = Math.max(high[i], p[i]);
  }
  for (const [a, b] of segments) {
    const delta = sub(b, a), squared = dot(delta, delta);
    if (squared < EPS * EPS) continue;
    if ([0, 1].some(i => Math.max(a[i], b[i]) < low[i] - EPS || Math.min(a[i], b[i]) > high[i] + EPS)) {
      outside.push([a, b]); continue;
    }
    const fractions = [0, 1];
    for (const [c, d] of edges) {
      const edge = sub(d, c), offset = sub(c, a), denominator = cross(delta, edge);
      if (Math.abs(denominator) > 1e-10) {
        const t = cross(offset, edge) / denominator, u = cross(offset, delta) / denominator;
        if (t > 0 && t < 1 && u >= -EPS && u <= 1 + EPS) fractions.push(t);
      } else if (Math.abs(cross(offset, delta)) < EPS) {
        fractions.push(clamp(dot(offset, delta) / squared), clamp(dot(sub(d, a), delta) / squared));
      }
    }
    const cuts = [...new Set(fractions.map(f => Math.round(f * 1e12) / 1e12))].sort((x, y) => x - y);
    for (let i = 0; i < cuts.length - 1; i++) {
      const first = add(a, mul(delta, cuts[i])), last = add(a, mul(delta, cuts[i + 1]));
      if (distance(first, last) < EPS) continue;
      (contains(area, add(a, mul(delta, (cuts[i] + cuts[i + 1]) / 2))) ? inside : outside).push([first, last]);
    }
  }
  return [outside, inside];
}

export function touchesRectangle(segment, first, last) {
  let enter = 0, leave = 1;
  for (let axis = 0; axis < 2; axis++) {
    const low = Math.min(first[axis], last[axis]), high = Math.max(first[axis], last[axis]);
    const origin = segment[0][axis], speed = segment[1][axis] - origin;
    if (Math.abs(speed) < EPS) { if (origin < low || origin > high) return false; }
    else {
      const values = [(low - origin) / speed, (high - origin) / speed].sort((a, b) => a - b);
      enter = Math.max(enter, values[0]); leave = Math.min(leave, values[1]);
      if (enter > leave) return false;
    }
  }
  return true;
}

export function segmentKey(segment, integer = false) {
  const round = n => integer ? Math.round(n) : Math.round(n * 1e6) / 1e6;
  return segment.map(p => p.map(round).join(",")).sort().join("|");
}

export function uniqueSegments(segments) {
  const seen = new Set();
  return segments.filter(s => {
    const key = segmentKey(s);
    if (distance(...s) < EPS || seen.has(key)) return false;
    seen.add(key); return true;
  });
}

export function mergeCollinear(segments) {
  const groups = new Map();
  for (const [a, b] of uniqueSegments(segments)) {
    let axis = mul(sub(b, a), 1 / distance(a, b));
    if (axis[0] < -EPS || (Math.abs(axis[0]) < EPS && axis[1] < 0)) axis = mul(axis, -1);
    const normal = [-axis[1], axis[0]], offset = dot(a, normal);
    const key = [axis[0], axis[1], offset].map(v => Math.round(v * 1e5)).join(",");
    if (!groups.has(key)) groups.set(key, {axis, normal, offset, intervals: []});
    groups.get(key).intervals.push([dot(a, axis), dot(b, axis)].sort((x, y) => x - y));
  }
  const result = [];
  for (const {axis, normal, offset, intervals} of groups.values()) {
    intervals.sort((a, b) => a[0] - b[0]);
    let current = intervals[0];
    const emit = () => result.push(current.map(t => add(mul(axis, t), mul(normal, offset))));
    for (const next of intervals.slice(1)) {
      if (next[0] <= current[1] + EPS) current[1] = Math.max(current[1], next[1]);
      else { emit(); current = next; }
    }
    emit();
  }
  return result;
}

export function splitAtPoints(segments, points) {
  if (!points.length) return segments;
  const result = [];
  for (const [a, b] of segments) {
    const delta = sub(b, a), squared = dot(delta, delta), fractions = [0, 1];
    if (squared < EPS * EPS) continue;
    for (const p of points) {
      const t = dot(sub(p, a), delta) / squared;
      if (t > EPS && t < 1 - EPS && distance(add(a, mul(delta, t)), p) < EPS) fractions.push(t);
    }
    fractions.sort((x, y) => x - y);
    for (let i = 0; i < fractions.length - 1; i++) {
      if (fractions[i + 1] - fractions[i] > EPS) result.push([add(a, mul(delta, fractions[i])), add(a, mul(delta, fractions[i + 1]))]);
    }
  }
  return result;
}

function jambs(removed, span) {
  const result = [], axis = mul(sub(span[1], span[0]), 1 / distance(...span)), length = distance(...span);
  for (const point of removed.flat()) {
    const along = dot(sub(point, span[0]), axis);
    const target = Math.abs(along) < 1e-5 ? span[0] : Math.abs(along - length) < 1e-5 ? span[1] : null;
    if (target && distance(point, target) > EPS) result.push([point, target]);
  }
  return result;
}

export class PaintGeometry {
  constructor(size, base = {}, state = {}) {
    this.size = size;
    this.base = {walls: [], invisible: [], doors: [], windows: [], ...base};
    this.areas = state.areas ?? {walls: [], invisible: []};
    this.coverage = state.coverage ?? {walls: [], invisible: []};
    // Explicit rectangles and click-to-click lines must not be clipped by ink
    // painted before them. Only a subsequent paint or erase operation edits them.
    this.lines = state.lines ?? {walls: [], invisible: []};
    this.portals = state.portals ?? [];
    this.diameter = state.diameter ?? 1;
  }

  static restore(data) {
    if (!data || data.version !== 1 || !Array.isArray(data.size)) throw new Error("Unsupported brush state.");
    return new PaintGeometry(data.size, data.base, data);
  }

  serialize() {
    return structuredClone({version: 1, size: this.size, base: this.base, areas: this.areas,
      coverage: this.coverage, lines: this.lines, portals: this.portals, diameter: this.diameter});
  }

  copy() { return PaintGeometry.restore(this.serialize()); }

  paint(layer, points, diameter, replaceOther = true) {
    const portal = layer === "doors" || layer === "windows";
    let area = brushArea(points, diameter, this.size, portal);
    if (!area.length) return this;
    const next = this.copy(); next.diameter = Math.max(next.diameter, diameter);
    if (portal) {
      let span = [points[0], points.at(-1)], ends = [...span];
      const axis = mul(sub(span[1], span[0]), 1 / distance(...span));
      const overlaps = (existing, material) => {
        if (!existing.every(p => Math.abs(cross(axis, sub(p, span[0]))) < EPS)) return false;
        const positions = existing.map(p => dot(sub(p, span[0]), axis));
        const low = Math.min(...positions), high = Math.max(...positions);
        return material === layer ? high >= -EPS && low <= distance(...span) + EPS : high > EPS && low < distance(...span) - EPS;
      };
      // Expand repeatedly so a bridge can merge several adjacent portal passes.
      let changed;
      do {
        changed = false;
        for (const material of ["doors", "windows"]) next.base[material] = next.base[material].filter(existing => {
          if (!overlaps(existing, material)) return true;
          ends.push(...existing); area = union(area, brushArea(existing, diameter, this.size, true)); changed = true; return false;
        });
        next.portals = next.portals.filter(existing => {
          if (!overlaps(existing.span, existing.layer)) return true;
          ends.push(...existing.span); area = union(area, existing.area); changed = true; return false;
        });
        const values = ends.map(p => dot(sub(p, points[0]), axis));
        span = [Math.min(...values), Math.max(...values)].map(t => add(points[0], mul(axis, t)));
      } while (changed);
      next.portals.push({layer, area, span});
      for (const material of ["walls", "invisible"]) {
        const [outside, removed] = splitSegments(next.lines[material], area);
        next.lines[material] = uniqueSegments([...outside, ...jambs(removed, span)]);
      }
    } else {
      if (!["walls", "invisible"].includes(layer)) throw new Error("Unknown brush material.");
      next.areas[layer] = union(next.areas[layer], area);
      next.coverage[layer] = union(next.coverage[layer], area);
      next.lines[layer] = splitSegments(next.lines[layer], area)[0];
      if (replaceOther) {
        const other = layer === "walls" ? "invisible" : "walls";
        next.areas[other] = difference(next.areas[other], area);
        next.coverage[other] = union(next.coverage[other], area);
        next.lines[other] = splitSegments(next.lines[other], area)[0];
      }
    }
    return next;
  }

  addRectangle(layer, first, last) {
    return this.addLines(layer, areaSegments(rectangle(first, last)));
  }

  addLines(layer, segments) {
    if (!["walls", "invisible"].includes(layer)) throw new Error("Unknown line material.");
    if (!segments.every(s => s.length === 2 && s.every(p => p.length === 2 && p.every(Number.isFinite)))) {
      throw new Error("Invalid line coordinates.");
    }
    const next = this.copy();
    next.lines[layer] = mergeCollinear([...next.lines[layer], ...segments]);
    return next;
  }

  erase(first, last, visible) {
    const next = this.copy(), area = rectangle(first, last);
    for (const layer of ["walls", "invisible", "doors", "windows"]) {
      if (!visible[layer]) continue;
      next.base[layer] = next.base[layer].filter(s => !touchesRectangle(s, first, last));
      if (layer in next.areas) {
        next.areas[layer] = difference(next.areas[layer], area);
        next.lines[layer] = next.lines[layer].filter(s => !touchesRectangle(s, first, last));
      }
    }
    next.portals = next.portals.filter(p => !visible[p.layer] || !touchesRectangle(p.span, first, last));
    return next;
  }

  compose() {
    const portals = [...this.portals];
    for (const layer of ["doors", "windows"]) for (const span of this.base[layer]) {
      portals.push({layer, span, area: brushArea(span, Math.max(2, this.diameter * 2), this.size, true), base: true});
    }
    const explicit = union(...portals.filter(p => !p.base).map(p => p.area));
    const cuts = union(...portals.flatMap(p => [p.area, brushArea(p.span, Math.max(2, this.diameter * 2), this.size, true)]));
    const result = {doors: uniqueSegments(portals.filter(p => p.layer === "doors").map(p => p.span)),
      windows: uniqueSegments(portals.filter(p => p.layer === "windows").map(p => p.span))};
    for (const layer of ["walls", "invisible"]) {
      // Baseline native documents are already authoritative. Only new paint
      // and newly drawn portals may change them; opening the editor never does.
      const uncovered = splitSegments(this.base[layer], this.coverage[layer])[0];
      const [outside, removed] = splitSegments(uncovered, explicit);
      const connections = portals.filter(p => !p.base).flatMap(p => jambs(splitSegments(removed, p.area)[1], p.span));
      // Simplify ink before cutting doorways so portal ends stay exact.
      const tolerance = Math.max(1, Math.min(6, this.diameter * 0.08));
      const ink = this.areas[layer].map(poly => poly.map(ring => {
        const reduced = simplify(ring, tolerance);
        return reduced.length >= 4 ? reduced : ring;
      }));
      const boundary = areaSegments(difference(ink, cuts));
      const added = [...connections, ...mergeCollinear(boundary)];
      const newPoints = [...this.portals.flatMap(p => p.span), ...added.flat()];
      result[layer] = uniqueSegments([...splitAtPoints(outside, newPoints),
        ...splitAtPoints(added, [...outside.flat(), ...portals.flatMap(p => p.span)]), ...this.lines[layer]]);
    }
    return result;
  }
}
