import {validateSettings} from "./settings.mjs";
import {splitSegments, distance, nearest, uniqueSegments, simplify, splitAtPoints} from "./geometry.mjs";

function rgbHSV([r, g, b]) {
  r /= 255; g /= 255; b /= 255;
  const high = Math.max(r, g, b), low = Math.min(r, g, b), delta = high - low;
  let hue = 0;
  if (delta) {
    hue = high === r ? (g - b) / delta : high === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
    hue = (hue * 30 + 180) % 180;
  }
  return [Math.round(hue) % 180, high ? Math.round(delta / high * 255) : 0, Math.round(high * 255)];
}
const hueDistance = (a, b) => Math.min(Math.abs(a - b), 180 - Math.abs(a - b));

function rectangleBox(center, length, thickness, angle) {
  const axis = [Math.cos(angle), Math.sin(angle)], n = [-axis[1], axis[0]];
  const ring = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
    [center[0] + a * axis[0] * length / 2 + b * n[0] * thickness / 2,
      center[1] + a * axis[1] * length / 2 + b * n[1] * thickness / 2]);
  ring.push(ring[0]); return [[ring]];
}

function attachDoors(walls, doors, scale, centerline, requireSupport) {
  const accepted = [], indices = [], additions = [];
  for (let i = 0; i < doors.length; i++) {
    const span = doors[i], length = distance(...span), axis = span[0].map((v, j) => (span[1][j] - v) / length);
    const center = span[0].map((v, j) => (span[1][j] + v) / 2), reach = Math.max(3 * scale, length * 0.4);
    const endpoints = [], connections = [];
    for (let end = 0; end < 2; end++) {
      const sign = end ? 1 : -1, expected = sign * length / 2;
      let best = null, bestDistance = Infinity;
      for (const [a, b] of walls) {
        const v = [b[0] - a[0], b[1] - a[1]], off = [a[0] - center[0], a[1] - center[1]];
        const cross = axis[0] * v[1] - axis[1] * v[0];
        if (Math.abs(cross) < 1e-9) continue;
        const along = (off[0] * v[1] - off[1] * v[0]) / cross;
        const fraction = (off[0] * axis[1] - off[1] * axis[0]) / cross;
        if (fraction < -1e-6 || fraction > 1 + 1e-6 || sign * along <= 0 || Math.abs(along - expected) > reach) continue;
        if (Math.abs(along) < bestDistance) { bestDistance = Math.abs(along); best = center.map((v, j) => v + along * axis[j]); }
      }
      if (!best && centerline) {
        const corner = nearest(walls, span[end], reach);
        if (corner) {
          const along = (corner[0] - center[0]) * axis[0] + (corner[1] - center[1]) * axis[1];
          best = center.map((v, j) => v + along * axis[j]);
          if (distance(best, corner) > 1e-6) connections.push([corner, best]);
        }
      }
      endpoints.push(best);
    }
    if (!requireSupport) for (let end = 0; end < 2; end++) endpoints[end] ??= span[end];
    if (endpoints.every(Boolean) && distance(...endpoints) > 1) {
      accepted.push(endpoints); indices.push(i);
      additions.push(...connections);
    }
  }
  return {doors: accepted, indices, additions};
}

function clipDoorways(walls, boxes, spans, preserveJambs) {
  let result = walls;
  for (let i = 0; i < boxes.length; i++) {
    const span = spans[i], length = distance(...span), axis = span[0].map((v, j) => (span[1][j] - v) / length);
    const center = span[0].map((v, j) => (span[1][j] + v) / 2);
    const preserved = [], candidates = [];
    for (const segment of result) {
      const along = segment.map(p => (p[0] - center[0]) * axis[0] + (p[1] - center[1]) * axis[1]);
      const side = segment.map(p => -(p[0] - center[0]) * axis[1] + (p[1] - center[1]) * axis[0]);
      const v = [segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]];
      const nearEnd = Math.min(...along) > length * 0.35 || Math.max(...along) < -length * 0.35;
      const perpendicular = Math.abs(v[0] * axis[0] + v[1] * axis[1]) <= 0.35 * distance(...segment);
      const crosses = Math.min(...side) <= 1e-6 && Math.max(...side) >= -1e-6 && Math.abs(side[0] - side[1]) > 1e-6;
      (preserveJambs && nearEnd && (perpendicular || crosses) ? preserved : candidates).push(segment);
    }
    result = [...preserved, ...splitSegments(candidates, boxes[i])[0]];
  }
  return result;
}

function skeletonSegments(mask, width, height, tolerance) {
  // Zhang–Suen thinning in typed arrays. This runs only in the detection worker.
  // A zero border lets edge-touching walls thin just like interior walls.
  const originalWidth = width, originalHeight = height;
  width += 2; height += 2;
  const data = new Uint8Array(width * height), remove = [];
  for (let y = 0; y < originalHeight; y++) for (let x = 0; x < originalWidth; x++) {
    data[(y + 1) * width + x + 1] = mask[y * originalWidth + x] ? 1 : 0;
  }
  const offsets = [-width, -width + 1, 1, width + 1, width, width - 1, -1, -width - 1];
  let changed;
  do {
    changed = false;
    for (let phase = 0; phase < 2; phase++) {
      remove.length = 0;
      for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        if (!data[i]) continue;
        let count = 0, transitions = 0;
        for (let k = 0; k < 8; k++) {
          const v = data[i + offsets[k]]; count += v;
          if (!v && data[i + offsets[(k + 1) % 8]]) transitions++;
        }
        if (count < 2 || count > 6 || transitions !== 1) continue;
        const n = data[i - width], e = data[i + 1], s = data[i + width], w = data[i - 1];
        if (phase ? n * e * w || n * s * w : n * e * s || e * s * w) continue;
        remove.push(i);
      }
      if (remove.length) changed = true;
      for (const i of remove) data[i] = 0;
    }
  } while (changed);
  const neighbors = i => {
    const list = [], x = i % width, y = Math.floor(i / width);
    for (const [dx, dy] of [[0, -1], [-1, 0], [1, 0], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height) continue;
      const j = i + dy * width + dx;
      if (!data[j] || (dx && dy && (data[i + dx] || data[i + dy * width]))) continue;
      list.push(j);
    }
    return list;
  };
  const visited = new Uint8Array(data.length), paths = [];
  const mark = (i, j) => {
    const a = offsets.indexOf(j - i), b = offsets.indexOf(i - j);
    visited[i] |= 1 << a; visited[j] |= 1 << b;
  };
  const used = (i, j) => visited[i] & (1 << offsets.indexOf(j - i));
  const trace = (start, next) => {
    const path = [start]; let previous = start, current = next;
    mark(previous, current);
    while (true) {
      path.push(current); const adjacent = neighbors(current);
      if (current === start || adjacent.length !== 2) break;
      const last = adjacent.find(j => j !== previous);
      if (used(current, last)) break;
      mark(current, last); previous = current; current = last;
    }
    paths.push(path.map(i => [i % width - 1, Math.floor(i / width) - 1]));
  };
  for (let i = 0; i < data.length; i++) if (data[i]) {
    const adjacent = neighbors(i);
    if (adjacent.length !== 2) for (const j of adjacent) if (!used(i, j)) trace(i, j);
  }
  for (let i = 0; i < data.length; i++) if (data[i]) for (const j of neighbors(i)) if (!used(i, j)) trace(i, j);
  return paths.flatMap(path => {
    const points = simplify(path, tolerance);
    return points.slice(1).map((p, i) => [points[i], p]);
  });
}

/** Pure image-to-vectors entry point, also exercised in Node with the bundled WASM. */
export function detectRGBA(cv, rgba, width, height, values, progress = () => {}) {
  const settings = validateSettings(values), started = performance.now(), owned = [];
  const own = value => { owned.push(value); return value; };
  const mat = () => own(new cv.Mat());
  const mask = () => own(cv.Mat.zeros(height, width, cv.CV_8UC1));
  const scale = Math.max(width, height) / 2048;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
      width > 4096 || height > 4096 || rgba.length !== width * height * 4) throw new Error("Invalid detection image (maximum 4096 pixels per edge).");
  try {
    progress("Reading image");
    const source = own(cv.matFromArray(height, width, cv.CV_8UC4, rgba)), image = mat(), hsv = mat();
    cv.cvtColor(source, image, cv.COLOR_RGBA2RGB); cv.cvtColor(image, hsv, cv.COLOR_RGB2HSV);
    const walls = mask(), wallData = walls.data;
    let pixels = image.data, colors = hsv.data;
    // WASM heap growth detaches earlier TypedArray views. Refresh only after
    // native allocations, then use the views throughout each pure JS pass.
    const refresh = () => { pixels = image.data; colors = hsv.data; };
    let stoneValue = 140;
    if (settings.wall_mode === "stone") {
      const histogram = new Uint32Array(256);
      for (let i = 0; i < width * height; i++) if (colors[i * 3 + 1] <= 38 && colors[i * 3 + 2] >= 65 && colors[i * 3 + 2] <= 220) histogram[colors[i * 3 + 2]]++;
      for (let v = 65; v <= 220; v++) if (histogram[v] > histogram[stoneValue]) stoneValue = v;
    }
    progress("Detecting walls");
    if (settings.wall_mode === "edges") {
      const gray = mat(), blurred = mat(); cv.cvtColor(image, gray, cv.COLOR_RGB2GRAY);
      cv.GaussianBlur(gray, blurred, new cv.Size(3, 3), 0); cv.Canny(blurred, walls, 50, 150);
    } else for (let i = 0; i < width * height; i++) {
      const j = i * 3;
      const selected = settings.wall_mode === "stone" ? colors[j + 1] <= 55 && Math.abs(colors[j + 2] - stoneValue) <= settings.wall_tolerance :
        settings.wall_color.every((c, k) => Math.abs(pixels[j + k] - c) <= settings.wall_tolerance);
      wallData[i] = selected ? 255 : 0;
    }
    const close = (input, amount) => {
      if (amount <= 0) return;
      const radius = Math.max(1, Math.round(scale * amount));
      const kernel = own(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(radius * 2 + 1, radius * 2 + 1)));
      cv.morphologyEx(input, input, cv.MORPH_CLOSE, kernel);
    };
    close(walls, settings.wall_gap_close);
    refresh(); let wallPixels = walls.data;
    for (let i = 0; i < width * height; i++) {
      if (rgba[i * 4 + 3] === 0 || settings.void_color.every((c, k) => pixels[i * 3 + k] === c)) wallPixels[i] = 0;
    }
    const labels = mat(), stats = mat(), centers = mat();
    const count = cv.connectedComponentsWithStats(walls, labels, stats, centers, 8, cv.CV_32S);
    const keep = new Uint8Array(count), wallStats = stats.data32S, wallLabels = labels.data32S;
    wallPixels = walls.data;
    for (let i = 1; i < count; i++) if (wallStats[i * 5 + 4] >= settings.min_wall_area * scale * scale) keep[i] = 255;
    for (let i = 0; i < width * height; i++) wallPixels[i] = keep[wallLabels[i]];

    const doorMask = mask(), [dh, ds, dv] = rgbHSV(settings.door_color);
    refresh(); let doorPixels = doorMask.data;
    if (settings.doors || settings.lights) for (let i = 0; i < width * height; i++) {
      const j = i * 3, h = colors[j], s = colors[j + 1], v = colors[j + 2];
      const selected = settings.door_match === "rgb" ? settings.door_color.every((c, k) => Math.abs(pixels[j + k] - c) <= settings.door_color_tolerance) :
        ds < 40 ? s <= 55 && Math.abs(v - dv) <= 35 : hueDistance(h, dh) <= settings.door_hue_tolerance &&
          s >= Math.min(90, Math.max(35, ds - 70)) && v >= Math.max(20, dv - 85) && v <= Math.min(255, dv + 85);
      doorPixels[i] = selected && rgba[i * 4 + 3] ? 255 : 0;
    }
    close(doorMask, settings.door_gap_close);
    doorPixels = doorMask.data; wallPixels = walls.data;
    if (settings.door_gap_close > 0) for (let i = 0; i < width * height; i++) if (wallPixels[i]) doorPixels[i] = 0;
    progress("Checking door jambs");
    const doorContours = own(new cv.MatVector()), hierarchy = mat();
    cv.findContours(doorMask, doorContours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const doors = [], boxes = [];
    for (let i = 0; i < doorContours.size(); i++) {
      const contour = own(doorContours.get(i)), rect = cv.minAreaRect(contour);
      const length = Math.max(rect.size.width, rect.size.height), thickness = Math.min(rect.size.width, rect.size.height);
      if (length < settings.door_min_length * scale || length > settings.door_max_length * scale ||
          thickness < Math.max(2, scale * 2) || length / thickness < 1.25 || length / thickness > 6 ||
          cv.contourArea(contour) / (length * thickness) < settings.door_rectangularity) continue;
      let angle = rect.angle + (rect.size.height > rect.size.width ? 90 : 0);
      if (Math.abs(angle - Math.round(angle / 90) * 90) <= 3) angle = Math.round(angle / 90) * 90;
      const theta = angle * Math.PI / 180, axis = [Math.cos(theta), Math.sin(theta)], center = [rect.center.x, rect.center.y];
      const ends = [-1, 1].map(sign => center.map((v, k) => v + sign * axis[k] * length / 2));
      wallPixels = walls.data;
      if (settings.door_wall_support && !ends.every((p, end) => {
        const reach = Math.max(3 * scale, thickness * 0.7), steps = Math.max(3, Math.ceil(reach) + 2);
        for (let sample = 0; sample < steps; sample++) {
          const offset = -scale + (reach + scale) * sample / (steps - 1), sign = end ? 1 : -1;
          const x = Math.round(p[0] + sign * axis[0] * offset), y = Math.round(p[1] + sign * axis[1] * offset);
          if (x >= 0 && y >= 0 && x < width && y < height && wallPixels[y * width + x]) return true;
        }
        return false;
      })) continue;
      const extension = Math.max(2 * scale, thickness * 0.18);
      doors.push(ends.map((p, end) => p.map((v, k) => Math.max(0, Math.min((k ? height : width) - 1, v + (end ? 1 : -1) * axis[k] * extension)))));
      boxes.push(rectangleBox(center, length + 4 * scale, thickness + 4 * scale, theta));
    }
    progress("Tracing walls");
    let segments;
    if (settings.wall_geometry === "centerline") segments = skeletonSegments(walls.data, width, height, settings.simplify * scale);
    else {
      const contours = own(new cv.MatVector()), hierarchy = mat();
      cv.findContours(walls, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE); segments = [];
      for (let i = 0; i < contours.size(); i++) {
        const contour = own(contours.get(i));
        if (cv.contourArea(contour) < Math.max(2, settings.min_wall_area * scale * scale * 0.1)) continue;
        const approximated = mat(); cv.approxPolyDP(contour, approximated, settings.simplify * scale, true);
        const points = [];
        for (let j = 0; j < approximated.data32S.length; j += 2) points.push([approximated.data32S[j], approximated.data32S[j + 1]]);
        for (let j = 0; j < points.length; j++) if (distance(points[j], points[(j + 1) % points.length]) > 1e-6) segments.push([points[j], points[(j + 1) % points.length]]);
      }
    }
    let portals = [], rejected = 0;
    if (settings.doors) {
      const centerline = settings.wall_geometry === "centerline";
      let cut = clipDoorways(segments, boxes, doors, !centerline);
      const attached = attachDoors(cut, doors, scale, centerline, settings.door_wall_support);
      rejected = doors.length - attached.doors.length;
      if (rejected) cut = clipDoorways(segments, attached.indices.map(i => boxes[i]), attached.indices.map(i => doors[i]), !centerline);
      segments = [...cut, ...attached.additions]; portals = attached.doors;
      segments = splitAtPoints(segments, portals.flat());
    }
    progress("Detecting lights");
    const lights = [];
    if (settings.lights) {
      const lightMask = mask(), bright = new Uint8Array(width * height), [lh, ls, lv] = rgbHSV(settings.light_color);
      refresh(); const lightPixels = lightMask.data;
      for (let i = 0; i < width * height; i++) {
        const j = i * 3, h = colors[j], s = colors[j + 1], v = colors[j + 2];
        bright[i] = (0.299 * pixels[j] + 0.587 * pixels[j + 1] + 0.114 * pixels[j + 2] >= settings.light_threshold && s >= settings.light_saturation) ? 1 : 0;
        const selected = settings.light_mode === "warm" ? h <= 40 && s >= 120 && v >= 180 :
          settings.light_mode === "cool" ? h >= 75 && h <= 115 && s >= 40 && v >= 170 :
            settings.light_mode === "color" ? hueDistance(h, lh) <= 15 && s >= Math.max(20, ls - 100) && v >= Math.max(80, lv - 70) : bright[i];
        lightPixels[i] = selected && rgba[i * 4 + 3] ? 255 : 0;
      }
      for (const box of boxes) {
        const vertices = own(cv.matFromArray(4, 1, cv.CV_32SC2, box[0][0].slice(0, 4).flat().map(Math.round)));
        cv.fillConvexPoly(lightMask, vertices, new cv.Scalar(0));
      }
      const count = cv.connectedComponentsWithStats(lightMask, labels, stats, centers, 8, cv.CV_32S);
      refresh();
      const lightLabels = labels.data32S, lightStats = stats.data32S, lightCenters = centers.data64F;
      const sums = Array.from({length: count}, () => [0, 0, 0, 0]);
      for (let i = 0; i < width * height; i++) {
        const label = lightLabels[i]; if (!label) continue;
        const sum = sums[label]; sum[0] += pixels[i * 3]; sum[1] += pixels[i * 3 + 1]; sum[2] += pixels[i * 3 + 2]; sum[3] += bright[i];
      }
      for (let i = 1; i < count; i++) {
        const area = lightStats[i * 5 + 4], w = lightStats[i * 5 + 2], h = lightStats[i * 5 + 3];
        if (area < settings.light_min_area * scale * scale || area > settings.light_max_area * scale * scale || area / (w * h) < settings.light_min_fill ||
            sums[i][3] < Math.max(1, Math.min(3, settings.light_min_area) * scale * scale, area * settings.light_core_fraction)) continue;
        const color = `#${sums[i].slice(0, 3).map(c => Math.round(c / area).toString(16).padStart(2, "0")).join("")}`;
        lights.push({x: lightCenters[i * 2], y: lightCenters[i * 2 + 1], color, range: settings.light_range});
      }
    }
    return {walls: uniqueSegments(segments), doors: portals, lights, width, height,
      elapsed: performance.now() - started, rejected,
      warnings: segments.length ? [] : ["No walls found. Sample the actual wall color or change the palette."]};
  } finally {
    for (let i = owned.length - 1; i >= 0; i--) owned[i].delete();
  }
}
