export const MODULE_ID = "map-enhancer";
export const DEFAULTS = Object.freeze({scale: 2, maxEdge: 8192, backend: "auto", sharpen: 0.4, sharpRadius: 1,
  smooth: 0, smoothRadius: 2, edgeTolerance: 0.08, quality: 0.85});

export function settings(values = {}) {
  if (!values || typeof values !== "object") values = {};
  const result = {...DEFAULTS};
  const ranges = {sharpen: [0, 2], sharpRadius: [1, 4], smooth: [0, 1], smoothRadius: [1, 4],
    edgeTolerance: [0.01, 0.4], quality: [0.5, 1]};
  if ([1, 2, 4].includes(values.scale)) result.scale = values.scale;
  if (["auto", "cpu"].includes(values.backend)) result.backend = values.backend;
  if ([4096, 8192, 12288, 16383].includes(values.maxEdge)) result.maxEdge = values.maxEdge;
  for (const [key, [low, high]] of Object.entries(ranges)) {
    if (Number.isFinite(values[key])) result[key] = Math.max(low, Math.min(high, values[key]));
  }
  result.sharpRadius = Math.round(result.sharpRadius);
  result.smoothRadius = Math.round(result.smoothRadius);
  return result;
}

// WebP has a 16383-pixel dimension limit. Bound total memory as well as edge size.
export function outputSize(width, height, options, textureLimit = 16383) {
  if (![width, height].every(n => Number.isInteger(n) && n > 0)) throw new Error("Invalid map dimensions.");
  const {scale, maxEdge} = settings(options);
  if (!Number.isFinite(textureLimit) || textureLimit < 1) textureLimit = 16383;
  const factor = Math.min(scale, Math.min(maxEdge, textureLimit, 16383) / Math.max(width, height),
    Math.sqrt(64 * 1024 * 1024 / (width * height)));
  return {width: Math.max(1, Math.floor(width * factor)), height: Math.max(1, Math.floor(height * factor)),
    limited: factor < scale - 1e-6};
}

export function previewRegion(width, height, center = [0.5, 0.5]) {
  const w = Math.min(width, 160), h = Math.min(height, 128);
  const x = Math.round(Math.max(0, Math.min(width - w, center[0] * width - w / 2)));
  const y = Math.round(Math.max(0, Math.min(height - h, center[1] * height - h / 2)));
  return {x, y, width: w, height: h};
}
