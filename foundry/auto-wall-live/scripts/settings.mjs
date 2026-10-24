export const MODULE_ID = "auto-wall-live";

// Match the desktop palette detector. ML settings are deliberately absent.
export const DEFAULTS = Object.freeze({
  max_dimension: 2048, wall_mode: "color", wall_color: [160, 160, 160],
  wall_tolerance: 24, min_wall_area: 1000, simplify: 1.25, wall_gap_close: 0,
  wall_geometry: "outline", void_color: [0, 0, 0], doors: true,
  door_color: [112, 32, 144], door_match: "hue", door_color_tolerance: 24,
  door_hue_tolerance: 16, door_min_length: 12, door_max_length: 160,
  door_gap_close: 2, door_rectangularity: 0.78, door_wall_support: true,
  lights: true, light_mode: "warm", light_color: [255, 190, 50],
  light_threshold: 190, light_min_area: 100, light_max_area: 5000,
  light_core_fraction: 0.06, light_min_fill: 0.25, light_saturation: 60,
  light_range: 6
});

const ENUMS = {
  wall_mode: ["color", "stone", "edges"], wall_geometry: ["outline", "centerline"],
  door_match: ["hue", "rgb"], light_mode: ["warm", "cool", "color", "brightness"]
};
const RANGES = {
  max_dimension: [256, 4096], wall_tolerance: [0, 255], min_wall_area: [0, 1e8],
  simplify: [0, 100], wall_gap_close: [0, 100], door_color_tolerance: [0, 255],
  door_hue_tolerance: [0, 90], door_min_length: [0.1, 4096],
  door_max_length: [0.1, 4096], door_gap_close: [0, 100],
  door_rectangularity: [0.01, 1], light_threshold: [0, 255],
  light_min_area: [0.01, 1e8], light_max_area: [0.01, 1e8],
  light_core_fraction: [0, 1], light_min_fill: [0, 1], light_saturation: [0, 255],
  light_range: [0.1, 1000]
};
const INTEGERS = new Set(["max_dimension", "wall_tolerance", "door_color_tolerance",
  "door_hue_tolerance", "light_threshold", "light_saturation"]);

export function rgb(value) {
  if (typeof value === "string" && /^#[\da-f]{6}$/i.test(value)) {
    return [1, 3, 5].map(i => parseInt(value.slice(i, i + 2), 16));
  }
  if (Array.isArray(value) && value.length === 3 && value.every(v => Number.isInteger(v) && v >= 0 && v <= 255)) {
    return [...value];
  }
  throw new Error("Colors must be #RRGGBB or three integer RGB channels.");
}

export function hex(value) {
  return `#${rgb(value).map(v => v.toString(16).padStart(2, "0")).join("")}`;
}

export function validateSettings(values) {
  const settings = {...DEFAULTS, ...values};
  for (const [key, value] of Object.entries(settings)) {
    if (!Object.hasOwn(DEFAULTS, key)) throw new Error(`Unknown detection setting: ${key}`);
    if (key.endsWith("_color")) settings[key] = rgb(value);
    else if (key in ENUMS) {
      if (!ENUMS[key].includes(value)) throw new Error(`Invalid ${key}.`);
    } else if (key in RANGES) {
      const [low, high] = RANGES[key];
      if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high ||
          (INTEGERS.has(key) && !Number.isInteger(value))) throw new Error(`Invalid ${key}: expected ${low}–${high}.`);
    } else if (typeof value !== "boolean") throw new Error(`Invalid toggle: ${key}`);
  }
  if (settings.door_min_length > settings.door_max_length) throw new Error("Door minimum exceeds maximum length.");
  if (settings.light_min_area > settings.light_max_area) throw new Error("Light minimum exceeds maximum area.");
  return settings;
}

export function preset(name) {
  if (name === "foundry") return validateSettings({});
  if (name === "exact") return validateSettings({wall_tolerance: 0, door_match: "rgb",
    door_color_tolerance: 0, door_gap_close: 0, min_wall_area: 10, simplify: 0.5});
  if (name === "demon") return validateSettings({wall_mode: "stone", wall_color: [131, 131, 131],
    door_color: [255, 160, 65], wall_gap_close: 1, door_gap_close: 0,
    door_max_length: 60, light_mode: "brightness", light_min_area: 3,
    light_max_area: 180, light_core_fraction: 0, light_min_fill: 0});
  throw new Error("Unknown palette preset.");
}

export function importSettings(data, base = DEFAULTS) {
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("A profile must be a JSON object.");
  if ("palette" in data || "reserved_colors" in data) {
    const colors = {};
    for (const name of ["wall", "door", "void"]) {
      const reserved = data.reserved_colors?.[name];
      const candidates = [];
      if (reserved && !Array.isArray(reserved) && typeof reserved === "object") {
        for (const key of ["hex", "rgb"]) if (key in reserved) candidates.push(rgb(reserved[key]));
      } else if (reserved !== undefined) candidates.push(rgb(reserved));
      if (data.palette?.[name] !== undefined) candidates.push(rgb(data.palette[name]));
      if (!candidates.length || candidates.some(c => String(c) !== String(candidates[0]))) {
        throw new Error(`Missing or conflicting ${name} color.`);
      }
      colors[`${name}_color`] = candidates[0];
    }
    if (new Set(Object.values(colors).map(String)).size !== 3) throw new Error("Wall, door, and void colors must differ.");
    return validateSettings({...base, ...colors, wall_mode: "color", wall_gap_close: 0});
  }
  if (data.settings && ((data.version ?? 1) !== 1 || (data.format ?? "auto-wall-settings") !== "auto-wall-settings")) {
    throw new Error("Unsupported settings format.");
  }
  const values = data.settings ?? data;
  if (!values || typeof values !== "object" || Array.isArray(values)) throw new Error("Settings must be a JSON object.");
  // Desktop profiles may contain ML or source-image grid options; none enter the addon.
  const filtered = Object.fromEntries(Object.entries(values).filter(([key]) =>
    key !== "model_path" && key !== "grid_size" && !key.startsWith("ml_")));
  return validateSettings({...base, ...filtered});
}
