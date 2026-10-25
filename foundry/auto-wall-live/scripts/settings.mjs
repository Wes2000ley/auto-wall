export const MODULE_ID = "auto-wall-live";
export const LAYERS = ["walls", "doors", "invisible", "windows", "lights"];
export const MODES = ["rectangle", "brush", "line"];
export const DEFAULT_TOOLS = Object.freeze({
  layer: "walls", mode: "rectangle", diameter: 24,
  lightColor: "#ffce72", lightRange: 6
});

export function toolPreferences(values = {}) {
  const tools = {...DEFAULT_TOOLS};
  if (!values || typeof values !== "object") return tools;
  if (LAYERS.includes(values.layer)) tools.layer = values.layer;
  if (MODES.includes(values.mode)) tools.mode = values.mode;
  else if (typeof values.brush === "boolean") tools.mode = values.brush ? "brush" : "rectangle";
  if (Number.isFinite(values.diameter)) tools.diameter = Math.max(2, Math.min(2048, Math.round(values.diameter)));
  if (typeof values.lightColor === "string" && /^#[\da-f]{6}$/i.test(values.lightColor)) tools.lightColor = values.lightColor;
  if (Number.isFinite(values.lightRange) && values.lightRange > 0 && values.lightRange <= 1000) tools.lightRange = values.lightRange;
  return tools;
}
