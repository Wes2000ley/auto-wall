import {MODULE_ID} from "./settings.mjs";

export function dataFolder(value) {
  if (typeof value !== "string") throw new Error("Choose a relative Foundry Data folder.");
  const path = value.trim().replace(/^\/+|\/+$/g, "");
  if (!path || /[\\:*?"<>|\u0000-\u001f]/.test(path) || path.split("/").some(p => !p || p === "." || p === ".." || /[. ]$/.test(p))) {
    throw new Error("Choose a relative Foundry Data folder, such as assets/maps/encounters.");
  }
  if (["modules", "systems"].includes(path.split("/")[0])) throw new Error("Save maps in assets or a world folder, outside installed packages.");
  return path;
}

export function destination(source, baseURL, routePrefix = "") {
  const url = new URL(source, baseURL), base = new URL(baseURL);
  let path = decodeURIComponent(url.pathname);
  const prefix = routePrefix.replace(/^\/+|\/+$/g, "");
  if (prefix && path.startsWith(`/${prefix}/`)) path = path.slice(prefix.length + 1);
  path = path.replace(/^\//, "");
  const parts = path.split("/"), rawName = parts.pop() || "map";
  const name = rawName.replace(/\.[^.]+$/, "").replace(/[\\/:*?"<>|\u0000-\u001f]/g, "-").replace(/[. ]+$/, "").slice(0, 140) || "map";
  let folder = "assets/maps/enhanced";
  if (url.origin === base.origin && parts.length) {
    try { folder = dataFolder(parts.join("/")); } catch { /* Package and external assets use a writable Data folder. */ }
  }
  return {folder, name};
}

export function targetSnapshot(scene, level) {
  return {sceneId: scene.id, levelId: level.id, source: level.background.src,
    footprint: JSON.stringify({width: scene.width, height: scene.height, textures: level.textures})};
}

export function checkTarget(scene, level, expected) {
  if (!globalThis.game?.user?.isGM || !scene.canUserModify(game.user, "update")) throw new Error("Only a GM with scene access can save this map.");
  if (scene.levels.get(level.id) !== level || JSON.stringify(targetSnapshot(scene, level)) !== JSON.stringify(expected)) {
    throw new Error("The map or its placement changed. Reopen Map Enhancer before saving.");
  }
}

async function ensureFolder(picker, folder) {
  let current = "";
  for (const part of folder.split("/")) {
    current = current ? `${current}/${part}` : part;
    try { await picker.browse("data", current); }
    catch {
      try { await picker.createDirectory("data", current); }
      catch (error) {
        // A concurrent directory creation is harmless; permissions errors remain visible.
        try { await picker.browse("data", current); } catch { throw error; }
      }
    }
  }
}

export async function saveMap({scene, level, expected, blob, folder, name, options, dimensions, picker, unique}) {
  checkTarget(scene, level, expected);
  folder = dataFolder(folder);
  if (blob.type !== "image/webp" || !blob.size) throw new Error("The browser did not produce a valid WebP image.");
  if (typeof name !== "string" || !name || /[\\/:*?"<>|\u0000-\u001f]/.test(name) ||
      typeof unique !== "string" || !/^[a-zA-Z0-9-]+$/.test(unique)) throw new Error("Invalid output filename.");
  const filename = `${name}-enhanced-${dimensions.width}x${dimensions.height}-${unique}.webp`;
  await ensureFolder(picker, folder);
  checkTarget(scene, level, expected);
  const result = await picker.upload("data", folder, new File([blob], filename, {type: "image/webp"}), {}, {notify: false});
  if (!result?.path || result.error) throw new Error(result?.error || "Foundry could not save the image. The scene was not changed.");
  try {
    checkTarget(scene, level, expected);
    const history = [...(level.getFlag(MODULE_ID, "history") ?? []),
      {before: expected.source, after: result.path, settings: options, dimensions, at: new Date().toISOString()}].slice(-5);
    await level.update({"background.src": result.path, [`flags.${MODULE_ID}.history`]: history});
    if (level.background.src !== result.path) throw new Error("A scene update hook rejected the background change.");
  } catch (error) {
    throw new Error(`${error.message} The processed file is saved at ${result.path}.`);
  }
  return result.path;
}

export async function restoreMap(scene, level) {
  const history = level.getFlag(MODULE_ID, "history") ?? [], last = history.at(-1);
  if (!last || level.background.src !== last.after) throw new Error("There is no matching previous background to restore.");
  const expected = targetSnapshot(scene, level); checkTarget(scene, level, expected);
  await level.update({"background.src": last.before, [`flags.${MODULE_ID}.history`]: history.slice(0, -1)});
  if (level.background.src !== last.before) throw new Error("A scene update hook rejected restoring the previous background.");
}
