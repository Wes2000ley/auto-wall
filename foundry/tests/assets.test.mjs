import test from "node:test";
import assert from "node:assert/strict";
import {readFile, access} from "node:fs/promises";
import {createHash} from "node:crypto";
const root = new URL("../auto-wall-live/", import.meta.url);

test("every manifest entry is included and OpenCV stays out of the game bootstrap", async () => {
  const manifest = JSON.parse(await readFile(new URL("module.json", root), "utf8"));
  assert.equal(manifest.id, "auto-wall-live");
  const entries = [...manifest.scripts, ...manifest.esmodules, ...manifest.styles, manifest.license];
  for (const file of entries) await access(new URL(file, root));
  assert.equal(entries.some(file => file.includes("opencv")), false);
});

test("bundled runtimes match the documented SHA-256 pins", async () => {
  const documented = await readFile(new URL("THIRD-PARTY.md", root), "utf8");
  for (const file of ["vendor/opencv.js", "vendor/polygon-clipping.js"]) {
    const hash = createHash("sha256").update(await readFile(new URL(file, root))).digest("hex");
    assert.ok(documented.includes(`${hash}  ${file}`), `${file} does not match its documented pin`);
  }
});
