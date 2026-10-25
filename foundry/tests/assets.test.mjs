import test from "node:test";
import assert from "node:assert/strict";
import {readFile, access, readdir} from "node:fs/promises";
import {createHash} from "node:crypto";
const root = new URL("../auto-wall-live/", import.meta.url);

test("every manual module manifest entry is included", async () => {
  const manifest = JSON.parse(await readFile(new URL("module.json", root), "utf8"));
  assert.equal(manifest.id, "auto-wall-live");
  const entries = [...manifest.scripts, ...manifest.esmodules, ...manifest.styles, manifest.license];
  for (const file of entries) await access(new URL(file, root));
  assert.equal(entries.some(file => file.includes("opencv")), false);
});

test("the polygon runtime matches its documented SHA-256 pin", async () => {
  const documented = await readFile(new URL("THIRD-PARTY.md", root), "utf8");
  for (const file of ["vendor/polygon-clipping.js"]) {
    const hash = createHash("sha256").update(await readFile(new URL(file, root))).digest("hex");
    assert.ok(documented.includes(`${hash}  ${file}`), `${file} does not match its documented pin`);
  }
});

test("the installed source contains no detection worker, image kernels, or detection controls", async () => {
  for (const file of ["scripts/detection.mjs", "scripts/detector-worker.js", "vendor/opencv.js"]) {
    await assert.rejects(access(new URL(file, root)), {code: "ENOENT"});
  }
  for (const file of await readdir(new URL("scripts/", root))) {
    const source = await readFile(new URL(`scripts/${file}`, root), "utf8");
    assert.doesNotMatch(source, /new Worker|getImageData|backgroundSource|pickColor|detectRGBA|getSnappedPoint|canvas\.grid/);
  }
  const template = await readFile(new URL("templates/editor.hbs", root), "utf8");
  assert.doesNotMatch(template, /data-action="(?:detect|sample|importProfile)"|data-setting=|data-snap|Auto detection/);
});
