import {opencv, image, Scene, wall} from "./helpers.mjs";
import {detectRGBA} from "../auto-wall-live/scripts/detection.mjs";
import {preset} from "../auto-wall-live/scripts/settings.mjs";
import {planWalls, geometryFromDocuments, levelSnapshot} from "../auto-wall-live/scripts/documents.mjs";
import {PaintGeometry} from "../auto-wall-live/scripts/geometry.mjs";
const {cv} = await opencv();
const results = [];
for (const size of [1024, 2048, 4096]) {
  const img = image(size, size), gray = [160, 160, 160, 255];
  img.fill(50, 50, size - 50, 80, gray).fill(50, size - 80, size - 50, size - 50, gray)
    .fill(50, 50, 80, size - 50, gray).fill(size - 80, 50, size - 50, size - 50, gray);
  const times = [];
  for (let pass = 0; pass < 3; pass++) times.push(detectRGBA(cv, img.pixels, size, size, {...preset("exact"), lights: true, max_dimension: size}).elapsed);
  results.push({image: `${size}x${size}`, milliseconds: times.map(t => Math.round(t)), note: "synthetic palette room, includes vectorization; excludes worker startup and Foundry persistence"});
}
const scene = new Scene(Array.from({length: 10000}, (_, i) => wall(`wall${i}`, [i % 100 * 20, Math.floor(i / 100) * 20, i % 100 * 20 + 10, Math.floor(i / 100) * 20])));
const existing = levelSnapshot(scene, "ground").walls, desired = geometryFromDocuments(existing);
desired.walls.push([[2500, 100], [2700, 100]]);
const planStart = performance.now(), plan = planWalls(scene, "ground", existing, desired);
results.push({operation: "one new segment on a level with 10000 existing walls", milliseconds: Math.round(performance.now() - planStart), creates: plan.create.length, deletes: plan.remove.length});
let geometry = new PaintGeometry([4096, 4096]); const paintStart = performance.now();
for (let i = 0; i < 40; i++) geometry = geometry.paint("walls", [[100 + i * 40, 500], [180 + i * 40, 500]], 40);
results.push({operation: "40 overlapping wall strokes", milliseconds: Math.round(performance.now() - paintStart), finalSegments: geometry.compose().walls.length});
console.log(JSON.stringify(results, null, 2));
