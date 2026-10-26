import * as ort from "../vendor/ort/ort.all.min.mjs";
import {settings, outputSize} from "./settings.mjs";
import {FilterEngine, filterCanvas} from "./filters.mjs";
import {reflect, tiles, scaledTile} from "./tiles.mjs";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // Works on ordinary Foundry hosts without isolation headers.
ort.env.wasm.proxy = false;
ort.env.logLevel = "error";
let source, session, provider, sessionBackend, rawPreview, previewRegion, textureLimit;
const filters = new FilterEngine();
let queue = Promise.resolve();

function progress(id, message, completed = 0, total = 0) {
  self.postMessage({id, type: "progress", message, completed, total, provider});
}

async function model(backend, id) {
  if (session && sessionBackend === backend) return;
  if (session) { await session.release(); session = null; }
  progress(id, "Loading the bundled Digital Art model…");
  const path = new URL("../models/digital-art-4x.onnx", import.meta.url).href;
  provider = backend !== "cpu" && self.navigator.gpu ? "webgpu" : "wasm";
  try { session = await ort.InferenceSession.create(path, {executionProviders: [provider], graphOptimizationLevel: "all"}); }
  catch (error) {
    if (provider !== "webgpu") throw error;
    progress(id, `GPU initialization failed: ${error.message}. Loading the bundled CPU fallback…`);
    provider = "wasm";
    session = await ort.InferenceSession.create(path, {executionProviders: [provider], graphOptimizationLevel: "all"});
  }
  sessionBackend = backend;
}

function tensorTile(x, y, width, height, padding) {
  const left = Math.max(0, x - padding), top = Math.max(0, y - padding);
  const right = Math.min(source.width, x + width + padding), bottom = Math.min(source.height, y + height + padding);
  const image = source.getContext("2d", {willReadFrequently: true}).getImageData(left, top, right - left, bottom - top);
  const w = width + padding * 2, h = height + padding * 2, plane = w * h, values = new Float32Array(3 * plane);
  for (let ty = 0; ty < h; ty++) for (let tx = 0; tx < w; tx++) {
    const sx = reflect(x + tx - padding, source.width) - left, sy = reflect(y + ty - padding, source.height) - top;
    const i = (sy * image.width + sx) * 4, j = ty * w + tx;
    for (let c = 0; c < 3; c++) values[j + c * plane] = image.data[i + c] / 255;
  }
  return new ort.Tensor("float32", values, [1, 3, h, w]);
}

async function upscale(region, dimensions, options, id) {
  await model(options.backend, id);
  const edge = provider === "webgpu" ? 128 : 64, padding = 48;
  const output = new OffscreenCanvas(dimensions.width, dimensions.height), ctx = output.getContext("2d");
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  const total = Math.ceil(region.width / edge) * Math.ceil(region.height / edge); let done = 0;
  for (const tile of tiles(region.width, region.height, edge)) {
    progress(id, `Upscaling ${done + 1} of ${total} tiles (${provider === "webgpu" ? "GPU" : "CPU"})…`, done, total);
    const input = tensorTile(region.x + tile.x, region.y + tile.y, tile.width, tile.height, padding);
    let result;
    try {
      try { result = await session.run({[session.inputNames[0]]: input}); }
      catch (error) {
        if (provider !== "webgpu") throw error;
        await session.release(); session = null;
        progress(id, `GPU inference failed: ${error.message}. Retrying this tile on the CPU…`, done, total);
        await model("cpu", id);
        sessionBackend = options.backend;
        result = await session.run({[session.inputNames[0]]: input});
      }
      const tensor = result[session.outputNames[0]], values = await tensor.getData();
      const w = tensor.dims[3], h = tensor.dims[2], plane = w * h, rgba = new Uint8ClampedArray(plane * 4);
      for (let i = 0; i < plane; i++) {
        for (let c = 0; c < 3; c++) rgba[i * 4 + c] = values[c * plane + i] * 255;
        rgba[i * 4 + 3] = 255;
      }
      const patch = new OffscreenCanvas(w, h); patch.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
      const target = scaledTile(tile, region.width, region.height, dimensions.width, dimensions.height);
      if (target.width && target.height) ctx.drawImage(patch, padding * 4, padding * 4, tile.width * 4, tile.height * 4,
        target.x, target.y, target.width, target.height);
    } finally {
      input.dispose(); for (const tensor of Object.values(result ?? {})) tensor.dispose();
    }
    done++;
  }
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(source, region.x, region.y, region.width, region.height, 0, 0, output.width, output.height);
  ctx.globalCompositeOperation = "source-over";
  progress(id, "Upscale complete.", total, total);
  return output;
}

async function preview(id, options) {
  if (!rawPreview) throw new Error("Generate an upscale preview first.");
  const fullSize = outputSize(source.width, source.height, options, textureLimit);
  const resized = new OffscreenCanvas(Math.max(1, Math.round(previewRegion.width * fullSize.width / source.width)),
    Math.max(1, Math.round(previewRegion.height * fullSize.height / source.height)));
  const context = resized.getContext("2d"); context.imageSmoothingQuality = "high";
  context.drawImage(rawPreview, 0, 0, resized.width, resized.height);
  const processed = filterCanvas(resized, options, filters);
  const after = await createImageBitmap(await processed.convertToBlob({type: "image/webp", quality: options.quality}));
  const before = await createImageBitmap(source, previewRegion.x, previewRegion.y, previewRegion.width, previewRegion.height);
  self.postMessage({id, type: "result", before, after, provider, region: previewRegion}, [before, after]);
}

async function run(message) {
  const {id, type} = message, options = settings(message.options);
  if (type === "load") {
    const response = await fetch(message.source, {credentials: "same-origin"});
    if (!response.ok) throw new Error(`Could not read the map (${response.status}).`);
    const bitmap = await createImageBitmap(await response.blob());
    source = new OffscreenCanvas(bitmap.width, bitmap.height);
    source.getContext("2d", {willReadFrequently: true}).drawImage(bitmap, 0, 0); bitmap.close(); rawPreview = null;
    const factor = Math.min(1, 600 / source.width, 180 / source.height);
    const overview = new OffscreenCanvas(Math.max(1, Math.round(source.width * factor)), Math.max(1, Math.round(source.height * factor)));
    overview.getContext("2d").drawImage(source, 0, 0, overview.width, overview.height);
    const image = await createImageBitmap(overview);
    self.postMessage({id, type: "result", width: source.width, height: source.height, overview: image}, [image]); return;
  }
  if (!source) throw new Error("Load a scene map first.");
  if (type === "preview") {
    rawPreview = null;
    previewRegion = message.region;
    textureLimit = message.textureLimit;
    const dimensions = {width: previewRegion.width * 4, height: previewRegion.height * 4};
    rawPreview = await upscale(previewRegion, dimensions, options, id);
    await preview(id, options); return;
  }
  if (type === "filter") { await preview(id, options); return; }
  if (type === "export") {
    const dimensions = outputSize(source.width, source.height, options, message.textureLimit);
    const raw = await upscale({x: 0, y: 0, width: source.width, height: source.height}, dimensions, options, id);
    progress(id, "Applying smoothing and sharpening…");
    const processed = filterCanvas(raw, options, filters, (done, total) => progress(id, "Applying image filters…", done, total));
    progress(id, "Encoding WebP…");
    const blob = await processed.convertToBlob({type: "image/webp", quality: options.quality});
    self.postMessage({id, type: "result", blob, dimensions, provider}); return;
  }
  throw new Error("Unknown image operation.");
}

self.onmessage = ({data}) => {
  queue = queue.then(() => run(data)).catch(error => self.postMessage({id: data.id, type: "error", message: error.message || String(error)}));
};
