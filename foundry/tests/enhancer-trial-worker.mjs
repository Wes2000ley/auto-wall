// Test-only runner. Uses production tile helpers/runtime, never Foundry documents.
import * as ort from "../map-enhancer/vendor/ort/ort.all.min.mjs";
import {reflect, tiles} from "../map-enhancer/scripts/tiles.mjs";
ort.env.wasm.wasmPaths = new URL("../map-enhancer/vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.logLevel = "error";
let session, modelSHA256;

async function loadModel(model, backend) {
  const response = await fetch(model.url);
  if (!response.ok) throw new Error(`Could not load ${model.url}`);
  const bytes = await response.arrayBuffer();
  modelSHA256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), value => value.toString(16).padStart(2, "0")).join("");
  if (modelSHA256 !== model.sha256) throw new Error(`Model hash mismatch: ${model.id}`);
  return ort.InferenceSession.create(bytes, {executionProviders: [backend], graphOptimizationLevel: "all"});
}

async function read(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  canvas.getContext("2d").drawImage(bitmap, 0, 0); bitmap.close(); return canvas;
}

async function inference(source, region, model, edge) {
  const {scale, minimumSize = 1, inputMultiple = 1, inputRemainder = 0, outputOffset = 0} = model;
  const paddedSize = size => Math.ceil((Math.max(minimumSize, size) - inputRemainder) / inputMultiple) * inputMultiple + inputRemainder;
  const output = new OffscreenCanvas(region.width * scale, region.height * scale), context = output.getContext("2d");
  const image = source.getContext("2d", {willReadFrequently: true}).getImageData(0, 0, source.width, source.height);
  for (const tile of tiles(region.width, region.height, edge)) {
    const width = paddedSize(tile.width + 96), height = paddedSize(tile.height + 96);
    const plane = width * height, values = new Float32Array(3 * plane);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const sx = reflect(region.x + tile.x + x - 48, source.width), sy = reflect(region.y + tile.y + y - 48, source.height);
      for (let c = 0; c < 3; c++) values[c * plane + y * width + x] = image.data[(sy * source.width + sx) * 4 + c] / 255;
    }
    const input = new ort.Tensor("float32", values, [1, 3, height, width]); let result;
    try {
      result = await session.run({[session.inputNames[0]]: input});
      const tensor = result[session.outputNames[0]], data = await tensor.getData();
      if (tensor.dims.join() !== [1, 3, height * scale - 2 * outputOffset, width * scale - 2 * outputOffset].join()) throw new Error(`Wrong output dimensions: ${tensor.dims}`);
      const w = tensor.dims[3], h = tensor.dims[2], pixels = w * h, rgba = new Uint8ClampedArray(4 * pixels);
      for (let i = 0; i < pixels; i++) {
        for (let c = 0; c < 3; c++) {
          const value = data[c * pixels + i]; if (!Number.isFinite(value)) throw new Error("Non-finite model output");
          rgba[i * 4 + c] = value * 255;
        }
        rgba[i * 4 + 3] = 255;
      }
      const patch = new OffscreenCanvas(w, h); patch.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
      context.drawImage(patch, 48 * scale - outputOffset, 48 * scale - outputOffset, tile.width * scale, tile.height * scale,
        tile.x * scale, tile.y * scale, tile.width * scale, tile.height * scale);
    } finally { input.dispose(); for (const tensor of Object.values(result ?? {})) tensor.dispose(); }
  }
  return output;
}

function difference(a, b) {
  const left = a.getContext("2d").getImageData(0, 0, a.width, a.height).data;
  const right = b.getContext("2d").getImageData(0, 0, b.width, b.height).data;
  if (left.length !== right.length) throw new Error("Reference dimensions differ");
  let max = 0, sum = 0;
  for (let i = 0; i < left.length; i++) if (i % 4 !== 3) { const error = Math.abs(left[i] - right[i]); max = Math.max(max, error); sum += error; }
  return {max, mean: sum / (left.length * .75)};
}

async function resized(image, width, height, region) {
  const canvas = new OffscreenCanvas(width, height), ctx = canvas.getContext("2d"); ctx.imageSmoothingQuality = "high";
  if (region) ctx.drawImage(image, region.x, region.y, region.width, region.height, 0, 0, width, height);
  else ctx.drawImage(image, 0, 0, width, height);
  return canvas.convertToBlob({type: "image/png"});
}

self.onmessage = async ({data: {model, input, backend, validateReference}}) => {
  try {
    self.postMessage({type: "progress", message: `${model.label}: ${input.name} · ${backend}`});
    if (!session) session = await loadModel(model, backend);
    let referenceError = null;
    if (validateReference && model.id !== "digital-art-4x") {
      const original = await read("/comparison/models/reference-input.png"), expected = await read(`/comparison/models/${model.id}-reference.png`);
      const actual = await inference(original, {x: 0, y: 0, width: original.width, height: original.height}, model, 128);
      referenceError = difference(actual, expected);
      if (referenceError.mean > .5 || referenceError.max > 3) throw new Error(`Browser/native reference mismatch: ${JSON.stringify(referenceError)}`);
    }
    const image = await read(`/comparison/${input.file}`), edge = backend === "webgpu" ? 128 : 64;
    let start = performance.now(); await inference(image, input.region, model, edge); const firstMs = performance.now() - start;
    start = performance.now(); const output = await inference(image, input.region, model, edge); const warmMs = performance.now() - start;
    const whole = await inference(image, input.region, model, Math.max(input.region.width, input.region.height));
    const seamError = difference(output, whole), images = {};
    for (const scale of [2, 4]) {
      const width = input.region.width * scale, height = input.region.height * scale;
      images[`image${scale}`] = await resized(output, width, height);
      images[`whole${scale}`] = await resized(whole, width, height);
      if (model.id === "digital-art-4x") images[`original${scale}`] = await resized(image, width, height, input.region);
    }
    self.postMessage({type: "result", provider: backend, modelSHA256, firstMs, warmMs, referenceError, seamError, ...images});
  } catch (error) { self.postMessage({type: "error", message: error.message || String(error)}); }
};
