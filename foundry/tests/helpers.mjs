import {createRequire} from "node:module";
const require = createRequire(import.meta.url);
globalThis.polygonClipping = require("../auto-wall-live/vendor/polygon-clipping.js");

export async function opencv() {
  return new Promise(resolve => require("../auto-wall-live/vendor/opencv.js").then(cv => resolve({cv})));
}

export {Scene, wall} from './mock-scene.mjs';

export function image(width, height) {
  const pixels = new Uint8Array(width * height * 4);
  for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255;
  return {width, height, pixels, fill(x0, y0, x1, y1, color) {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 4; pixels.set(color, i);
    }
    return this;
  }};
}

// Collision invariants are shared by Node tests and the isolated browser harness.
export {crosses} from "./browser-ray.mjs";
