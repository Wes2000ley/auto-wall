import {createRequire} from "node:module";
const require = createRequire(import.meta.url);
globalThis.polygonClipping = require("../auto-wall-live/vendor/polygon-clipping.js");

export {Scene, wall} from './mock-scene.mjs';

// Collision invariants are shared by Node tests and the isolated browser harness.
export {crosses} from "./browser-ray.mjs";
