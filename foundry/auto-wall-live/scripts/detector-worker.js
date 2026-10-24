/* Classic worker: the bundled libraries load locally, with no CDN or service. */
importScripts("../vendor/polygon-clipping.js", "../vendor/opencv.js");
// OpenCV exposes a legacy thenable which resolves to itself. Wrap it rather
// than passing it to Promise resolution (that would recurse indefinitely).
const ready = new Promise(resolve => cv.then(library => resolve({cv: library}))).then(async loaded => ({
  ...loaded, detect: (await import("./detection.mjs")).detectRGBA
}));

self.onmessage = async ({data}) => {
  const bitmap = data.bitmap;
  try {
    const {cv, detect} = await ready;
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", {willReadFrequently: true});
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height);
    const result = detect(cv, pixels.data, bitmap.width, bitmap.height, data.settings,
      stage => self.postMessage({stage}));
    self.postMessage({result});
  } catch (error) {
    self.postMessage({error: error instanceof Error ? error.message : String(error)});
  } finally {
    bitmap.close();
  }
};
