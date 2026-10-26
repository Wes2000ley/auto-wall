import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const readJSON = async relative => JSON.parse(await readFile(new URL(relative, import.meta.url), "utf8"));
const catalog = await readJSON("../tools/anime-trials.json");
const report = await readJSON("./model-trial-results.json");
const pins = await readJSON("../map-enhancer/asset-pins.json");
const sha256 = /^[0-9a-f]{64}$/;

test("research results account for every catalog candidate and its pinned source", () => {
  const sources = new Map(catalog.map(model => [model.id, model.source.sha256]));
  assert.equal(sources.size, catalog.length, "duplicate catalog IDs");
  const results = [...report.models, ...report.nativeFailures];
  assert.equal(new Set(results.map(model => model.id)).size, results.length, "duplicate results");
  assert.deepEqual(results.map(model => model.id).sort(), [...sources.keys()].sort());
  for (const model of report.models) {
    assert.equal(model.sourceSHA256, sources.get(model.id));
    assert.match(model.preparedSHA256, sha256);
    assert.match(model.outputDigestSHA256, sha256);
    assert.ok(model.preparedBytes > 0);
    assert.ok(model.nativeMaxAbsoluteError >= 0 && model.nativeMaxAbsoluteError <= report.method.nativeMaxAbsoluteErrorTolerance);
    assert.match(model.nativeReference, /^(independent PyTorch|author ONNX graph; no independent PyTorch comparison)$/);
  }
  for (const model of report.nativeFailures) {
    assert.equal(model.sourceSHA256, sources.get(model.id));
    assert.ok(model.error);
  }
});

test("browser comparison totals and numerical gates agree with recorded batches", () => {
  const batches = new Map(report.batches.map(batch => [batch.id, batch]));
  assert.equal(batches.size, report.batches.length);
  assert.equal(report.counts.catalogModels, catalog.length);
  assert.equal(report.counts.nativePassed, report.models.length);
  assert.equal(report.counts.nativeFailed, report.nativeFailures.length);
  assert.equal(report.counts.candidateMapComparisons, report.models.reduce((sum, model) => sum + model.successfulMapComparisons, 0));
  assert.equal(report.counts.candidatePNGsVerified, report.models.reduce((sum, model) => sum + model.savedPNGs, 0));
  assert.equal(report.counts.baselineMapComparisons, report.batches.length * report.inputs.length);
  for (const batch of report.batches) {
    const models = report.models.filter(model => model.batch === batch.id);
    assert.equal(batch.successfulMapComparisons, (models.length + 1) * report.inputs.length);
    assert.equal(typeof batch.browserVerifiedModelHashes, "boolean");
    for (const model of models) {
      assert.equal(model.successfulMapComparisons, report.inputs.length);
      assert.equal(model.savedPNGs, report.inputs.length * report.method.displayScales.length * (batch.wholeCropPNGsSaved ? 2 : 1));
      assert.equal(model.browserVerifiedModelHashes, batch.browserVerifiedModelHashes);
      assert.equal(model.requestedProvider, "webgpu");
      assert.ok(Number.isFinite(model.medianWarmCropMs) && model.medianWarmCropMs > 0);
      assert.ok(model.browserNativeReference.max >= 0 && model.browserNativeReference.max <= report.method.browserReferenceMaxRGBError);
      assert.ok(model.browserNativeReference.mean >= 0 && model.browserNativeReference.mean <= report.method.browserReferenceMeanRGBError);
    }
  }
  assert.ok(report.models.every(model => batches.has(model.batch)));
});

test("research baseline matches the shipped asset and inputs omit private file paths", () => {
  assert.equal(report.baseline.sha256, pins.files["models/digital-art-4x.onnx"].sha256);
  assert.equal(report.environment.browserRuntime, pins.runtime);
  assert.equal(report.baseline.repeatedBatchMeasurements.length, report.batches.length);
  for (const input of report.inputs) {
    assert.match(input.sourceSHA256, sha256);
    assert.equal(input.source, undefined);
    assert.equal(input.file, undefined);
    assert.deepEqual(input.sourceCrop.slice(2), report.method.region);
    assert.deepEqual([input.region.width, input.region.height], report.method.region);
    assert.equal(input.region.x, report.method.sourceContextPixelsPerSide);
    assert.equal(input.region.y, report.method.sourceContextPixelsPerSide);
  }
});
