// Development only: candidate weights/results stay in the private comparison directory.
const status = document.getElementById("status"), summary = document.getElementById("summary");
const start = document.getElementById("compare"), cancel = document.getElementById("cancel");
const review = document.getElementById("review");
const assembly = document.getElementById("assembly");
let worker, cancelled = false, rejectPending;

async function save(name, body) {
  const response = await fetch(`/trial-results/${name}`, {method: "POST", body});
  if (!response.ok) throw new Error(`Could not save the local result: ${name}`);
}

function request(data) {
  return new Promise((resolve, reject) => {
    rejectPending = reject;
    worker.onmessage = ({data}) => {
      if (data.type === "progress") { status.textContent = data.message; return; }
      rejectPending = null;
      if (data.type === "error") reject(new Error(data.message)); else resolve(data);
    };
    worker.onerror = event => { event.preventDefault(); rejectPending = null; reject(new Error(event.message)); };
    worker.postMessage(data);
  });
}

async function cell(row, blob, label, key, scale) {
  await save(`${key}-${scale}x.png`, blob);
  if (scale !== Number(document.getElementById("scale").value)) return;
  figure(row, label, key);
}

function figure(row, label, key) {
  const figure = document.createElement("figure"), img = document.createElement("img"), caption = document.createElement("figcaption");
  img.dataset.result = key; img.src = resultURL(key); img.alt = label;
  caption.textContent = label; figure.append(img, caption); row.append(figure);
}

function resultURL(key) {
  const suffix = key.endsWith("-original") ? "" : assembly.value;
  return `/comparison/results/${key}-${document.getElementById("scale").value}x${suffix}.png`;
}

function makeRows(inputs) {
  document.getElementById("results").replaceChildren();
  return inputs.map(input => {
    const heading = document.createElement("h2"), row = document.createElement("div"); row.className = "row";
    heading.textContent = input.name; document.getElementById("results").append(heading, row); return row;
  });
}

review.onclick = async () => {
  try {
    const responses = await Promise.all([fetch("/comparison/inputs.json"), fetch("/comparison/models/models.json"), fetch("/comparison/results/browser-results.json")]);
    if (responses.some(response => !response.ok)) throw new Error("No saved comparison found in this private directory.");
    const [inputs, models, report] = await Promise.all(responses.map(response => response.json()));
    assembly.disabled = !report.hasWholeImages;
    if (assembly.disabled) assembly.value = "";
    const rows = makeRows(inputs), labels = new Map(models.map(model => [model.id, model.label]));
    labels.set("digital-art-4x", "Digital Art (baseline)");
    for (const record of report.results) {
      if (record.error) continue;
      const index = inputs.findIndex(input => input.name === record.map);
      if (index < 0) continue;
      const input = inputs[index];
      if (record.model === "digital-art-4x") figure(rows[index], "Original · browser high-quality resize", `${input.id}-original`);
      figure(rows[index], `${labels.get(record.model) ?? record.model} · ${(record.warmMs / 1000).toFixed(2)}s warm · ${record.provider}`, `${input.id}-${record.model}`);
    }
    summary.textContent = JSON.stringify(report, null, 2);
    status.textContent = "Saved results loaded. Display can switch between 2× and 4× without rerunning inference.";
  } catch (error) { status.textContent = error.message; }
};

cancel.onclick = () => { cancelled = true; worker?.terminate(); rejectPending?.(new Error("Comparison cancelled; partial results are retained.")); };
document.getElementById("scale").onchange = assembly.onchange = () => {
  for (const image of document.querySelectorAll("img[data-result]")) image.src = resultURL(image.dataset.result);
};
start.onclick = async () => {
  start.disabled = true; review.disabled = true; cancel.disabled = false; cancelled = false;
  for (const select of document.querySelectorAll("select")) select.disabled = true;
  document.getElementById("results").replaceChildren(); summary.textContent = "";
  const results = [], backend = document.getElementById("backend").value;
  const report = {backend, createdAt: new Date().toISOString(), hasWholeImages: true,
    output: "Lossless PNG; no sharpening/smoothing; native AI pass then resize to 2x and 4x; tiled and whole crop", results};
  try {
    const responses = await Promise.all([fetch("/comparison/inputs.json"), fetch("/comparison/models/models.json"), fetch("/modules/map-enhancer/asset-pins.json")]);
    if (responses.some(response => !response.ok)) throw new Error("Prepare private inputs/models and pass the comparison directory to serve-harness.mjs.");
    const [inputs, candidates, pins] = await Promise.all(responses.map(response => response.json()));
    report.runtime = pins.runtime;
    report.modelHashesVerified = true;
    results.push(...candidates.filter(model => model.error).map(model => ({model: model.id, phase: "preparation", error: model.error})));
    summary.textContent = JSON.stringify(report, null, 2);
    await save("browser-results.json", summary.textContent);
    const models = [{id: "digital-art-4x", label: "Digital Art (baseline)", scale: 4,
      sha256: pins.files["models/digital-art-4x.onnx"].sha256, url: "/modules/map-enhancer/models/digital-art-4x.onnx"},
      ...candidates.filter(model => !model.error).map(model => ({...model, url: `/comparison/models/${model.file}`}))];
    const rows = makeRows(inputs);
    for (const model of models) {
      worker = new Worker(new URL("./enhancer-trial-worker.mjs", import.meta.url), {type: "module"});
      try {
        for (const [index, input] of inputs.entries()) {
          if (cancelled) throw new Error("Comparison cancelled.");
          const output = await request({model, input, backend, validateReference: index === 0});
          for (const scale of [2, 4]) {
            if (model.id === "digital-art-4x") await cell(rows[index], output[`original${scale}`], "Original · browser high-quality resize", `${input.id}-original`, scale);
            await save(`${input.id}-${model.id}-${scale}x-whole.png`, output[`whole${scale}`]);
            await cell(rows[index], output[`image${scale}`], `${model.label} · ${(output.warmMs / 1000).toFixed(2)}s warm · ${output.provider}`, `${input.id}-${model.id}`, scale);
          }
          const {image2, image4, whole2, whole4, original2, original4, ...metrics} = output;
          results.push({map: input.name, sourceSHA256: input.sourceSHA256, sourceCrop: input.sourceCrop,
            model: model.id, modelSHA256: model.sha256, ...metrics});
          summary.textContent = JSON.stringify(report, null, 2);
          await save("browser-results.json", summary.textContent);
        }
      } catch (error) {
        results.push({model: model.id, phase: "browser", error: error.message});
        summary.textContent = JSON.stringify(report, null, 2); await save("browser-results.json", summary.textContent);
        if (cancelled) throw error;
      } finally { worker.terminate(); worker = null; }
    }
    status.textContent = `COMPARISON COMPLETE. ${results.filter(row => row.error).length} model failures. Results saved locally. Timings are crop inference, not whole-map estimates.`;
  } catch (error) { status.textContent = `STOPPED: ${error.message}`; }
  finally {
    worker?.terminate(); worker = null; rejectPending = null; start.disabled = false; review.disabled = false; cancel.disabled = true;
    for (const select of document.querySelectorAll("select")) select.disabled = false;
  }
};
