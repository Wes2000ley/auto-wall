export class Processor {
  constructor(progress) {
    this.worker = new Worker(new URL("./processor-worker.mjs", import.meta.url), {type: "module"});
    this.pending = new Map(); this.serial = 0;
    this.worker.onmessage = ({data}) => {
      if (data.type === "progress") { progress(data); return; }
      const request = this.pending.get(data.id);
      if (!request) { data.before?.close(); data.after?.close(); data.overview?.close(); return; }
      this.pending.delete(data.id);
      if (data.type === "error") request.reject(new Error(data.message)); else request.resolve(data);
    };
    this.worker.onerror = event => {
      event.preventDefault();
      this.stop(new Error(event.message || "The local image worker could not start."));
    };
  }

  request(type, values = {}) {
    if (!this.worker) return Promise.reject(new Error("Processing was cancelled. Reload the map to continue."));
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      try { this.worker.postMessage({id, type, ...values}); }
      catch (error) { this.pending.delete(id); reject(error); }
    });
  }

  stop(error = new Error("Processing cancelled. The scene was not changed.")) {
    this.worker?.terminate(); this.worker = null;
    for (const {reject} of this.pending.values()) reject(error);
    this.pending.clear();
  }
}
