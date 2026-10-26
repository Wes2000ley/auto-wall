export class Comparison {
  constructor(canvas) {
    this.canvas = canvas; this.zoom = 1; this.pan = [0, 0]; this.split = 0.5;
    this.abort = new AbortController(); const signal = this.abort.signal;
    canvas.addEventListener("wheel", event => {
      event.preventDefault(); this.zoom = Math.max(1, Math.min(8, this.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15))); this.draw();
    }, {signal, passive: false});
    canvas.addEventListener("pointerdown", event => {
      this.drag = {x: event.clientX, y: event.clientY, pan: [...this.pan]}; canvas.setPointerCapture(event.pointerId);
    }, {signal});
    canvas.addEventListener("pointermove", event => {
      if (this.drag) { this.pan = [this.drag.pan[0] + event.clientX - this.drag.x, this.drag.pan[1] + event.clientY - this.drag.y]; this.draw(); }
    }, {signal});
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(type, () => { this.drag = null; }, {signal});
    this.resize = new ResizeObserver(() => this.draw()); this.resize.observe(canvas);
  }

  set(before, after) {
    this.before?.close(); this.after?.close(); this.before = before; this.after = after; this.draw();
  }

  reset() { this.zoom = 1; this.pan = [0, 0]; this.draw(); }

  clear() { this.before?.close(); this.after?.close(); this.before = null; this.after = null; this.reset(); }

  draw() {
    const bounds = this.canvas.getBoundingClientRect(), ratio = devicePixelRatio || 1;
    if (!bounds.width || !bounds.height) return;
    this.canvas.width = Math.round(bounds.width * ratio); this.canvas.height = Math.round(bounds.height * ratio);
    const ctx = this.canvas.getContext("2d"); ctx.scale(ratio, ratio);
    ctx.fillStyle = "#15191f"; ctx.fillRect(0, 0, bounds.width, bounds.height);
    if (!this.after) {
      ctx.fillStyle = "#c8ced8"; ctx.font = "15px sans-serif"; ctx.textAlign = "center";
      ctx.fillText("Choose a map detail, then generate a preview", bounds.width / 2, bounds.height / 2); return;
    }
    const factor = Math.min(bounds.width / this.after.width, bounds.height / this.after.height) * this.zoom;
    const w = this.after.width * factor, h = this.after.height * factor;
    const left = (bounds.width - w) / 2 + this.pan[0], top = (bounds.height - h) / 2 + this.pan[1];
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.before, left, top, w, h);
    ctx.save(); ctx.beginPath(); ctx.rect(bounds.width * this.split, 0, bounds.width, bounds.height); ctx.clip();
    ctx.drawImage(this.after, left, top, w, h); ctx.restore();
    ctx.strokeStyle = "#f4c86a"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(bounds.width * this.split, 0); ctx.lineTo(bounds.width * this.split, bounds.height); ctx.stroke();
    ctx.font = "bold 12px sans-serif";
    for (const [text, x] of [["Original", 10], ["Processed", Math.max(bounds.width * this.split + 10, bounds.width - 85)]]) {
      ctx.fillStyle = "#161a20dd"; ctx.fillRect(x - 4, 6, 80, 25); ctx.fillStyle = "#fff"; ctx.fillText(text, x, 23);
    }
  }

  dispose() { this.abort.abort(); this.resize.disconnect(); this.clear(); }
}
