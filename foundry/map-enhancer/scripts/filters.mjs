import {settings} from "./settings.mjs";

// Pure CPU path also makes the filters available without WebGL in a worker.
export function filterCPU(pixels, width, height, values) {
  const options = settings(values);
  let source = new Uint8ClampedArray(pixels);
  if (source.length !== width * height * 4) throw new Error("Invalid filter image dimensions.");
  if (options.smooth > 0) {
    const result = source.slice(), radius = options.smoothRadius;
    const spatial = [], range = 2 * (options.edgeTolerance * 255) ** 2 * 3;
    for (let y = -radius; y <= radius; y++) for (let x = -radius; x <= radius; x++) {
      spatial.push([x, y, Math.exp(-(x * x + y * y) / (2 * radius * radius))]);
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4, sum = [0, 0, 0]; let total = 0;
      for (const [dx, dy, weight] of spatial) {
        const j = (Math.max(0, Math.min(height - 1, y + dy)) * width + Math.max(0, Math.min(width - 1, x + dx))) * 4;
        const d = (source[i] - source[j]) ** 2 + (source[i + 1] - source[j + 1]) ** 2 + (source[i + 2] - source[j + 2]) ** 2;
        const w = weight * Math.exp(-d / range) * source[j + 3] / 255;
        total += w; for (let c = 0; c < 3; c++) sum[c] += source[j + c] * w;
      }
      if (total > 0) for (let c = 0; c < 3; c++) result[i + c] = source[i + c] + options.smooth * (sum[c] / total - source[i + c]);
    }
    source = result;
  }
  if (options.sharpen > 0) {
    const radius = options.sharpRadius, kernel = [];
    for (let i = -radius; i <= radius; i++) kernel.push(Math.exp(-i * i / (2 * (radius / 2 + 0.5) ** 2)));
    let blur = new Float32Array(source.length);
    for (let i = 0; i < source.length; i += 4) {
      for (let c = 0; c < 3; c++) blur[i + c] = source[i + c] * source[i + 3] / 255;
      blur[i + 3] = source[i + 3] / 255;
    }
    for (const horizontal of [true, false]) {
      const next = new Float32Array(blur.length);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        for (let k = -radius; k <= radius; k++) {
          const j = (Math.max(0, Math.min(height - 1, y + (horizontal ? 0 : k))) * width +
            Math.max(0, Math.min(width - 1, x + (horizontal ? k : 0)))) * 4;
          for (let c = 0; c < 4; c++) next[i + c] += blur[j + c] * kernel[k + radius];
        }
      }
      blur = next;
    }
    const result = source.slice();
    for (let i = 0; i < source.length; i += 4) if (blur[i + 3] > 0) {
      for (let c = 0; c < 3; c++) result[i + c] = source[i + c] + options.sharpen * (source[i + c] - blur[i + c] / blur[i + 3]);
    }
    source = result;
  }
  return source;
}

const VERTEX = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform int radius;
uniform int mode;
uniform float amount;
uniform float tolerance;
out vec4 result;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), size = textureSize(image, 0);
  vec4 center = texelFetch(image, p, 0);
  vec3 sum = vec3(0.0); float total = 0.0;
  float sigma = mode == 0 ? float(radius) : float(radius) / 2.0 + 0.5;
  for (int y = -4; y <= 4; y++) for (int x = -4; x <= 4; x++) {
    if (abs(x) > radius || abs(y) > radius) continue;
    vec4 other = texelFetch(image, clamp(p + ivec2(x,y), ivec2(0), size-1), 0);
    vec3 delta = other.rgb - center.rgb;
    float weight = exp(-float(x*x+y*y)/(2.0*sigma*sigma));
    if (mode == 0) weight *= exp(-dot(delta,delta)/(6.0*tolerance*tolerance));
    weight *= other.a;
    sum += other.rgb * weight; total += weight;
  }
  vec3 mean = total > 0.0 ? sum / total : center.rgb;
  vec3 color = mode == 0 ? mix(center.rgb, mean, amount) : center.rgb + amount * (center.rgb - mean);
  result = vec4(clamp(color,0.0,1.0),center.a);
}`;

export class FilterEngine {
  constructor() {
    this.gl = null;
    try {
      const gl = new OffscreenCanvas(1, 1).getContext("webgl2", {alpha: true, premultipliedAlpha: false, antialias: false});
      if (!gl) return;
      const compile = (type, source) => {
        const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
        return shader;
      };
      const program = gl.createProgram(), vertex = compile(gl.VERTEX_SHADER, VERTEX), fragment = compile(gl.FRAGMENT_SHADER, FRAGMENT);
      gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
      gl.deleteShader(vertex); gl.deleteShader(fragment);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      this.gl = gl; this.program = program; this.framebuffer = gl.createFramebuffer();
      this.textures = [gl.createTexture(), gl.createTexture()];
      this.locations = Object.fromEntries(["radius", "mode", "amount", "tolerance"].map(n => [n, gl.getUniformLocation(program, n)]));
    } catch { this.gl = null; }
  }

  process(pixels, width, height, values) {
    const options = settings(values), gl = this.gl;
    if (!gl || gl.isContextLost()) return filterCPU(pixels, width, height, options);
    if (!options.smooth && !options.sharpen) return new Uint8ClampedArray(pixels);
    gl.canvas.width = width; gl.canvas.height = height; gl.viewport(0, 0, width, height);
    for (const texture of this.textures) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength));
    gl.useProgram(this.program); gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    let source = 0;
    for (const [mode, amount, radius] of [[0, options.smooth, options.smoothRadius], [1, options.sharpen, options.sharpRadius]]) {
      if (!amount) continue;
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures[1 - source], 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) return filterCPU(pixels, width, height, options);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[source]);
      gl.uniform1i(this.locations.mode, mode); gl.uniform1i(this.locations.radius, radius);
      gl.uniform1f(this.locations.amount, amount); gl.uniform1f(this.locations.tolerance, options.edgeTolerance);
      gl.drawArrays(gl.TRIANGLES, 0, 3); source = 1 - source;
    }
    const result = new Uint8Array(pixels.length);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, result);
    if (gl.getError() !== gl.NO_ERROR) return filterCPU(pixels, width, height, options);
    return new Uint8ClampedArray(result.buffer);
  }
}

export function filterCanvas(source, options, engine, progress = () => {}) {
  if (!options.smooth && !options.sharpen) return source;
  const output = new OffscreenCanvas(source.width, source.height), input = source.getContext("2d", {willReadFrequently: true});
  const context = output.getContext("2d"), edge = 512, halo = options.smoothRadius + options.sharpRadius;
  const count = Math.ceil(source.width / edge) * Math.ceil(source.height / edge); let done = 0;
  for (let y = 0; y < source.height; y += edge) for (let x = 0; x < source.width; x += edge) {
    const left = Math.max(0, x - halo), top = Math.max(0, y - halo);
    const w = Math.min(source.width, x + edge + halo) - left, h = Math.min(source.height, y + edge + halo) - top;
    const image = input.getImageData(left, top, w, h), filtered = engine.process(image.data, w, h, options);
    context.putImageData(new ImageData(filtered, w, h), left, top, x - left, y - top, Math.min(edge, source.width - x), Math.min(edge, source.height - y));
    progress(++done, count);
  }
  return output;
}
