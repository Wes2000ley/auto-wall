export function crosses(segment, start, end) {
  const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
  const v = [end[0] - start[0], end[1] - start[1]], w = [segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]];
  const delta = [segment[0][0] - start[0], segment[0][1] - start[1]], denominator = cross(v, w);
  if (Math.abs(denominator) < 1e-8) return false;
  const t = cross(delta, w) / denominator, u = cross(delta, v) / denominator;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}
