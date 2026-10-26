export function reflect(index, size) {
  if (size <= 1) return 0;
  const period = 2 * (size - 1), n = ((index % period) + period) % period;
  return n < size ? n : period - n;
}

export function* tiles(width, height, edge) {
  for (let y = 0; y < height; y += edge) for (let x = 0; x < width; x += edge) {
    yield {x, y, width: Math.min(edge, width - x), height: Math.min(edge, height - y)};
  }
}

// Exact shared boundaries avoid gaps when 4x tiles are resampled to a smaller output.
export function scaledTile(tile, inputWidth, inputHeight, outputWidth, outputHeight) {
  const left = Math.round(tile.x * outputWidth / inputWidth), top = Math.round(tile.y * outputHeight / inputHeight);
  return {x: left, y: top, width: Math.round((tile.x + tile.width) * outputWidth / inputWidth) - left,
    height: Math.round((tile.y + tile.height) * outputHeight / inputHeight) - top};
}
