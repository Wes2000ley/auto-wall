"""Decode physical light centers without merging nearby independent sources."""

import cv2
import numpy as np


def light_center_points(heatmap, threshold):
    """Return local maxima as XY points, with one center per equal-valued plateau."""
    if (heatmap.ndim != 2 or not heatmap.size or not np.isfinite(heatmap).all()
            or heatmap.min() < 0 or heatmap.max() > 1):
        raise ValueError("Light heatmaps must contain finite scores between zero and one.")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Light center threshold must be greater than zero and at most one.")
    heatmap = np.ascontiguousarray(heatmap, dtype=np.float32)
    maxima = cv2.dilate(heatmap, np.ones((7, 7), np.uint8))
    mask = ((heatmap >= threshold) & (heatmap == maxima)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points = []
    for index in range(1, count):
        x, y, width, height, _ = stats[index]
        ys, xs = np.nonzero(labels[y:y + height, x:x + width] == index)
        points.append((x + xs.mean(), y + ys.mean()))
    return np.asarray(points, dtype=float).reshape(-1, 2)
