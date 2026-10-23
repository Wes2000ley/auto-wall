from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.core.fast_pipeline import FastSettings, _detect_light_centers, detect_map, to_uvtt
from src.core.light_geometry import light_center_points


def test_plateaus_are_single_sources_and_neighboring_lights_stay_separate():
    heatmap = np.zeros((40, 40), np.float32)
    heatmap[10:12, 10:12] = .8
    heatmap[10, 19] = .7
    heatmap[0, 0] = .9
    points = light_center_points(heatmap, .5)
    np.testing.assert_array_equal(points, [[0, 0], [10.5, 10.5], [19, 10]])
    assert len(light_center_points(heatmap, .85)) == 1


def test_rejecting_a_door_highlight_cannot_create_peaks_on_its_cut_edges():
    yy, xx = np.mgrid[:64, :64]
    heatmap = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 100).astype(np.float32)
    heatmap[5, 5] = 1
    original = heatmap.copy()
    image = np.full((64, 64, 3), (20, 100, 240), np.uint8)
    footprint = np.array([[27, 27], [33, 27], [33, 33], [27, 33]])
    lights = _detect_light_centers(image, heatmap, [footprint], FastSettings())
    assert len(lights) == 1 and lights[0]["position"] == {"x": 5., "y": 5.}
    assert lights[0]["color"] == "fff06414"
    np.testing.assert_array_equal(heatmap, original)


def test_points_replace_fragmented_pixel_lights_without_changing_geometry():
    image = np.zeros((512, 512, 3), np.uint8)
    labels = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(labels, (30, 30), (480, 480), 1, 8)
    cv2.rectangle(labels, (250, 26), (268, 34), 2, -1)
    cv2.line(labels, (90, 200), (200, 200), 4, 5)
    for x in (120, 130, 140):
        cv2.circle(labels, (x, 120), 2, 3, -1)
    confidence = np.ones(labels.shape, np.float32)
    heatmap = np.zeros(labels.shape, np.float32)
    heatmap[119:121, 129:131] = .9
    settings = replace(FastSettings(), light_min_area=1)
    baseline = detect_map(image, settings, prediction=(labels, confidence))
    result = detect_map(image, settings, prediction=(labels, confidence, heatmap))
    assert len(baseline.lights) == 3 and len(result.lights) == 1
    assert result.walls == baseline.walls and result.doors == baseline.doors
    assert result.transparent_barriers == baseline.transparent_barriers
    assert result.lights[0]["position"] == {"x": 129.5, "y": 119.5}
    assert to_uvtt(result, 100)["lights"][0]["position"] == {"x": 1.295, "y": 1.195}
    disabled = detect_map(image, replace(settings, lights=False), prediction=(labels, confidence, heatmap))
    assert not disabled.lights and disabled.walls == result.walls
    for invalid in (np.zeros((5, 5), np.float32), np.full(labels.shape, np.nan, np.float32),
                    np.full(labels.shape, 1.1, np.float32)):
        with pytest.raises(ValueError, match="Light heatmaps"):
            detect_map(image, settings, prediction=(labels, confidence, invalid))
