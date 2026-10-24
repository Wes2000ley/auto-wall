"""Prediction contracts for queued CUDA tiles and the synchronous CPU path."""
# ruff: noqa: E402

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from src.core.fast_pipeline import DetectionCancelled
from src.ml.model import predict
from src.ml.network import BattlemapNet, WINDOW_CLASSES


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"))]


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", DEVICES)
def test_odd_tile_count_has_complete_outputs_and_ordered_progress(device):
    model = BattlemapNet(8, 1, WINDOW_CLASSES).to(device).eval()
    # Three tiles exercise a full CUDA queue followed by its final odd tile.
    image = np.random.default_rng(3).integers(0, 256, (17, 129, 3), np.uint8)
    progress = []
    labels, confidence, heatmap = predict(model, image, tile_size=64,
                                          progress=progress.append, include_light_centers=True)
    assert progress == [f"ML segmentation: {i}/3 tiles" for i in range(1, 4)]
    assert labels.shape == confidence.shape == (17, 129)
    assert labels.dtype == np.uint8 and confidence.dtype == np.float32
    assert labels.max() < len(WINDOW_CLASSES)
    assert np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1))
    assert heatmap is None
    np.testing.assert_array_equal(predict(model, image, tile_size=64)[0], labels)


@pytest.mark.parametrize("device", DEVICES)
def test_progress_cancellation_does_not_report_queued_results(device):
    model = BattlemapNet(8).to(device).eval()
    progress = []
    with pytest.raises(DetectionCancelled):
        predict(model, np.zeros((17, 129, 3), np.uint8), tile_size=64,
                cancelled=lambda: bool(progress), progress=progress.append)
    assert progress == ["ML segmentation: 1/3 tiles"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failure", ["probabilities", "light scores"])
def test_nonfinite_outputs_are_rejected_even_outside_retained_region(device, failure):
    class InvalidModel(torch.nn.Module):
        context_dilation = 1

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.local = SimpleNamespace(head=self.forward)
            self.light_head = self.light_scores

        def forward(self, image):
            result = image.new_zeros((1, 5, *image.shape[-2:]))
            if failure == "probabilities":
                result[0, 0, -1, -1] = float("nan")
            return result

        def forward_features(self, image, include_context):
            return image, None

        def light_scores(self, features, image, context):
            result = image.new_zeros((1, 1, *image.shape[-2:]))
            if failure == "light scores":
                result[0, 0, -1, -1] = float("nan")
            return result

    with pytest.raises(RuntimeError, match=f"nonfinite {failure}"):
        predict(InvalidModel().to(device).eval(), np.zeros((9, 17, 3), np.uint8),
                include_light_centers=True)
