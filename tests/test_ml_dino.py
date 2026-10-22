"""Optional DINO checks with random local tensors; no weights or network downloads."""
# Optional packages must be checked before importing the model implementation.
# ruff: noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from src.ml.dino import DinoBattlemap
from src.ml.model import load_model, predict, save_model
from scripts.train_ml import training_optimizer


def test_dino_trains_decoder_freezes_context_and_roundtrips_safely(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = DinoBattlemap().train()
        logits = model(torch.rand(2, 3, 32, 40))
        assert logits.shape == (2, 5, 32, 40) and logits.isfinite().all()
        logits.square().mean().backward()
        assert model.project.weight.grad.abs().sum() > 0
        assert model.local.head.weight.grad.abs().sum() > 0
        assert not model.backbone.training
        assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
        path = tmp_path / "dino.safetensors"
        save_model(path, model.eval(), {"test": "random tensors"})
        loaded, details = load_model(path, "cpu")
        assert isinstance(loaded, DinoBattlemap) and details == {"test": "random tensors"}
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
        labels, confidence = predict(loaded, np.zeros((9, 17, 3), np.uint8), tile_size=8)
        assert labels.shape == confidence.shape == (9, 17) and np.isfinite(confidence).all()
    finally:
        torch.set_num_threads(previous)


def test_dino_adapts_only_selected_final_blocks_and_respects_no_grad():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = DinoBattlemap().train()
        optimizer = training_optimizer(model, .0002, backbone_layers=1)
        assert [group["initial_lr"] for group in optimizer.param_groups] == [.0002, .00002]
        image = torch.rand(2, 3, 32, 40)
        model(image).square().mean().backward()
        expected = ("encoder.layer.11.", "layernorm.")
        for name, parameter in model.backbone.named_parameters():
            if name.startswith(expected):
                assert parameter.requires_grad and parameter.grad is not None
                assert parameter.grad.isfinite().all()
            else:
                assert not parameter.requires_grad and parameter.grad is None
        assert any(p.grad.abs().sum() > 0 for p in model.backbone.encoder.layer[-1].parameters())
        assert model.local.head.weight.grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            before = model(image)
        model.train_backbone_layers(0)
        assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
        with torch.no_grad():
            torch.testing.assert_close(model(image), before, rtol=0, atol=0)
        for invalid in (-1, 5, True, 1.5):
            with pytest.raises(ValueError, match="zero to four"):
                model.train_backbone_layers(invalid)
    finally:
        torch.set_num_threads(previous)
