"""Small light-instance head sharing frozen DINO segmentation features."""

import torch
from torch import nn
from torch.nn import functional as F

from src.ml.dino import sample_patch_context


class LightCenterHead(nn.Module):
    """A 60-pixel RGB context distinguishes an entire fire from its flame tips."""

    def __init__(self, semantic_context=False):
        super().__init__()
        self.semantic_context = semantic_context
        self.context_projection = nn.Conv2d(384, 16, 1) if semantic_context else None
        self.layers = nn.Sequential(
            nn.Conv2d(35 if semantic_context else 19, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(24, 1, 1),
        )
        nn.init.constant_(self.layers[-1].bias, -4.6)

    def forward(self, features, image, context=None):
        parts = [F.avg_pool2d(features, 4), F.avg_pool2d(image, 4)]
        if self.semantic_context:
            if context is None:
                raise ValueError("This light head requires DINO patch context from the same image pass.")
            context = F.gelu(self.context_projection(context))
            parts.append(sample_patch_context(context, parts[0].shape[-2:], image.shape[-2:]))
        return self.layers(torch.cat(parts, dim=1))

    @torch.no_grad()
    def with_semantic_context(self):
        """Copy a trained local head; start new context weights at zero influence."""
        if self.semantic_context:
            raise ValueError("This light head already uses semantic context.")
        parameter = self.layers[0].weight
        upgraded = LightCenterHead(semantic_context=True).to(device=parameter.device, dtype=parameter.dtype)
        state = upgraded.state_dict()
        for name, value in self.state_dict().items():
            if name == "layers.0.weight":
                state[name].zero_()
                state[name][:, :19].copy_(value)
            else:
                state[name].copy_(value)
        return upgraded.train(self.training)
