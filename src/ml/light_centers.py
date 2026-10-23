"""Small light-instance head sharing frozen DINO segmentation features."""

import torch
from torch import nn
from torch.nn import functional as F


class LightCenterHead(nn.Module):
    """A 60-pixel RGB context distinguishes an entire fire from its flame tips."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(19, 24, 3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(24, 1, 1),
        )
        nn.init.constant_(self.layers[-1].bias, -4.6)

    def forward(self, features, image):
        features = F.avg_pool2d(features, 4)
        image = F.avg_pool2d(image, 4)
        return self.layers(torch.cat((features, image), dim=1))
