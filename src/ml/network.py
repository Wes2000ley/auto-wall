"""Small U-Net for solid walls, doors, lights, and sight-transparent barriers."""

import torch
from torch import nn
from torch.nn import functional as F


CLASSES = ("background", "wall", "door", "light", "transparent_barrier")


def block(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class BattlemapNet(nn.Module):
    def __init__(self, width=16, context_dilation=1):
        super().__init__()
        self.width = width
        self.context_dilation = context_dilation
        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.enc3 = block(width * 2, width * 4)
        self.bottom = block(width * 4, width * 8, context_dilation)
        self.dec3 = block(width * 12, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, len(CLASSES), 1)

    def forward(self, image):
        a = self.enc1(image)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        x = self.bottom(F.max_pool2d(c, 2))
        for skip, decoder in ((c, self.dec3), (b, self.dec2), (a, self.dec1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)
