"""Small U-Net for solid walls, doors, lights, and sight-transparent barriers."""

import torch
from torch import nn
from torch.nn import functional as F


LEGACY_CLASSES = ("background", "wall", "door", "light", "transparent_barrier")
WINDOW_CLASSES = LEGACY_CLASSES + ("window",)
CLASS_PROFILES = {"legacy": LEGACY_CLASSES, "windows": WINDOW_CLASSES}
CLASS_SETS = (LEGACY_CLASSES, WINDOW_CLASSES)
# Keep the historical default for existing callers and checkpoint metadata.
CLASSES = LEGACY_CLASSES


def validate_classes(classes):
    """Return one supported ordered class set, rejecting ambiguous variants."""
    try:
        classes = tuple(classes)
    except TypeError as error:
        raise ValueError("Classes must be the exact legacy or windows class set.") from error
    if classes not in CLASS_SETS:
        raise ValueError("Classes must be the exact legacy or windows class set.")
    return classes


def block(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class BattlemapNet(nn.Module):
    architecture = "compact"

    def __init__(self, width=16, context_dilation=1, classes=CLASSES):
        super().__init__()
        self.classes = validate_classes(classes)
        self.width = width
        self.context_dilation = context_dilation
        self.enc1 = block(3, width)
        self.enc2 = block(width, width * 2)
        self.enc3 = block(width * 2, width * 4)
        self.bottom = block(width * 4, width * 8, context_dilation)
        self.dec3 = block(width * 12, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, len(self.classes), 1)

    def forward(self, image):
        a = self.enc1(image)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        x = self.bottom(F.max_pool2d(c, 2))
        for skip, decoder in ((c, self.dec3), (b, self.dec2), (a, self.dec1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)


def expand_legacy_head(model):
    """Expand a five-class model in place, preserving all existing output channels."""
    if validate_classes(model.classes) != LEGACY_CLASSES:
        raise ValueError("Only a legacy five-class model can be expanded for windows.")
    decoder = getattr(model, "local", model)
    old_head = decoder.head
    if old_head.out_channels != len(LEGACY_CLASSES):
        raise ValueError("The legacy segmentation head has an unexpected shape.")
    new_head = nn.Conv2d(
        old_head.in_channels,
        len(WINDOW_CLASSES),
        old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
        dilation=old_head.dilation,
        groups=old_head.groups,
        bias=old_head.bias is not None,
        padding_mode=old_head.padding_mode,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )
    with torch.no_grad():
        new_head.weight[:len(LEGACY_CLASSES)].copy_(old_head.weight)
        if old_head.bias is not None:
            new_head.bias[:len(LEGACY_CLASSES)].copy_(old_head.bias)
    new_head.weight.requires_grad_(old_head.weight.requires_grad)
    if old_head.bias is not None:
        new_head.bias.requires_grad_(old_head.bias.requires_grad)
    new_head.train(old_head.training)
    decoder.head = new_head
    decoder.classes = WINDOW_CLASSES
    return model
