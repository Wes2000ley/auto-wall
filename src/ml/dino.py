"""DINOv2 context plus a small trainable decoder for precise boundaries.

No weights or remote Python are fetched here. The model loader supplies all tensors
from one local safetensors file. DINOv2 backbone weights require their Apache 2.0
notice; see models/licenses/DINOv2.txt when distributing a derived checkpoint.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.ml.network import BattlemapNet


class DinoBattlemap(nn.Module):
    architecture = "dino"
    width = 16
    context_dilation = 4

    def __init__(self):
        super().__init__()
        try:
            from transformers import Dinov2Config, Dinov2Model
        except ImportError as error:
            raise RuntimeError("DINO models need the optional requirements-ml-dino.txt dependencies.") from error
        config = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                              image_size=518, patch_size=14, layerscale_value=1.0,
                              use_swiglu_ffn=False)
        self.backbone = Dinov2Model(config).requires_grad_(False)
        self.backbone_training_layers = 0
        self.local = BattlemapNet(self.width, self.context_dilation)
        self.project = nn.Conv2d(384, 128, 1)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train_backbone_layers(self, count):
        """Optionally adapt the final blocks; loading for inference stays frozen."""
        if type(count) is not int or not 0 <= count <= 4:
            raise ValueError("Choose zero to four final DINO blocks to fine-tune.")
        self.backbone.requires_grad_(False)
        for parameter in self.backbone.parameters():
            parameter.grad = None
        if count:
            for block in self.backbone.encoder.layer[-count:]:
                block.requires_grad_(True)
            self.backbone.layernorm.requires_grad_(True)
        self.backbone_training_layers = count

    def forward(self, image):
        local = self.local
        a = local.enc1(image)
        b = local.enc2(F.max_pool2d(a, 2))
        c = local.enc3(F.max_pool2d(b, 2))
        x = local.bottom(F.max_pool2d(c, 2))
        padded = F.pad(image, (0, -image.shape[3] % 14, 0, -image.shape[2] % 14), mode="replicate")
        self.backbone.eval()
        # Frozen prefixes need no activation graph. PyTorch records operations
        # only once it reaches the explicitly unfrozen final blocks.
        train_context = torch.is_grad_enabled() and self.training and self.backbone_training_layers > 0
        with torch.set_grad_enabled(train_context):
            tokens = self.backbone((padded - self.mean) / self.std).last_hidden_state[:, 1:]
            features = tokens.transpose(1, 2).reshape(image.shape[0], 384, padded.shape[2] // 14, padded.shape[3] // 14)
        features = self.project(features)
        # Sample context at decoder cell centers, preserving padding alignment.
        # Expanding all 128 channels to full resolution first wastes GPU memory.
        gy = (torch.arange(x.shape[2], device=x.device, dtype=torch.float32) + .5) / x.shape[2]
        gx = (torch.arange(x.shape[3], device=x.device, dtype=torch.float32) + .5) / x.shape[3]
        yy, xx = torch.meshgrid(gy * image.shape[2] / padded.shape[2] * 2 - 1,
                               gx * image.shape[3] / padded.shape[3] * 2 - 1, indexing="ij")
        grid = torch.stack((xx, yy), -1)[None].expand(image.shape[0], -1, -1, -1)
        x = x + F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=False)
        for skip, decoder in ((c, local.dec3), (b, local.dec2), (a, local.dec1)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), 1))
        return local.head(x)
