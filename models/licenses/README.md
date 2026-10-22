# Pretrained backbone attribution

DINOv2 is by Meta Platforms, Inc. and affiliates. The upstream project distributes
its code and weights under Apache 2.0. `DINOv2.txt` is an unchanged copy of the
[upstream LICENSE](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
The inspected upstream root did not include a separate NOTICE file.

The optional Auto-Wall DINO architecture adds a local U-Net decoder and a context
projection to the pretrained backbone. Those additions are trained for battlemap
classes; the distributed v1 checkpoint keeps the backbone frozen. The trainer can
optionally fine-tune its final blocks in later experiments. This is a modified
model, not an official Meta battlemap detector. Model cards must identify the exact source revision,
weight hash, changes, and training provenance for each distributed checkpoint.

The compact `battlemap-v1.safetensors` baseline was trained from scratch and
contains no DINOv2 weights. No CubiCasa or SegFormer weights are distributed here.
