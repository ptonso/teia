"""``CropNodes`` — cut one fixed-size crop per box out of an image.

Source: common knowledge

Description:
  Bilinear ROI crops (``torchvision.ops.roi_align``) of every box, resized to ``crop_size`` and returned as
  uint8. Each box grows by ``pad`` of its own width and height before cropping, so a component keeps a
  little of its surroundings. Multi-input and per-image, so it has no export kernel.
"""

from __future__ import annotations

import torch

from teia.base.data import Transform
from teia.core.deps import require_dependency


class CropNodes(Transform):
    """``(item.image[3,H,W] uint8, item.boxes[N,4] xywh px) → item.node_image[N,3,S,S]`` uint8."""

    def __init__(self, *, crop_size: int = 64, pad: float = 0.1) -> None:
        require_dependency("torchvision", "CropNodes")
        self.crop_size = int(crop_size)
        self.pad = float(pad)

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import roi_align

        x, y, w, h = boxes.unbind(-1)
        rois = torch.stack([x - self.pad * w, y - self.pad * h, x + (1 + self.pad) * w, y + (1 + self.pad) * h], -1)
        crops = roi_align(image[None].float(), [rois], self.crop_size, aligned=True, sampling_ratio=2)
        return crops.round().clamp(0, 255).to(torch.uint8)
