"""``NodeBoxes`` — box geometry as node features.

Source: common knowledge

Description:
  Converts pixel ``xywh`` boxes to ``cxcywh`` normalised by the image size, so the position of a component
  on the board is a scale-free node feature.
"""

from __future__ import annotations

import torch

from teia.base.data import Transform


class NodeBoxes(Transform):
    """``(item.image[3,H,W], item.boxes[N,4] xywh px) → item.node_box[N,4]`` float32 ``cxcywh`` in ``[0,1]``."""

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        x, y, w, h = boxes.unbind(-1)
        scale = boxes.new_tensor([width, height, width, height])
        return torch.stack([x + w / 2, y + h / 2, w, h], -1) / scale
