"""Random horizontal flip over the item workspace (mirrors boxes, keypoints, polygons).

Source: common knowledge

Description:
  Normalized-coordinate analog of the record-level flip: unlike a bare image flip it also mirrors
  ``item.boxes`` (cxcywh), ``item.keypoints`` (with optional left/right ``flip_idx`` reorder) and
  ``item.polys`` so it is safe for det/pose/obb/inst-seg.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torch

from teia.node.data.transform.augment import ItemAugment


class RandomHFlip(ItemAugment):
    """Mirror the sample horizontally with probability ``p``."""

    def __init__(self, *, p: float = 0.5, flip_idx: Sequence[int] | None = None) -> None:
        self.p = float(p)
        self.flip_idx = list(flip_idx) if flip_idx is not None else None

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        if random.random() >= self.p:
            return record
        record["image"] = record["image"].flip(-1)

        boxes = record.get("boxes")
        if boxes is not None and boxes.numel():
            flipped = boxes.clone()
            flipped[:, 0] = 1.0 - boxes[:, 0]
            record["boxes"] = flipped

        kpts = record.get("keypoints")
        if kpts is not None and kpts.numel():
            flipped = kpts.clone()
            flipped[..., 0] = 1.0 - kpts[..., 0]
            if self.flip_idx is not None and len(self.flip_idx) == flipped.shape[1]:
                flipped = flipped[:, self.flip_idx]
            record["keypoints"] = flipped

        polys = record.get("polys")
        if polys is not None:
            record["polys"] = [torch.stack([1.0 - poly[:, 0], poly[:, 1]], dim=1) for poly in polys]

        mask = record.get("mask")
        if mask is not None:
            record["mask"] = mask.flip(-1)

        return record
