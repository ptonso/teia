"""Box hygiene augments — clamp to the canvas and drop degenerate instances.

Source: common knowledge

Description:
  Normalized-cxcywh analogs of torchvision ``ClampBoundingBoxes`` / ``SanitizeBoundingBoxes``.
  ``SanitizeBoxes`` measures ``min_size`` / ``min_area`` in pixels against the current ``item.image``
  side and drops the failing rows across every aligned per-instance key it is wired to
  (``item.box_cls`` / ``item.keypoints`` / ``item.polys``).
"""

from __future__ import annotations

from typing import Any

import torch

from teia.node.data.transform.augment import ItemAugment

_INSTANCE_TENSOR_KEYS = ("box_cls", "keypoints")


class ClampBoxes(ItemAugment):
    """Clamp each box to the unit canvas by clipping its corners, preserving cxcywh."""

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        boxes = record.get("boxes")
        if boxes is None or not boxes.numel():
            return record
        cx, cy, w, h = boxes.unbind(1)
        x1 = (cx - w / 2).clamp(0.0, 1.0)
        y1 = (cy - h / 2).clamp(0.0, 1.0)
        x2 = (cx + w / 2).clamp(0.0, 1.0)
        y2 = (cy + h / 2).clamp(0.0, 1.0)
        record["boxes"] = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)
        return record


class SanitizeBoxes(ItemAugment):
    """Drop boxes below ``min_size`` (side, px) or ``min_area`` (px²), keeping aligned keys in sync."""

    def __init__(self, *, min_size: float = 2.0, min_area: float = 4.0) -> None:
        self.min_size = float(min_size)
        self.min_area = float(min_area)

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        boxes = record.get("boxes")
        if boxes is None or not boxes.numel():
            return record
        side = float(record["image"].shape[-1])
        w = boxes[:, 2] * side
        h = boxes[:, 3] * side
        keep = (w > self.min_size) & (h > self.min_size) & (w * h > self.min_area)
        n = int(keep.shape[0])
        record["boxes"] = boxes[keep]
        for key in _INSTANCE_TENSOR_KEYS:
            value = record.get(key)
            if torch.is_tensor(value) and value.shape[0] == n:
                record[key] = value[keep]
        polys = record.get("polys")
        if polys is not None and len(polys) == n:
            record["polys"] = [poly for poly, flag in zip(polys, keep.tolist()) if flag]
        return record
