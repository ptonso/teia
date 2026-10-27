"""
Grid Detection Activation.

Source:
  - title: "You Only Look Once: Unified, Real-Time Object Detection"
    url: "https://arxiv.org/abs/1506.02640"
    year: 2016

Description:
  Turns a grid of detection candidates into boxes and class scores. Each cell proposes one box, scored by its objectness times its class probability. The kernel then keeps the confident candidates, suppresses overlaps, undoes the letterbox and emits fractional boxes, plus the angle, the keypoints or the box polygon of each instance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor

from teia.base.net import BaseActivation
from teia.node.net._activation_utils import SquareFrameParams, grid_postprocess


class GridDetActivation(SquareFrameParams, BaseActivation):
    """Grid candidates to boxes: ``score = σ(obj) · softmax(cls)``, ``center = (cell + offset) / S``, ``size = (w, h)``, offset and size clamped to ``[0, 1]``.

    Args:
        extra: ``none`` for boxes, ``angle`` for rotated boxes (the raw fifth box value), ``keypoints`` for boxes with keypoints
            (offsets from the box center in box sizes), ``polygon`` for boxes whose instance polygon is the box rectangle.
    """

    def __init__(self, extra: Literal["none", "angle", "keypoints", "polygon"] = "none") -> None:
        super().__init__()
        if extra not in ("none", "angle", "keypoints", "polygon"):
            raise ValueError(f"unknown extra {extra!r}; expected 'none', 'angle', 'keypoints' or 'polygon'.")
        self.extra = extra

    def params(self) -> dict[str, Any]:
        return {**super().params(), "extra": self.extra}

    def activation(self, obj: Tensor, cls: Tensor, box: Tensor, *points: Tensor) -> dict[str, Tensor]:
        grid = obj.shape[-1]
        scores = (torch.sigmoid(obj).unsqueeze(1) * torch.softmax(cls, dim=1)).flatten(2).transpose(1, 2)
        rows, cols = torch.meshgrid(*[torch.arange(grid, device=obj.device, dtype=box.dtype)] * 2, indexing="ij")
        rows, cols = rows.reshape(1, -1), cols.reshape(1, -1)
        values = box.flatten(2).transpose(1, 2)
        cx, cy = (cols + values[..., 0].clamp(0, 1)) / grid, (rows + values[..., 1].clamp(0, 1)) / grid
        w, h = values[..., 2].clamp(0, 1), values[..., 3].clamp(0, 1)
        activated = {"boxes": torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1), "scores": scores}
        if self.extra == "angle":
            activated["angles"] = values[..., 4:5]
        if self.extra == "keypoints":
            keypoints = points[0].flatten(3).permute(0, 3, 1, 2)
            xy = torch.stack([cx.unsqueeze(-1) + keypoints[..., 0] * w.unsqueeze(-1), cy.unsqueeze(-1) + keypoints[..., 1] * h.unsqueeze(-1)], dim=-1)
            activated["keypoints"] = torch.cat([xy, keypoints[..., 2:]], dim=-1)
        return activated

    @staticmethod
    def kernel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        return grid_postprocess(activated, ctx)
