"""
Grid Detection Loss.

Source:
  - title: "You Only Look Once: Unified, Real-Time Object Detection"
    url: "https://arxiv.org/abs/1506.02640"
    year: 2016

Description:
  Trains a grid of detection candidates without anchors or matching. Each ground-truth box belongs to the grid cell holding its center, and the larger box wins a cell two boxes share. Every cell learns whether it holds an object, positive cells also learn the class and the box inside the cell, and the loss is divided by the number of positive cells.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss


class GridDetLoss(BaseLoss):
    """Objectness, class and geometry loss over an ``S x S`` grid, for boxes and their per-instance extras.

        L = [ Σ_cells w_cell BCE(obj) + Σ_pos CE(cls) + λ Σ_pos |g - ĝ| ] / N_pos

    The geometry ``g`` of a positive cell is the box center inside the cell, ``(cx S - col, cy S - row)``, and the box size
    ``(w, h)``, all regressed directly so the gradient never vanishes. A rotated box adds its angle, and keypoints are offsets
    from the box center in units of the box size.

    Args:
        extra: ``none`` for boxes, ``angle`` for rotated boxes (a fifth box value), ``keypoints`` for boxes with keypoints.
        background_weight: ``w_cell`` of a cell without an object; positive cells weigh 1.
        box_weight: ``λ``.
    """

    def __init__(self, extra: Literal["none", "angle", "keypoints"] = "none", background_weight: float = 0.5, box_weight: float = 5.0) -> None:
        super().__init__()
        if extra not in ("none", "angle", "keypoints"):
            raise ValueError(f"unknown extra {extra!r}; expected 'none', 'angle' or 'keypoints'.")
        self.extra = extra
        self.background_weight = float(background_weight)
        self.box_weight = float(box_weight)

    def forward(self, obj: Tensor, cls: Tensor, box: Tensor, bboxes: Tensor, labels: Tensor, batch_idx: Tensor, *extras: Tensor) -> Tensor:
        batch, grid = obj.shape[0], obj.shape[-1]
        cells = batch * grid * grid
        bboxes, labels, batch_idx = bboxes.float(), labels.reshape(-1).long(), batch_idx.reshape(-1).long()
        col = (bboxes[:, 0] * grid).long().clamp(0, grid - 1)
        row = (bboxes[:, 1] * grid).long().clamp(0, grid - 1)
        cell = batch_idx * grid * grid + row * grid + col
        area = bboxes[:, 2] * bboxes[:, 3]
        order = torch.arange(len(area), device=area.device)
        largest = area.new_full((cells,), -1.0).scatter_reduce(0, cell, area, "amax")
        winner_of = order.new_full((cells,), len(area)).scatter_reduce(0, cell[area == largest[cell]], order[area == largest[cell]], "amin")
        pos_cell = (winner_of < len(area)).nonzero().squeeze(1)
        gt = winner_of[pos_cell]
        is_pos = torch.zeros(cells, dtype=torch.bool, device=obj.device)
        is_pos[pos_cell] = True

        target = [bboxes[gt, 0] * grid - col[gt], bboxes[gt, 1] * grid - row[gt], bboxes[gt, 2], bboxes[gt, 3]]
        if self.extra == "angle":
            target.append(bboxes[gt, 4])
        geometry = (box.permute(0, 2, 3, 1).reshape(cells, -1)[pos_cell] - torch.stack(target, dim=1)).abs().sum()
        if self.extra == "keypoints":
            points, truth = extras
            truth = truth[gt].float().clone()
            truth[..., 0] = (truth[..., 0] - bboxes[gt, 0, None]) / bboxes[gt, 2, None].clamp(min=1e-6)
            truth[..., 1] = (truth[..., 1] - bboxes[gt, 1, None]) / bboxes[gt, 3, None].clamp(min=1e-6)
            geometry = geometry + (points.permute(0, 3, 4, 1, 2).reshape(cells, *truth.shape[1:])[pos_cell] - truth).abs().sum()

        weight = torch.where(is_pos, 1.0, self.background_weight)
        objectness = (F.binary_cross_entropy_with_logits(obj.reshape(-1), is_pos.float(), reduction="none") * weight).sum()
        classes = F.cross_entropy(cls.permute(0, 2, 3, 1).reshape(cells, -1)[pos_cell], labels[gt], reduction="sum")
        return (objectness + classes + self.box_weight * geometry) / max(len(pos_cell), 1)


__all__ = ["GridDetLoss"]
