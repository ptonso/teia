"""
Node Pack.

Source: common knowledge

Description:
  Turns the flat per-node feature rows of a batch of boards (``M = sum N_i`` rows) into a padded
  ``[B, N, D']`` node-feature tensor, and appends a learned projection of each node's box geometry. It is
  the seam between a per-crop encoder and a graph encoder: padded slots stay exactly zero.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode


class NodePack(TeiaNode):
    """``in: [feat.<rows> [M,D], batch.node_box [B,N,4], batch.node_mask [B,N]]``  ``out: feat.<nodes> [B,N,D+box_dim]``

        h_k = [x_{cumsum(mask)_k - 1}, MLP(box_k)] · mask_k

    Rows fill the real slots in row-major node order, so board ``i`` receives its own ``N_i`` rows.

    Args:
        box_dim: Width of the box projection concatenated to the features.
    """

    def build_module(self, box_dim: int = 32, **kwargs: Any) -> None:
        box_features = int(self.in_shape[1][-1])
        self.box_proj = nn.Sequential(nn.Linear(box_features, box_dim), nn.GELU(), nn.Linear(box_dim, box_dim))

    def forward(self, rows: Tensor, boxes: Tensor, node_mask: Tensor) -> Tensor:
        node_mask = node_mask.bool()

        # slot k takes row cumsum(mask)_k - 1; padded slots are zeroed
        row = node_mask.flatten().cumsum(0).sub(1).clamp(0, rows.shape[0] - 1)
        feats = rows[row].view(*node_mask.shape, -1) * node_mask.unsqueeze(-1)

        # append the projected box
        return torch.cat([feats, self.box_proj(boxes) * node_mask.unsqueeze(-1)], dim=-1)
