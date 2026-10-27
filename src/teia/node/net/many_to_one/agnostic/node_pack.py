"""
Node Pack.

Source: common knowledge

Description:
  Turns the flat per-node feature rows of a batch of graphs (``M = sum N_i`` rows) into a padded
  ``[B, N, D']`` node-feature tensor, and appends a learned projection of each node's attributes (box geometry, features). It is
  the seam between a per-crop encoder and a graph encoder: padded slots stay exactly zero.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode


class NodePack(TeiaNode):
    """``in: [feat.<rows> [M,D], batch.<attrs> [B,N,A], batch.node_mask [B,N]]``  ``out: feat.<nodes> [B,N,D+attr_dim]``

        h_k = [x_{cumsum(mask)_k - 1}, MLP(attr_k)] · mask_k

    Rows fill the real slots in row-major node order, so graph ``i`` receives its own ``N_i`` rows.

    Args:
        attr_dim: Width of the attribute projection concatenated to the features.
    """

    def build_module(self, attr_dim: int = 32, **kwargs: Any) -> None:
        attr_features = int(self.in_shape[1][-1])
        self.attr_proj = nn.Sequential(nn.Linear(attr_features, attr_dim), nn.GELU(), nn.Linear(attr_dim, attr_dim))

    def forward(self, rows: Tensor, attrs: Tensor, node_mask: Tensor) -> Tensor:
        node_mask = node_mask.bool()

        # slot k takes row cumsum(mask)_k - 1; padded slots are zeroed
        row = node_mask.flatten().cumsum(0).sub(1).clamp(0, rows.shape[0] - 1)
        feats = rows[row].view(*node_mask.shape, -1) * node_mask.unsqueeze(-1)

        # append the projected attributes
        return torch.cat([feats, self.attr_proj(attrs) * node_mask.unsqueeze(-1)], dim=-1)
