"""
Node Unpack.

Source: common knowledge

Description:
  Selects the real nodes of a padded ``[B, N, C]`` prediction, giving one row per node, so a per-row loss
  and activation run unchanged on the nodes of every board in the batch.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from teia.base.net import TeiaNode


class NodeUnpack(TeiaNode):
    """``in: [pred.<padded> [B,N,C], batch.node_mask [B,N]]``  ``out: pred.<rows> [M,C]``, row-major node order."""

    def build_module(self, **kwargs: Any) -> None:
        return None

    def forward(self, padded: Tensor, node_mask: Tensor) -> Tensor:
        return padded[node_mask.bool()]
