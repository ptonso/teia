"""
Pointwise MLP.

Source: common knowledge

Description:
  A feed-forward stack applied to the last axis of a tensor of any rank, with the same weights at every
  leading position. On ``[B, N, D]`` node features it is one shared per-node head, so the output for a node
  never depends on which slot the node occupies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode


class PointwiseMLP(TeiaNode):
    """``in: [..., D]``  ``out: [..., out_shape[-1]]``, Linear → GELU → LayerNorm → dropout per hidden width."""

    def build_module(self, hidden: Sequence[int] = (128,), dropout: float = 0.0, **kwargs: Any) -> None:
        layers: list[nn.Module] = []
        width = int(self.in_shape[-1])
        for size in hidden:
            layers += [nn.Linear(width, int(size)), nn.GELU(), nn.LayerNorm(int(size)), nn.Dropout(float(dropout))]
            width = int(size)
        layers.append(nn.Linear(width, int(self.out_shape[-1])))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
