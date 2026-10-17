"""
Graph COMBINE Functions.

Source:
  - title: "Do Transformers Really Perform Bad for Graph Representation?"
    url: "https://arxiv.org/abs/2106.05234"
    year: 2021
  - title: "Inductive Representation Learning on Large Graphs"
    url: "https://arxiv.org/abs/1706.02216"
    year: 2017
  - title: "How Powerful are Graph Neural Networks?"
    url: "https://arxiv.org/abs/1810.00826"
    year: 2019

Description:
  The COMBINE options of ``GraphEncoder``: each merges a node's message ``a_i`` into its feature ``h_i``.

Adaptations:
  - The residual COMBINE is post-LN by default, and ``norm_order="pre"`` gives the pre-LN block of Graphormer.
  - The L2 normalisation of the concat COMBINE can be turned off with ``normalize``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Combine(nn.Module):
    """A COMBINE option. ``prepare`` maps the features before AGGREGATE reads them, which only a pre-LN block changes."""

    def prepare(self, h: Tensor) -> Tensor:
        return h


class ResidualCombine(Combine):
    """Transformer block, post-LN by default.

        post: h ← LN(h + a),  h ← LN(h + FFN(h))
        pre:  a is computed from LN(h),  h ← h + a,  h ← h + FFN(LN(h))
    """

    def __init__(self, dim: int, ffn_ratio: int, dropout: float, norm_order: str = "post", **_: Any) -> None:
        super().__init__()
        self.pre = norm_order == "pre"
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_ratio * dim), nn.GELU(), nn.Linear(ffn_ratio * dim, dim))
        self.drop = nn.Dropout(dropout)

    def prepare(self, h: Tensor) -> Tensor:
        return self.norm1(h) if self.pre else h

    def forward(self, h: Tensor, a: Tensor) -> Tensor:
        if self.pre:
            h = h + self.drop(a)
            return h + self.drop(self.ffn(self.norm2(h)))

        h = self.norm1(h + self.drop(a))

        return self.norm2(h + self.drop(self.ffn(h)))


class ConcatCombine(Combine):
    """Concatenate and project (GraphSAGE Alg. 1).

        h ← ReLU(W [h ‖ a])
        h ← h / ‖h‖₂      (``normalize``)
    """

    def __init__(self, dim: int, normalize: bool = True, **_: Any) -> None:
        super().__init__()
        self.w, self.normalize = nn.Linear(2 * dim, dim), normalize

    def forward(self, h: Tensor, a: Tensor) -> Tensor:
        h = F.relu(self.w(torch.cat([h, a], dim=-1)))

        return F.normalize(h, dim=-1) if self.normalize else h


class GinCombine(Combine):
    """GIN-ε (Eq. 4.1): ``h ← MLP((1 + ε) h + a)``, with a learned ``ε``."""

    def __init__(self, dim: int, ffn_ratio: int, **_: Any) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(dim, ffn_ratio * dim), nn.ReLU(), nn.Linear(ffn_ratio * dim, dim))

    def forward(self, h: Tensor, a: Tensor) -> Tensor:
        return self.mlp((1 + self.eps) * h + a)


COMBINE = {"residual": ResidualCombine, "concat": ConcatCombine, "gin": GinCombine}
