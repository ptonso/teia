"""
Graph AGGREGATE Functions.

Source:
  - title: "Do Transformers Really Perform Bad for Graph Representation?"
    url: "https://arxiv.org/abs/2106.05234"
    year: 2021
  - title: "How Attentive are Graph Attention Networks?"
    url: "https://arxiv.org/abs/2105.14491"
    year: 2021
  - title: "Inductive Representation Learning on Large Graphs"
    url: "https://arxiv.org/abs/1706.02216"
    year: 2017
  - title: "How Powerful are Graph Neural Networks?"
    url: "https://arxiv.org/abs/1810.00826"
    year: 2019

Description:
  The AGGREGATE options of ``GraphEncoder``: each gathers a message ``a_i`` from the nodes ``i`` reads.

Adaptations:
  - GATv2 heads are concatenated and projected by ``W_O``, as in Graphormer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from teia.node.net.many_to_one.graph.graph_encoder import GraphStructure


def _head_dim(dim: int, heads: int) -> int:
    if dim % heads:
        raise ValueError(f"GraphEncoder width {dim} is not divisible by heads={heads}.")
    return dim // heads


class DotAttention(nn.Module):
    """Dot-product attention with a structural bias (Graphormer Eq. 6).

        A_ij = (h_i W_Q)(h_j W_K)ᵀ / √d + b_φ(i,j)
        a_i  = W_O Σ_j softmax_j(A_ij) h_j W_V
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.head_dim = _head_dim(dim, heads)
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h: Tensor, s: GraphStructure) -> Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)

        mask = s.bias.masked_fill(~s.attend[:, None], float("-inf")).to(q.dtype)

        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)

        return self.out(a.transpose(1, 2).reshape(b, n, d))


class GATv2Attention(nn.Module):
    """Dynamic attention (GATv2 Eq. 7).

        e_ij = aᵀ LeakyReLU(W_dst h_i + W_src h_j)
        a_i  = W_O Σ_j softmax_j(e_ij) W_src h_j
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.head_dim = _head_dim(dim, heads)
        self.heads = heads
        self.w_dst = nn.Linear(dim, dim)
        self.w_src = nn.Linear(dim, dim)
        self.attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(heads, self.head_dim)))
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, s: GraphStructure) -> Tensor:
        b, n, _ = h.shape
        dst = self.w_dst(h).view(b, n, 1, self.heads, self.head_dim)
        src = self.w_src(h).view(b, 1, n, self.heads, self.head_dim)

        e = (F.leaky_relu(dst + src, 0.2) * self.attn).sum(-1)  # [B, N(i), N(j), H]
        alpha = self.drop(e.masked_fill(~s.attend.unsqueeze(-1), float("-inf")).softmax(dim=2))

        a = torch.einsum("bijh,bjhc->bihc", alpha, src.squeeze(1)).reshape(b, n, -1)

        return self.out(a)


class MeanAggregate(nn.Module):
    """Neighbour mean (GraphSAGE Eq. 2): ``a_i = 1/|N(i)| Σ_{j∈N(i)} h_j``."""

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(self, h: Tensor, s: GraphStructure) -> Tensor:
        adj = s.neighbors.to(h.dtype)
        return adj @ h / adj.sum(-1, keepdim=True).clamp(min=1)


class SumAggregate(nn.Module):
    """Neighbour sum (GIN Eq. 4.1): ``a_i = Σ_{j∈N(i)} h_j``."""

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(self, h: Tensor, s: GraphStructure) -> Tensor:
        return s.neighbors.to(h.dtype) @ h


AGGREGATE = {"dot": DotAttention, "gatv2": GATv2Attention, "mean": MeanAggregate, "sum": SumAggregate}
ATTENTION = ("dot", "gatv2")
