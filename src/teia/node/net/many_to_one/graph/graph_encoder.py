"""
Graph Encoder.

Source:
  - title: "Do Transformers Really Perform Bad for Graph Representation?"
    url: "https://arxiv.org/abs/2106.05234"
    year: 2021

Description:
  Message passing on dense padded graphs: each layer AGGREGATEs a message from a node's neighbours and
  COMBINEs it with the node's own feature (Graphormer §2). The two steps are chosen by name, so leaves
  reproduce Graphormer, GATv2, GraphSAGE and GIN.

Adaptations:
  - No edge encoding, virtual node, or in/out-degree split.
  - Padded nodes are masked from every neighbourhood and zeroed after every layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode
from teia.node.net.many_to_one.graph.aggregate import AGGREGATE, ATTENTION
from teia.node.net.many_to_one.graph.combine import COMBINE


@dataclass(frozen=True)
class GraphStructure:
    """Adjacency-derived tensors shared by every layer."""

    neighbors: Tensor  # [B,N,N] j ∈ N(i), self excluded
    attend: Tensor     # [B,N,N] keys attention reads, self included
    bias: Tensor       # [B|1,H|1,N,N] b_φ(i,j)
    node_mask: Tensor  # [B,N]


def shortest_path(neighbors: Tensor, hops: int) -> Tensor:
    """Shortest-path distance ``φ(i,j)`` (Graphormer Eq. 6), capped at ``hops + 1`` for far or unreachable pairs."""
    n = neighbors.shape[-1]
    step = neighbors.float()

    reached = torch.eye(n, dtype=torch.bool, device=neighbors.device).expand_as(neighbors)
    spd = torch.full_like(neighbors, hops + 1, dtype=torch.long).masked_fill(reached, 0)

    for k in range(1, hops + 1):
        frontier = (reached.float() @ step > 0) & ~reached
        spd = spd.masked_fill(frontier, k)
        reached = reached | frontier

    return spd


class GraphLayer(nn.Module):
    """One message-passing layer (Graphormer §2).

        a_i = AGGREGATE({h_j : j ∈ N(i)})
        h_i ← COMBINE(h_i, a_i)
    """

    def __init__(self, aggregate: nn.Module, combine: nn.Module) -> None:
        super().__init__()
        self.aggregate, self.combine = aggregate, combine

    def forward(self, h: Tensor, s: GraphStructure) -> Tensor:
        a = self.aggregate(self.combine.prepare(h), s)

        h = self.combine(h, a)

        return h * s.node_mask.unsqueeze(-1)


class GraphEncoder(TeiaNode):
    """``in: [feat.<nodes> [B,N,D], batch.adj [B,N,N], batch.node_mask [B,N]]``  ``out: feat.<graph> [B,N,D]``

        h⁽⁰⁾_i = x_i + z_deg(i)
        h⁽ˡ⁾_i = COMBINE(h⁽ˡ⁻¹⁾_i, AGGREGATE({h⁽ˡ⁻¹⁾_j : j ∈ N(i)}))

    Args:
        aggregate: ``dot`` (Graphormer), ``gatv2``, ``mean`` (GraphSAGE), ``sum`` (GIN).
        combine: ``residual`` (transformer), ``concat`` (GraphSAGE), ``gin``.
        norm_order: Layer norm placement of the ``residual`` combine, ``post`` or ``pre``.
        normalize: L2 normalisation of the ``concat`` combine output. ``concat`` only.
        scope: Nodes attention reads, ``all`` or ``neighbors``.
        heads: Attention heads.
        num_layers: Layers ``L``.
        ffn_ratio: Hidden width of the FFN or GIN MLP, in multiples of ``D``.
        dropout: On attention weights and residual branches.
        spatial_hops: Hops with a learned bias ``b_φ``, ``0`` disables. ``dot`` only.
        max_degree: Degree cap of ``z_deg``, ``0`` disables.
    """

    def build_module(
        self,
        aggregate: str = "dot",
        combine: str = "residual",
        norm_order: str = "post",
        normalize: bool = True,
        scope: str = "all",
        heads: int = 4,
        num_layers: int = 2,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        spatial_hops: int = 0,
        max_degree: int = 0,
        **kwargs: Any,
    ) -> None:
        _check("aggregate", aggregate, AGGREGATE)
        _check("combine", combine, COMBINE)
        _check("norm_order", norm_order, ("post", "pre"))
        _check("scope", scope, ("all", "neighbors"))
        if scope == "all" and aggregate not in ATTENTION:
            raise ValueError(f"GraphEncoder.scope='all' needs an attention aggregate {ATTENTION}; got {aggregate!r}.")
        if not normalize and combine != "concat":
            raise ValueError(f"GraphEncoder.normalize=False needs combine='concat'; got {combine!r}.")
        if spatial_hops and aggregate != "dot":
            raise ValueError(f"GraphEncoder.spatial_hops needs aggregate='dot'; got {aggregate!r}.")

        dim = int(self.in_shape[0][-1])
        self.scope, self.spatial_hops, self.max_degree = scope, int(spatial_hops), int(max_degree)

        self.centrality = nn.Embedding(self.max_degree + 1, dim) if self.max_degree else None
        self.spatial = nn.Embedding(self.spatial_hops + 2, int(heads)) if self.spatial_hops else None

        self.layers = nn.ModuleList(
            GraphLayer(
                AGGREGATE[aggregate](dim=dim, heads=int(heads), dropout=float(dropout)),
                COMBINE[combine](dim=dim, ffn_ratio=int(ffn_ratio), dropout=float(dropout), norm_order=norm_order, normalize=normalize),
            )
            for _ in range(int(num_layers))
        )

    def structure(self, adj: Tensor, node_mask: Tensor) -> GraphStructure:
        n = adj.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=adj.device)
        real = node_mask[:, :, None] & node_mask[:, None, :]

        # N(i): connected real nodes, self excluded
        neighbors = (adj > 0) & real & ~eye

        # keys attention reads: the scope, plus i itself so no row is empty
        attend = (real if self.scope == "all" else neighbors) | eye

        # spatial encoding b_φ(i,j), learned per head and hop count
        bias = (
            self.spatial(shortest_path(neighbors, self.spatial_hops)).permute(0, 3, 1, 2)
            if self.spatial is not None
            else adj.new_zeros(1, 1, n, n)
        )

        return GraphStructure(neighbors, attend, bias, node_mask)

    def forward(self, nodes: Tensor, adj: Tensor, node_mask: Tensor) -> Tensor:
        # graph structure, shared by every layer
        s = self.structure(adj, node_mask.bool())

        # h⁽⁰⁾ = x + z_deg: centrality encoding, padded rows zeroed
        h = nodes
        if self.centrality is not None:
            h = h + self.centrality(s.neighbors.sum(-1).clamp(max=self.max_degree))
        h = h * s.node_mask.unsqueeze(-1)

        # h⁽ˡ⁾ = COMBINE(h⁽ˡ⁻¹⁾, AGGREGATE(neighbours of h⁽ˡ⁻¹⁾))
        for layer in self.layers:
            h = layer(h, s)

        return h


def _check(name: str, value: str, options: Any) -> None:
    if value not in options:
            raise ValueError(f"GraphEncoder.{name} must be one of {sorted(options)}; got {value!r}.")
