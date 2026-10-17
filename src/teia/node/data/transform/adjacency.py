"""Board adjacency transforms — turn an edge list or box geometry into a per-board adjacency matrix.

Source: common knowledge

Description:
  Both nodes emit ``item.adj[N,N]`` bool, symmetric and with self-loops, so the graph collate and every
  graph encoder can rely on that form. ``EdgeAdjacency`` builds it from an annotated edge list, and
  ``KnnAdjacency`` from box centres, the stand-in graph when no edge annotations exist and the control
  a wiring graph has to beat.
"""

from __future__ import annotations

import torch

from teia.base.data import Transform


class EdgeAdjacency(Transform):
    """``(item.edges[E,2] node indices, item.boxes[N,4]) → item.adj[N,N]`` bool, symmetric, self-loops."""

    def __call__(self, edges: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        n = boxes.shape[0]
        adj = torch.eye(n, dtype=torch.bool)
        adj[edges[:, 0], edges[:, 1]] = True
        adj[edges[:, 1], edges[:, 0]] = True
        return adj


class KnnAdjacency(Transform):
    """``item.boxes[N,4] → item.adj[N,N]`` bool: each node linked to its ``k`` nearest centres, symmetric, self-loops."""

    def __init__(self, *, k: int = 4) -> None:
        self.k = int(k)

    def __call__(self, boxes: torch.Tensor) -> torch.Tensor:
        n = boxes.shape[0]
        centres = boxes[:, :2] + boxes[:, 2:] / 2
        dist = torch.cdist(centres, centres)
        dist.fill_diagonal_(float("inf"))
        nearest = dist.topk(min(self.k, n - 1), largest=False).indices if n > 1 else torch.zeros((n, 0), dtype=torch.int64)
        adj = torch.eye(n, dtype=torch.bool)
        adj[torch.arange(n)[:, None], nearest] = True
        return adj | adj.T
