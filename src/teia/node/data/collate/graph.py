"""Graph collates — assemble a batch of boards into node tensors, dense adjacency and a node mask.

Contract: one sample is a board with ``N_i`` nodes. Node images are flat-concatenated over the batch
(``M = sum N_i`` rows), while everything the graph layers read is padded to the largest board of the
*batch* (``N = max N_i``), so ``N`` changes from batch to batch and no weight depends on it. ``node_mask``
is 1 for a real node and 0 for padding, and rows of ``cls`` follow the node order of ``node_mask``
(``logits[node_mask]`` lines up with ``cls``).

Registers ``node_image``, ``node_box``, ``adj`` and ``node_mask`` in ``teia.base.fields.FIELD_VOCAB`` on
import, and reuses ``cls`` and ``batch_idx``.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.fields import register_field
from teia.base.data import Collate

register_field("node_image", structure="grid", optional=False)
register_field("node_box", structure=None, optional=False)
register_field("adj", structure="graph", optional=False)
register_field("node_mask", structure=None, optional=False)


class StackNodeImages(Collate):
    """``item.node_image[Ni,3,S,S] → batch.node_image[M,3,S,S]`` (flat over the batch)."""

    field = "node_image"
    structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        return torch.cat(field_batch, 0), {}


class PadNodeBoxes(Collate):
    """``item.node_box[Ni,4] → batch.node_box[B,N,4]`` zero-padded."""

    field = "node_box"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        out = field_batch[0].new_zeros(len(field_batch), max(b.shape[0] for b in field_batch), field_batch[0].shape[1])
        for i, boxes in enumerate(field_batch):
            out[i, : boxes.shape[0]] = boxes
        return out, {}


class PadAdjacency(Collate):
    """``item.adj[Ni,Ni] bool → batch.adj[B,N,N]`` float32, zero-padded."""

    field = "adj"
    structure = "graph"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        n = max(a.shape[0] for a in field_batch)
        out = torch.zeros(len(field_batch), n, n)
        for i, adj in enumerate(field_batch):
            out[i, : adj.shape[0], : adj.shape[0]] = adj.float()
        return out, {}


class NodeMask(Collate):
    """``item.node_box[Ni,4] → batch.node_mask[B,N]`` bool, True for real nodes."""

    field = "node_mask"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        sizes = torch.tensor([b.shape[0] for b in field_batch])
        return torch.arange(int(sizes.max()))[None] < sizes[:, None], {}


class FlatNodeClasses(Collate):
    """``item.cls[Ni] int64 → batch.cls[M]`` (flat over the batch, node order)."""

    field = "cls"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        return torch.cat(field_batch, 0), {}


class NodeBatchIndex(Collate):
    """``item.node_box[Ni,4] → batch.batch_idx[M]`` int64, the board index of each node row."""

    field = "batch_idx"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        return torch.cat([torch.full((b.shape[0],), i, dtype=torch.int64) for i, b in enumerate(field_batch)]), {}
