"""Vision graph node classification task contract.

Source: common knowledge

Description:
  A fused task: the targets and captures of ``GraphNodeCls``, with an image per node as a second input. A batch
  flat-concatenates the node images of every graph (``M = sum N_i`` rows, in ``node_mask`` row-major order, as
  ``cls``). See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import Ragged
from teia.task.graph_node_cls import GraphNodeCls


class VisionGraphNodeCls(GraphNodeCls):
    """Node features, node images and adjacency → ``capture.cls.{scores,label}`` per node, ground truth ``batch.cls``."""

    batch = {**GraphNodeCls.batch, "node_image": Ragged("N 3 h w", index="batch_idx", dtype="uint8")}
