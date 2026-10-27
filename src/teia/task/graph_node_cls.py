"""Graph node classification task contract.

Source: common knowledge

Description:
  Graphs with a feature vector per node in; one class per node out. A batch pads what the graph layers read
  to the largest graph of the batch (``N``), with ``node_mask`` marking real nodes, and flat-concatenates the
  classes of every node (``M = sum N_i`` rows, ``batch_idx`` names the graph). Captured predictions and the
  ground truth ``cls`` are one row per real node, in ``node_mask`` row-major order. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import Ragged, SingleLabel, Tensor


class GraphNodeCls(SingleLabel):
    """Node features and adjacency → ``capture.cls.{scores,label}`` per node, ground truth ``batch.cls``.

    ``capture.cls.*`` rows are nodes (the leading ``B`` of the inherited specs counts node rows here).
    """

    batch = {
        "node_feat": Tensor("B N F", "float32"),
        "adj": Tensor("B N N", "float32"),
        "node_mask": Tensor("B N", "bool"),
        "cls": Ragged("N", index="batch_idx", dtype="int64", target=True),
        "batch_idx": Ragged("N", index="batch_idx", dtype="int64", target=True),
    }
