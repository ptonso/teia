"""Graph node classification task contract.

Source: common knowledge

Description:
  Boards, each a graph of boxed components, in; one class per node out. A batch flat-concatenates the crops of
  every node (``M = sum N_i`` rows, ``batch_idx`` names the board) and pads what the graph layers read to the
  largest board of the batch (``N``), with ``node_mask`` marking real nodes. Captured predictions and the
  ground truth ``cls`` are one row per real node, in ``node_mask`` row-major order. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import Ragged, SingleLabel, Tensor


class GraphNodeCls(SingleLabel):
    """Node crops, box geometry and adjacency → ``capture.cls.{scores,label}`` per node, ground truth ``batch.cls``.

    ``capture.cls.*`` rows are nodes (the leading ``B`` of the inherited specs counts node rows here).
    """

    batch = {
        "node_image": Ragged("N 3 h w", index="batch_idx", dtype="uint8"),
        "node_box": Tensor("B N 4", "float32"),
        "adj": Tensor("B N N", "float32"),
        "node_mask": Tensor("B N", "bool"),
        "cls": Ragged("N", index="batch_idx", dtype="int64", target=True),
        "batch_idx": Ragged("N", index="batch_idx", dtype="int64", target=True),
    }
