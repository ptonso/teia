"""Tabular multi-label classification task contract.

Source: common knowledge

Description:
  The typed tabular slots in; a multi-hot label vector out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import TABULAR, MultiLabel, Tensor


class TabularMultiCls(MultiLabel):
    """Tabular slots → ``capture.cls.{scores,labels}``, ground truth multi-hot ``batch.cls``."""

    batch = {**TABULAR, "cls": Tensor("B num_classes", target=True)}
