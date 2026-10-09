"""Tabular classification task contract.

Source: common knowledge

Description:
  The typed tabular slots in; one class per row out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import TABULAR, SingleLabel, Tensor


class TabularCls(SingleLabel):
    """Tabular slots → ``capture.cls.{scores,label}``, ground truth ``batch.cls``."""

    batch = {**TABULAR, "cls": Tensor("B", "int64", target=True)}
