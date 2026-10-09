"""Single-label image classification task contract.

Source: common knowledge

Description:
  An RGB image in; one class per image out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import IMAGE, SingleLabel, Tensor


class VisionCls(SingleLabel):
    """``batch.image`` → ``capture.cls.{scores,label}``, ground truth ``batch.cls``."""

    batch = {**IMAGE, "cls": Tensor("B", "int64", target=True)}
