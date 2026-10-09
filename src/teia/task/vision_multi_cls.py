"""Multi-label image classification task contract.

Source: common knowledge

Description:
  An RGB image in; a multi-hot class vector out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import IMAGE, MultiLabel, Tensor


class VisionMultiCls(MultiLabel):
    """``batch.image`` → ``capture.cls.{scores,labels}``, ground truth multi-hot ``batch.cls``."""

    batch = {**IMAGE, "cls": Tensor("B num_classes", target=True)}
