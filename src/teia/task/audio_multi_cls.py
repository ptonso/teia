"""Multi-label audio classification task contract.

Source: common knowledge

Description:
  A mel-spectrogram grid in; a multi-hot class vector out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import MultiLabel, Tensor


class AudioMultiCls(MultiLabel):
    """``batch.audio`` → ``capture.cls.{scores,labels}``, ground truth multi-hot ``batch.cls``."""

    batch = {"audio": Tensor("B *"), "cls": Tensor("B num_classes", target=True)}
