"""Single-label audio classification (spectrogram) task contract.

Source: common knowledge

Description:
  A mel-spectrogram grid in; one class per clip out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import SingleLabel, Tensor


class AudioCls(SingleLabel):
    """``batch.audio`` → ``capture.cls.{scores,label}``, ground truth ``batch.cls``."""

    batch = {"audio": Tensor("B *"), "cls": Tensor("B", "int64", target=True)}
