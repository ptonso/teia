"""Single-label audio classification (waveform) task contract.

Source: common knowledge

Description:
  A raw waveform sequence in; one class per clip out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import SingleLabel, Tensor


class AudioWaveCls(SingleLabel):
    """``batch.waveform`` → ``capture.cls.{scores,label}``, ground truth ``batch.cls``."""

    batch = {"waveform": Tensor("B *"), "cls": Tensor("B", "int64", target=True)}
