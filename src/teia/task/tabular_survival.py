"""Tabular survival analysis task contract.

Source: common knowledge

Description:
  The typed tabular slots in; a monotone risk score per row out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import TABULAR, TaskContract, Tensor


class TabularSurvival(TaskContract):
    """Tabular slots → ``capture.survival.risk``, ground truth ``batch.{time,event}``."""

    batch = {**TABULAR, "time": Tensor("B *", target=True), "event": Tensor("B *", target=True)}
    capture = {"survival.risk": Tensor("B", "float32")}
