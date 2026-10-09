"""Tabular regression task contract.

Source: common knowledge

Description:
  The typed tabular slots in; a Gaussian mean and std per target out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import TABULAR, Dim, Names, TaskContract, Tensor


class TabularReg(TaskContract):
    """Tabular slots → ``capture.reg.{mean,std}``, ground truth ``batch.target``."""

    meta = {"num_targets": Dim(), "target_names": Names("num_targets")}
    batch = {**TABULAR, "target": Tensor("B num_targets", target=True)}
    capture = {"reg.mean": Tensor("B num_targets", "float32"), "reg.std": Tensor("B num_targets", "float32")}
