"""Tabular forecasting task contract.

Source: common knowledge

Description:
  Windowed or lagged tabular slots in; quantile forecasts over the horizon out. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import TABULAR, TaskContract, Tensor


class TabularForecast(TaskContract):
    """Tabular slots → ``capture.forecast.quantiles``, ground truth ``batch.target``."""

    batch = {**TABULAR, "target": Tensor("B *", target=True)}
    capture = {"forecast.quantiles": Tensor("B *", "float32")}
