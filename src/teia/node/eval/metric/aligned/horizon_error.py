"""
HorizonError metric: quantile-forecast error over a horizon.

Source: common knowledge

Description:
  Median-quantile selection, overall and per-horizon-step MAE/RMSE, and the per-horizon MAE curve
  array for a ``MultiCurveBundle`` or ``ThresholdCurve`` view.

Param naming: as in ``residuals.py``, the forecast objective's ``targets`` field maps to
capture column ``batch.target`` (singular); ``target`` and ``targets`` are both accepted.
``levels`` (the quantile-level array, ``[0.1, 0.5, 0.9]``) is a per-run constant, not a
per-sample column, so it is both a constructor default and an optional wired-in override.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric
from teia.node.eval.metric.aligned._shared import to_rows

__all__ = ["HorizonError"]


class HorizonError(Metric):
    """Median-quantile forecast error, overall and per horizon step.

    Constructor kwarg: ``levels``, the default quantile levels used when not overridden by a
    wired ``levels`` input (default ``[0.5]``).
    """

    #: No ``"fitness"`` alias; wire ``monitor: val/mae`` or ``val/rmse`` directly (see
    #: ``Residuals``'s identical note).
    def __init__(self, levels: list[float] | None = None) -> None:
        self.default_levels = list(levels) if levels else [0.5]
        self.MONITORS = ("mae", "rmse")
        self.reset()

    # Streaming: buffer raw per-sample kwargs across batches, replay from_source at compute time.

    def update(self, *, quantiles: Any = None, target: Any = None, targets: Any = None, levels: Any = None, **_: Any) -> None:
        truth_raw = target if target is not None else targets
        if quantiles is not None:
            self._buf_quantiles.extend(to_rows(quantiles))
        if truth_raw is not None:
            self._buf_target.extend(to_rows(truth_raw))
        if levels is not None and len(list(levels)) > 0:
            self._levels = list(levels)

    def compute(self) -> dict[str, Any]:
        if not self._buf_quantiles:
            return {"n_samples": 0}
        return self.from_source(None, quantiles=self._buf_quantiles, target=self._buf_target, levels=self._levels)

    def reset(self) -> None:
        self._buf_quantiles: list[Any] = []
        self._buf_target: list[Any] = []
        self._levels: list[float] | None = None

    def from_source(
        self,
        source: Any,
        *,
        quantiles: Any,
        target: Any = None,
        targets: Any = None,
        levels: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        del source
        truth_raw = target if target is not None else targets
        if truth_raw is None:
            raise ValueError("HorizonError.from_source requires a ground-truth atom wired as 'target' (or 'targets').")

        q = np.asarray(quantiles, dtype=float)
        if q.size == 0:
            return {"n_samples": 0}
        resolved_levels = list(levels) if levels is not None and len(list(levels)) > 0 else self.default_levels
        median_index = int(np.argmin(np.abs(np.asarray(resolved_levels, dtype=float) - 0.5)))
        median = q[..., median_index]
        tgt = np.asarray(truth_raw, dtype=float)
        median = median.reshape(median.shape[0], -1)
        tgt = tgt.reshape(tgt.shape[0], -1)
        err = median - tgt

        per_step = {
            f"step_{index}": {
                "mae": round(float(np.mean(np.abs(err[:, index]))), 6),
                "rmse": round(float(np.sqrt(np.mean(err[:, index] ** 2))), 6),
            }
            for index in range(err.shape[1])
        }
        overall = {"mae": round(float(np.mean(np.abs(err))), 6), "rmse": round(float(np.sqrt(np.mean(err**2))), 6)}
        mae_per_horizon = np.mean(np.abs(err), axis=0)

        return {
            "n_samples": int(median.shape[0]),
            "overall": overall,
            # Top-level duplicates of `overall`'s two keys: the streaming compute() path needs a
            # flat dict for MONITORS (see StreamingMonitorRunner.compute()).
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "per_step": per_step,
            # Pre-shaped for MultiCurveBundle's `curves: list[{"label","x","y"}]` kwarg.
            "horizon_mae_curves": [{"label": "MAE", "x": list(range(1, len(mae_per_horizon) + 1)), "y": mae_per_horizon.tolist()}],
        }
