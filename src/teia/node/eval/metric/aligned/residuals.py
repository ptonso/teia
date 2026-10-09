"""
Residuals metric: regression error breakdown.

Source: common knowledge

Description:
  Overall and per-target RMSE/MAE/bias/R2, plus flat actual/pred/residual arrays for a
  ``ScatterPlot`` or ``GroupedHistogram`` view.

Param naming: the ``reg`` objective's ``targets`` field maps to capture column ``batch.target``
(singular), so the literal atom is ``target``. This metric accepts both spellings, ``target``
primary and ``targets`` as an alias.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric
from teia.node.eval.metric.aligned._shared import to_rows

__all__ = ["Residuals"]


class Residuals(Metric):
    """Overall + per-target regression error. Constructor kwarg: ``target_names`` (optional
    fallback names used when the wired class-name list is shorter than the target count)."""

    #: No ``"fitness"`` alias: a pipeline wanting early stopping wires ``monitor: val/r2``
    #: (mode max) or ``val/rmse``/``val/mae`` (mode min) directly, like any other named metric.
    def __init__(self, target_names: list[str] | None = None) -> None:
        self.target_names = list(target_names) if target_names else None
        self.MONITORS = ("rmse", "mae", "r2")
        self.reset()

    # Streaming: buffer raw per-sample kwargs across batches, replay from_source at compute time.

    def update(self, *, pred: Any = None, target: Any = None, targets: Any = None, target_names: list[str] | None = None, **_: Any) -> None:
        truth_raw = target if target is not None else targets
        if pred is not None:
            self._buf_pred.extend(to_rows(pred))
        if truth_raw is not None:
            self._buf_target.extend(to_rows(truth_raw))
        if target_names:
            self._target_names = list(target_names)

    def compute(self) -> dict[str, Any]:
        if not self._buf_pred:
            return {"n_samples": 0}
        return self.from_source(None, pred=self._buf_pred, target=self._buf_target, target_names=self._target_names)

    def reset(self) -> None:
        self._buf_pred: list[Any] = []
        self._buf_target: list[Any] = []
        self._target_names: list[str] | None = None

    def from_source(
        self,
        source: Any,
        *,
        pred: Any,
        target: Any = None,
        targets: Any = None,
        target_names: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        del source
        truth_raw = target if target is not None else targets
        if truth_raw is None:
            raise ValueError("Residuals.from_source requires a ground-truth atom wired as 'target' (or 'targets').")
        names = list(target_names) if target_names else (self.target_names or [])

        preds = np.asarray([[float(v) for v in row] for row in pred], dtype=float)
        tgts = np.asarray([[float(v) for v in row] for row in truth_raw], dtype=float)
        if preds.size == 0:
            return {"n_samples": 0}
        preds = preds.reshape(preds.shape[0], -1)
        tgts = tgts.reshape(tgts.shape[0], -1)
        err = preds - tgts

        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((tgts - tgts.mean(axis=0)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot

        n_targets = preds.shape[1]
        resolved_names = names if len(names) >= n_targets else [f"target_{index}" for index in range(n_targets)]
        metrics = {"n_samples": int(preds.shape[0]), "rmse": round(rmse, 6), "mae": round(mae, 6), "bias": round(bias, 6), "r2": round(r2, 6)}

        target_rows = []
        for index in range(n_targets):
            column_err = err[:, index]
            target_rows.append(
                {
                    "target": resolved_names[index],
                    "rmse": round(float(np.sqrt(np.mean(column_err**2))), 6),
                    "mae": round(float(np.mean(np.abs(column_err))), 6),
                    "bias": round(float(np.mean(column_err)), 6),
                }
            )

        flat_actual = tgts.reshape(-1).tolist()
        return {
            **metrics,
            "target_names": resolved_names,
            "target_rows": target_rows,
            "actual": flat_actual,
            "pred": preds.reshape(-1).tolist(),
            "residual": err.reshape(-1).tolist(),
            # A regression run has no natural per-point identity, so each flattened point is
            # labeled by its position.
            "index_labels": [str(index) for index in range(len(flat_actual))],
            # Pre-shaped for GroupedHistogram's `groups: dict[str, list[float]]` kwarg.
            "residual_groups": {"residual": err.reshape(-1).tolist()},
        }
