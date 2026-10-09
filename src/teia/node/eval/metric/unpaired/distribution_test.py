"""
DistributionTest: per-dimension two-sample Kolmogorov-Smirnov distance.

Source:
  - title: "Table for Estimating the Goodness of Fit of Empirical Distributions"
    url: "https://doi.org/10.1214/aoms/1177730256"
    year: 1948

Description:
  Unlike :class:`~.frechet_distance.FrechetDistance` and
  :class:`~.kernel_distance.KernelDistance`, which compare embeddings from a feature extractor,
  this metric compares raw numeric feature arrays directly, with no feature extractor. It is the
  general-purpose unpaired comparison: usable standalone on any per-sample numeric quantity, or
  downstream of an embedding step if a bundle wires one. It reports the KS statistic itself, not a
  p-value (see ``_shared.ks_statistic``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric
from teia.node.eval.metric.unpaired._shared import ks_statistic

__all__ = ["DistributionTest"]


class DistributionTest(Metric):
    """Constructor kwarg: ``feature_names`` (optional, labels the per-dimension results;
    defaults to ``feature_0``, ``feature_1``, ... when not given or too short)."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = list(feature_names) if feature_names else None
        self._pred: list[np.ndarray] = []
        self._real: list[np.ndarray] = []

    def update(self, *, pred: Any = None, real: Any = None, **_: Any) -> None:
        if pred is not None:
            self._pred.append(np.atleast_2d(np.asarray(pred, dtype=float)))
        if real is not None:
            self._real.append(np.atleast_2d(np.asarray(real, dtype=float)))

    def compute(self) -> dict[str, Any]:
        if not self._pred or not self._real:
            return {"n_pred": sum(len(a) for a in self._pred), "n_real": sum(len(a) for a in self._real)}
        pred = np.concatenate(self._pred, axis=0)
        real = np.concatenate(self._real, axis=0)
        n_features = pred.shape[1]
        names = self.feature_names if self.feature_names and len(self.feature_names) >= n_features else [f"feature_{i}" for i in range(n_features)]

        per_feature = [
            {"feature": names[index], "ks_statistic": round(ks_statistic(pred[:, index], real[:, index]), 6)}
            for index in range(n_features)
        ]
        return {
            "n_pred": int(pred.shape[0]),
            "n_real": int(real.shape[0]),
            "mean_ks_statistic": round(float(np.mean([row["ks_statistic"] for row in per_feature])), 6),
            "per_feature": per_feature,
        }

    def reset(self) -> None:
        self._pred = []
        self._real = []

    def from_source(self, source: Any, *, pred: Any = None, real: Any = None, **_: Any) -> dict[str, Any]:
        del source
        self.reset()
        self.update(pred=pred, real=real)
        return self.compute()
