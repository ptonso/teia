"""
KmEstimate metric: Kaplan-Meier curves split by median risk.

Source:
  - title: "Nonparametric Estimation from Incomplete Observations"
    url: "https://doi.org/10.1080/01621459.1958.10501452"
    year: 1958

Description:
  Splits samples into high/low risk groups at the median risk score and returns one survival step
  curve per group, in the ``{label, x, y}`` shape a ``MultiCurveBundle`` view consumes.

Adaptations:
  - This is NOT the Kaplan-Meier product-limit estimator its name suggests. It ignores censoring
    entirely: it takes no ``event`` indicator and returns the plain empirical survival function
    ``1 - i/n`` over sorted times, whereas Kaplan-Meier accumulates ``prod(1 - d_i/n_i)`` over
    observed event times only. With censored data the two disagree, and this one is biased low.
    :mod:`.concordance` has the censoring-aware estimator used for the IPCW c-index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric

__all__ = ["KmEstimate"]


class KmEstimate(Metric):
    """Median-risk-split KM curves. No constructor kwargs."""

    def from_source(self, source: Any, *, time: Any, risk: Any, **_: Any) -> dict[str, Any]:
        del source
        time_arr = np.asarray(time, dtype=float).reshape(-1)
        risk_arr = np.asarray(risk, dtype=float).reshape(-1)
        if time_arr.size == 0:
            return {"n_samples": 0, "curves": []}

        median = float(np.median(risk_arr))
        curves: list[dict[str, Any]] = []
        for label, mask in (("high risk", risk_arr >= median), ("low risk", risk_arr < median)):
            group = np.sort(time_arr[mask])
            if group.size == 0:
                continue
            surv = 1.0 - np.arange(1, group.size + 1) / group.size
            curves.append({"label": label, "x": group.tolist(), "y": surv.tolist()})

        return {"n_samples": int(time_arr.size), "curves": curves}
