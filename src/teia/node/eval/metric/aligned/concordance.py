"""
Concordance metric: survival c-index and IPCW c-index.

Source:
  - title: "Evaluating the Yield of Medical Tests"
    url: "https://doi.org/10.1001/jama.1982.03320430047030"
    year: 1982
  - title: "On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data"
    url: "https://doi.org/10.1002/sim.4154"
    year: 2011
  - title: "Nonparametric Estimation from Incomplete Observations"
    url: "https://doi.org/10.1080/01621459.1958.10501452"
    year: 1958

Description:
  Censoring fit: the IPCW c-index needs a censoring distribution ``G(t)``. Ideally this is fit on
  the training split, since fitting it on the eval split is self-referential. The eval plane never
  has a live datamodule, so a train-split ``(time, event)`` pair reaches this metric only if a
  capture-time side-artifact provides it, wired in as the optional ``train_time``/``train_event``
  kwargs. When those are supplied they are used for censoring; when absent, censoring is fit on
  the eval split's own ``(time, event)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric
from teia.node.eval.metric.aligned._shared import to_rows

__all__ = ["Concordance"]


def _concordance_index(time: Any, risk: Any, event: Any) -> float:
    concordant = permissible = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible else float("nan")


def _censoring_survival(time: Any, event: Any) -> tuple[Any, Any]:
    times = np.sort(np.unique(time))
    survival = []
    current = 1.0
    for value in times:
        at_risk = float(np.count_nonzero(time >= value))
        censored = float(np.count_nonzero((time == value) & (event == 0)))
        current *= 1.0 - censored / at_risk
        survival.append(current)
    return times, np.array(survival, dtype=float)


def _step_value(times: Any, values: Any, query: Any) -> float:
    index = int(np.searchsorted(times, query, side="right") - 1)
    return 1.0 if index < 0 else float(values[index])


def _ipcw_concordance_index(time: Any, risk: Any, event: Any, censoring: tuple[Any, Any]) -> float:
    if not (len(time) == len(risk) == len(event)):
        raise ValueError("Survival metric arrays must have matching lengths.")
    if not (np.isfinite(time).all() and np.isfinite(risk).all()):
        raise ValueError("Survival metric arrays must be finite.")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Survival event indicators must be 0/1.")
    km_time, km_survival = censoring
    concordant = permissible = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        weight_base = _step_value(km_time, km_survival, time[i])
        if weight_base <= 0.0:
            continue
        weight = 1.0 / (weight_base * weight_base)
        for j in range(n):
            if time[j] <= time[i]:
                continue
            permissible += weight
            if risk[i] > risk[j]:
                concordant += weight
            elif risk[i] == risk[j]:
                concordant += 0.5 * weight
    return float(concordant / permissible) if permissible else float("nan")


class Concordance(Metric):
    """Harrell's c-index and IPCW c-index. No constructor kwargs.

    No ``"fitness"`` alias; wire ``monitor: val/c_index`` directly. C-index is O(n^2) in the
    split size, so the streaming driver buffers the whole split's ``(time, event, risk)`` and
    pays that cost once per val epoch, the same cost profile ``from_source`` has for the offline
    report.
    """

    def __init__(self) -> None:
        self.MONITORS = ("c_index", "ipcw_c_index")
        self.reset()

    # Streaming: buffer raw per-sample kwargs across batches, replay from_source at compute time.

    def update(
        self,
        *,
        time: Any = None,
        event: Any = None,
        risk: Any = None,
        train_time: Any = None,
        train_event: Any = None,
        **_: Any,
    ) -> None:
        if time is not None:
            self._buf_time.extend(to_rows(time))
        if event is not None:
            self._buf_event.extend(to_rows(event))
        if risk is not None:
            self._buf_risk.extend(to_rows(risk))
        if train_time is not None:
            self._buf_train_time.extend(to_rows(train_time))
        if train_event is not None:
            self._buf_train_event.extend(to_rows(train_event))

    def compute(self) -> dict[str, Any]:
        if not self._buf_time:
            return {"n_samples": 0}
        return self.from_source(
            None,
            time=self._buf_time,
            event=self._buf_event,
            risk=self._buf_risk,
            train_time=self._buf_train_time or None,
            train_event=self._buf_train_event or None,
        )

    def reset(self) -> None:
        self._buf_time: list[Any] = []
        self._buf_event: list[Any] = []
        self._buf_risk: list[Any] = []
        self._buf_train_time: list[Any] = []
        self._buf_train_event: list[Any] = []

    def from_source(
        self,
        source: Any,
        *,
        time: Any,
        event: Any,
        risk: Any,
        train_time: Any = None,
        train_event: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        del source
        time_arr = np.asarray(time, dtype=float).reshape(-1)
        event_arr = np.asarray(event, dtype=int).reshape(-1)
        risk_arr = np.asarray(risk, dtype=float).reshape(-1)
        if time_arr.size == 0:
            return {"n_samples": 0}

        if train_time is not None and train_event is not None:
            censoring_time = np.asarray(train_time, dtype=float).reshape(-1)
            censoring_event = np.asarray(train_event, dtype=int).reshape(-1)
        else:
            censoring_time, censoring_event = time_arr, event_arr
        censoring = _censoring_survival(censoring_time, censoring_event)

        c_index = _concordance_index(time_arr, risk_arr, event_arr)
        ipcw_c_index = _ipcw_concordance_index(time_arr, risk_arr, event_arr, censoring)
        return {
            "n_samples": int(len(time_arr)),
            "c_index": c_index,
            "ipcw_c_index": ipcw_c_index,
            # Pre-shaped for Table's `rows: list[dict]` kwarg.
            "rows": [{"metric": "c_index", "value": c_index}, {"metric": "ipcw_c_index", "value": ipcw_c_index}],
        }
