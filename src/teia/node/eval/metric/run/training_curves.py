"""
``training_curves`` metric: per-epoch loss/metric and learning-rate series from the run's CSV logger.

Source: common knowledge

Description:
  Reads ``EvalSource.curves()`` (the per-epoch ``metrics.csv``) and splits the numeric columns into
  a loss/metric family and a learning-rate family, each as ``{label, x, y}`` curve dicts a
  ``MultiCurveBundle`` view renders. Offline only — there is no streaming driver; the CSV does not
  exist until fit ends.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from teia.base.eval import Metric

__all__ = ["TrainingCurves"]

_METRIC_HINTS = ("loss", "metric", "map", "precision", "recall", "acc", "f1", "iou")


class TrainingCurves(Metric):
    """Constructor kwargs: ``logger`` (CSV logger name, default ``"teia"``; falls back to whichever
    logger has the most rows), ``step_key`` (x-axis column, default ``"epoch"``)."""

    #: ``lr_curves`` is only ever non-empty if a ``LearningRateMonitor`` callback actually ran
    #: during training and logged ``lr-*`` columns; without it this metric still runs (loss
    #: curves are unaffected) but silently produces an empty LR plot, so the requirement is
    #: declared here for the pre-flight callback gate rather than failing loudly on its own.
    REQUIRES_CALLBACK = ("lightning.pytorch.callbacks.LearningRateMonitor",)

    def __init__(self, logger: str = "teia", step_key: str = "epoch") -> None:
        self.logger = logger
        self.step_key = step_key

    def from_source(self, source: Any, **_: Any) -> dict[str, Any]:
        rows = self._rows(source)
        if not rows:
            return {"n_rows": 0, "loss_curves": [], "lr_curves": [], "series": []}
        columns = _numeric_columns(rows)
        x_key = self.step_key if self.step_key in columns else next(iter(columns), None)
        x = columns.get(x_key) if x_key else None
        if x_key == "epoch" and "step" in columns:
            x = _epoch_by_step(rows, epoch_key=x_key, step_key="step") or x

        loss_curves = [
            _curve(name, x, values)
            for name, values in columns.items()
            if name != x_key and any(hint in name.lower() for hint in _METRIC_HINTS) and not _is_lr(name)
        ]
        lr_curves = [_curve(name, x, values) for name, values in columns.items() if _is_lr(name)]
        return {
            "n_rows": len(rows),
            "loss_curves": loss_curves,
            "lr_curves": lr_curves,
            "series": [curve["label"] for curve in (*loss_curves, *lr_curves)],
        }

    def _rows(self, source: Any) -> list[dict[str, str]]:
        curves = source.curves() or {}
        if self.logger in curves:
            return curves[self.logger]
        return max(curves.values(), key=len) if curves else []


def _is_lr(name: str) -> bool:
    """The learning rate itself, not the ``LearningRateMonitor`` sibling series it also logs
    under the same ``lr-`` prefix (momentum, weight decay) — those live on a different scale and
    would otherwise squash the real LR curve flat on a shared axis."""
    lowered = name.lower()
    return lowered.startswith("lr-") and not lowered.endswith(("-momentum", "-weight_decay"))


def _parse_float(raw: Any) -> float:
    if raw in ("", "nan", "NaN", "None", None):
        return math.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def _numeric_columns(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    for key in rows[0]:
        values: list[float] = []
        for row in rows:
            raw = row.get(key, "")
            if raw not in ("", "nan", "NaN", "None", None):
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    break
            else:
                values.append(math.nan)
        else:
            columns[key] = values
    return columns


def _epoch_by_step(rows: list[dict[str, str]], *, epoch_key: str, step_key: str) -> list[float] | None:
    """Every row's epoch, inferred from ``step`` rather than read as-is.

    A CSV logger flushes different ``self.log`` calls into separate rows: only the row where an
    epoch-aggregated metric is flushed carries a non-empty ``epoch``, so a per-step series (LR,
    ``*_step`` losses) would otherwise fall back to its raw row position as x, unrelated to
    training progress. Steps are anchored to the epoch of the nearest known-epoch row at or after
    them (an epoch's flush lands on its last step), with trailing rows past the last anchor (e.g.
    a standalone test pass) carried forward from that last known epoch.
    """
    steps = [_parse_float(row.get(step_key, "")) for row in rows]
    epochs = [_parse_float(row.get(epoch_key, "")) for row in rows]
    anchors = sorted({(s, e) for s, e in zip(steps, epochs) if not math.isnan(s) and not math.isnan(e)})
    if not anchors or any(math.isnan(step) for step in steps):
        return None
    anchor_steps = [step for step, _ in anchors]
    filled: list[float] = []
    for step in steps:
        idx = bisect.bisect_left(anchor_steps, step)
        filled.append(anchors[idx][1] if idx < len(anchors) else anchors[-1][1])
    return filled


def _curve(name: str, x: list[float] | None, values: list[float]) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, float) and math.isnan(value):
            continue
        xs.append(float(x[index]) if x and index < len(x) and not math.isnan(x[index]) else float(index))
        ys.append(float(value))
    return {"label": name, "x": xs, "y": ys}
