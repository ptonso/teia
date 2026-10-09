"""
``target_summary`` metric: a per-run sanity check over the captured ground-truth atoms.

Source: common knowledge

Description:
  Reads whichever ``batch.*`` target atoms the capture holds (``cls``, ``target``, ``time`` +
  ``event``) and emits one summary row — sample count, target dimensionality, per-dimension
  mean/std/min/max, or survival event rate. A ``Table`` view writes it to ``target_summary.csv``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric

__all__ = ["TargetSummary"]


class TargetSummary(Metric):
    """One summary row per captured ground-truth atom; no constructor kwargs, no streaming
    driver (offline only)."""

    def from_source(self, source: Any, **_: Any) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for atom in ("batch.cls", "batch.target"):
            if source.has(atom):
                rows.append(_numeric_row(atom.split(".")[1], np.asarray(source.column(atom), dtype=float)))
        if source.has("batch.time") and source.has("batch.event"):
            time = np.asarray(source.column("batch.time"), dtype=float).reshape(-1)
            event = np.asarray(source.column("batch.event"), dtype=float).reshape(-1)
            rows.append(
                {
                    "atom": "time+event",
                    "n_samples": int(time.size),
                    "time_mean": _round(time.mean()) if time.size else 0.0,
                    "event_rate": _round(event.mean()) if event.size else 0.0,
                }
            )
        return {"n_atoms": len(rows), "rows": rows}


def _numeric_row(name: str, array: np.ndarray) -> dict[str, Any]:
    if array.size == 0:
        return {"atom": name, "n_samples": 0}
    array = array.reshape(array.shape[0], -1)
    return {
        "atom": name,
        "n_samples": int(array.shape[0]),
        "n_dims": int(array.shape[1]),
        "mean": [_round(v) for v in array.mean(axis=0)],
        "std": [_round(v) for v in array.std(axis=0)],
        "min": [_round(v) for v in array.min(axis=0)],
        "max": [_round(v) for v in array.max(axis=0)],
    }


def _round(value: Any) -> float:
    return round(float(value), 6)
