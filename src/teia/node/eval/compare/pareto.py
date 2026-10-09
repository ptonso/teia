"""
``pareto`` comparison: the non-dominated frontier between two metrics across source runs.

Source: common knowledge

Description:
  A run is Pareto-optimal when no other run is at least as good on both metrics and strictly
  better on one. Direction per metric is set by ``minimize_x``/``minimize_y`` (default ``False``,
  higher is better, matching every accuracy/AP/F1-shaped metric; set ``True`` for a cost/error
  metric like ``rmse``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from teia.base.eval import Comparison, EvalSource
from teia.node.eval.compare._shared import flat_metrics, run_id
from teia.node.eval.view._shared import slugify
from teia.node.eval.view.scatter_plot import ScatterPlot

__all__ = ["Pareto"]


class Pareto(Comparison):
    """Config (``__init__``):

    - ``x_metric``/``y_metric``: ``eval/summary.yaml`` keys (``"det.ap"``,
      ``"det.map_50_95"``).
    - ``minimize_x``/``minimize_y``: direction per axis (default ``False``, higher is better).
    - ``title``/``filename``: default to ``"<y_metric> vs <x_metric> (Pareto)"`` / a slug of that.
    """

    def __init__(
        self,
        *,
        x_metric: str,
        y_metric: str,
        minimize_x: bool = False,
        minimize_y: bool = False,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        self.x_metric = x_metric
        self.y_metric = y_metric
        self.minimize_x = minimize_x
        self.minimize_y = minimize_y
        self._title = title or f"{y_metric} vs {x_metric} (Pareto)"
        self._filename = filename or f"{slugify(self._title)}.csv"

    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        points: list[dict[str, Any]] = []
        for source in sources:
            metrics = flat_metrics(source)
            x_value, y_value = metrics.get(self.x_metric), metrics.get(self.y_metric)
            if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
                continue
            points.append({"run_id": run_id(source), "x": float(x_value), "y": float(y_value)})

        frontier = _pareto_frontier(points, minimize_x=self.minimize_x, minimize_y=self.minimize_y)
        frontier_ids = {point["run_id"] for point in frontier}

        csv_path = out_dir / self._filename
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", self.x_metric, self.y_metric, "pareto_optimal"])
            writer.writeheader()
            for point in points:
                writer.writerow(
                    {
                        "run_id": point["run_id"],
                        self.x_metric: point["x"],
                        self.y_metric: point["y"],
                        "pareto_optimal": point["run_id"] in frontier_ids,
                    }
                )

        written = [csv_path]
        if points:
            plot = ScatterPlot(title=self._title, x_label=self.x_metric, y_label=self.y_metric, filename=f"{Path(self._filename).stem}.png")
            written.extend(
                plot.render(
                    out_dir,
                    x=[p["x"] for p in points],
                    y=[p["y"] for p in points],
                    labels=[p["run_id"] for p in points],
                )
            )
        return written


def _pareto_frontier(points: list[dict[str, Any]], *, minimize_x: bool, minimize_y: bool) -> list[dict[str, Any]]:
    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ax, ay = (a["x"], a["y"])
        bx, by = (b["x"], b["y"])
        if minimize_x:
            ax, bx = -ax, -bx
        if minimize_y:
            ay, by = -ay, -by
        at_least_as_good = ax >= bx and ay >= by
        strictly_better = ax > bx or ay > by
        return at_least_as_good and strictly_better

    return [point for point in points if not any(dominates(other, point) for other in points if other is not point)]
