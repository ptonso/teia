"""
``factor_diff`` comparison: attach whichever config override actually varies across the.

Source: common knowledge

Description:
  source runs, derived by diffing ``config/overrides.yaml`` (``EvalSource.overrides()``) rather
  than a hardcoded key, as a column next to every run's metrics.

Zero-config by default: with no ``metric`` given, the CSV carries every metric from
:class:`~.metric_table.MetricTable` plus one extra ``factor``/``factor_value`` column, which is
what a comparison run with an unknown objective needs. Pass ``metric`` to narrow to one column
and also get a scatter plot of that metric against the derived factor (falls back to run index
on the x-axis when the factor's values are not numeric).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from teia.base.eval import Comparison, EvalSource
from teia.node.eval.compare._shared import SEED_KEYS, flat_metrics, override_map, run_id
from teia.node.eval.view._shared import slugify
from teia.node.eval.view.scatter_plot import ScatterPlot

__all__ = ["FactorDiff"]


class FactorDiff(Comparison):
    """Config (``__init__``):

    - ``metric``: the ``eval/summary.yaml`` key to isolate and plot
      (``"cls.accuracy"``). Default ``None``: every metric is carried instead, no plot.
    - ``factor``: an explicit override key to use as the varying axis. If ``None`` (default), it
      is derived as the one override key whose value differs across the sources. Seed-like keys
      are ignored unless nothing else varies; ties among the rest are broken lexicographically
      (more than one varying key means this run set is not a clean single-factor sweep, but the
      CSV still records every source's value).
    - ``title``/``filename``: default to ``"<metric> vs <factor>"`` (or ``"metrics vs <factor>"``
      with no ``metric``) / a slug of that title.
    """

    def __init__(self, *, metric: str | None = None, factor: str | None = None, title: str | None = None, filename: str | None = None) -> None:
        self.metric = metric
        self.factor = factor
        self._title = title
        self._filename = filename

    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        overrides_by_source = [override_map(source.overrides()) for source in sources]
        factor_key = self.factor or _derive_varying_key(overrides_by_source)
        label = self.metric or "metrics"
        title = self._title or f"{label} vs {factor_key or 'run'}"
        filename = self._filename or f"{slugify(title)}.csv"

        if self.metric is not None:
            written = self._write_single_metric(sources, overrides_by_source, out_dir, factor_key=factor_key, title=title, filename=filename)
        else:
            written = [self._write_all_metrics(sources, overrides_by_source, out_dir, factor_key=factor_key, filename=filename)]
        return written

    def _write_all_metrics(
        self, sources: list[EvalSource], overrides_by_source: list[dict[str, str]], out_dir: Path, *, factor_key: str | None, filename: str
    ) -> Path:
        rows: list[dict[str, Any]] = []
        columns = ["run_id", "factor_key", "factor_value"]
        seen = set(columns)
        for source, overrides in zip(sources, overrides_by_source, strict=True):
            row: dict[str, Any] = {
                "run_id": run_id(source),
                "factor_key": factor_key or "",
                "factor_value": overrides.get(factor_key, "") if factor_key else "",
                **flat_metrics(source),
            }
            for column in row:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
            rows.append(row)

        csv_path = out_dir / filename
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        return csv_path

    def _write_single_metric(
        self,
        sources: list[EvalSource],
        overrides_by_source: list[dict[str, str]],
        out_dir: Path,
        *,
        factor_key: str | None,
        title: str,
        filename: str,
    ) -> list[Path]:
        rows: list[dict[str, Any]] = []
        for source, overrides in zip(sources, overrides_by_source, strict=True):
            rows.append(
                {
                    "run_id": run_id(source),
                    "factor": overrides.get(factor_key, "") if factor_key else "",
                    "metric": flat_metrics(source).get(self.metric, ""),
                }
            )

        csv_path = out_dir / filename
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "factor", "metric"])
            writer.writeheader()
            writer.writerows(rows)

        written = [csv_path]
        xs, ys, labels = [], [], []
        for index, row in enumerate(rows):
            if row["metric"] == "":
                continue
            try:
                ys.append(float(row["metric"]))
            except (TypeError, ValueError):
                continue
            try:
                xs.append(float(row["factor"]))
            except (TypeError, ValueError):
                xs.append(float(index))
            labels.append(row["run_id"])
        if xs:
            plot = ScatterPlot(title=title, x_label=str(factor_key or "run"), y_label=str(self.metric), filename=f"{Path(filename).stem}.png")
            written.extend(plot.render(out_dir, x=xs, y=ys, labels=labels))
        return written


def _derive_varying_key(overrides_by_source: list[dict[str, str]]) -> str | None:
    all_keys: set[str] = set()
    for overrides in overrides_by_source:
        all_keys.update(overrides.keys())

    def varying_among(keys: set[str]) -> list[str]:
        return sorted(
            key for key in keys if len({overrides.get(key) for overrides in overrides_by_source if key in overrides}) > 1
        )

    non_seed = varying_among(all_keys - SEED_KEYS)
    if non_seed:
        return non_seed[0]
    seed_only = varying_among(all_keys & SEED_KEYS)
    return seed_only[0] if seed_only else None
