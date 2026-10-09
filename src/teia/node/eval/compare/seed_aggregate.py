"""
``seed_aggregate`` comparison: mean and std of every metric across runs that share every.

Source: common knowledge

Description:
  override except ``runtime.seed``, the "same config, different seed" sweep.

The grouping key is every other override (``config/overrides.yaml``, via
``EvalSource.overrides()``), not the composed config: two runs' composed configs differ in
thousands of resolved defaults, but their override lists are exactly what was asked to vary.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from teia.base.eval import Comparison, EvalSource
from teia.node.eval.compare._shared import SEED_KEYS, flat_metrics, override_map, run_id

__all__ = ["SeedAggregate"]


class SeedAggregate(Comparison):
    """Config (``__init__``): ``filename`` (default ``seed_aggregate.csv``)."""

    def __init__(self, *, filename: str = "seed_aggregate.csv") -> None:
        self.filename = filename

    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        groups: dict[tuple[tuple[str, str], ...], list[EvalSource]] = {}
        for source in sources:
            overrides = {k: v for k, v in override_map(source.overrides()).items() if k not in SEED_KEYS}
            key = tuple(sorted(overrides.items()))
            groups.setdefault(key, []).append(source)

        rows: list[dict[str, Any]] = []
        columns: list[str] = ["group", "n_seeds", "run_ids"]
        seen_columns = set(columns)
        for key, group_sources in groups.items():
            row: dict[str, Any] = {
                "group": ", ".join(f"{k}={v}" for k, v in key) or "(no non-seed overrides)",
                "n_seeds": len(group_sources),
                "run_ids": ";".join(run_id(source) for source in group_sources),
            }
            per_metric: dict[str, list[float]] = {}
            for source in group_sources:
                for metric_key, value in flat_metrics(source).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        per_metric.setdefault(metric_key, []).append(float(value))
            for metric_key, values in per_metric.items():
                mean_column, std_column = f"{metric_key}.mean", f"{metric_key}.std"
                row[mean_column] = round(statistics.fmean(values), 6)
                row[std_column] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
                for column in (mean_column, std_column):
                    if column not in seen_columns:
                        seen_columns.add(column)
                        columns.append(column)
            rows.append(row)

        dst = out_dir / self.filename
        with dst.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        return [dst]
