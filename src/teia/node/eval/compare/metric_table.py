"""
``metric_table`` comparison: join every source run's ``eval/summary.yaml`` (written.

Source: common knowledge

Description:
  by the ``summary`` view) into one CSV, one row per run.

One column per summary key; a run missing a key leaves that cell blank.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from teia.base.eval import Comparison, EvalSource
from teia.node.eval.compare._shared import flat_metrics, run_id

__all__ = ["MetricTable"]


class MetricTable(Comparison):
    """Config (``__init__``): ``filename`` (default ``metric_table.csv``)."""

    def __init__(self, *, filename: str = "metric_table.csv") -> None:
        self.filename = filename

    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        columns: list[str] = ["run_id"]
        seen_columns = {"run_id"}
        for source in sources:
            row: dict[str, Any] = {"run_id": run_id(source), **flat_metrics(source)}
            for column in row:
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
