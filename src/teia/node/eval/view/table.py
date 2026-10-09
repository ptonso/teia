"""
``table`` view: writes a row-oriented CSV table.

Source: common knowledge

Description:
  Unlike the eight plotting views, this one is not a matplotlib geometry. It terminates a metric
  whose natural output is tabular (per-sample worst-K rows with sample IDs and filenames, say).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from teia.base.eval import View
from teia.node.eval.view._shared import slugify, write_description_sidecar


class Table(View):
    """Writes ``rows`` to a CSV file.

    Config (``__init__``): ``columns`` (fixed column order; if ``None``, columns are inferred
    from the first-seen key order across all rows at render time), ``filename`` (defaults to
    ``.csv``), ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``rows`` (``list[dict]``).
    """

    def __init__(
        self,
        *,
        title: str,
        columns: list[str] | None = None,
        filename: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        blindspot: str | None = None,
    ) -> None:
        self.title = title
        self.columns = list(columns) if columns is not None else None
        self.filename = filename or f"{slugify(title)}.csv"
        self.description = description
        self.interpretation = interpretation
        self.blindspot = blindspot

    def render(self, out_dir: Path, *, rows: list[dict[str, Any]]) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / self.filename

        columns = self.columns if self.columns is not None else _infer_columns(rows)
        with dst.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})

        write_description_sidecar(
            out_dir,
            Path(self.filename).stem,
            title=self.title,
            description=self.description,
            interpretation=self.interpretation,
            blindspot=self.blindspot,
        )
        return [dst]


def _infer_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    inferred: list[str] = []
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            inferred.append(key)
    return inferred
