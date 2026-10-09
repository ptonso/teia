"""CSV writer: one row per inferred sample.

Source: common knowledge

Description:
  Writes ``predictions.csv`` with one column per ``in`` atom, named by the atom's last key
  segment (``capture.cls.label`` → ``label``). A vector atom spreads over ``<name>_<j>`` columns,
  named by class when ``meta.class_names`` is also an input, which also maps ``label`` indices to
  names. Rows follow the inferred split's order, so they join to the source by position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from teia.base.data import Writer


class CsvWriter(Writer):
    """``(capture.* atoms[, meta.class_names]) → dst/predictions.csv``."""

    def __init__(self, *, filename: str = "predictions.csv") -> None:
        self.filename = filename

    def __call__(self, *inputs: Any, dst: Path) -> list[Path]:
        import pandas as pd

        named = dict(zip(self.in_key, inputs))
        class_names = [str(name) for name in named.pop("meta.class_names", [])]
        columns: dict[str, Any] = {}
        for key, value in named.items():
            name = key.rsplit(".", 1)[-1]
            array = np.asarray(value)
            if array.ndim <= 1:
                columns[name] = [class_names[int(v)] for v in array] if name == "label" and class_names else array
                continue
            flat = array.reshape(array.shape[0], -1)
            labels = class_names if len(class_names) == flat.shape[1] else [str(j) for j in range(flat.shape[1])]
            columns.update({f"{name}_{label}": flat[:, j] for j, label in enumerate(labels)})
        path = Path(dst) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns).to_csv(path, index=False)
        return [path]


__all__ = ["CsvWriter"]
