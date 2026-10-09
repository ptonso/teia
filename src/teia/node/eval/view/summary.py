"""
``summary`` view: flattens every wired metric's scalar top-level fields into one.

Source: common knowledge

Description:
  ``eval/summary.yaml``, the flat scalar map cross-run comparison joins on
  (``metric_table`` reads exactly this file, one per objective).

Each ``in`` entry is wired to a whole metric result dict in list form; the trailing-atom kwarg
name is discarded, only the values matter. Only scalar fields (int/float/str/bool, never a
list, dict, or None) are kept; a metric's array/table/curve fields are what the other views in
the same bundle render, so summary.yaml does not duplicate them. A key present in more than one
wired metric (``n_samples``) takes the last-wired value, which is the same quantity by
construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.base.eval import View

__all__ = ["Summary"]

_SCALAR_TYPES = (int, float, str, bool)


class Summary(View):
    """Config (``__init__``): ``filename`` (default ``summary.yaml``).

    Data (``render(**inputs)``): each kwarg is one metric's whole result dict; only its scalar
    top-level fields are kept.
    """

    def __init__(self, *, filename: str = "summary.yaml") -> None:
        self.filename = filename

    def render(self, out_dir: Path, **inputs: Any) -> list[Path]:
        import yaml

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        flat: dict[str, Any] = {}
        for value in inputs.values():
            if not isinstance(value, dict):
                continue
            for key, field in value.items():
                if isinstance(field, _SCALAR_TYPES):
                    flat[key] = field

        dst = out_dir / self.filename
        dst.write_text(yaml.safe_dump(flat, sort_keys=False), encoding="utf-8")
        return [dst]
