"""
``summary`` view: flattens every wired metric's scalar top-level fields into one.

Source: common knowledge

Description:
  ``eval/summary.yaml``, the flat scalar map cross-run comparison joins on
  (``metric_table`` reads exactly this file, one per objective).

Each ``in`` entry is wired to a whole metric result dict in list form; its kwarg name is the trailing
atom of the ``in`` key (``eval.gen.rfid`` gives ``rfid``). Only scalar fields (int/float/str/bool,
never a list, dict, or None) are kept; a metric's array/table/curve fields are what the other views
in the same bundle render, so summary.yaml does not duplicate them. With ``namespace: true`` each
field is written as ``<atom>.<field>``, which lets two instances of one metric (an rFID and a gFID)
sit side by side. A key written twice with different values raises; the same value written twice
(``n_samples`` of metrics over one split) is kept once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.base.eval import View

__all__ = ["Summary"]

_SCALAR_TYPES = (int, float, str, bool)


class Summary(View):
    """Config (``__init__``): ``filename`` (default ``summary.yaml``) and ``namespace`` (prefix each
    field with its input's atom name).

    Data (``render(**inputs)``): each kwarg is one metric's whole result dict; only its scalar
    top-level fields are kept.
    """

    def __init__(self, *, filename: str = "summary.yaml", namespace: bool = False) -> None:
        self.filename = filename
        self.namespace = bool(namespace)

    def render(self, out_dir: Path, **inputs: Any) -> list[Path]:
        import yaml

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        flat: dict[str, Any] = {}
        for name, value in inputs.items():
            if not isinstance(value, dict):
                continue
            for key, field in value.items():
                if not isinstance(field, _SCALAR_TYPES):
                    continue
                key = f"{name}.{key}" if self.namespace else key
                if key in flat and flat[key] != field:
                    raise ValueError(f"summary: key {key!r} is {flat[key]!r} and {field!r} in two inputs; set `namespace: true`.")
                flat[key] = field

        dst = out_dir / self.filename
        dst.write_text(yaml.safe_dump(flat, sort_keys=False), encoding="utf-8")
        return [dst]
