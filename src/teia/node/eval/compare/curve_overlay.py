"""
``curve_overlay`` comparison: one metric's logger curve, overlaid across runs, one line per.

Source: common knowledge

Description:
  source run from each run's own CSV logger (``EvalSource.curves()``).

Reuses :class:`~teia.node.eval.view.multi_curve_bundle.MultiCurveBundle` for the drawing: a
run-labeled curve is the same rendering geometry as a metric's per-class curve bundle, sourced
from N runs instead of N classes within one run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.base.eval import Comparison, EvalSource
from teia.node.eval.compare._shared import run_id
from teia.node.eval.view._shared import slugify
from teia.node.eval.view.multi_curve_bundle import MultiCurveBundle

__all__ = ["CurveOverlay"]


class CurveOverlay(Comparison):
    """Config (``__init__``):

    - ``metric``: the logger column to plot. Default ``"val/loss"``, the one monitor key every
      task publishes regardless of objective, so a zero-config comparison still gets a
      training-curve overlay. Pass an objective-specific key (``"val/macro_f1"``,
      ``"test/loss.det"``) to compare something else.
    - ``logger``: which logger's CSV to read (default ``"teia"``, this repo's CSV logger name).
    - ``step_key``: the x-axis column (default ``"step"``; ``"epoch"`` is the other common choice).
    - ``title``/``filename``: default to ``"<metric> across runs"`` / a slug of that title.
    """

    def __init__(
        self, *, metric: str = "val/loss", logger: str = "teia", step_key: str = "step", title: str | None = None, filename: str | None = None
    ) -> None:
        self.metric = metric
        self.logger = logger
        self.step_key = step_key
        self.title = title or f"{metric} across runs"
        self.filename = filename or f"{slugify(self.title)}.png"

    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]:
        curves: list[dict[str, Any]] = []
        for source in sources:
            rows = (source.curves() or {}).get(self.logger) or []
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                raw_step, raw_value = row.get(self.step_key), row.get(self.metric)
                if raw_step in (None, "") or raw_value in (None, ""):
                    continue
                try:
                    step, value = float(raw_step), float(raw_value)
                except ValueError:
                    continue
                xs.append(step)
                ys.append(value)
            if xs:
                curves.append({"label": run_id(source), "x": xs, "y": ys})

        view = MultiCurveBundle(title=self.title, x_label=self.step_key, y_label=self.metric, filename=self.filename)
        return view.render(Path(out_dir), curves=curves)
