"""
``threshold_curve``: one or more named series plotted against a shared x axis, typically a.

Source: common knowledge

Description:
  decision-threshold sweep. Not paginated.

Examples this view serves: precision/recall/F1 vs confidence threshold, micro/macro F1 vs
shared label-score threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _apply_report_style,
    _display_label,
    _wrap_label,
    plt,
    slugify,
    write_description_sidecar,
)


class ThresholdCurve(View):
    """Draws one or more labeled series against a shared x axis.

    Config (``__init__``): ``title``, ``x_label``, ``y_label``,
    ``bounded``/``bounded_x``/``xlim``/``ylim``/``figsize`` (styling overrides), ``filename``,
    ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``x`` (shared x values), ``series`` (list of
    ``{"label": str, "y": list[float]}`` dicts).
    """

    def __init__(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        bounded: bool = False,
        bounded_x: bool = False,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        filename: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        blindspot: str | None = None,
    ) -> None:
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.bounded = bounded
        self.bounded_x = bounded_x
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
        self.filename = filename or f"{slugify(title)}.png"
        self.description = description
        self.interpretation = interpretation
        self.blindspot = blindspot

    def render(self, out_dir: Path, *, x: list[float], series: list[dict[str, Any]]) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / self.filename

        fig, ax = plt.subplots(figsize=self.figsize or (8, 5))
        for one_series in series:
            ax.plot(x, one_series.get("y", []), label=_display_label(one_series.get("label"), width=30, max_lines=3))
        ax.set_xlabel(str(self.x_label))
        ax.set_ylabel(str(self.y_label))
        if series:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        _apply_report_style(fig, ax, bounded=self.bounded, bounded_x=self.bounded_x, xlim=self.xlim, ylim=self.ylim)
        ax.set_title(_wrap_label(self.title, width=64, max_lines=2))
        fig.tight_layout()
        fig.savefig(dst, dpi=160, bbox_inches="tight")
        plt.close(fig)

        write_description_sidecar(
            out_dir,
            Path(self.filename).stem,
            title=self.title,
            description=self.description,
            interpretation=self.interpretation,
            blindspot=self.blindspot,
        )
        return [dst]
