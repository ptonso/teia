"""
``grouped_histogram``: overlaid histograms, one per named group. Not paginated.

Source: common knowledge

Description:
  Examples this view serves: confidence histogram for correct vs incorrect predictions, IoU
  histogram for matched detections, angle-error histogram for matched oriented boxes.
"""

from __future__ import annotations

from pathlib import Path

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _apply_report_style,
    _wrap_label,
    plt,
    slugify,
    write_description_sidecar,
)


class GroupedHistogram(View):
    """Draws overlaid histograms for each named group in one axes.

    Config (``__init__``): ``title``, ``x_label``, ``y_label`` (default "Count"), ``bins``,
    ``bounded``/``bounded_x``/``xlim``/``ylim``/``figsize`` (styling overrides), ``filename``,
    ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``groups`` (``dict[str, list[float]]``, one value list per
    group name).
    """

    def __init__(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str = "Count",
        bins: int = 20,
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
        self.bins = bins
        self.bounded = bounded
        self.bounded_x = bounded_x
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
        self.filename = filename or f"{slugify(title)}.png"
        self.description = description
        self.interpretation = interpretation
        self.blindspot = blindspot

    def render(self, out_dir: Path, *, groups: dict[str, list[float]]) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / self.filename

        fig, ax = plt.subplots(figsize=self.figsize or (8, 5))
        for name, values in (groups or {}).items():
            ax.hist(values, bins=self.bins, alpha=0.6, label=name)
        ax.set_xlabel(str(self.x_label))
        ax.set_ylabel(str(self.y_label))
        if groups:
            ax.legend(fontsize=8)

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
