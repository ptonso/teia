"""
``multi_curve_bundle``: many named (x, y) curves in one axes, with an optional diagonal.

Source: common knowledge

Description:
  baseline and a legend placed outside the plot area.

Examples this view serves: macro plus worst-K per-class PR curves, one-vs-all ROC curves per
label, Kaplan-Meier curves by risk group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _apply_report_style,
    _balanced_chunks,
    _display_label,
    _page_filename,
    _page_title,
    _wrap_label,
    plt,
    slugify,
    write_description_sidecar,
)

CURVE_PAGE_SIZE = 12


class MultiCurveBundle(View):
    """Draws every curve into one axes per page (paginates beyond ``CURVE_PAGE_SIZE`` curves;
    a curve labeled "macro", case-insensitive, is pinned onto every page).

    Config (``__init__``): ``title``, ``x_label``, ``y_label``, ``baseline`` (``"diagonal"`` to
    draw the ``y=x`` reference line, else ``None``),
    ``bounded``/``bounded_x``/``xlim``/``ylim``/``figsize`` (styling overrides), ``filename``,
    ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``curves`` (list of ``{"label": str, "x": list[float],
    "y": list[float]}`` dicts).
    """

    def __init__(
        self,
        *,
        title: str,
        x_label: str,
        y_label: str,
        baseline: str | None = None,
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
        self.baseline = baseline
        self.bounded = bounded
        self.bounded_x = bounded_x
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
        self.filename = filename or f"{slugify(title)}.png"
        self.description = description
        self.interpretation = interpretation
        self.blindspot = blindspot

    def render(self, out_dir: Path, *, curves: list[dict[str, Any]]) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        curves = list(curves)

        pages = self._paginate(curves)
        written: list[Path] = []
        for page_index, page_curves in enumerate(pages):
            filename = _page_filename(self.filename, page_index, len(pages))
            title = _page_title(self.title, page_index, len(pages))
            dst = out_dir / filename
            self._draw(dst, title, page_curves)
            written.append(dst)
            write_description_sidecar(
                out_dir,
                Path(filename).stem,
                title=title,
                description=self.description,
                interpretation=self.interpretation,
                blindspot=self.blindspot,
            )
        return written

    def _paginate(self, curves: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if len(curves) <= CURVE_PAGE_SIZE:
            return [curves]
        pinned = [curve for curve in curves if str(curve.get("label", "")).lower() == "macro"]
        rest = [curve for curve in curves if curve not in pinned]
        capacity = max(1, CURVE_PAGE_SIZE - len(pinned))
        chunks = _balanced_chunks(rest, capacity)
        return [pinned + chunk for chunk in chunks]

    def _draw(self, dst: Path, title: str, curves: list[dict[str, Any]]) -> None:
        fig, ax = plt.subplots(figsize=self.figsize or (8, 5))
        for curve in curves:
            ax.plot(curve.get("x", []), curve.get("y", []), label=_display_label(curve.get("label"), width=30, max_lines=3))
        if self.baseline == "diagonal":
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel(str(self.x_label))
        ax.set_ylabel(str(self.y_label))
        if curves:
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

        _apply_report_style(fig, ax, bounded=self.bounded, bounded_x=self.bounded_x, xlim=self.xlim, ylim=self.ylim)
        ax.set_title(_wrap_label(title, width=64, max_lines=2))
        fig.tight_layout()
        fig.savefig(dst, dpi=160, bbox_inches="tight")
        plt.close(fig)
