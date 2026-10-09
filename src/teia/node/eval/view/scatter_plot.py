"""
``scatter_plot``: a labeled 2D scatter, optionally size-encoded (by support, say), with.

Source: common knowledge

Description:
  direct-annotation or numbered-legend labeling depending on point density.

Examples this view serves: per-class precision vs recall (sized by support), per-label
prevalence vs F1, support vs AP@[.50:.95].
"""

from __future__ import annotations

from pathlib import Path

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _PALETTE,
    _apply_report_style,
    _balanced_chunks,
    _display_labels,
    _limits,
    _page_filename,
    _page_title,
    _scatter_can_label_directly,
    _scatter_label_order,
    _scatter_label_y,
    _wrap_label,
    plt,
    slugify,
    write_description_sidecar,
)

SCATTER_PAGE_SIZE = 40


class ScatterPlot(View):
    """Draws one scatter per page (paginates beyond ``SCATTER_PAGE_SIZE`` points, pinning
    shared axis limits across pages so points remain comparable page to page).

    Config (``__init__``): ``title``, ``x_label``, ``y_label``, ``bounded``/``bounded_x``/
    ``xlim``/``ylim``/``figsize`` (styling overrides), ``filename``,
    ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``x``, ``y``, ``labels`` (all same length), optional ``sizes``
    (marker sizes, e.g. by support; defaults to a fixed size).
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

    def render(
        self,
        out_dir: Path,
        *,
        x: list[float],
        y: list[float],
        labels: list[str],
        sizes: list[float] | None = None,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        x = list(x)
        y = list(y)
        labels = list(labels)
        sizes = list(sizes) if sizes else [60.0] * len(labels)

        pages = self._paginate(x, y, labels, sizes)
        written: list[Path] = []
        for page_index, (page_x, page_y, page_labels, page_sizes, xlim, ylim) in enumerate(pages):
            filename = _page_filename(self.filename, page_index, len(pages))
            title = _page_title(self.title, page_index, len(pages))
            dst = out_dir / filename
            self._draw(dst, title, page_x, page_y, page_labels, page_sizes, xlim, ylim)
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

    def _paginate(self, x: list[float], y: list[float], labels: list[str], sizes: list[float]):
        if len(labels) <= SCATTER_PAGE_SIZE:
            return [(x, y, labels, sizes, self.xlim, self.ylim)]
        order = sorted(range(len(labels)), key=lambda index: (float(y[index]), float(x[index]), str(labels[index])))
        chunks = _balanced_chunks(order, SCATTER_PAGE_SIZE)
        page_xlim = self.xlim if self.bounded_x else _limits(x)
        page_ylim = self.ylim if self.bounded else _limits(y)
        pages = []
        for chunk in chunks:
            pages.append(
                (
                    [x[index] for index in chunk],
                    [y[index] for index in chunk],
                    [labels[index] for index in chunk],
                    [sizes[index] for index in chunk],
                    page_xlim,
                    page_ylim,
                )
            )
        return pages

    def _draw(
        self,
        dst: Path,
        title: str,
        x: list[float],
        y: list[float],
        labels: list[str],
        sizes: list[float],
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
    ) -> None:
        figsize = self.figsize
        if figsize is None:
            label_count = len(labels)
            figsize = (9.6 if label_count > 8 else 8.0, max(5.0, min(11.5, 0.14 * label_count + 4.0)))
        fig, ax = plt.subplots(figsize=figsize)

        marker_sizes = [min(max(float(size), 36.0), 480.0) for size in sizes]
        ax.scatter(x, y, s=marker_sizes, alpha=0.75, color=_PALETTE[0], edgecolor="white", linewidth=0.5)
        wrapped = _display_labels(labels, width=28, max_lines=3)
        if _scatter_can_label_directly(x, y, labels):
            for index in range(len(labels)):
                ax.annotate(wrapped[index], (x[index], y[index]), fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
        else:
            ordered = _scatter_label_order(x, y, labels)
            fontsize = 7 if len(ordered) <= 24 else 6
            for marker, index in enumerate(ordered, start=1):
                label_y = _scatter_label_y(marker - 1, len(ordered))
                ax.annotate(
                    str(marker),
                    xy=(x[index], y[index]),
                    xycoords="data",
                    xytext=(1.02, label_y),
                    textcoords=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=fontsize,
                    weight="bold",
                    annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "connectionstyle": "arc3,rad=0.08", "linewidth": 0.35, "alpha": 0.28, "color": "#555555", "shrinkA": 0, "shrinkB": 2},
                )
                ax.text(1.06, label_y, wrapped[index].replace("\n", ""), transform=ax.transAxes, va="center", ha="left", fontsize=fontsize)
        ax.set_xlabel(str(self.x_label))
        ax.set_ylabel(str(self.y_label))

        _apply_report_style(fig, ax, scatter=True, bounded=self.bounded, bounded_x=self.bounded_x, xlim=xlim, ylim=ylim)
        ax.set_title(_wrap_label(title, width=64, max_lines=2))
        fig.tight_layout()
        fig.savefig(dst, dpi=160, bbox_inches="tight")
        plt.close(fig)
