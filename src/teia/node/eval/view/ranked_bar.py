"""
``ranked_bar``: a bar chart over named categories, optionally sorted, with optional per-bar.

Source: common knowledge

Description:
  support annotations.

Examples this view serves: per-class F1 sorted ascending, per-label average precision, top
confusion pairs, worst-K samples by a per-sample metric. Each of those is a wiring of this one
class, not a separate class.
"""

from __future__ import annotations

from pathlib import Path

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _all_same_nonzero,
    _apply_report_style,
    _balanced_chunks,
    _display_labels,
    _line_count,
    _page_filename,
    _page_title,
    _wrap_label,
    plt,
    slugify,
    write_description_sidecar,
)

RANKED_BAR_PAGE_SIZE = 30


class RankedBar(View):
    """Draws one bar chart per page (paginates when a horizontal chart would exceed
    ``RANKED_BAR_PAGE_SIZE`` bars; vertical orientation is never paginated).

    Config (``__init__``): ``title``, ``orientation`` ("horizontal"/"vertical"), ``sort``
    (ascending-by-value when drawn), ``ylabel``, ``xlabel``, ``bounded_x``, ``support_label``
    (unit label for the optional per-bar support annotation, e.g. "n"), ``figsize``,
    ``filename``, ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``labels``, ``values``, optional ``support`` (same length as
    ``labels``, drawn as small annotations next to each bar).
    """

    def __init__(
        self,
        *,
        title: str,
        orientation: str = "horizontal",
        sort: bool = True,
        ylabel: str | None = None,
        xlabel: str | None = None,
        bounded_x: bool = False,
        support_label: str = "n",
        figsize: tuple[float, float] | None = None,
        filename: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        blindspot: str | None = None,
    ) -> None:
        self.title = title
        self.orientation = orientation
        self.sort = sort
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.bounded_x = bounded_x
        self.support_label = support_label
        self.figsize = figsize
        self.filename = filename or f"{slugify(title)}.png"
        self.description = description
        self.interpretation = interpretation
        self.blindspot = blindspot

    def render(self, out_dir: Path, *, labels: list[str], values: list[float], support: list[float] | None = None) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = list(labels)
        values = list(values)
        support = list(support or [])

        pages = self._paginate(labels, values, support)
        written: list[Path] = []
        for page_index, (page_labels, page_values, page_support, sort_flag) in enumerate(pages):
            filename = _page_filename(self.filename, page_index, len(pages))
            title = _page_title(self.title, page_index, len(pages))
            dst = out_dir / filename
            self._draw(dst, title, page_labels, page_values, page_support, sort_flag)
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

    def _paginate(
        self, labels: list[str], values: list[float], support: list[float]
    ) -> list[tuple[list[str], list[float], list[float], bool]]:
        if self.orientation == "vertical" or len(labels) <= RANKED_BAR_PAGE_SIZE:
            return [(labels, values, support, self.sort)]
        entries = list(zip(labels, values, support if support else [None] * len(labels), strict=False))
        import math
        def safe_sort_key(item):
            val = item[1]
            return float('-inf') if val is None or math.isnan(val) else val

        if self.sort:
            entries.sort(key=safe_sort_key, reverse=True)
        chunks = _balanced_chunks(entries, RANKED_BAR_PAGE_SIZE)
        pages: list[tuple[list[str], list[float], list[float], bool]] = []
        for chunk in chunks:
            page_entries = sorted(chunk, key=safe_sort_key) if self.sort else chunk
            page_labels = [item[0] for item in page_entries]
            page_values = [item[1] for item in page_entries]
            page_support = [item[2] for item in page_entries] if support else []
            # Already ordered correctly for drawing; _draw must not re-sort.
            pages.append((page_labels, page_values, page_support, False))
        return pages

    def _draw(self, dst: Path, title: str, labels: list[str], values: list[float], support: list[float], sort_flag: bool) -> None:
        has_support = bool(support)
        entries = list(zip(labels, values, support if has_support else [0.0] * len(labels), strict=False))
        import math
        def safe_sort_key(item):
            val = item[1]
            return float('-inf') if val is None or math.isnan(val) else val

        if sort_flag:
            entries.sort(key=safe_sort_key)
        labels = [item[0] for item in entries]
        values = [item[1] for item in entries]
        support = [item[2] for item in entries]

        figsize = self.figsize
        if figsize is None:
            if self.orientation == "vertical":
                figsize = (max(8, 0.55 * len(labels) + 2), 5)
            else:
                display = _display_labels(labels, width=34, max_lines=3)
                extra_lines = sum(max(0, _line_count(label) - 1) for label in display)
                figsize = (8.8, max(3.2, min(16.0, 0.27 * len(display) + 0.16 * extra_lines + 1.4)))
        fig, ax = plt.subplots(figsize=figsize)

        support_note = ""
        if self.orientation == "vertical":
            ax.bar(range(len(labels)), values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(_display_labels(labels, width=16, max_lines=3), rotation=45, ha="right", fontsize=8)
        else:
            ax.barh(range(len(labels)), values)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(_display_labels(labels, width=34, max_lines=3), fontsize=8)
            ax.margins(y=0.01)
            if self.bounded_x:
                ax.set_xlim(0.0, 1.0)
            elif values:
                high = max(float(value) for value in values)
                ax.set_xlim(0.0, high * 1.08 if high > 0 else 1.0)
            if has_support and _all_same_nonzero(support) and len(support) > 3:
                support_note = f"support {self.support_label}={int(support[0])} each"
            elif has_support:
                for idx, item in enumerate(support):
                    ax.text(1.01, idx, f"{self.support_label}={int(item)}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7, clip_on=False)
        if self.xlabel:
            ax.set_xlabel(f"{self.xlabel} ({support_note})" if support_note else str(self.xlabel))
        elif support_note:
            ax.set_xlabel(support_note)
        if self.ylabel:
            ax.set_ylabel(str(self.ylabel))

        _apply_report_style(fig, ax, bounded_x=self.bounded_x)
        ax.set_title(_wrap_label(title, width=64, max_lines=2))
        fig.tight_layout()
        fig.savefig(dst, dpi=160, bbox_inches="tight")
        plt.close(fig)
