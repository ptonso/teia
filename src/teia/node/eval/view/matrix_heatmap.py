"""
``matrix_heatmap``: an annotated 2D matrix (confusion matrix, distance matrix) with.

Source: common knowledge

Description:
  independent row and column label lists.

Examples this view serves: a row-normalized confusion matrix over class names, a TN-masked
one-vs-all grid. Any annotated matrix with labeled rows and columns is a ``MatrixHeatmap``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _apply_report_style,
    _balanced_chunks,
    _display_labels,
    _page_filename,
    _page_title,
    _wrap_label,
    np,
    plt,
    slugify,
    write_description_sidecar,
)

HEATMAP_TILE_SIZE = 30
HEATMAP_TILE_THRESHOLD = 40


class MatrixHeatmap(View):
    """Draws one annotated heatmap per (possibly tiled) page.

    Config (``__init__``): ``title``, ``annotation_format``, ``x_label``/``y_label`` (axis
    captions like "Predicted"/"True", distinct from the per-cell tick labels, which are data),
    ``bounded``/``bounded_x``/``xlim``/``ylim``/``figsize`` (styling overrides), ``filename``,
    ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``matrix`` (2D list/array), ``x_labels``, ``y_labels`` (tick
    label lists, one entry per column/row of ``matrix``).
    """

    def __init__(
        self,
        *,
        title: str,
        annotation_format: str = ".2f",
        x_label: str = "Predicted",
        y_label: str = "True",
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
        self.annotation_format = annotation_format
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

    def render(self, out_dir: Path, *, matrix: Any, x_labels: list[str], y_labels: list[str]) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        matrix = [list(row) for row in matrix]
        x_labels = list(x_labels)
        y_labels = list(y_labels)

        pages = self._paginate(matrix, x_labels, y_labels)
        written: list[Path] = []
        for page_index, (page_matrix, page_x_labels, page_y_labels) in enumerate(pages):
            filename = _page_filename(self.filename, page_index, len(pages))
            title = _page_title(self.title, page_index, len(pages))
            dst = out_dir / filename
            self._draw(dst, title, page_matrix, page_x_labels, page_y_labels)
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
        self, matrix: list[list[float]], x_labels: list[str], y_labels: list[str]
    ) -> list[tuple[list[list[float]], list[str], list[str]]]:
        if max(len(x_labels), len(y_labels)) <= HEATMAP_TILE_THRESHOLD:
            return [(matrix, x_labels, y_labels)]
        row_chunks = _balanced_chunks(list(range(len(y_labels))), HEATMAP_TILE_SIZE)
        col_chunks = _balanced_chunks(list(range(len(x_labels))), HEATMAP_TILE_SIZE)
        pages: list[tuple[list[list[float]], list[str], list[str]]] = []
        for rows in row_chunks:
            for cols in col_chunks:
                pages.append(
                    (
                        [[matrix[row][col] for col in cols] for row in rows],
                        [x_labels[index] for index in cols],
                        [y_labels[index] for index in rows],
                    )
                )
        return pages

    def _draw(self, dst: Path, title: str, matrix: list[list[float]], x_labels: list[str], y_labels: list[str]) -> None:
        matrix_arr = np.asarray(matrix)
        display_x = _display_labels(x_labels, width=18, max_lines=3)
        display_y = _display_labels(y_labels, width=24, max_lines=3)
        annotation_fontsize = 8 if matrix_arr.size <= 100 else 5
        tick_fontsize = 8 if max(matrix_arr.shape) <= 20 else 6

        figsize = self.figsize
        if figsize is None:
            side = max(6, min(14, 0.36 * max(len(x_labels), len(y_labels)) + 3))
            figsize = (side, side * 0.85)
        fig, ax = plt.subplots(figsize=figsize)

        image = ax.imshow(matrix_arr, cmap="Blues", aspect="auto")
        fig.colorbar(image, ax=ax)
        for y_index in range(matrix_arr.shape[0]):
            for x_index in range(matrix_arr.shape[1]):
                ax.text(x_index, y_index, format(matrix_arr[y_index, x_index], self.annotation_format), ha="center", va="center", fontsize=annotation_fontsize)
        ax.set_xticks(range(len(display_x)))
        ax.set_xticklabels(display_x, rotation=45, ha="right", fontsize=tick_fontsize)
        ax.set_yticks(range(len(display_y)))
        ax.set_yticklabels(display_y, fontsize=tick_fontsize)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        _apply_report_style(fig, ax, bounded=self.bounded, bounded_x=self.bounded_x, xlim=self.xlim, ylim=self.ylim)
        ax.set_title(_wrap_label(title, width=64, max_lines=2))
        fig.tight_layout()
        fig.savefig(dst, dpi=160, bbox_inches="tight")
        plt.close(fig)
