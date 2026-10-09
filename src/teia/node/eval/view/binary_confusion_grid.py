"""
``binary_confusion_grid``: a small-multiples grid of one-vs-all TP/FP/FN tiles (TN masked.

Source: common knowledge

Description:
  out), one tile per category.

Serves the TN-masked one-vs-all confusion tiles used for multi-label classification, where a
full multi-class confusion matrix would double-count background.
"""

from __future__ import annotations

from pathlib import Path

from teia.base.eval import View
from teia.node.eval.view._shared import (
    _balanced_chunks,
    _display_labels,
    _line_count,
    _page_filename,
    _page_title,
    _wrap_label,
    np,
    plt,
    slugify,
    write_description_sidecar,
)

BINARY_CONFUSION_PAGE_SIZE = 12


class BinaryConfusionGrid(View):
    """Draws one 2x2 (TN masked) tile per category, small-multiples style (paginates beyond
    ``BINARY_CONFUSION_PAGE_SIZE`` categories). Unlike the other plotting views, this one does
    not go through ``_apply_report_style``: each tile is its own miniature axes with its own
    fixed styling.

    Config (``__init__``): ``title``, ``filename``,
    ``description``/``interpretation``/``blindspot``. No ``figsize``/``bounded*``/``xlim``/
    ``ylim``: the grid's figure size is derived from the category count and label length.

    Data (``render(**inputs)``): ``labels``, ``tp``, ``fp``, ``fn`` (equal-length lists, one
    entry per category; each cell is normalized by ``tp+fp+fn`` for that category).
    """

    def __init__(
        self,
        *,
        title: str,
        filename: str | None = None,
        description: str | None = None,
        interpretation: str | None = None,
        blindspot: str | None = None,
    ) -> None:
        self.title = title
        self.filename = filename or f"{slugify(title)}.png"
        self.description = description
        self.interpretation = interpretation
        self.blindspot = blindspot

    def render(self, out_dir: Path, *, labels: list[str], tp: list[float], fp: list[float], fn: list[float]) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = list(labels)
        tp = list(tp)
        fp = list(fp)
        fn = list(fn)

        pages = self._paginate(labels, tp, fp, fn)
        written: list[Path] = []
        for page_index, (page_labels, page_tp, page_fp, page_fn) in enumerate(pages):
            filename = _page_filename(self.filename, page_index, len(pages))
            title = _page_title(self.title, page_index, len(pages))
            dst = out_dir / filename
            self._draw(dst, title, page_labels, page_tp, page_fp, page_fn)
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

    def _paginate(self, labels: list[str], tp: list[float], fp: list[float], fn: list[float]):
        if len(labels) <= BINARY_CONFUSION_PAGE_SIZE:
            return [(labels, tp, fp, fn)]
        entries = list(zip(labels, tp, fp, fn, strict=False))
        chunks = _balanced_chunks(entries, BINARY_CONFUSION_PAGE_SIZE)
        return [
            (
                [label for label, _, _, _ in chunk],
                [value for _, value, _, _ in chunk],
                [value for _, _, value, _ in chunk],
                [value for _, _, _, value in chunk],
            )
            for chunk in chunks
        ]

    def _draw(self, dst: Path, title: str, labels: list[str], tp: list[float], fp: list[float], fn: list[float]) -> None:
        display_labels = _display_labels(labels, width=34, max_lines=2)
        n = max(len(labels), 1)
        max_label_lines = max((_line_count(label) for label in display_labels), default=1)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, (2.35 + 0.18 * max_label_lines) * nrows + 0.4), squeeze=False)
        fig.set_facecolor("white")
        for index in range(nrows * ncols):
            ax = axes[index // ncols][index % ncols]
            if index >= len(labels):
                ax.axis("off")
                continue
            total = tp[index] + fp[index] + fn[index]
            denom = total if total > 0 else 1.0
            # rows: true neg/pos, cols: pred neg/pos. TN cell is masked.
            cell = np.array([[np.nan, fp[index] / denom], [fn[index] / denom, tp[index] / denom]], dtype=float)
            masked = np.ma.masked_invalid(cell)
            cmap = plt.cm.Blues.copy()
            cmap.set_bad(color="#dddddd")
            ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
            annotations = [["TN", f"FP\n{fp[index] / denom:.2f}"], [f"FN\n{fn[index] / denom:.2f}", f"TP\n{tp[index] / denom:.2f}"]]
            for r in range(2):
                for c in range(2):
                    ax.text(c, r, annotations[r][c], ha="center", va="center", fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["pred-", "pred+"], fontsize=7)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["true-", "true+"], fontsize=7)
            ax.set_title(display_labels[index], fontsize=7 if max_label_lines > 2 else 8)
        fig.suptitle(_wrap_label(title, width=72, max_lines=2))
        fig.tight_layout()
        fig.savefig(dst, dpi=160, bbox_inches="tight")
        plt.close(fig)
