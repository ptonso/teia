"""
``calibration_diagram``: mean predicted confidence vs empirical accuracy per bin, against.

Source:
  - title: "The Comparison and Evaluation of Forecasters"
    url: "https://doi.org/10.2307/2987588"
    year: 1983
  - title: "On Calibration of Modern Neural Networks"
    url: "https://arxiv.org/abs/1706.04599"
    year: 2017

Description:
  the identity diagonal, annotated with bin counts.

Not paginated: a calibration diagram has a fixed, small number of bins by construction. Serves
the reliability diagram used across classification objectives.
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


class CalibrationDiagram(View):
    """Draws mean-confidence vs empirical-accuracy points against the identity diagonal.

    Config (``__init__``): ``title``, ``bounded``/``bounded_x``/``xlim``/``ylim``/``figsize``
    (styling overrides), ``filename``, ``description``/``interpretation``/``blindspot``.

    Data (``render(**inputs)``): ``mean_confidence``, ``empirical_accuracy``, ``counts``, three
    equal-length lists, one entry per confidence bin.
    """

    def __init__(
        self,
        *,
        title: str,
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
        mean_confidence: list[float],
        empirical_accuracy: list[float],
        counts: list[int],
    ) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / self.filename

        fig, ax = plt.subplots(figsize=self.figsize or (8, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.plot(mean_confidence, empirical_accuracy, marker="o")
        for x_val, y_val, count in zip(mean_confidence, empirical_accuracy, counts, strict=False):
            ax.text(x_val, y_val, str(count), fontsize=7)
        ax.set_xlabel("Mean Confidence")
        ax.set_ylabel("Empirical Accuracy")

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
