"""
Calibration metric: single-label confidence calibration.

Source:
  - title: "On Calibration of Modern Neural Networks"
    url: "https://arxiv.org/abs/1706.04599"
    year: 2017
  - title: "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
    url: "https://doi.org/10.1609/aaai.v29i1.9602"
    year: 2015

Description:
  Produces expected calibration error, the confidence histogram, and reliability-diagram data.
  Single-label only; multi-label classification has no calibration section.

Param naming: ``cls`` is the literal capture-store atom for classification targets; ``truth``
is accepted as an alias.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric
from teia.node.eval.metric.aligned._shared import expected_calibration_error

__all__ = ["Calibration"]


class Calibration(Metric):
    """Expected calibration error + reliability-diagram data, single-label classification.

    Constructor kwarg: ``bins``, the number of confidence bins (default ``10``).
    """

    def __init__(self, bins: int = 10) -> None:
        self.bins = int(bins)

    def from_source(self, source: Any, *, scores: Any, cls: Any = None, truth: Any = None, **_: Any) -> dict[str, Any]:
        del source
        truth_raw = cls if cls is not None else truth
        if truth_raw is None:
            raise ValueError("Calibration.from_source requires a ground-truth atom wired as 'cls' (or 'truth').")

        y_true = [int(t) for t in np.asarray(truth_raw).reshape(-1)]
        if not y_true:
            return {"n_samples": 0}
        score_vectors = np.asarray(scores, dtype=float).reshape(len(y_true), -1).tolist()

        calibration_error, calibration_data = expected_calibration_error(y_true, [list(row) for row in score_vectors], bins=self.bins)

        score_array = np.asarray(score_vectors, dtype=float)
        confidences = score_array.max(axis=1)
        predictions = score_array.argmax(axis=1)
        correct_mask = predictions == np.asarray(y_true)

        return {
            "n_samples": len(y_true),
            "calibration_error": round(float(calibration_error), 6),
            "reliability_diagram": {
                "mean_confidence": calibration_data["mean_confidence"],
                "empirical_accuracy": calibration_data["empirical_accuracy"],
                "counts": calibration_data["counts"],
            },
            "confidence_histogram": {
                "correct": confidences[correct_mask].tolist(),
                "incorrect": confidences[~correct_mask].tolist(),
            },
        }
