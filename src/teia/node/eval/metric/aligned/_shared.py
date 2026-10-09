"""
Private numeric helpers shared by ``teia.node.eval.metric.aligned.*`` components.

Source:
  - title: "Microsoft COCO: Common Objects in Context"
    url: "https://arxiv.org/abs/1405.0312"
    year: 2014
  - title: "The PASCAL Visual Object Classes (VOC) Challenge"
    url: "https://doi.org/10.1007/s11263-009-0275-4"
    year: 2010
  - title: "On Calibration of Modern Neural Networks"
    url: "https://arxiv.org/abs/1706.04599"
    year: 2017
  - title: "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
    url: "https://doi.org/10.1609/aaai.v29i1.9602"
    year: 2015

Description:
  Not a component: no ``_target_``, no ``conf/eval/metric/*.yaml`` leaf. Just shared code every
  ``aligned/*`` metric imports from.

Third-party imports (numpy, sklearn) are at module top level, not deferred into methods.
``teia.core.deps.collect_missing`` detects a missing dependency by trial-importing the component
module, which only reports the dependency if the import is visible at module scope. See
``packages/teia/specs/layout.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sklearn_metrics

__all__ = [
    "np",
    "sklearn_metrics",
    "expected_calibration_error",
    "metric_curve_from_scores",
    "interpolated_ap",
    "precision_recall_f1",
    "label_binarize_indices",
    "to_rows",
]


def to_rows(value: Any) -> list[Any]:
    """Convert one batch's per-sample array kwarg (tensor, ndarray, or plain list) into a flat
    list of per-sample rows that can be appended to a streaming buffer.

    Used by every buffered ``update`` driver in this package (Confusion, Residuals,
    HorizonError, Concordance): each batch is converted here and appended to a running buffer,
    then the whole buffer is handed to ``from_source`` unchanged at ``compute`` time, so the
    same formula runs whether the samples arrived in one pass or many.
    """
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.dtype == torch.bfloat16:
                # numpy has no native bfloat16 dtype; widen before conversion.
                value = value.float()
            value = value.cpu().numpy()
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return list(np.asarray(value))


def label_binarize_indices(indices: list[int], n_classes: int) -> Any:
    """One-hot binarize a list of class indices into an ``(N, n_classes)`` float array."""
    if not indices:
        return np.zeros((0, n_classes), dtype=float)
    eye = np.eye(n_classes, dtype=float)
    return eye[list(indices)]


def expected_calibration_error(
    true_indices: list[int], score_vectors: list[list[float]], *, bins: int = 10
) -> tuple[float, dict[str, list[float]]]:
    """Expected calibration error over equal-width confidence bins, plus the reliability-diagram
    arrays (per-bin mean confidence, empirical accuracy and count).

    ECE is the count-weighted mean of ``|accuracy - confidence|`` across non-empty bins, so a
    perfectly calibrated model scores 0. Empty bins are skipped rather than counted as 0 error.
    """
    probabilities = np.asarray(score_vectors, dtype=float)
    truths = np.asarray(true_indices, dtype=int)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == truths).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mean_confidence: list[float] = []
    empirical_accuracy: list[float] = []
    counts: list[int] = []
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not mask.any():
            continue
        conf = float(confidences[mask].mean())
        acc = float(correctness[mask].mean())
        count = int(mask.sum())
        ece += abs(acc - conf) * (count / max(len(confidences), 1))
        mean_confidence.append(conf)
        empirical_accuracy.append(acc)
        counts.append(count)
    return float(ece), {
        "mean_confidence": mean_confidence,
        "empirical_accuracy": empirical_accuracy,
        "counts": counts,
    }


def metric_curve_from_scores(y_true: Any, y_score: Any) -> dict[str, Any]:
    """PR and ROC curve arrays for one binary score column, with their summary scalars
    (average precision, ROC AUC)."""
    precision, recall, _ = sklearn_metrics.precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = sklearn_metrics.roc_curve(y_true, y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "average_precision": float(sklearn_metrics.average_precision_score(y_true, y_score)),
        "roc_auc": float(sklearn_metrics.roc_auc_score(y_true, y_score)),
    }


def interpolated_ap(recalls: Any, precisions: Any) -> float:
    """Average precision on COCO's 101-point recall grid.

    Precision is replaced by its suffix maximum (the monotone envelope), then sampled at
    ``recall = 0, 0.01, ..., 1`` and averaged. Recall points beyond the observed range
    contribute 0, which is what makes missed ground truth cost score.
    """
    recall_values = np.asarray(recalls, dtype=float)
    precision_values = np.asarray(precisions, dtype=float)
    if recall_values.size == 0 or precision_values.size == 0:
        return 0.0
    recall_grid = np.linspace(0.0, 1.0, 101)
    suffix_max = np.maximum.accumulate(precision_values[::-1])[::-1]
    indices = np.searchsorted(recall_values, recall_grid, side="left")
    out = np.zeros_like(recall_grid)
    valid = indices < suffix_max.size
    out[valid] = suffix_max[indices[valid]]
    return float(out.mean())


def precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from raw counts. Each is 0 where its denominator is 0."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
