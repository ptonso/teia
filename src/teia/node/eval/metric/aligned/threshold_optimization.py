"""
ThresholdOptimization: a decision rule fitted from captured scores, single- or multi-label.

Source: common knowledge

Description:
  Fits the rule on one capture split (the node's ``split``, ``val`` in the shipped evalmodules) and
  reruns every decision-dependent node of the graph with it on the report split, into
  ``<out_dir>/threshold-opt/`` (``after_pass``). The paradigm follows ``mode``: ``multi_cls`` fits one
  threshold per label, ``single_cls`` fits one score weight per class (prediction
  ``argmax(weights * probs)``, coordinate ascent on the macro objective), ``auto`` picks from the target
  layout (a multi-hot matrix is multi-label, a class-index vector is single-label).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric

__all__ = ["ThresholdOptimization", "optimize_multi_label_thresholds", "optimize_single_label_weights"]

_MODES = ("auto", "single_cls", "multi_cls")
_TARGETS = ("f1", "f_beta", "balanced_accuracy")
_DEFAULT_FALLBACK_THRESHOLD = 0.5
_RERUN_DIR = "threshold-opt"


class ThresholdOptimization(Metric):
    """Fits the decision rule matching the target layout and reruns the nodes that accept it.

    Constructor kwargs (leaf/bundle config):
      - ``mode``: ``auto`` | ``single_cls`` | ``multi_cls``. A forced mode that contradicts the
        captured layout fails fast.
      - ``target``: ``f1`` | ``f_beta`` | ``balanced_accuracy``, optimized per label (multi-label) or
        macro-averaged over classes (single-label).
      - ``beta``: the F-beta weight, only valid with ``target=f_beta``.
      - ``fallback_threshold``: the threshold of a degenerate multi-label class, one with no positive
        or no negative support (default ``0.5``).
    """

    def __init__(
        self,
        mode: str = "auto",
        target: str = "f1",
        beta: float | None = None,
        fallback_threshold: float = _DEFAULT_FALLBACK_THRESHOLD,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"ThresholdOptimization mode must be one of {_MODES}, got {mode!r}.")
        if target not in _TARGETS:
            raise ValueError(f"ThresholdOptimization target must be one of {_TARGETS}, got {target!r}.")
        if (beta is not None) != (target == "f_beta"):
            raise ValueError("ThresholdOptimization `beta` is required by target=f_beta and invalid with any other target.")
        self.mode = mode
        self.target = target
        self.beta = 1.0 if beta is None else float(beta)
        self.fallback_threshold = float(fallback_threshold)

    def from_source(
        self,
        source: Any,
        *,
        scores: Any,
        cls: Any = None,
        truth: Any = None,
        class_names: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        del source
        truth_raw = cls if cls is not None else truth
        if truth_raw is None:
            raise ValueError("ThresholdOptimization.from_source requires a ground-truth atom wired as 'cls' (or 'truth').")
        names = list(class_names) if class_names else None
        detected = "single_cls" if _is_single_label(scores, truth_raw) else "multi_cls"
        if self.mode not in ("auto", detected):
            raise ValueError(f"ThresholdOptimization mode={self.mode!r} contradicts the captured target layout ({detected}).")
        if detected == "single_cls":
            return optimize_single_label_weights(probabilities=scores, targets=truth_raw, label_names=names, target=self.target, beta=self.beta)
        return optimize_multi_label_thresholds(
            probabilities=scores,
            targets=truth_raw,
            label_names=names,
            target=self.target,
            beta=self.beta,
            fallback_threshold=self.fallback_threshold,
        )

    def after_pass(self, graph: Any) -> None:
        """Rerun every node that accepts a ``rule`` plus its consumers with the fitted rule."""
        targets = [record.name for record in graph.records if "rule" in graph.component_class(record.name).ACCEPTS]
        if not targets:
            return
        rule = graph.results[self.out_key[0]]["rule"]
        graph.rerun(
            targets + graph.downstream(targets),
            out_dir=graph.out_dir / _RERUN_DIR,
            extra={name: {"rule": rule} for name in targets},
        )


def _is_single_label(probabilities: Any, targets: Any) -> bool:
    scores = np.asarray(probabilities, dtype=np.float64)
    return scores.ndim == 2 and scores.shape[1] > 1 and np.asarray(targets).size == scores.shape[0]


def _score(target: str, beta: float, tp: Any, fp: Any, fn: Any, tn: Any) -> np.ndarray:
    """The objective from confusion counts (scalars or equal-length arrays); 0 where undefined."""
    tp, fp, fn, tn = (np.asarray(value, dtype=np.float64) for value in (tp, fp, fn, tn))
    if target == "balanced_accuracy":
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp), where=(tn + fp) > 0)
        return (recall + specificity) / 2
    weight = beta**2
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denominator = weight * precision + recall
    return np.divide((1 + weight) * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)


def optimize_multi_label_thresholds(
    *,
    probabilities: Any,
    targets: Any,
    label_names: list[str] | None = None,
    target: str = "f1",
    beta: float = 1.0,
    fallback_threshold: float = _DEFAULT_FALLBACK_THRESHOLD,
) -> dict[str, Any]:
    """Fit one decision threshold per label by sweeping candidate cuts for the best ``target``.

    The free function behind :class:`ThresholdOptimization`. Labels with no positive or no negative
    example are degenerate and take ``fallback_threshold``.
    """
    probability_rows = _rows_as_lists(probabilities)
    target_rows = _rows_as_lists(targets)
    num_classes = max(
        len(label_names or []),
        max((len(row) for row in probability_rows), default=0),
        max((len(row) for row in target_rows), default=0),
    )
    resolved_names = list(label_names or [])
    if len(resolved_names) < num_classes:
        resolved_names.extend([f"class_{index}" for index in range(len(resolved_names), num_classes)])

    summaries: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for index in range(num_classes):
        scores = _column_values(probability_rows, index)
        truths = [1 if float(value) >= 0.5 else 0 for value in _column_values(target_rows, index)]
        summary = _optimize_binary_threshold(
            scores=scores, truths=truths, fallback_threshold=float(fallback_threshold), target=target, beta=beta
        )
        summary["index"] = index
        summary["label"] = resolved_names[index]
        summaries.append(summary)
        baselines.append(_binary_metrics(scores=scores, truths=truths, threshold=_DEFAULT_FALLBACK_THRESHOLD))

    thresholds = [float(summary["threshold"]) for summary in summaries]
    return {
        "mode": "multi_cls",
        "objective": target,
        "fallback_threshold": float(fallback_threshold),
        "fit_baseline_macro_f1": float(np.mean([baseline["f1"] for baseline in baselines])),
        "fit_tuned_macro_f1": float(np.mean([summary["f1"] for summary in summaries])),
        "fit_baseline_objective": float(np.mean([_score(target, beta, b["tp"], b["fp"], b["fn"], b["tn"]) for b in baselines])),
        "fit_tuned_objective": float(np.mean([summary["objective"] for summary in summaries])),
        "rule": {"mode": "multi_cls", "objective": target, "label_names": resolved_names, "thresholds": thresholds},
        "label_names": resolved_names,
        "thresholds": thresholds,
        "per_class": summaries,
        # Pre-shaped for Table's `rows: list[dict]` kwarg.
        "rows": summaries,
    }


def optimize_single_label_weights(
    *,
    probabilities: Any,
    targets: Any,
    label_names: list[str] | None = None,
    target: str = "f1",
    beta: float = 1.0,
    rounds: int = 3,
    max_candidates: int = 256,
) -> dict[str, Any]:
    """Fit one positive weight per class so ``argmax(weights * probs)`` maximizes the macro ``target``.

    Coordinate ascent: each class weight moves in turn while the others stay fixed. A sample flips
    between class ``k`` and its best rival exactly when ``log w_k`` crosses ``log(rival / p_k)``, so
    only those crossings (thinned to ``max_candidates``) plus the current value can change the
    predictions. The current value is always a candidate, so the objective never decreases. Classes
    without support cannot be scored and keep weight 1.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(targets).reshape(-1).astype(np.int64)
    if truth.size == 0:
        raise ValueError("optimize_single_label_weights requires at least one sample.")
    num_classes = probs.shape[1]
    names = list(label_names or [])
    names.extend(f"class_{index}" for index in range(len(names), num_classes))

    support = np.bincount(truth, minlength=num_classes)
    log_probs = np.log(np.clip(probs, 1e-12, None))
    log_weights = np.zeros(num_classes)
    baseline_predictions = probs.argmax(axis=1)
    baseline = _macro_score(truth, baseline_predictions, support, target, beta)
    best = baseline

    for _ in range(rounds):
        start = best
        for k in np.flatnonzero(support):
            rivals = np.delete(log_probs + log_weights, k, axis=1)
            rival_score = rivals.max(axis=1)
            rival_class = np.delete(np.arange(num_classes), k)[rivals.argmax(axis=1)]
            crossings = np.unique(rival_score - log_probs[:, k])
            midpoints = (crossings[1:] + crossings[:-1]) / 2
            grid = np.concatenate([[crossings[0] - 1.0], midpoints, [crossings[-1] + 1.0]])
            grid = grid[np.unique(np.linspace(0, len(grid) - 1, min(len(grid), max_candidates)).astype(int))]
            candidates = np.append(grid, log_weights[k])
            scores = np.array(
                [
                    _macro_score(truth, np.where(candidate + log_probs[:, k] > rival_score, k, rival_class), support, target, beta)
                    for candidate in candidates
                ]
            )
            winner = int(np.lexsort((-np.abs(candidates - log_weights[k]), scores))[-1])
            log_weights[k] = candidates[winner]
            best = float(scores[winner])
        if best <= start:
            break

    predictions = (log_probs + log_weights).argmax(axis=1)
    true_positive = np.bincount(truth[predictions == truth], minlength=num_classes)
    predicted = np.bincount(predictions, minlength=num_classes)
    rows = [
        {
            "label": names[index],
            "index": index,
            "weight": float(np.exp(log_weights[index])),
            "support": int(support[index]),
            "precision": float(true_positive[index] / predicted[index]) if predicted[index] else 0.0,
            "recall": float(true_positive[index] / support[index]) if support[index] else 0.0,
            "f1": float(2 * true_positive[index] / (predicted[index] + support[index])) if predicted[index] + support[index] else 0.0,
            "degenerate": not support[index],
            "degenerate_reason": None if support[index] else "missing_positive_support",
        }
        for index in range(num_classes)
    ]
    weights = [row["weight"] for row in rows]
    return {
        "mode": "single_cls",
        "objective": target,
        "fit_baseline_macro_f1": _macro_score(truth, baseline_predictions, support, "f1", 1.0),
        "fit_tuned_macro_f1": _macro_score(truth, predictions, support, "f1", 1.0),
        "fit_baseline_objective": float(baseline),
        "fit_tuned_objective": float(best),
        "rule": {"mode": "single_cls", "objective": target, "label_names": names, "weights": weights},
        "label_names": names,
        "weights": weights,
        "per_class": rows,
        "rows": rows,
    }


def _macro_score(truth: np.ndarray, predictions: np.ndarray, support: np.ndarray, target: str, beta: float) -> float:
    """One-vs-rest ``target`` averaged over the classes that have support."""
    num_classes = len(support)
    true_positive = np.bincount(truth[predictions == truth], minlength=num_classes)
    predicted = np.bincount(predictions, minlength=num_classes)
    false_positive = predicted - true_positive
    false_negative = support - true_positive
    true_negative = len(truth) - true_positive - false_positive - false_negative
    return float(_score(target, beta, true_positive, false_positive, false_negative, true_negative)[support > 0].mean())


def _optimize_binary_threshold(*, scores: list[float], truths: list[int], fallback_threshold: float, target: str, beta: float) -> dict[str, Any]:
    positives = sum(1 for truth in truths if int(truth) == 1)
    negatives = max(0, len(truths) - positives)
    if not scores or positives == 0 or negatives == 0:
        metrics = _binary_metrics(scores=scores, truths=truths, threshold=fallback_threshold)
        metrics.update(
            {
                "threshold": float(fallback_threshold),
                "degenerate": True,
                "degenerate_reason": _degenerate_reason(scores=scores, positives=positives, negatives=negatives),
                "objective": float(_score(target, beta, metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])),
            }
        )
        return metrics

    threshold, counts = _best_threshold(scores=scores, truths=truths, fallback_threshold=fallback_threshold, target=target, beta=beta)
    metrics = _metrics_from_counts(tp=counts[0], fp=counts[1], fn=counts[2], tn=counts[3], support=positives)
    metrics.update(
        {
            "threshold": threshold,
            "degenerate": False,
            "degenerate_reason": None,
            "objective": float(_score(target, beta, counts[0], counts[1], counts[2], counts[3])),
        }
    )
    return metrics


def _best_threshold(*, scores: list[float], truths: list[int], fallback_threshold: float, target: str, beta: float) -> tuple[float, tuple[int, int, int, int]]:
    """Sweep every candidate threshold in one vectorized pass.

    Candidates are the distinct scores plus the fallback (a prediction is positive iff
    ``score >= threshold``, so only those values can change the confusion matrix). Sorting
    descending once and taking cumulative sums gives every candidate's tp/fp in O(n log n).

    Ranking is lexicographic on ``(objective, precision, recall, threshold)``: ranks are total
    (threshold is unique per candidate), so the winner is unambiguous.
    """
    score_array = np.asarray(scores, dtype=np.float64)
    truth_array = np.asarray(truths, dtype=np.int64)
    total_positive = int(truth_array.sum())
    total = int(score_array.size)

    order = np.argsort(-score_array, kind="stable")
    truth_desc = truth_array[order]
    score_desc = score_array[order]

    _, run_lengths = np.unique(-score_desc, return_counts=True)
    kept = np.cumsum(run_lengths)
    candidates = score_desc[kept - 1]
    true_positive = np.cumsum(truth_desc)[kept - 1]

    fallback = float(fallback_threshold)
    if not np.isin(fallback, candidates):
        kept = np.append(kept, int(np.count_nonzero(score_array >= fallback)))
        candidates = np.append(candidates, fallback)
        true_positive = np.append(true_positive, int(truth_array[score_array >= fallback].sum()))

    false_positive = kept - true_positive
    false_negative = total_positive - true_positive
    true_negative = total - total_positive - false_positive
    predicted = true_positive + false_positive
    actual = true_positive + false_negative

    precision = np.divide(true_positive, predicted, out=np.zeros(len(kept)), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros(len(kept)), where=actual > 0)
    objective = _score(target, beta, true_positive, false_positive, false_negative, true_negative)

    best = int(np.lexsort((candidates, recall, precision, objective))[-1])
    return (
        float(candidates[best]),
        (int(true_positive[best]), int(false_positive[best]), int(false_negative[best]), int(true_negative[best])),
    )


def _metrics_from_counts(*, tp: int, fp: int, fn: int, tn: int, support: int) -> dict[str, Any]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    positive_union = tp + fp + fn
    return {
        "support": int(support),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_union": int(positive_union),
        "iou_jaccard": float(tp / positive_union) if positive_union else 0.0,
    }


def _binary_metrics(*, scores: list[float], truths: list[int], threshold: float) -> dict[str, Any]:
    predictions = [1 if float(score) >= float(threshold) else 0 for score in scores]
    tp = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == 1 and truth == 1)
    tn = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == 0 and truth == 0)
    fp = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == 1 and truth == 0)
    fn = sum(1 for prediction, truth in zip(predictions, truths, strict=False) if prediction == 0 and truth == 1)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    positive_union = tp + fp + fn
    return {
        "support": int(sum(1 for truth in truths if truth == 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_union": int(positive_union),
        "iou_jaccard": float(tp / positive_union) if positive_union else 0.0,
    }


def _degenerate_reason(*, scores: list[float], positives: int, negatives: int) -> str:
    if not scores:
        return "missing_scores"
    if positives == 0:
        return "missing_positive_support"
    if negatives == 0:
        return "missing_negative_support"
    return "unknown"


def _rows_as_lists(rows: Any) -> list[list[float]]:
    if hasattr(rows, "tolist") and not isinstance(rows, list):
        rows = rows.tolist()
    normalized: list[list[float]] = []
    for row in rows or []:
        if hasattr(row, "tolist") and not isinstance(row, list):
            row = row.tolist()
        normalized.append([float(value) for value in row])
    return normalized


def _column_values(rows: list[list[float]], index: int) -> list[float]:
    values: list[float] = []
    for row in rows:
        if index < len(row):
            values.append(float(row[index]))
        else:
            values.append(0.0)
    return values
