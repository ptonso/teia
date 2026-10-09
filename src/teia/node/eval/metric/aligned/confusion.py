"""
Confusion metric: single-label and multi-label classification confusion breakdowns.

Source: common knowledge

Description:
  Single-label produces accuracy, macro-F1, balanced accuracy, the confusion matrix and per-class
  rows. Multi-label produces micro/macro/sample F1, exact-match accuracy, Hamming loss and
  per-class TN-masked rows. AP, ROC, PR curves, log-loss and top-k live in
  :mod:`.ranked_average_precision`; calibration lives in :mod:`.calibration`. This metric covers
  only the threshold-or-argmax confusion breakdown.

Param naming: the wired capture-store atom for both ``cls`` and ``multi-cls`` targets is
``batch.cls``, so ``_atom_name`` resolves that ``in_key`` to the kwarg ``cls``. The metric
accepts ``cls`` as the primary name and ``truth`` as an alias, so a bundle wiring either
convention works.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sklearn_metrics

from teia.base.eval import Metric
from teia.node.eval.metric.aligned._shared import to_rows

__all__ = ["Confusion"]


def _resolve_thresholds(n_classes: int, raw_threshold: float | list[float] | tuple[float, ...] | None) -> list[float]:
    fallback = 0.5 if raw_threshold is None else raw_threshold
    if isinstance(fallback, (int, float)):
        return [float(fallback)] * n_classes
    values = [float(value) for value in fallback]
    if len(values) >= n_classes:
        return values[:n_classes]
    return values + [0.5] * (n_classes - len(values))


class Confusion(Metric):
    """Threshold/argmax-based confusion breakdown, single-label or multi-label.

    Constructor kwargs (leaf config, not wired ``in`` keys):
      - ``multi_label``: selects the report shape (default ``False``, single-label).
      - ``worst_k``: how many top confusion pairs (single-label) to report (default ``5``).
      - ``threshold``: per-class or scalar decision threshold, multi-label only (default ``0.5``).

    Optional ``rule`` kwarg of ``from_source`` (a fitted decision rule, see ``ThresholdOptimization``):
    per-class ``thresholds`` replace ``threshold`` (multi-label), per-class ``weights`` scale the
    scores before the argmax (single-label). Reported scores stay the raw probabilities.
    """

    ACCEPTS = frozenset({"rule"})

    #: ``fitness`` is accuracy for single-label and macro-F1 for multi-label, so a
    #: ``vision_fitness.yaml``-wired pipeline's ``EarlyStopping``/``ModelCheckpoint`` monitoring
    #: ``val/fitness`` keeps working. macro-F1 uses the conventional definition (zero-support
    #: classes count as 0, via sklearn's ``zero_division=0``), so ``val/macro_f1`` can shift on
    #: datasets with unseen classes relative to a "drop unsupported classes" definition.
    def __init__(self, multi_label: bool = False, worst_k: int = 5, threshold: float | list[float] | None = None) -> None:
        self.multi_label = bool(multi_label)
        self.worst_k = int(worst_k)
        self.threshold = threshold
        self.MONITORS = (
            ("micro_f1", "macro_f1", "exact_match_accuracy", "fitness")
            if self.multi_label
            else ("accuracy", "macro_f1", "fitness")
        )
        self.reset()

    # Streaming: buffer raw per-sample kwargs across batches, replay from_source at compute time.

    def update(
        self,
        *,
        scores: Any = None,
        cls: Any = None,
        truth: Any = None,
        class_names: list[str] | None = None,
        sample_ids: list[str] | None = None,
        **_: Any,
    ) -> None:
        truth_raw = cls if cls is not None else truth
        if scores is not None:
            self._buf_scores.extend(to_rows(scores))
        if truth_raw is not None:
            self._buf_truth.extend(to_rows(truth_raw))
        if sample_ids is not None:
            self._buf_sample_ids.extend(list(sample_ids))
        if class_names:
            self._class_names = list(class_names)

    def compute(self) -> dict[str, Any]:
        if not self._buf_scores:
            return {"n_samples": 0}
        outputs = self.from_source(
            None,
            scores=self._buf_scores,
            truth=self._buf_truth,
            class_names=self._class_names,
            sample_ids=self._buf_sample_ids or None,
        )
        primary = "macro_f1" if self.multi_label else "accuracy"
        outputs["fitness"] = outputs.get(primary, 0.0)
        return outputs

    def reset(self) -> None:
        self._buf_scores: list[Any] = []
        self._buf_truth: list[Any] = []
        self._buf_sample_ids: list[Any] = []
        self._class_names: list[str] | None = None

    def from_source(
        self,
        source: Any,
        *,
        scores: Any,
        cls: Any = None,
        truth: Any = None,
        class_names: list[str] | None = None,
        sample_ids: list[str] | None = None,
        rule: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        del source
        truth_raw = cls if cls is not None else truth
        if truth_raw is None:
            raise ValueError("Confusion.from_source requires a ground-truth atom wired as 'cls' (or 'truth').")
        expected_mode = "multi_cls" if self.multi_label else "single_cls"
        if rule is not None and rule["mode"] != expected_mode:
            raise ValueError(f"Confusion(multi_label={self.multi_label}) cannot apply a {rule['mode']} decision rule.")
        if self.multi_label:
            thresholds = self.threshold if rule is None else rule["thresholds"]
            return self._from_multi_label(scores=scores, truth=truth_raw, class_names=class_names, sample_ids=sample_ids, threshold=thresholds)
        weights = None if rule is None else rule["weights"]
        return self._from_single_label(scores=scores, truth=truth_raw, class_names=class_names, sample_ids=sample_ids, weights=weights)

    # -- single-label -----------------------------------------------------------------------

    def _from_single_label(
        self, *, scores: Any, truth: Any, class_names: list[str] | None, sample_ids: list[str] | None, weights: list[float] | None = None
    ) -> dict[str, Any]:
        y_true = [int(t) for t in np.asarray(truth).reshape(-1)]
        score_vectors = np.asarray(scores, dtype=float).reshape(len(y_true), -1).tolist() if y_true else []
        worst_k = self.worst_k

        resolved_classes = list(class_names) if class_names else []
        if not resolved_classes:
            num_classes = max((len(row) for row in score_vectors), default=0) or ((max(y_true) + 1) if y_true else 0)
            resolved_classes = [f"class_{index}" for index in range(num_classes)]

        if not y_true:
            return {"n_samples": 0}

        y_pred: list[int] = []
        sample_quality: list[tuple[str, float]] = []
        sample_rows: list[dict[str, Any]] = []
        for index, (truth_index, raw_scores) in enumerate(zip(y_true, score_vectors, strict=False)):
            score_vector = [float(v) for v in raw_scores]
            ranked = score_vector if weights is None else [score * weight for score, weight in zip(score_vector, weights, strict=True)]
            pred_index = int(max(range(len(ranked)), key=lambda i: ranked[i])) if ranked else 0
            confidence = float(score_vector[pred_index]) if score_vector else None
            pred_label = resolved_classes[pred_index] if 0 <= pred_index < len(resolved_classes) else str(pred_index)
            y_pred.append(pred_index)
            sample_id = str(sample_ids[index]) if sample_ids is not None and index < len(sample_ids) else str(index)
            quality = float(score_vector[truth_index]) if (score_vector and 0 <= truth_index < len(score_vector)) else float(truth_index == pred_index)
            sample_quality.append((sample_id, quality))
            truth_label = resolved_classes[truth_index] if 0 <= truth_index < len(resolved_classes) else str(truth_index)
            sample_rows.append(
                {"sample_id": sample_id, "truth": truth_label, "pred": pred_label, "confidence": confidence, "correct": int(truth_index == pred_index)}
            )

        metrics: dict[str, Any] = {
            "n_samples": len(y_true),
            "accuracy": round(float(sklearn_metrics.accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(sklearn_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
            "balanced_accuracy": round(float(sklearn_metrics.balanced_accuracy_score(y_true, y_pred)), 6),
        }

        per_class_precision, per_class_recall, per_class_f1, support = sklearn_metrics.precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(resolved_classes))), zero_division=0
        )
        class_rows: list[dict[str, Any]] = []
        for index, class_name in enumerate(resolved_classes):
            class_rows.append(
                {
                    "class_name": class_name,
                    "support": int(support[index]),
                    "precision": round(float(per_class_precision[index]), 6),
                    "recall": round(float(per_class_recall[index]), 6),
                    "f1": round(float(per_class_f1[index]), 6),
                }
            )

        confusion = sklearn_metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(resolved_classes)))).astype(float)
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = confusion / np.clip(row_sums, 1.0, None)

        off_diagonal = []
        for true_index, true_name in enumerate(resolved_classes):
            for pred_index, pred_name in enumerate(resolved_classes):
                if true_index == pred_index:
                    continue
                count = float(confusion[true_index, pred_index])
                if count > 0:
                    off_diagonal.append((f"{true_name}→{pred_name}", count))
        off_diagonal.sort(key=lambda item: item[1], reverse=True)
        top_pairs = off_diagonal[: max(worst_k, 1)]

        worst_samples = sorted(sample_quality, key=lambda item: item[1])[: max(worst_k, 1)]

        return {
            **metrics,
            "class_names": resolved_classes,
            "confusion_matrix": confusion.tolist(),
            "confusion_matrix_normalized": normalized.tolist(),
            "class_rows": class_rows,
            "top_confusion_pairs": [{"pair": name, "count": count} for name, count in top_pairs],
            "sample_rows": sample_rows,
            "worst_samples": [{"sample_id": label, "value": value} for label, value in worst_samples],
        }

    # -- multi-label --------------------------------------------------------------------------

    def _from_multi_label(
        self, *, scores: Any, truth: Any, class_names: list[str] | None, sample_ids: list[str] | None, threshold: float | list[float] | None
    ) -> dict[str, Any]:
        truth_rows = np.asarray(truth, dtype=int).tolist()
        score_rows = np.asarray(scores, dtype=float).tolist()
        worst_k = self.worst_k

        resolved_classes = list(class_names) if class_names else []
        if not resolved_classes:
            num_classes = max((len(row) for row in score_rows), default=0) or max((len(row) for row in truth_rows), default=0)
            resolved_classes = [f"class_{index}" for index in range(num_classes)]
        thresholds = _resolve_thresholds(len(resolved_classes), threshold)

        if not truth_rows:
            return {"n_samples": 0}

        pred_rows: list[list[int]] = [
            [int(value >= threshold) for value, threshold in zip(score_vector, thresholds, strict=False)] for score_vector in score_rows
        ]
        sample_rows: list[dict[str, Any]] = []
        for index, (truth_vector, score_vector, pred_vector) in enumerate(zip(truth_rows, score_rows, pred_rows, strict=False)):
            sample_id = str(sample_ids[index]) if sample_ids is not None and index < len(sample_ids) else str(index)
            row = {
                "sample_id": sample_id,
                "true_label_count": int(sum(truth_vector)),
                "pred_label_count": int(sum(pred_vector)),
                "exact_match": int(list(truth_vector) == list(pred_vector)),
                "label_cardinality_error": abs(int(sum(pred_vector)) - int(sum(truth_vector))),
            }
            for class_name, truth_value, pred_value, score_value in zip(resolved_classes, truth_vector, pred_vector, score_vector, strict=False):
                row[f"truth_{class_name}"] = int(truth_value)
                row[f"pred_{class_name}"] = int(pred_value)
                row[f"score_{class_name}"] = float(score_value)
            sample_rows.append(row)

        truth_array = np.asarray(truth_rows, dtype=int)
        pred_array = np.asarray(pred_rows, dtype=int)

        metrics: dict[str, Any] = {
            "n_samples": int(len(sample_rows)),
            "micro_f1": round(float(sklearn_metrics.f1_score(truth_array, pred_array, average="micro", zero_division=0)), 6),
            "macro_f1": round(float(sklearn_metrics.f1_score(truth_array, pred_array, average="macro", zero_division=0)), 6),
            "sample_f1": round(float(sklearn_metrics.f1_score(truth_array, pred_array, average="samples", zero_division=0)), 6),
            "exact_match_accuracy": round(float(np.mean(np.all(truth_array == pred_array, axis=1))), 6),
            "hamming_loss": round(float(sklearn_metrics.hamming_loss(truth_array, pred_array)), 6),
            "label_cardinality_error": round(float(np.mean(np.abs(truth_array.sum(axis=1) - pred_array.sum(axis=1)))), 6),
        }

        supports = truth_array.sum(axis=0)
        class_tp: list[float] = []
        class_fp: list[float] = []
        class_fn: list[float] = []
        class_rows: list[dict[str, Any]] = []
        for index, class_name in enumerate(resolved_classes):
            tp = float(((truth_array[:, index] == 1) & (pred_array[:, index] == 1)).sum())
            fp = float(((truth_array[:, index] == 0) & (pred_array[:, index] == 1)).sum())
            fn = float(((truth_array[:, index] == 1) & (pred_array[:, index] == 0)).sum())
            class_tp.append(tp)
            class_fp.append(fp)
            class_fn.append(fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            class_rows.append(
                {
                    "class_name": class_name,
                    "support": int(supports[index]),
                    "prevalence": round(float(supports[index] / max(len(sample_rows), 1)), 6),
                    "precision": round(float(precision), 6),
                    "recall": round(float(recall), 6),
                    "f1": round(float(f1), 6),
                    "threshold": round(float(thresholds[index]), 6),
                }
            )

        sample_f1 = []
        for index, row in enumerate(sample_rows):
            truth_vector = truth_rows[index]
            pred_vector = pred_rows[index]
            tp = float(sum(t == 1 and p == 1 for t, p in zip(truth_vector, pred_vector, strict=False)))
            fp = float(sum(t == 0 and p == 1 for t, p in zip(truth_vector, pred_vector, strict=False)))
            fn = float(sum(t == 1 and p == 0 for t, p in zip(truth_vector, pred_vector, strict=False)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            sample_f1.append((str(row["sample_id"] or index), f1))
        worst_samples = sorted(sample_f1, key=lambda item: item[1])[: max(worst_k, 1)]

        return {
            **metrics,
            "class_names": resolved_classes,
            "class_rows": class_rows,
            "confusion_grid": {"tp": class_tp, "fp": class_fp, "fn": class_fn},
            "sample_rows": sample_rows,
            "worst_samples": [{"sample_id": label, "value": value} for label, value in worst_samples],
        }
