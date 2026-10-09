"""
RankedAveragePrecision metric: the score-ranked AP/ROC family under aligned pairing.

Source:
  - title: "The PASCAL Visual Object Classes (VOC) Challenge"
    url: "https://doi.org/10.1007/s11263-009-0275-4"
    year: 2010
  - title: "Microsoft COCO: Common Objects in Context"
    url: "https://arxiv.org/abs/1405.0312"
    year: 2014

Description:
  Covers AP, ROC, PR curves, top-k accuracy and log-loss for single-label and multi-label
  classification. Calibration (ECE, reliability diagram) is out of scope; see :mod:`.calibration`.

The metric also accepts a ``matched`` input, the ``rows`` and per-category ``ground_truth`` counts of
an upstream :class:`~teia.node.eval.metric.aligned.instance_match.InstanceMatch`
returns, so ``det``/``obb``/``inst-seg``/``pose`` reuse this same metric for their AP rather
than defining a separate detection-AP metric.

Param naming: ``cls`` is the literal capture-store atom for both ``cls`` and ``multi-cls``
targets; ``truth`` is accepted as an alias.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sklearn_metrics

from teia.base.eval import Metric
from teia.node.eval.metric.aligned._shared import interpolated_ap, label_binarize_indices, metric_curve_from_scores

__all__ = ["RankedAveragePrecision"]


class RankedAveragePrecision(Metric):
    """Score-ranked average precision / ROC AUC, single-label, multi-label, or matched-rows.

    Constructor kwargs (leaf config):
      - ``multi_label``: selects the ``scores``/``cls`` report shape (default ``False``).
      - ``top_k``: top-k accuracy cutoff, single-label only (default ``3``).
      - ``worst_k``: how many worst-by-AP per-class PR curves to keep, plus the macro curve
        (default ``5``).
      - ``is_fitness``: publish ``"fitness"`` (default ``False``). det/obb/inst-seg/pose each
        wire one ``RankedAveragePrecision`` per route; the route that should own ``val/fitness``
        sets this kwarg, so its mAP becomes the monitored fitness with no task-string table.
        cls/multi-cls leave it ``False`` (their fitness, if any, comes from ``Confusion``).
    """

    def __init__(self, multi_label: bool = False, top_k: int = 3, worst_k: int = 5, is_fitness: bool = False) -> None:
        self.multi_label = bool(multi_label)
        self.top_k = int(top_k)
        self.worst_k = int(worst_k)
        self.MONITORS = ("mean_average_precision", "fitness") if is_fitness else ("mean_average_precision",)
        self.reset()

    # Streaming: AP is inherently whole-split (it ranks every prediction against every other), so
    # nothing meaningful is computable per batch. This node is fed once, at epoch end, via the
    # streaming runner's synthetic update() call (see core/eval/streaming.py): `matched` is then
    # InstanceMatch.compute()'s buffered whole-split rows.

    def update(self, *, matched: Any = None, **_: Any) -> None:
        if matched is not None:
            self._matched = matched

    def compute(self) -> dict[str, Any]:
        if self._matched is None:
            return {"n_samples": 0}
        outputs = self._from_matched(self._matched)
        outputs["fitness"] = outputs.get("mean_average_precision", 0.0)
        return outputs

    def reset(self) -> None:
        self._matched: Any = None

    def from_source(
        self,
        source: Any,
        *,
        scores: Any = None,
        cls: Any = None,
        truth: Any = None,
        matched: Any = None,
        class_names: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        del source
        if matched is not None:
            return self._from_matched(matched)
        truth_raw = cls if cls is not None else truth
        if scores is None or truth_raw is None:
            raise ValueError("RankedAveragePrecision.from_source requires either 'matched' or ('scores' and 'cls').")
        if self.multi_label:
            return self._from_multi_label(scores=scores, truth=truth_raw, class_names=class_names)
        return self._from_single_label(scores=scores, truth=truth_raw, class_names=class_names)

    # det-family path: matched rows -> per-category ranked AP, averaged over overlap thresholds

    def _from_matched(self, matched: Any) -> dict[str, Any]:
        """``matched``: the dict an upstream ``InstanceMatch`` returns. ``rows`` holds one
        ``{score, category, is_true_positive, threshold}`` per prediction per threshold, and
        ``ground_truth`` the instance count per category, which is the recall denominator (a
        missed instance costs recall). Per threshold and category the rows are ranked by score
        and scored with interpolated AP. Category AP averages over thresholds, and
        ``mean_average_precision`` averages categories (those with ground truth) and thresholds,
        the COCO ``AP@[.50:.95]`` when the matcher ran the ten COCO thresholds. ``per_threshold``
        and ``ap50`` / ``ap75`` read the same matrix.
        """
        if not isinstance(matched, dict) or "ground_truth" not in matched:
            raise ValueError("matched must be the InstanceMatch output: a dict with 'rows' and 'ground_truth'.")
        ground_truth = {str(category): int(count) for category, count in matched["ground_truth"].items()}
        by_cell: dict[tuple[float, str], list[tuple[float, bool]]] = {}
        for row in matched["rows"]:
            cell = (float(row["threshold"]), str(row["category"]))
            by_cell.setdefault(cell, []).append((float(row["score"]), bool(row["is_true_positive"])))
        thresholds = sorted({threshold for threshold, _ in by_cell})
        categories = sorted(category for category, count in ground_truth.items() if count > 0)

        ap = np.zeros((len(thresholds), len(categories)))
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for t_index, threshold in enumerate(thresholds):
            for c_index, category in enumerate(categories):
                ranked = sorted(by_cell.get((threshold, category), []), key=lambda item: item[0], reverse=True)
                tp_flags = np.array([1.0 if is_tp else 0.0 for _, is_tp in ranked], dtype=float)
                cum_tp = np.cumsum(tp_flags)
                cum_fp = np.cumsum(1.0 - tp_flags)
                recalls = cum_tp / ground_truth[category]
                precisions = cum_tp / np.clip(cum_tp + cum_fp, 1e-9, None)
                ap[t_index, c_index] = interpolated_ap(recalls, precisions)
                if t_index == 0:
                    curves[category] = (recalls, precisions)

        category_ap = ap.mean(axis=0) if thresholds else np.zeros(len(categories))
        per_category = [
            {
                "category": category,
                "n_ground_truth": ground_truth[category],
                "n_true_positive": int(sum(is_tp for _, is_tp in by_cell.get((thresholds[0], category), []))) if thresholds else 0,
                "average_precision": round(float(category_ap[c_index]), 6),
                "recall": curves[category][0].tolist() if category in curves else [],
                "precision": curves[category][1].tolist() if category in curves else [],
            }
            for c_index, category in enumerate(categories)
        ]
        per_threshold = {f"{threshold:g}": round(float(ap[t_index].mean()), 6) for t_index, threshold in enumerate(thresholds)} if categories else {}
        result: dict[str, Any] = {
            "mean_average_precision": round(float(ap.mean()), 6) if ap.size else 0.0,
            "per_threshold": per_threshold,
            "per_category": per_category,
            # Pre-shaped for MultiCurveBundle's `curves: list[{"label","x","y"}]` kwarg (first threshold).
            "pr_curves": [
                {"label": f"{row['category']} (AP={row['average_precision']:.3f})", "x": row["recall"], "y": row["precision"]}
                for row in per_category
            ],
        }
        for name, value in (("ap50", 0.5), ("ap75", 0.75)):
            key = next((k for k in per_threshold if abs(float(k) - value) < 1e-9), None)
            if key is not None:
                result[name] = per_threshold[key]
        return result

    # -- single-label -----------------------------------------------------------------------

    def _from_single_label(self, *, scores: Any, truth: Any, class_names: list[str] | None) -> dict[str, Any]:
        y_true = [int(t) for t in np.asarray(truth).reshape(-1)]
        score_vectors = np.asarray(scores, dtype=float).reshape(len(y_true), -1).tolist() if y_true else []

        resolved_classes = list(class_names) if class_names else []
        if not resolved_classes:
            num_classes = max((len(row) for row in score_vectors), default=0) or ((max(y_true) + 1) if y_true else 0)
            resolved_classes = [f"class_{index}" for index in range(num_classes)]

        if not y_true or len(score_vectors) != len(y_true):
            return {"n_samples": len(y_true)}

        score_array = np.asarray([[float(v) for v in row] for row in score_vectors], dtype=float)
        truth_bin = label_binarize_indices(y_true, len(resolved_classes))
        top_k_used = min(self.top_k, len(resolved_classes))
        top_indices = np.argsort(score_array, axis=1)[:, -top_k_used:]

        metrics: dict[str, Any] = {"n_samples": len(y_true)}
        metrics[f"top_{self.top_k}_accuracy"] = round(
            float(np.mean([int(true_index in top_indices[row_index]) for row_index, true_index in enumerate(y_true)])), 6
        )
        ap_values = sklearn_metrics.average_precision_score(truth_bin, score_array, average=None)
        metrics["macro_average_precision"] = round(float(np.mean(ap_values)), 6)
        metrics["log_loss"] = round(float(sklearn_metrics.log_loss(y_true, score_array, labels=list(range(len(resolved_classes))))), 6)

        class_rows: list[dict[str, Any]] = [
            {"class_name": name, "average_precision": float("nan"), "roc_auc": float("nan")}
            for name in resolved_classes
        ]
        curves = []
        recall_grid = np.linspace(0.0, 1.0, 101)
        macro_precision = np.zeros_like(recall_grid)
        order = sorted(range(len(resolved_classes)), key=lambda idx: float(ap_values[idx]))
        worst_class_indices = order[: min(self.worst_k, len(order))]
        for class_index, class_name in enumerate(resolved_classes):
            y_true_bin = truth_bin[:, class_index]
            if y_true_bin.max() == y_true_bin.min():
                continue
            curve = metric_curve_from_scores(y_true_bin, score_array[:, class_index])
            precision = np.asarray(curve["precision"])
            recall = np.asarray(curve["recall"])
            unique_recall, unique_index = np.unique(recall, return_index=True)
            interp_precision = np.interp(recall_grid, unique_recall, precision[unique_index], left=precision[0], right=precision[-1])
            macro_precision += interp_precision
            class_rows[class_index]["average_precision"] = round(float(curve["average_precision"]), 6)
            class_rows[class_index]["roc_auc"] = round(float(curve["roc_auc"]), 6)
            if class_index in worst_class_indices:
                curves.append({"label": f"{class_name} (AP={curve['average_precision']:.3f})", "x": curve["recall"], "y": curve["precision"]})
        if curves:
            macro_precision /= max(len(resolved_classes), 1)
            curves.insert(0, {"label": "macro", "x": recall_grid.tolist(), "y": macro_precision.tolist()})

        return {**metrics, "class_names": resolved_classes, "class_rows": class_rows, "pr_curves": curves}

    # -- multi-label --------------------------------------------------------------------------

    def _from_multi_label(self, *, scores: Any, truth: Any, class_names: list[str] | None) -> dict[str, Any]:
        truth_rows = np.asarray(truth, dtype=int).tolist()
        score_rows = np.asarray(scores, dtype=float).tolist()

        resolved_classes = list(class_names) if class_names else []
        if not resolved_classes:
            num_classes = max((len(row) for row in score_rows), default=0) or max((len(row) for row in truth_rows), default=0)
            resolved_classes = [f"class_{index}" for index in range(num_classes)]

        if not truth_rows:
            return {"n_samples": 0}

        truth_array = np.asarray(truth_rows, dtype=int)
        score_array = np.asarray(score_rows, dtype=float)

        ap_values = sklearn_metrics.average_precision_score(truth_array, score_array, average=None)
        metrics: dict[str, Any] = {
            "n_samples": int(truth_array.shape[0]),
            "mean_average_precision": round(float(np.mean(ap_values)), 6),
        }

        class_rows: list[dict[str, Any]] = [
            {"class_name": name, "average_precision": float("nan"), "roc_auc": float("nan")}
            for name in resolved_classes
        ]
        class_has_roc: list[float] = []
        roc_curves: list[dict[str, Any]] = []
        for index, class_name in enumerate(resolved_classes):
            class_rows[index]["average_precision"] = round(float(ap_values[index]), 6)
            if truth_array[:, index].max() != truth_array[:, index].min():
                roc_auc = sklearn_metrics.roc_auc_score(truth_array[:, index], score_array[:, index])
                class_rows[index]["roc_auc"] = round(float(roc_auc), 6)
                class_has_roc.append(float(roc_auc))
                fpr, tpr, _ = sklearn_metrics.roc_curve(truth_array[:, index], score_array[:, index])
                roc_curves.append({"label": f"{class_name} (AUC={roc_auc:.3f})", "x": fpr.tolist(), "y": tpr.tolist()})
        if class_has_roc:
            metrics["macro_roc_auc"] = round(float(np.mean(class_has_roc)), 6)
        try:
            metrics["label_ranking_loss"] = round(float(sklearn_metrics.label_ranking_loss(truth_array, score_array)), 6)
        except ValueError:
            pass

        sweep_thresholds = np.linspace(0.05, 0.95, 19)
        threshold_rows: list[dict[str, Any]] = []
        micro_f1_curve: list[float] = []
        macro_f1_curve: list[float] = []
        for sweep in sweep_thresholds:
            sweep_pred = (score_array >= sweep).astype(int)
            micro_f1 = float(sklearn_metrics.f1_score(truth_array, sweep_pred, average="micro", zero_division=0))
            macro_f1 = float(sklearn_metrics.f1_score(truth_array, sweep_pred, average="macro", zero_division=0))
            threshold_rows.append({"threshold": round(float(sweep), 6), "micro_f1": round(micro_f1, 6), "macro_f1": round(macro_f1, 6)})
            micro_f1_curve.append(micro_f1)
            macro_f1_curve.append(macro_f1)

        curves = []
        recall_grid = np.linspace(0.0, 1.0, 101)
        macro_precision = np.zeros_like(recall_grid)
        order = sorted(range(len(resolved_classes)), key=lambda idx: float(ap_values[idx]))
        worst_indices = order[: min(self.worst_k, len(order))]
        valid_curve_count = 0
        for index, class_name in enumerate(resolved_classes):
            if truth_array[:, index].max() == truth_array[:, index].min():
                continue
            precision, recall, _ = sklearn_metrics.precision_recall_curve(truth_array[:, index], score_array[:, index])
            unique_recall, unique_index = np.unique(recall, return_index=True)
            interp_precision = np.interp(recall_grid, unique_recall, precision[unique_index], left=precision[0], right=precision[-1])
            macro_precision += interp_precision
            valid_curve_count += 1
            if index in worst_indices:
                curves.append({"label": f"{class_name} (AP={ap_values[index]:.3f})", "x": recall.tolist(), "y": precision.tolist()})
        if valid_curve_count:
            macro_precision /= valid_curve_count
            curves.insert(0, {"label": "macro", "x": recall_grid.tolist(), "y": macro_precision.tolist()})

        return {
            **metrics,
            "class_names": resolved_classes,
            "class_rows": class_rows,
            "pr_curves": curves,
            "roc_curves": roc_curves,
            "threshold_sweep": threshold_rows,
            "threshold_sweep_series": [{"label": "micro_f1", "y": micro_f1_curve}, {"label": "macro_f1", "y": macro_f1_curve}],
        }
