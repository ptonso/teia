"""Correctness tests for teia.node.eval.metric.aligned.*.

These were originally written as parity tests against the pre-eval-plane
``teia.core.report.objectives.*`` modules (proving the Part 4b lift preserved every formula
exactly); that legacy code was deleted once the Part 4c cutover replaced its only callers (see
the Part 4 cleanup pass). The parity was verified once, then. What's left worth keeping is
verifying the *formulas themselves* are correct — so each test now computes its expected value
directly (via the same sklearn/numpy primitives the metric internally uses), rather than via
code that no longer exists.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sklearn_metrics

from teia.node.eval.metric.aligned.calibration import Calibration
from teia.node.eval.metric.aligned.concordance import Concordance
from teia.node.eval.metric.aligned.confusion import Confusion
from teia.node.eval.metric.aligned.horizon_error import HorizonError
from teia.node.eval.metric.aligned.km_estimate import KmEstimate
from teia.node.eval.metric.aligned.ranked_average_precision import RankedAveragePrecision
from teia.node.eval.metric.aligned.residuals import Residuals


# --------------------------------------------------------------------------------------------
# Single-label classification: Confusion + RankedAveragePrecision + Calibration
# --------------------------------------------------------------------------------------------


def _cls_fixture():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    score_vectors = [
        [0.7, 0.3],
        [0.4, 0.6],
        [0.9, 0.1],
        [0.2, 0.8],
        [0.55, 0.45],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.1, 0.9],
    ]
    sample_ids = [str(i) for i in range(len(y_true))]
    class_names = ["a", "b"]
    return y_true, score_vectors, sample_ids, class_names


class TestClsMetrics:
    def test_confusion_scalar_metrics_match_sklearn(self) -> None:
        y_true, score_vectors, sample_ids, class_names = _cls_fixture()
        y_pred = [int(np.argmax(row)) for row in score_vectors]

        new = Confusion(multi_label=False, worst_k=5).from_source(None, scores=score_vectors, cls=y_true, class_names=class_names, sample_ids=sample_ids)

        assert new["accuracy"] == pytest.approx(sklearn_metrics.accuracy_score(y_true, y_pred))
        assert new["macro_f1"] == pytest.approx(sklearn_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))
        assert new["balanced_accuracy"] == pytest.approx(sklearn_metrics.balanced_accuracy_score(y_true, y_pred))

        expected_confusion = sklearn_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        assert new["confusion_matrix"] == expected_confusion.tolist()

        precision, recall, f1, support = sklearn_metrics.precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        for index, row in enumerate(new["class_rows"]):
            assert row["support"] == int(support[index])
            assert row["precision"] == pytest.approx(precision[index])
            assert row["recall"] == pytest.approx(recall[index])
            assert row["f1"] == pytest.approx(f1[index])

    def test_ranked_average_precision_matches_sklearn(self) -> None:
        y_true, score_vectors, sample_ids, class_names = _cls_fixture()
        score_array = np.asarray(score_vectors)
        truth_bin = np.eye(2)[y_true]

        new = RankedAveragePrecision(multi_label=False, top_k=3, worst_k=5).from_source(None, scores=score_vectors, cls=y_true, class_names=class_names)

        ap_values = sklearn_metrics.average_precision_score(truth_bin, score_array, average=None)
        assert new["macro_average_precision"] == pytest.approx(float(np.mean(ap_values)))
        assert new["log_loss"] == pytest.approx(sklearn_metrics.log_loss(y_true, score_array, labels=[0, 1]))

        for index, row in enumerate(new["class_rows"]):
            assert row["average_precision"] == pytest.approx(ap_values[index])

        # macro PR curve is the mean of each class's precision, interpolated onto a 101-point
        # recall grid — sanity-check its endpoints rather than re-deriving the full curve.
        macro_curve = next(c for c in new["pr_curves"] if c["label"] == "macro")
        assert macro_curve["x"][0] == pytest.approx(0.0)
        assert macro_curve["x"][-1] == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in macro_curve["y"])

    def test_calibration_error_matches_hand_binned_computation(self) -> None:
        y_true, score_vectors, sample_ids, class_names = _cls_fixture()
        score_array = np.asarray(score_vectors)
        confidences = score_array.max(axis=1)
        predictions = score_array.argmax(axis=1)
        correct = (predictions == np.asarray(y_true)).astype(float)

        new = Calibration(bins=10).from_source(None, scores=score_vectors, cls=y_true)

        # Expected calibration error: sum over occupied bins of |bin_accuracy - bin_confidence|
        # weighted by bin occupancy — recomputed directly against the same confidence/correctness
        # arrays the metric itself derives.
        edges = np.linspace(0.0, 1.0, 11)
        expected_ece = 0.0
        for index in range(10):
            left, right = edges[index], edges[index + 1]
            mask = (confidences >= left) & (confidences < right) if index < 9 else (confidences >= left) & (confidences <= right)
            if not mask.any():
                continue
            expected_ece += abs(float(correct[mask].mean()) - float(confidences[mask].mean())) * (mask.sum() / len(confidences))
        assert new["calibration_error"] == pytest.approx(expected_ece)

        assert sorted(new["confidence_histogram"]["correct"]) == pytest.approx(sorted(confidences[correct == 1].tolist()))
        assert sorted(new["confidence_histogram"]["incorrect"]) == pytest.approx(sorted(confidences[correct == 0].tolist()))


# --------------------------------------------------------------------------------------------
# Multi-label classification: Confusion(multi_label=True) + RankedAveragePrecision(multi_label=True)
# --------------------------------------------------------------------------------------------


def _multi_cls_fixture():
    truth_rows = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]]
    score_rows = [
        [0.8, 0.2, 0.1],
        [0.3, 0.7, 0.2],
        [0.6, 0.55, 0.1],
        [0.2, 0.1, 0.9],
        [0.1, 0.6, 0.65],
        [0.7, 0.2, 0.55],
    ]
    class_names = ["red", "green", "blue"]
    sample_ids = [str(i) for i in range(len(truth_rows))]
    return truth_rows, score_rows, sample_ids, class_names


class TestMultiClsMetrics:
    def test_confusion_scalar_metrics_match_sklearn(self) -> None:
        truth_rows, score_rows, sample_ids, class_names = _multi_cls_fixture()
        truth_array = np.asarray(truth_rows)
        pred_array = (np.asarray(score_rows) >= 0.5).astype(int)

        new = Confusion(multi_label=True, worst_k=5, threshold=0.5).from_source(None, scores=score_rows, cls=truth_rows, class_names=class_names, sample_ids=sample_ids)

        assert new["micro_f1"] == pytest.approx(sklearn_metrics.f1_score(truth_array, pred_array, average="micro", zero_division=0))
        assert new["macro_f1"] == pytest.approx(sklearn_metrics.f1_score(truth_array, pred_array, average="macro", zero_division=0))
        assert new["sample_f1"] == pytest.approx(sklearn_metrics.f1_score(truth_array, pred_array, average="samples", zero_division=0))
        assert new["exact_match_accuracy"] == pytest.approx(float(np.mean(np.all(truth_array == pred_array, axis=1))))
        assert new["hamming_loss"] == pytest.approx(sklearn_metrics.hamming_loss(truth_array, pred_array))

        for index, row in enumerate(new["class_rows"]):
            tp = float(((truth_array[:, index] == 1) & (pred_array[:, index] == 1)).sum())
            fp = float(((truth_array[:, index] == 0) & (pred_array[:, index] == 1)).sum())
            fn = float(((truth_array[:, index] == 1) & (pred_array[:, index] == 0)).sum())
            expected_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            expected_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            assert row["support"] == int(truth_array[:, index].sum())
            assert row["precision"] == pytest.approx(expected_precision)
            assert row["recall"] == pytest.approx(expected_recall)

        assert new["confusion_grid"]["tp"][0] == float(((truth_array[:, 0] == 1) & (pred_array[:, 0] == 1)).sum())

    def test_ranked_average_precision_matches_sklearn(self) -> None:
        truth_rows, score_rows, sample_ids, class_names = _multi_cls_fixture()
        truth_array = np.asarray(truth_rows)
        score_array = np.asarray(score_rows)

        new = RankedAveragePrecision(multi_label=True, worst_k=5).from_source(None, scores=score_rows, cls=truth_rows, class_names=class_names)

        ap_values = sklearn_metrics.average_precision_score(truth_array, score_array, average=None)
        assert new["mean_average_precision"] == pytest.approx(float(np.mean(ap_values)))
        for index, row in enumerate(new["class_rows"]):
            assert row["average_precision"] == pytest.approx(ap_values[index])

        assert len(new["threshold_sweep"]) == 19
        sweep = new["threshold_sweep"][9]  # middle of linspace(0.05, 0.95, 19) ~= 0.5
        sweep_pred = (score_array >= sweep["threshold"]).astype(int)
        expected_micro = float(sklearn_metrics.f1_score(truth_array, sweep_pred, average="micro", zero_division=0))
        assert sweep["micro_f1"] == pytest.approx(expected_micro)


# --------------------------------------------------------------------------------------------
# Regression: Residuals
# --------------------------------------------------------------------------------------------


class TestRegMetrics:
    def test_residuals_matches_hand_computation(self) -> None:
        pred = [[1.1], [2.2], [2.9], [4.1]]
        targets = [[1.0], [2.0], [3.0], [4.0]]
        target_names = ["y"]

        new = Residuals().from_source(None, pred=pred, target=targets, target_names=target_names)

        preds = np.asarray(pred).reshape(-1)
        tgts = np.asarray(targets).reshape(-1)
        err = preds - tgts
        assert new["rmse"] == pytest.approx(float(np.sqrt(np.mean(err**2))), abs=1e-5)
        assert new["mae"] == pytest.approx(float(np.mean(np.abs(err))), abs=1e-5)
        assert new["bias"] == pytest.approx(float(np.mean(err)), abs=1e-5)
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((tgts - tgts.mean()) ** 2))
        assert new["r2"] == pytest.approx(1.0 - ss_res / ss_tot, abs=1e-5)

        assert new["actual"] == pytest.approx(tgts.tolist())
        assert new["pred"] == pytest.approx(preds.tolist())
        assert new["target_rows"][0]["rmse"] == pytest.approx(new["rmse"])


# --------------------------------------------------------------------------------------------
# Forecast: HorizonError
# --------------------------------------------------------------------------------------------


class TestForecastMetrics:
    def test_horizon_error_matches_hand_computation(self) -> None:
        levels = [0.1, 0.5, 0.9]
        quantiles = [
            [[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]],
            [[1.9, 2.0, 2.1], [2.9, 3.0, 3.1]],
            [[0.8, 1.2, 1.6], [1.8, 2.2, 2.6]],
            [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]],
        ]
        targets = [[1.0, 2.0], [2.0, 3.0], [1.0, 2.0], [1.5, 2.5]]

        new = HorizonError(levels=levels).from_source(None, quantiles=quantiles, target=targets)

        # level 0.5 is the median index (1); horizon step 0's median predictions are the
        # middle quantile value at each sample.
        median = np.asarray(quantiles)[:, :, 1]
        tgt = np.asarray(targets)
        err = median - tgt
        assert new["overall"]["mae"] == pytest.approx(float(np.mean(np.abs(err))), abs=1e-6)
        assert new["overall"]["rmse"] == pytest.approx(float(np.sqrt(np.mean(err**2))), abs=1e-6)
        assert new["per_step"]["step_0"]["mae"] == pytest.approx(float(np.mean(np.abs(err[:, 0]))), abs=1e-6)

        mae_per_horizon = np.mean(np.abs(err), axis=0)
        assert new["horizon_mae_curves"][0]["y"] == pytest.approx(mae_per_horizon.tolist(), abs=1e-6)


# --------------------------------------------------------------------------------------------
# Survival: Concordance + KmEstimate
# --------------------------------------------------------------------------------------------


class TestSurvivalMetrics:
    def test_concordance_matches_pinned_regression_values(self) -> None:
        """Pinned against a hand-worked concordance/IPCW-concordance computation for this
        5-sample fixture (also cross-checked in tests/test_eval_bundle_smoke.py's survival
        bundle end-to-end test)."""
        time = np.array([1, 2, 3, 4, 5], dtype=float)
        event = np.array([0, 1, 0, 1, 1], dtype=int)
        risk = np.array([0.1, 0.5, 0.4, 0.6, 0.7], dtype=float)

        new = Concordance().from_source(None, time=time, event=event, risk=risk)
        assert new["c_index"] == pytest.approx(0.25)
        assert new["ipcw_c_index"] == pytest.approx(4.0 / 21.0)

    def test_concordance_uses_train_split_when_given(self) -> None:
        """train_time/train_event, when supplied, are used for censoring instead of the eval
        split's own (time, event) — proving the Part-6 side-artifact hook works, even though no
        writer produces it yet. Cross-checked: using the eval split's own censoring (no
        train_time/train_event) gives a different ipcw_c_index for this fixture, confirming the
        override actually takes effect rather than silently falling back."""
        time = np.array([1, 2, 3, 4, 5], dtype=float)
        event = np.array([0, 1, 0, 1, 1], dtype=int)
        risk = np.array([0.1, 0.5, 0.4, 0.6, 0.7], dtype=float)
        train_time = np.array([1, 1, 2, 3, 5, 8], dtype=float)
        train_event = np.array([1, 0, 1, 1, 0, 1], dtype=int)

        without_override = Concordance().from_source(None, time=time, event=event, risk=risk)
        with_override = Concordance().from_source(None, time=time, event=event, risk=risk, train_time=train_time, train_event=train_event)

        assert with_override["ipcw_c_index"] != pytest.approx(without_override["ipcw_c_index"])

    def test_km_estimate_splits_into_two_risk_groups(self) -> None:
        time = np.array([1, 2, 3, 4, 5], dtype=float)
        risk = np.array([0.1, 0.5, 0.4, 0.6, 0.7], dtype=float)

        new = KmEstimate().from_source(None, time=time, risk=risk)

        labels = {c["label"] for c in new["curves"]}
        assert labels == {"high risk", "low risk"}
        for curve in new["curves"]:
            # A step survival curve starts below 1.0 and is non-increasing.
            assert curve["y"][0] < 1.0
            assert all(a >= b for a, b in zip(curve["y"], curve["y"][1:]))


# --------------------------------------------------------------------------------------------
# RankedAveragePrecision's future det-family ``matched`` path (nothing wires this yet).
# --------------------------------------------------------------------------------------------


def _rows(threshold: float, *entries: tuple[float, str, bool]) -> list[dict]:
    return [{"score": score, "category": category, "is_true_positive": tp, "threshold": threshold} for score, category, tp in entries]


class TestRankedAveragePrecisionMatchedPath:
    def test_rows_grouped_and_ranked_by_category(self) -> None:
        # category "a" is perfectly ranked; "b" has one FP ahead of its lone TP.
        matched = {
            "rows": _rows(0.5, (0.95, "a", True), (0.9, "a", True), (0.4, "a", False), (0.99, "b", False), (0.8, "b", True)),
            "ground_truth": {"a": 2, "b": 1},
        }
        result = RankedAveragePrecision().from_source(None, matched=matched)

        by_category = {row["category"]: row for row in result["per_category"]}
        assert by_category["a"]["average_precision"] == pytest.approx(1.0)
        assert by_category["b"]["average_precision"] < 1.0
        assert result["mean_average_precision"] == pytest.approx((by_category["a"]["average_precision"] + by_category["b"]["average_precision"]) / 2.0)

    def test_missed_ground_truth_costs_recall(self) -> None:
        # one true positive out of four ground-truth instances: recall tops out at 0.25, so AP is 0.26 (26 of 101 recall points).
        matched = {"rows": _rows(0.5, (0.9, "x", True)), "ground_truth": {"x": 4}}
        result = RankedAveragePrecision().from_source(None, matched=matched)
        assert result["per_category"][0]["average_precision"] == pytest.approx(26 / 101, abs=1e-6)

    def test_category_without_predictions_scores_zero(self) -> None:
        matched = {"rows": _rows(0.5, (0.9, "a", True)), "ground_truth": {"a": 1, "b": 3}}
        result = RankedAveragePrecision().from_source(None, matched=matched)
        by_category = {row["category"]: row["average_precision"] for row in result["per_category"]}
        assert by_category == {"a": pytest.approx(1.0), "b": 0.0}
        assert result["mean_average_precision"] == pytest.approx(0.5)

    def test_average_over_overlap_thresholds(self) -> None:
        # the prediction matches at 0.5 but not at 0.75: AP is 1.0 and 0.0, and the mean is 0.5.
        matched = {
            "rows": _rows(0.5, (0.9, "x", True)) + _rows(0.75, (0.9, "x", False)),
            "ground_truth": {"x": 1},
        }
        result = RankedAveragePrecision().from_source(None, matched=matched)
        assert result["per_threshold"] == {"0.5": pytest.approx(1.0), "0.75": 0.0}
        assert result["ap50"] == pytest.approx(1.0)
        assert result["ap75"] == 0.0
        assert result["mean_average_precision"] == pytest.approx(0.5)

    def test_rows_without_ground_truth_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="ground_truth"):
            RankedAveragePrecision().from_source(None, matched=_rows(0.5, (0.9, "x", True)))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
