"""Threshold resolution/application (core: consumed
by report rendering + prediction materialization, not evaluation) stays tested against
``teia.core.capture.thresholds``; the pure F1 sweep that moved to ``teia`` as the
``ThresholdOptimization`` metric is tested against its new home directly.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


import numpy as np

from teia.node.eval.metric.aligned.confusion import Confusion
from teia.node.eval.metric.aligned.threshold_optimization import (
    ThresholdOptimization,
    _binary_metrics,
    _optimize_binary_threshold,
    optimize_multi_label_thresholds,
    optimize_single_label_weights,
)
from teia.node.net._activation_utils import decode_singlelabel


class ClassificationThresholdTests(unittest.TestCase):
    def test_optimize_multi_label_thresholds_picks_expected_per_class_f1_thresholds(self) -> None:
        result = optimize_multi_label_thresholds(
            probabilities=[
                [0.9, 0.1],
                [0.7, 0.8],
                [0.1, 0.6],
            ],
            targets=[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            label_names=["alpha", "beta"],
        )

        self.assertEqual(result["thresholds"], [0.9, 0.6])
        self.assertEqual(result["per_class"][0]["label"], "alpha")
        self.assertAlmostEqual(result["per_class"][0]["f1"], 1.0)
        self.assertAlmostEqual(result["per_class"][1]["f1"], 1.0)

    def test_optimize_multi_label_thresholds_falls_back_for_degenerate_classes(self) -> None:
        result = optimize_multi_label_thresholds(
            probabilities=[
                [0.2, 0.9],
                [0.4, 0.7],
            ],
            targets=[
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            label_names=["alpha", "beta"],
            fallback_threshold=0.5,
        )

        self.assertEqual(result["thresholds"], [0.5, 0.5])
        self.assertTrue(result["per_class"][0]["degenerate"])
        self.assertEqual(result["per_class"][0]["degenerate_reason"], "missing_positive_support")
        self.assertTrue(result["per_class"][1]["degenerate"])
        self.assertEqual(result["per_class"][1]["degenerate_reason"], "missing_negative_support")

    def test_optimize_matches_exhaustive_sweep_including_tied_scores(self) -> None:
        """The sweep is vectorized (cumsum over descending scores) rather than rescanning
        every candidate. Pin it against a naive exhaustive reference, with score
        distributions that force ties — where an off-by-one in the run-length grouping
        would otherwise hide."""
        import random

        rng = random.Random(20260726)

        def reference(scores: list[float], truths: list[int], fallback: float) -> tuple:
            candidates = sorted({float(s) for s in scores} | {float(fallback)})
            best = None
            for threshold in candidates:
                metrics = _binary_metrics(scores=scores, truths=truths, threshold=threshold)
                rank = (metrics["f1"], metrics["precision"], metrics["recall"], threshold)
                if best is None or rank > best:
                    best = rank
            return best

        compared = 0
        for trial in range(400):
            size = rng.randint(2, 30)
            if trial % 3 == 0:
                scores = [round(rng.random(), 3) for _ in range(size)]
            elif trial % 3 == 1:
                scores = [rng.choice([0.0, 0.5, 0.5, 1.0]) for _ in range(size)]
            else:
                scores = [0.2] * size
            truths = [rng.randint(0, 1) for _ in range(size)]
            if not (0 < sum(truths) < size):
                continue  # degenerate classes take the fallback path, covered above
            fallback = rng.choice([0.5, 0.2, 0.0, 1.0])

            result = _optimize_binary_threshold(
                scores=scores, truths=truths, fallback_threshold=fallback, target="f1", beta=1.0
            )
            expected = reference(scores, truths, fallback)
            self.assertEqual(result["threshold"], expected[3])
            self.assertAlmostEqual(result["f1"], expected[0])
            self.assertEqual(result["tp"] + result["fp"] + result["fn"] + result["tn"], size)
            compared += 1

        self.assertGreater(compared, 200)

    def test_metric_picks_paradigm_from_target_layout(self) -> None:
        scores = [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]
        metric = ThresholdOptimization()

        single = metric.from_source(None, scores=scores, cls=[0, 1, 0])
        multi = metric.from_source(None, scores=scores, cls=[[1, 0], [0, 1], [1, 0]])

        self.assertEqual(single["mode"], "single_cls")
        self.assertEqual(len(single["weights"]), 2)
        self.assertEqual(multi["mode"], "multi_cls")
        self.assertEqual(len(multi["thresholds"]), 2)

    def test_single_label_weights_rescue_a_suppressed_minority_class(self) -> None:
        rng = np.random.default_rng(0)
        majority = np.column_stack([rng.uniform(0.55, 0.95, 90), rng.uniform(0.05, 0.45, 90)])
        minority = np.column_stack([rng.uniform(0.40, 0.70, 10), rng.uniform(0.30, 0.60, 10)])
        probs = np.vstack([majority, minority])
        probs = probs / probs.sum(axis=1, keepdims=True)
        truth = np.array([0] * 90 + [1] * 10)

        result = optimize_single_label_weights(probabilities=probs, targets=truth, label_names=["a", "b"])

        self.assertGreater(result["fit_tuned_macro_f1"], result["fit_baseline_macro_f1"])
        self.assertGreater(result["weights"][1] / result["weights"][0], 1.0)

    def test_single_label_weights_leave_unsupported_class_untouched(self) -> None:
        probs = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1]]

        result = optimize_single_label_weights(probabilities=probs, targets=[0, 1, 0])

        self.assertEqual(result["weights"][2], 1.0)
        self.assertTrue(result["per_class"][2]["degenerate"])
        self.assertGreaterEqual(result["fit_tuned_macro_f1"], result["fit_baseline_macro_f1"])

    def test_forced_mode_contradicting_the_layout_fails_fast(self) -> None:
        scores = [[0.7, 0.3], [0.4, 0.6]]

        with self.assertRaisesRegex(ValueError, "contradicts"):
            ThresholdOptimization(mode="multi_cls").from_source(None, scores=scores, cls=[0, 1])
        with self.assertRaisesRegex(ValueError, "contradicts"):
            ThresholdOptimization(mode="single_cls").from_source(None, scores=scores, cls=[[1, 0], [0, 1]])
        self.assertEqual(ThresholdOptimization(mode="single_cls").from_source(None, scores=scores, cls=[0, 1])["mode"], "single_cls")

    def test_target_and_beta_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "target"):
            ThresholdOptimization(target="accuracy")
        with self.assertRaisesRegex(ValueError, "beta"):
            ThresholdOptimization(target="f1", beta=2.0)
        with self.assertRaisesRegex(ValueError, "beta"):
            ThresholdOptimization(target="f_beta")

    def test_targets_change_the_fitted_threshold(self) -> None:
        scores = [0.9, 0.8, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2]
        truths = [1, 1, 0, 1, 0, 1, 0, 0]

        def fit(target: str, beta: float) -> float:
            return _optimize_binary_threshold(scores=scores, truths=truths, fallback_threshold=0.5, target=target, beta=beta)["threshold"]

        self.assertLessEqual(fit("f_beta", 4.0), fit("f1", 1.0))
        self.assertGreaterEqual(fit("f_beta", 0.25), fit("f1", 1.0))
        self.assertIsInstance(fit("balanced_accuracy", 1.0), float)

    def test_single_label_target_is_respected(self) -> None:
        probs = [[0.6, 0.4], [0.55, 0.45], [0.7, 0.3], [0.52, 0.48]]
        truth = [0, 1, 0, 1]

        for target, beta in (("f1", 1.0), ("f_beta", 2.0), ("balanced_accuracy", 1.0)):
            result = optimize_single_label_weights(probabilities=probs, targets=truth, target=target, beta=beta)
            self.assertEqual(result["objective"], target)
            self.assertGreaterEqual(result["fit_tuned_objective"], result["fit_baseline_objective"])

    def test_after_pass_reruns_accepting_nodes_with_the_fitted_rule(self) -> None:
        calls: dict[str, object] = {}

        class Record:
            def __init__(self, name: str) -> None:
                self.name = name

        class Graph:
            records = [Record("confusion"), Record("ap"), Record("view")]
            results = {"eval.decision_rule": {"rule": {"mode": "multi_cls", "thresholds": [0.3]}}}
            out_dir = Path("eval")

            def component_class(self, name: str) -> type:
                return Confusion if name == "confusion" else ThresholdOptimization

            def downstream(self, names: list[str]) -> list[str]:
                return ["view"]

            def rerun(self, names: list[str], *, out_dir: Path, extra: dict) -> None:
                calls.update(names=names, out_dir=out_dir, extra=extra)

        metric = ThresholdOptimization()
        metric.out_key = ["eval.decision_rule"]
        metric.after_pass(Graph())

        self.assertEqual(calls["names"], ["confusion", "view"])
        self.assertEqual(calls["out_dir"], Path("eval") / "threshold-opt")
        self.assertEqual(calls["extra"], {"confusion": {"rule": {"mode": "multi_cls", "thresholds": [0.3]}}})

    def test_confusion_applies_the_rule_in_both_modes(self) -> None:
        scores = [[0.6, 0.4], [0.55, 0.45]]

        single = Confusion(multi_label=False).from_source(
            None, scores=scores, cls=[0, 1], rule={"mode": "single_cls", "weights": [1.0, 2.0]}
        )
        multi = Confusion(multi_label=True).from_source(
            None, scores=scores, cls=[[1, 0], [0, 1]], rule={"mode": "multi_cls", "thresholds": [0.7, 0.3]}
        )

        self.assertEqual([row["pred"] for row in single["sample_rows"]], ["class_1", "class_1"])
        self.assertEqual([row["pred_class_0"] for row in multi["sample_rows"]], [0, 0])
        self.assertEqual([row["pred_class_1"] for row in multi["sample_rows"]], [1, 1])
        with self.assertRaisesRegex(ValueError, "cannot apply"):
            Confusion(multi_label=False).from_source(None, scores=scores, cls=[0, 1], rule={"mode": "multi_cls", "thresholds": [0.5, 0.5]})

    def test_decode_singlelabel_applies_weights(self) -> None:
        activated = {"probs": np.array([[0.6, 0.4]])}

        self.assertEqual(decode_singlelabel(activated, {})[0]["label"], 0)
        self.assertEqual(decode_singlelabel(activated, {"weights": [1.0, 2.0]})[0]["label"], 1)


if __name__ == "__main__":
    unittest.main()
