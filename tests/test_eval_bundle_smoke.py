"""Smoke tests (compose, graph builds, runs end-to-end, produces real files) for the task
evalmodules not covered by a dedicated numeric test: each confirms the evalmodule's wiring is
internally consistent (view kwarg names, field paths into each metric's return shape)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _store_helpers import load_evalmodule, write_store

from teia.core.eval.graph import build_eval_graph, discover_node_entries
from teia.core.eval.runner import run_eval

class BundleSmokeTests(unittest.TestCase):
    def _run(self, task: str, columns: dict, **meta) -> dict[str, list[Path]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_store(root, "test", columns, **meta)
            write_store(root, "val", columns, **meta)
            graph = load_evalmodule(task)
            build_eval_graph(discover_node_entries(graph))  # cycle/producer/duplicate-out sanity
            written = run_eval(root, graph=graph, split="test", out_dir=root / "eval")
            for paths in written.values():
                for path in paths:
                    self.assertTrue(path.exists(), f"{task}: {path} was not written")
                    self.assertGreater(path.stat().st_size, 0, f"{task}: {path} is empty")
            self.rerun_files = sorted(path.name for path in (root / "eval" / "threshold-opt").glob("*")) if (root / "eval" / "threshold-opt").exists() else []
            self.flat_files = sorted(path.name for path in (root / "eval").glob("*") if path.is_file())
            return written

    def test_multi_cls_bundle_runs_end_to_end(self) -> None:
        written = self._run(
            "vision-multi-cls",
            {"capture.cls.scores": [[0.9, 0.1, 0.2], [0.1, 0.8, 0.9], [0.6, 0.6, 0.1], [0.2, 0.1, 0.7]], "batch.cls": [[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]]},
            class_names=["a", "b", "c"],
        )
        self.assertEqual(
            self.rerun_files,
            ["label_prevalence_vs_f1_scatter.png", "per_label_f1_bar_sorted.png", "sample_metrics.csv", "summary.yaml", "tn_masked_one_vs_all_confusion_matrices.png"],
        )
        self.assertIn("decision_rule.csv", self.flat_files)
        self.assertIn("pr_curves_macro_and_worst_k_labels.png", self.flat_files)
        self.assertNotIn("pr_curves_macro_and_worst_k_labels.png", self.rerun_files)
        self.assertIn("tn_masked_confusion_grid", written)
        self.assertIn("threshold_sweep_view", written)

    def test_reg_bundle_runs_end_to_end(self) -> None:
        written = self._run(
            "tabular-reg",
            {"capture.reg.mean": [[1.1], [2.2], [2.9], [4.1]], "capture.reg.std": [[1.0]] * 4, "batch.target": [[1.0], [2.0], [3.0], [4.0]]},
            target_names=["y"],
        )
        self.assertIn("predicted_vs_actual_scatter", written)
        self.assertIn("residual_histogram", written)

    def test_survival_bundle_runs_end_to_end(self) -> None:
        written = self._run(
            "tabular-survival",
            {"batch.time": [5.0, 8.0, 3.0, 10.0, 6.0], "batch.event": [1, 0, 1, 1, 0], "capture.survival.risk": [0.8, 0.3, 0.9, 0.2, 0.5]},
        )
        self.assertIn("concordance_table", written)
        self.assertIn("km_by_risk_view", written)


if __name__ == "__main__":
    unittest.main()
