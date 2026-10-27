from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _store_helpers import load_evalmodule, write_store

from teia.core.eval.runner import run_eval


class TabularObjectiveReportTests(unittest.TestCase):
    """Tabular task evalmodules render through the shared eval components, like vision."""

    def _render(self, family: str, columns: dict, **meta) -> Path:
        root = Path(tempfile.mkdtemp())
        write_store(root, "test", columns, **meta)
        write_store(root, "val", columns, **meta)
        run_eval(root, graph=load_evalmodule(family), split="test", out_dir=root / "eval")
        return root

    def test_tabular_cls_renders_shared_objective_report(self) -> None:
        root = self._render(
            "cls",
            {"capture.cls.scores": [[0.7, 0.3], [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]], "batch.cls": [0, 1, 0, 1]},
            class_names=["a", "b"],
        )
        plots = {p.name for p in (root / "eval").glob("*.png")}
        self.assertIn("normalized_confusion_matrix.png", plots)
        self.assertIn("pr_curves_macro_and_worst_k_classes.png", plots)

    def test_tabular_reg_renders_shared_objective_report(self) -> None:
        root = self._render(
            "reg",
            {"capture.reg.mean": [[1.1], [2.2], [2.9], [4.1]], "capture.reg.std": [[1.0]] * 4, "batch.target": [[1.0], [2.0], [3.0], [4.0]]},
            target_names=["y"],
        )
        plots = {p.name for p in (root / "eval").glob("*.png")}
        self.assertIn("regression_predicted_vs_actual_scatter.png", plots)
        self.assertIn("regression_residual_histogram.png", plots)


if __name__ == "__main__":
    unittest.main()
