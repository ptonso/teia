"""End-to-end proof that the ``vision-cls`` evalmodule runs metric -> view through the real,
composed config and produces real report files, with pinned expected values."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _store_helpers import load_evalmodule, write_store

from teia.core.eval.graph import build_eval_graph, discover_node_entries
from teia.core.eval.runner import RunEvalSource, run_eval
from teia.node.eval.metric.aligned.confusion import Confusion

class ClsBundleEndToEndTests(unittest.TestCase):
    def test_bundle_yaml_composes_and_produces_a_valid_graph(self) -> None:
        graph = load_evalmodule("vision-cls")
        records = build_eval_graph(discover_node_entries(graph))
        # Metrics before views (toposort); every node resolved to a real, importable component.
        kinds = [r.kind for r in records]
        self.assertIn("metric", kinds)
        self.assertIn("view", kinds)
        self.assertLess(kinds.index("metric"), len(kinds) - kinds[::-1].index("view") - 1 + 1)

    def test_bundle_runs_end_to_end_and_matches_pinned_values(self) -> None:
        columns = {
            "capture.cls.scores": [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1]],
            "batch.cls": [0, 1, 2, 0, 2, 1],
        }
        class_names = ["a", "b", "c"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_store(root, "test", columns, class_names=class_names)
            write_store(root, "val", columns, class_names=class_names)

            graph = load_evalmodule("vision-cls")
            out_dir = root / "eval"
            written = run_eval(root, graph=graph, split="test", out_dir=out_dir)

            self.assertIn("confusion_matrix_view", written)
            self.assertTrue(written["confusion_matrix_view"][0].exists())
            self.assertTrue((out_dir / "sample_metrics.csv").exists())
            self.assertTrue((out_dir / "decision_rule.csv").exists())
            self.assertEqual(
                sorted(path.name for path in (out_dir / "threshold-opt").glob("*")),
                ["normalized_confusion_matrix.png", "per_class_f1_bar_sorted.png", "sample_metrics.csv", "summary.yaml", "top_confusion_pairs_bar.png"],
            )
            self.assertFalse((out_dir / "threshold-opt" / "reliability_diagram.png").exists())

            source = RunEvalSource(root, split="test")
            result = Confusion(multi_label=False).from_source(
                source, scores=source.column("capture.cls.scores"), cls=source.column("batch.cls"), class_names=class_names
            )
            # Every prediction in this fixture is argmax-correct.
            self.assertAlmostEqual(result["accuracy"], 1.0, places=6)
            self.assertAlmostEqual(result["macro_f1"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
