"""Tests for the ``summary`` view, ``teia.node.eval.compare.*``
cross-run comparison components, and the ``teia eval`` CLI command's single-run + multi-run paths.
"""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import yaml

from teia.core.commands.eval import run as teia_eval_run
from teia.core.eval.compare_runner import run_compare
from teia.core.eval.runner import RunEvalSource, run_eval
from teia.core.capture.store import CaptureWriter
from teia.node.eval.compare.factor_diff import FactorDiff
from teia.node.eval.compare.metric_table import MetricTable
from teia.node.eval.compare.pareto import Pareto
from teia.node.eval.compare.seed_aggregate import SeedAggregate
from teia.node.eval.view.summary import Summary

from _store_helpers import load_evalmodule



def _make_run(root: Path, *, run_id: str, accuracy: float, overrides: list[str]) -> None:
    (root / "eval").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "artifacts").mkdir(parents=True)
    (root / "eval" / "summary.yaml").write_text(yaml.safe_dump({"accuracy": accuracy, "n_samples": 10}))
    (root / "config" / "overrides.yaml").write_text(yaml.safe_dump({"overrides": overrides}))
    (root / "artifacts" / "run.yaml").write_text(yaml.safe_dump({"run_id": run_id, "status": "ok"}))


class SummaryViewTests(unittest.TestCase):
    def test_flattens_scalar_fields_across_wired_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            written = Summary().render(
                out_dir,
                confusion={"n_samples": 3, "accuracy": 1.0, "class_names": ["a", "b"], "confusion_matrix": [[1, 0], [0, 1]]},
                ap={"macro_average_precision": 0.9, "class_rows": [{"f1": 1.0}]},
            )
            self.assertEqual(len(written), 1)
            payload = yaml.safe_load(written[0].read_text())
            self.assertEqual(payload, {"n_samples": 3, "accuracy": 1.0, "macro_average_precision": 0.9})

    def test_ignores_non_dict_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = Summary().render(Path(tmp), stray=[1, 2, 3])
            self.assertEqual(yaml.safe_load(written[0].read_text()), {})


class ClsBundleWritesRealSummaryTests(unittest.TestCase):
    def test_run_eval_produces_report_route_summary_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("test", "val"):
                writer = CaptureWriter(root / "artifacts" / "capture" / split, split=split)
                writer.declare_numeric("capture.cls.scores", shape=(2,), dtype="float32")
                writer.declare_numeric("batch.cls", shape=(1,), dtype="int64")
                for scores, target in [([0.9, 0.1], 0), ([0.2, 0.8], 1), ([0.6, 0.4], 0)]:
                    writer.append_numeric("capture.cls.scores", [scores])
                    writer.append_numeric("batch.cls", [[target]])
                writer.set_meta(class_names=["a", "b"])
                writer.declare_route("cls", scores="capture.cls.scores", targets="batch.cls")
                writer.close()

            graph = load_evalmodule("vision-cls")
            out_dir = root / "eval"
            written = run_eval(root, graph=graph, split="test", out_dir=out_dir)
            self.assertIn("summary", written)

            summary = yaml.safe_load((out_dir / "summary.yaml").read_text())
            self.assertEqual(summary["n_samples"], 3)
            self.assertAlmostEqual(summary["accuracy"], 1.0)
            self.assertIn("macro_f1", summary)
            self.assertIn("calibration_error", summary)


class CompareComponentTests(unittest.TestCase):
    def test_metric_table_joins_summaries_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.9, overrides=["trainer.max_epochs=10"])
            _make_run(root / "b", run_id="b", accuracy=0.92, overrides=["trainer.max_epochs=10"])
            sources = [RunEvalSource(root / name, split="test") for name in ("a", "b")]

            written = MetricTable().compare(sources, root / "out")
            rows = (root / "out" / "metric_table.csv").read_text().splitlines()
            self.assertEqual(rows[0], "run_id,accuracy,n_samples")
            self.assertIn("a,0.9,10", rows[1])
            self.assertIn("b,0.92,10", rows[2])
            self.assertEqual(written, [root / "out" / "metric_table.csv"])

    def test_seed_aggregate_groups_by_non_seed_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.90, overrides=["trainer.max_epochs=10", "runtime.seed=1"])
            _make_run(root / "b", run_id="b", accuracy=0.92, overrides=["trainer.max_epochs=10", "runtime.seed=2"])
            _make_run(root / "c", run_id="c", accuracy=0.80, overrides=["trainer.max_epochs=20", "runtime.seed=1"])
            sources = [RunEvalSource(root / name, split="test") for name in ("a", "b", "c")]

            SeedAggregate().compare(sources, root / "out")
            text = (root / "out" / "seed_aggregate.csv").read_text()
            self.assertIn("trainer.max_epochs=10,2,a;b,0.91", text)
            self.assertIn("trainer.max_epochs=20,1,c,0.8", text)

    def test_factor_diff_derives_the_non_seed_varying_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.9, overrides=["trainer.max_epochs=10", "runtime.seed=1"])
            _make_run(root / "b", run_id="b", accuracy=0.95, overrides=["trainer.max_epochs=20", "runtime.seed=2"])
            sources = [RunEvalSource(root / name, split="test") for name in ("a", "b")]

            written = FactorDiff().compare(sources, root / "out")
            text = written[0].read_text()
            self.assertIn("factor_key", text)
            self.assertIn("trainer.max_epochs", text)
            self.assertIn("accuracy", text)

    def test_factor_diff_single_metric_mode_also_writes_a_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.9, overrides=["trainer.max_epochs=10"])
            _make_run(root / "b", run_id="b", accuracy=0.95, overrides=["trainer.max_epochs=20"])
            sources = [RunEvalSource(root / name, split="test") for name in ("a", "b")]

            written = FactorDiff(metric="accuracy").compare(sources, root / "out")
            suffixes = {path.suffix for path in written}
            self.assertEqual(suffixes, {".csv", ".png"})

    def test_pareto_flags_the_non_dominated_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def make(name: str, accuracy: float, n_samples: int) -> None:
                (root / name / "eval").mkdir(parents=True)
                (root / name / "artifacts").mkdir(parents=True)
                (root / name / "config").mkdir(parents=True)
                (root / name / "eval" / "summary.yaml").write_text(
                    yaml.safe_dump({"accuracy": accuracy, "n_samples": n_samples})
                )
                (root / name / "artifacts" / "run.yaml").write_text(yaml.safe_dump({"run_id": name}))
                (root / name / "config" / "overrides.yaml").write_text(yaml.safe_dump({"overrides": []}))

            make("a", 0.9, 5)
            make("b", 0.92, 8)
            make("c", 0.80, 20)  # worse accuracy, more samples -> non-dominated tradeoff
            sources = [RunEvalSource(root / name, split="test") for name in ("a", "b", "c")]

            Pareto(x_metric="accuracy", y_metric="n_samples").compare(sources, root / "out")
            csv_path = next((root / "out").glob("*.csv"))
            rows = {row.split(",")[0]: row for row in csv_path.read_text().splitlines()[1:]}
            self.assertEqual(rows["a"].split(",")[-1], "False")  # dominated by b (better on both)
            self.assertEqual(rows["b"].split(",")[-1], "True")
            self.assertEqual(rows["c"].split(",")[-1], "True")


class RunCompareTests(unittest.TestCase):
    def test_run_compare_drives_every_node_in_the_flat_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.9, overrides=[])
            _make_run(root / "b", run_id="b", accuracy=0.92, overrides=[])
            compare_cfg = {"metric_table": {"_target_": "teia.node.eval.compare.metric_table.MetricTable"}}
            written = run_compare([root / "a", root / "b"], compare=compare_cfg, split="test", out_dir=root / "out")
            self.assertEqual(written, [root / "out" / "metric_table.csv"])


class TeiaEvalCliTests(unittest.TestCase):
    def test_single_run_dir_rebuilds_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("test", "val"):
                writer = CaptureWriter(root / "artifacts" / "capture" / split, split=split)
                writer.declare_numeric("capture.cls.scores", shape=(2,), dtype="float32")
                writer.declare_numeric("batch.cls", shape=(1,), dtype="int64")
                for scores, target in [([0.9, 0.1], 0), ([0.2, 0.8], 1), ([0.6, 0.4], 0)]:
                    writer.append_numeric("capture.cls.scores", [scores])
                    writer.append_numeric("batch.cls", [[target]])
                writer.set_meta(class_names=["a", "b"])
                writer.declare_route("cls", scores="capture.cls.scores", targets="batch.cls")
                writer.close()

            (root / "config").mkdir(parents=True)
            graph = load_evalmodule("vision-cls")
            (root / "config" / "composed.yaml").write_text(yaml.safe_dump({"evalmodule": graph}))

            args = Namespace(run_dirs=[str(root)], split="test", output_dir=None, compare_path=None)
            rc = teia_eval_run(args)
            self.assertEqual(rc, 0)
            self.assertTrue((root / "eval" / "summary.yaml").exists())

    def test_multi_run_dir_uses_the_zero_config_default_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.9, overrides=["trainer.max_epochs=10"])
            _make_run(root / "b", run_id="b", accuracy=0.92, overrides=["trainer.max_epochs=20"])

            args = Namespace(
                run_dirs=[str(root / "a"), str(root / "b")],
                split="test",
                output_dir=str(root / "out"),
                compare_path=None,
            )
            rc = teia_eval_run(args)
            self.assertEqual(rc, 0)
            self.assertTrue((root / "out" / "metric_table.csv").exists())
            self.assertTrue((root / "out" / "seed_aggregate.csv").exists())

    def test_multi_run_dir_honors_an_explicit_compare_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "a", run_id="a", accuracy=0.9, overrides=[])
            _make_run(root / "b", run_id="b", accuracy=0.92, overrides=[])
            compare_path = root / "compare.yaml"
            compare_path.write_text(yaml.safe_dump({"compare": {"metric_table": {"_target_": "teia.node.eval.compare.metric_table.MetricTable"}}}))

            args = Namespace(run_dirs=[str(root / "a"), str(root / "b")], split="test", output_dir=str(root / "out"), compare_path=str(compare_path))
            rc = teia_eval_run(args)
            self.assertEqual(rc, 0)
            self.assertTrue((root / "out" / "metric_table.csv").exists())
            self.assertFalse((root / "out" / "seed_aggregate.csv").exists())  # not in the custom bundle


if __name__ == "__main__":
    unittest.main()
