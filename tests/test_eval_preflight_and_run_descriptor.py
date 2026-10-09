"""Tests for the pre-flight eval gates
(``runtime/validate.py``), ``deps.check_missing``, and the ``artifacts/run.yaml`` descriptor
writer (``runtime/run_descriptor.py``).
"""

from __future__ import annotations

from _store_helpers import load_evalmodule

import tempfile
import unittest
from pathlib import Path

from teia.core.deps import MissingDependencyError, check_missing
from teia.core.runtime.run_descriptor import finish_run_stage, run_descriptor_path, start_run_descriptor
from teia.core.runtime.validate import _validate_eval_callbacks, _validate_eval_deps, _validate_eval_wiring
from teia.core.utils import read_yaml


class CheckMissingTests(unittest.TestCase):
    def test_no_missing_dependency_is_a_noop(self) -> None:
        check_missing({"evalmodule": {}}, feature="Eval graph")  # must not raise

    def test_missing_module_raises_with_install_guidance(self) -> None:
        config = {"_target_": "teia.node.eval.metric.aligned.does_not_exist_module.DoesNotExist"}
        with self.assertRaises(MissingDependencyError) as ctx:
            check_missing(config, feature="Eval graph")
        self.assertIn("Eval graph", str(ctx.exception))


class EvalPreflightGateTests(unittest.TestCase):
    def test_deps_gate_is_a_noop_with_no_graph(self) -> None:
        _validate_eval_deps({})
        _validate_eval_deps({"evalmodule": {}})
        _validate_eval_deps({"evalmodule": {}})

    def test_deps_gate_raises_for_unresolvable_target(self) -> None:
        graph = {"bogus": {"_target_": "teia.node.eval.metric.aligned.does_not_exist_module.DoesNotExist", "out": "eval.x"}}
        with self.assertRaises(MissingDependencyError):
            _validate_eval_deps({"evalmodule": graph})

    def test_wiring_gate_is_a_noop_with_no_graph(self) -> None:
        _validate_eval_wiring({})
        _validate_eval_wiring({"evalmodule": {}})

    def test_wiring_gate_rejects_missing_producer(self) -> None:
        graph = {
            "bogus": {
                "_target_": "teia.node.eval.metric.aligned.confusion.Confusion",
                "in": ["eval.nonexistent"],
                "out": "eval.x",
            }
        }
        with self.assertRaisesRegex(ValueError, "not produced by any node"):
            _validate_eval_wiring({"evalmodule": graph})

    def test_wiring_gate_accepts_a_real_bundle_graph(self) -> None:
        import yaml

        graph = load_evalmodule("vision-cls", run_nodes=True)
        _validate_eval_deps({"evalmodule": graph})
        _validate_eval_wiring({"evalmodule": graph})  # must not raise


class EvalCallbacksGateTests(unittest.TestCase):
    """RunEvalSource.curves()'s bug class made concrete: a node's REQUIRES_CALLBACK is not
    satisfied by callbacks.items, and that must fail before trainer.fit, not silently degrade."""

    _GRAPH = {
        "training_curves": {
            "_target_": "teia.node.eval.metric.run.training_curves.TrainingCurves",
            "out": "eval.run.training_curves",
        }
    }

    def test_callbacks_gate_is_a_noop_with_no_graph(self) -> None:
        _validate_eval_callbacks({})
        _validate_eval_callbacks({"evalmodule": {}})

    def test_callbacks_gate_raises_when_required_callback_is_missing(self) -> None:
        config = {"evalmodule": self._GRAPH, "callbacks": {"items": []}}
        with self.assertRaisesRegex(ValueError, "LearningRateMonitor"):
            _validate_eval_callbacks(config)

    def test_callbacks_gate_accepts_when_required_callback_is_present(self) -> None:
        config = {
            "evalmodule": self._GRAPH,
            "callbacks": {"items": [{"_target_": "lightning.pytorch.callbacks.LearningRateMonitor"}]},
        }
        _validate_eval_callbacks(config)  # must not raise

    def test_callbacks_gate_ignores_metrics_with_no_requirement(self) -> None:
        graph = {"confusion": {"_target_": "teia.node.eval.metric.aligned.confusion.Confusion", "out": "eval.cls.confusion"}}
        _validate_eval_callbacks({"evalmodule": graph, "callbacks": {"items": []}})  # must not raise


class RunDescriptorTests(unittest.TestCase):
    def test_start_writes_incomplete_status_and_finish_marks_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = {"experiment_name": "exp1", "runtime": {"seed": 7}, "data_root": "data/coco8"}
            start_run_descriptor(run_dir=run_dir, config=config)

            path = run_descriptor_path(run_dir)
            self.assertTrue(path.exists())
            payload = read_yaml(path)
            self.assertEqual(payload["status"], "incomplete")
            self.assertEqual(payload["stages"], [])
            self.assertEqual(payload["seed"], 7)
            self.assertEqual(payload["group"], "exp1")
            self.assertEqual(payload["data_root"], "data/coco8")
            self.assertNotIn("/", payload["data_root"][:1])  # not silently made absolute

            finish_run_stage(run_dir=run_dir, stage="train", routes={"cls": ["scores", "label"]})
            payload = read_yaml(path)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["stages"], ["train"])
            self.assertEqual(payload["routes"], {"cls": ["scores", "label"]})

            finish_run_stage(run_dir=run_dir, stage="test")
            payload = read_yaml(path)
            self.assertEqual(payload["stages"], ["train", "test"])

    def test_start_is_idempotent_and_does_not_clobber_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            start_run_descriptor(run_dir=run_dir, config={"experiment_name": "first", "runtime": {"seed": 1}})
            start_run_descriptor(run_dir=run_dir, config={"experiment_name": "second", "runtime": {"seed": 99}})
            payload = read_yaml(run_descriptor_path(run_dir))
            self.assertEqual(payload["group"], "first")
            self.assertEqual(payload["seed"], 1)

    def test_finish_without_start_does_not_fabricate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            finish_run_stage(run_dir=run_dir, stage="train")
            self.assertFalse(run_descriptor_path(run_dir).exists())

    def test_a_crashed_run_stays_incomplete(self) -> None:
        """The whole point of writing at start and updating at end: a run that never reaches
        `finish_run_stage` (crashed, or trained but never tested) is left exactly as `start`
        wrote it — `status: incomplete` — not a missing file and not a stale success."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            start_run_descriptor(run_dir=run_dir, config={"experiment_name": "exp", "runtime": {"seed": 0}})
            payload = read_yaml(run_descriptor_path(run_dir))
            self.assertEqual(payload["status"], "incomplete")
            self.assertEqual(payload["stages"], [])


if __name__ == "__main__":
    unittest.main()
