"""Smoke + validation tests for the eval-graph substrate (teia.base.eval + teia.core.eval).

Mirrors tests/test_data_graph.py's style: tiny in-test node classes referenced by `_target_` (core
ships no concrete eval components), a happy-path 2-node graph, and the hard-error cases the
graph build must reject — proving `build_eval_graph` genuinely reuses the data graph's
`producer_map`/`_resolve_deps`/`_toposort` (cycle/missing-producer/duplicate-out_key) rather than
reimplementing them a third time.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from teia.base.eval import Comparison, EvalNode, EvalSource, Metric, View
from teia.core.eval.graph import build_eval_graph, discover_node_entries, parse_eval_node
from teia.core.eval.runner import RunEvalSource, required_splits, run_eval, run_eval_graph
from teia.core.capture.store import CaptureWriter


class SumMetric(Metric):
    """A metric's return dict is published *whole*, under its single ``out_key`` — a downstream
    node consuming that key receives this entire dict as one kwarg and pulls named fields out of
    it itself (see ``run_eval_graph::_assign_result``). Field name here is deliberately not
    ``total`` to keep the dict-vs-key distinction visible in ``RecordingView.written``."""

    MONITORS = ("total",)

    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source
        return {"sum": sum(inputs["scores"])}


class StreamingCountMetric(Metric):
    def __init__(self) -> None:
        self._count = 0

    def update(self, **inputs: Any) -> None:
        self._count += len(inputs.get("scores") or [])

    def compute(self) -> dict[str, Any]:
        return {"count": self._count}

    def reset(self) -> None:
        self._count = 0

    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source
        return {"count": len(inputs["scores"])}


class MonitoredCountMetric(Metric):
    # "fitness" is always published bare (see streaming.py::_flat_monitor_name), so two
    # differently-named nodes both declaring it collide on the same flat monitor key.
    MONITORS = ("fitness",)

    def __init__(self) -> None:
        self._count = 0

    def update(self, **inputs: Any) -> None:
        self._count += len(inputs.get("scores") or [])

    def compute(self) -> dict[str, Any]:
        return {"count": self._count}

    def reset(self) -> None:
        self._count = 0

    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source
        return {"count": len(inputs["scores"])}


class RecordingView(View):
    written: list[Any] = []

    def render(self, out_dir: Path, **inputs: Any) -> list[Path]:
        RecordingView.written.append(dict(inputs))
        path = out_dir / "recorded.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(inputs), encoding="utf-8")
        return [path]


class FactorMetric(Metric):
    """Accepts the optional ``factor`` kwarg a graph-aware metric can supply on a rerun."""

    ACCEPTS = frozenset({"factor"})

    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source
        return {"sum": sum(inputs["scores"]) * inputs.get("factor", 1)}


class RerunningMetric(Metric):
    """Finds the nodes accepting ``factor`` and reruns them plus their consumers into ``x10/``."""

    calls = 0

    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source, inputs
        return {}

    def after_pass(self, graph: Any) -> None:
        RerunningMetric.calls += 1
        targets = [record.name for record in graph.records if "factor" in graph.component_class(record.name).ACCEPTS]
        graph.rerun(targets + graph.downstream(targets), out_dir=graph.out_dir / "x10", extra={name: {"factor": 10} for name in targets})


class SplitProbe(Metric):
    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source
        return {"seen": list(inputs["scores"])}


class NotAnEvalNode:
    """Deliberately doesn't subclass Metric/View/Comparison, to test kind-inference rejection."""


class FakeComparison(Comparison):
    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]:
        return []


class RowMetric(Metric):
    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        del source, inputs
        return {"class_rows": [{"class_name": "a", "f1": 0.5}, {"class_name": "b", "f1": 0.9}]}


class FakeSource(EvalSource):
    def __init__(self, columns: dict[str, Any]) -> None:
        self.run_dir = Path(".")
        self._columns = columns

    def routes(self) -> list[str]:
        return sorted({key.split(".")[1] for key in self._columns if key.startswith("capture.")})

    def meta(self, key: str, default: Any = None) -> Any:
        return default

    def has(self, key: str) -> bool:
        return key in self._columns

    def column(self, key: str) -> Any:
        return self._columns[key]

    def config(self) -> dict[str, Any]:
        return {}

    def curves(self) -> Any:
        return {}

    def descriptor(self) -> dict[str, Any]:
        return {}

    def overrides(self) -> list[str]:
        return []


class EvalAbcTests(unittest.TestCase):
    def test_metric_requires_from_source(self) -> None:
        with self.assertRaises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_view_requires_render(self) -> None:
        with self.assertRaises(TypeError):
            View()  # type: ignore[abstract]

    def test_comparison_requires_compare(self) -> None:
        with self.assertRaises(TypeError):
            Comparison()  # type: ignore[abstract]

    def test_metric_without_streaming_driver_raises_on_update(self) -> None:
        metric = SumMetric()
        with self.assertRaises(NotImplementedError):
            metric.update(scores=[1])

    def test_streaming_metric_overrides_update_and_compute(self) -> None:
        metric = StreamingCountMetric()
        metric.update(scores=[1, 2, 3])
        self.assertEqual(metric.compute(), {"count": 3})
        metric.reset()
        self.assertEqual(metric.compute(), {"count": 0})

    def test_eval_node_base_declares_in_out_key(self) -> None:
        self.assertIn("in_key", EvalNode.__annotations__)
        self.assertIn("out_key", EvalNode.__annotations__)


class EvalGraphBuildTests(unittest.TestCase):
    def _entries(self, **overrides: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        entries = {
            "sum_metric": {
                "_target_": f"{__name__}.SumMetric",
                "in": ["capture.cls.scores"],
                "out": "eval.cls.total",
            },
            "view": {
                "_target_": f"{__name__}.RecordingView",
                "in": ["eval.cls.total"],
            },
        }
        entries.update(overrides)
        return discover_node_entries(entries)

    def test_two_node_graph_toposorts_metric_before_view(self) -> None:
        records = build_eval_graph(self._entries())
        self.assertEqual([r.name for r in records], ["sum_metric", "view"])
        self.assertEqual(records[0].kind, "metric")
        self.assertEqual(records[1].kind, "view")

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cycle"):
            build_eval_graph(
                discover_node_entries(
                    {
                        "a": {"_target_": f"{__name__}.SumMetric", "in": ["eval.b"], "out": "eval.a"},
                        "b": {"_target_": f"{__name__}.SumMetric", "in": ["eval.a"], "out": "eval.b"},
                    }
                )
            )

    def test_missing_producer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not produced"):
            build_eval_graph(self._entries(view={"_target_": f"{__name__}.RecordingView", "in": ["eval.absent"]}))

    def test_duplicate_out_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate out_key"):
            build_eval_graph(
                discover_node_entries(
                    {
                        "a": {"_target_": f"{__name__}.SumMetric", "in": ["capture.cls.scores"], "out": "eval.cls.total"},
                        "b": {"_target_": f"{__name__}.SumMetric", "in": ["capture.cls.scores"], "out": "eval.cls.total"},
                    }
                )
            )

    def test_view_declaring_out_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminal"):
            parse_eval_node(
                "view", {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.total"], "out": "eval.cls.rendered"}
            )

    def test_non_eval_node_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Metric/View/Comparison"):
            parse_eval_node("bad", {"_target_": f"{__name__}.NotAnEvalNode", "in": [], "out": []})


class EvalContractsBuildTimeChecksTests(unittest.TestCase):
    """The two pre-flight checks `_validate_eval_contracts` promotes from lazy (caught mid-fit
    by StreamingMonitorRunner) to build time (nodes.md §3/§5, eval.md §5)."""

    def test_non_streaming_metric_wired_as_monitor_is_rejected_at_build_time(self) -> None:
        entries = {
            "a": {"_target_": f"{__name__}.SumMetric", "in": ["capture.cls.scores"], "out": "eval.a", "monitor": True},
        }
        with self.assertRaisesRegex(ValueError, "requires a streaming driver"):
            build_eval_graph(discover_node_entries(entries))

    def test_streaming_metric_wired_as_monitor_is_accepted(self) -> None:
        entries = {
            "a": {"_target_": f"{__name__}.StreamingCountMetric", "in": ["capture.cls.scores"], "out": "eval.a", "monitor": True},
        }
        build_eval_graph(discover_node_entries(entries))  # must not raise

    def test_duplicate_monitor_name_across_nodes_is_rejected_at_build_time(self) -> None:
        entries = {
            "a": {"_target_": f"{__name__}.MonitoredCountMetric", "in": ["capture.cls.scores"], "out": "eval.a", "monitor": True},
            "b": {"_target_": f"{__name__}.MonitoredCountMetric", "in": ["capture.det.scores"], "out": "eval.b", "monitor": True},
        }
        with self.assertRaisesRegex(ValueError, "duplicate val-metric monitor name"):
            build_eval_graph(discover_node_entries(entries))


class EvalGraphRunnerTests(unittest.TestCase):
    def test_run_eval_graph_flows_capture_through_metric_to_view(self) -> None:
        RecordingView.written = []
        records = build_eval_graph(
            discover_node_entries(
                {
                    "sum_metric": {
                        "_target_": f"{__name__}.SumMetric",
                        "in": ["capture.cls.scores"],
                        "out": "eval.cls.total",
                    },
                    "view": {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.total"]},
                }
            )
        )
        source = FakeSource({"capture.cls.scores": [0.2, 0.3, 0.5]})
        with tempfile.TemporaryDirectory() as tmp:
            written = run_eval_graph(records, source=source, out_dir=Path(tmp))
        # "total" is the view's kwarg name (trailing atom of `eval.cls.total`); its value is
        # SumMetric's *entire* return dict, published whole under that one out_key.
        self.assertEqual(RecordingView.written, [{"total": {"sum": 1.0}}])
        self.assertEqual(list(written), ["view"])

    def test_mapping_form_in_extracts_named_fields_from_a_metrics_result_dict(self) -> None:
        """A metric publishes one result dict per out_key; a view wants specific named
        sub-fields as distinct kwargs (e.g. RankedBar's `labels`/`values`), not the whole dict
        under one name. Mapping-form `in` (`{kwarg: "source_key:field.path"}`) is how a bundle
        expresses that extraction."""
        RecordingView.written = []
        records = build_eval_graph(
            discover_node_entries(
                {
                    "sum_metric": {
                        "_target_": f"{__name__}.SumMetric",
                        "in": ["capture.cls.scores"],
                        "out": "eval.cls.total",
                    },
                    "view": {
                        "_target_": f"{__name__}.RecordingView",
                        "in": {"total": "eval.cls.total:sum"},
                    },
                }
            )
        )
        source = FakeSource({"capture.cls.scores": [0.2, 0.3, 0.5]})
        with tempfile.TemporaryDirectory() as tmp:
            written = run_eval_graph(records, source=source, out_dir=Path(tmp))
        self.assertEqual(RecordingView.written, [{"total": 1.0}])
        self.assertEqual(list(written), ["view"])

    def test_mapping_form_in_plucks_a_field_across_a_list_of_dicts(self) -> None:
        """The row-to-column pivot a view's parallel-array kwargs need from a metric's
        row-oriented output (e.g. RankedBar's `labels`/`values` from Confusion's `class_rows`)."""

        RecordingView.written = []
        records = build_eval_graph(
            discover_node_entries(
                {
                    "row_metric": {"_target_": f"{__name__}.RowMetric", "out": "eval.cls.confusion"},
                    "view": {
                        "_target_": f"{__name__}.RecordingView",
                        "in": {"labels": "eval.cls.confusion:class_rows[].class_name", "values": "eval.cls.confusion:class_rows[].f1"},
                    },
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_eval_graph(records, source=FakeSource({}), out_dir=Path(tmp))
        self.assertEqual(RecordingView.written, [{"labels": ["a", "b"], "values": [0.5, 0.9]}])

    def test_mapping_form_in_still_resolves_producer_dependency_for_toposort(self) -> None:
        records = build_eval_graph(
            discover_node_entries(
                {
                    "view": {
                        "_target_": f"{__name__}.RecordingView",
                        "in": {"total": "eval.cls.total:sum"},
                    },
                    "sum_metric": {
                        "_target_": f"{__name__}.SumMetric",
                        "in": ["capture.cls.scores"],
                        "out": "eval.cls.total",
                    },
                }
            )
        )
        self.assertEqual([r.name for r in records], ["sum_metric", "view"])

    def test_comparison_node_rejected_in_single_source_pass(self) -> None:
        records = build_eval_graph(discover_node_entries({"cmp": {"_target_": f"{__name__}.FakeComparison"}}))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "comparison nodes cannot run"):
                run_eval_graph(records, source=FakeSource({}), out_dir=Path(tmp))


class EvalPassTests(unittest.TestCase):
    def _records(self) -> list[Any]:
        return build_eval_graph(
            discover_node_entries(
                {
                    "factor": {"_target_": f"{__name__}.FactorMetric", "in": ["capture.cls.scores"], "out": "eval.cls.sum"},
                    "other": {"_target_": f"{__name__}.SumMetric", "in": ["capture.cls.scores"], "out": "eval.cls.other"},
                    "view": {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.sum"]},
                    "other_view": {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.other"]},
                    "rerunner": {"_target_": f"{__name__}.RerunningMetric", "in": [], "out": "eval.cls.rerunner"},
                }
            )
        )

    def test_after_pass_reruns_only_accepting_nodes_and_their_consumers_into_a_subfolder(self) -> None:
        RecordingView.written = []
        RerunningMetric.calls = 0
        source = FakeSource({"capture.cls.scores": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as tmp:
            written = run_eval_graph(self._records(), source=source, out_dir=Path(tmp))
            self.assertEqual((Path(tmp) / "x10" / "recorded.txt").exists(), True)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["recorded.txt", "x10"])
        base = [entry for entry in RecordingView.written if entry.get("sum") == {"sum": 3.0} or entry.get("other") == {"sum": 3.0}]
        rerun = [entry for entry in RecordingView.written if entry.get("sum") == {"sum": 30.0}]
        self.assertEqual(len(base), 2)
        self.assertEqual(len(rerun), 1)
        self.assertEqual(sorted(written), ["other_view", "view"])
        self.assertEqual(RerunningMetric.calls, 1)

    def test_downstream_follows_eval_edges_transitively(self) -> None:
        source = FakeSource({"capture.cls.scores": [1.0]})
        seen: dict[str, Any] = {}

        class Probe(RerunningMetric):
            def after_pass(self, graph: Any) -> None:
                seen["downstream"] = graph.downstream(["factor"])
                seen["none"] = graph.downstream(["view"])

        records = build_eval_graph(
            discover_node_entries(
                {
                    "factor": {"_target_": f"{__name__}.FactorMetric", "in": ["capture.cls.scores"], "out": "eval.cls.sum"},
                    "view": {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.sum"]},
                    "probe": {"_target_": f"{__name__}.SplitProbe", "in": ["capture.cls.scores"], "out": "eval.cls.probe"},
                }
            )
        )
        records[-1].component["_target_"] = f"{__name__}.Probe"
        globals()["Probe"] = Probe
        with tempfile.TemporaryDirectory() as tmp:
            run_eval_graph(records, source=source, out_dir=Path(tmp))
        self.assertEqual(seen, {"downstream": ["view"], "none": []})

    def test_rerun_rejects_extra_kwargs_for_nodes_outside_the_rerun_set(self) -> None:
        class Bad(RerunningMetric):
            def after_pass(self, graph: Any) -> None:
                graph.rerun(["view"], out_dir=graph.out_dir / "bad", extra={"factor": {"factor": 1}})

        globals()["Bad"] = Bad
        records = build_eval_graph(
            discover_node_entries(
                {
                    "factor": {"_target_": f"{__name__}.FactorMetric", "in": ["capture.cls.scores"], "out": "eval.cls.sum"},
                    "view": {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.sum"]},
                    "bad": {"_target_": f"{__name__}.Bad", "in": [], "out": "eval.cls.bad"},
                }
            )
        )
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "outside the rerun set"):
            run_eval_graph(records, source=FakeSource({"capture.cls.scores": [1.0]}), out_dir=Path(tmp))

    def test_node_split_reads_its_inputs_from_that_split(self) -> None:
        RecordingView.written = []
        records = build_eval_graph(
            discover_node_entries(
                {
                    "base": {"_target_": f"{__name__}.SplitProbe", "in": ["capture.cls.scores"], "out": "eval.cls.base"},
                    "fit": {"_target_": f"{__name__}.SplitProbe", "in": ["capture.cls.scores"], "out": "eval.cls.fit", "split": "val"},
                    "view": {"_target_": f"{__name__}.RecordingView", "in": ["eval.cls.base", "eval.cls.fit"]},
                }
            )
        )
        self.assertEqual({record.name: record.split for record in records}, {"base": None, "fit": "val", "view": None})
        sources = {"test": FakeSource({"capture.cls.scores": [1]}), "val": FakeSource({"capture.cls.scores": [2]})}
        with tempfile.TemporaryDirectory() as tmp:
            run_eval_graph(records, source=sources["test"], out_dir=Path(tmp), source_for=sources.__getitem__)
        self.assertEqual(RecordingView.written, [{"base": {"seen": [1]}, "fit": {"seen": [2]}}])

    def test_required_splits_collects_the_report_split_and_every_node_split(self) -> None:
        evalmodule = {"split": "test", "fit": {"_target_": f"{__name__}.SplitProbe", "split": "val"}, "base": {"_target_": f"{__name__}.SplitProbe"}}
        self.assertEqual(required_splits(evalmodule), {"test", "val"})
        self.assertEqual(required_splits(None), {"test"})

    def test_run_eval_fails_fast_naming_the_missing_capture_and_the_recovery_command(self) -> None:
        graph = {"fit": {"_target_": f"{__name__}.SplitProbe", "in": ["capture.cls.scores"], "out": "eval.cls.fit", "split": "val"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
            writer.declare_numeric("capture.cls.scores", shape=(1,), dtype="float32")
            writer.append_numeric("capture.cls.scores", [[0.5]])
            writer.declare_route("cls", scores="capture.cls.scores")
            writer.close()
            with self.assertRaisesRegex(FileNotFoundError, r"'val' capture.*teia test --run-dir"):
                run_eval(root, graph=graph, split="test")


class RunEvalSourceTests(unittest.TestCase):
    def test_reads_route_keyed_capture_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
            writer.declare_numeric("capture.cls.scores", shape=(2,), dtype="float32")
            writer.append_numeric("capture.cls.scores", [[0.4, 0.6]])
            writer.declare_route("cls", scores="capture.cls.scores")
            writer.close()

            source = RunEvalSource(root, split="test")
            self.assertEqual(source.routes(), ["cls"])
            self.assertTrue(source.has("capture.cls.scores"))
            self.assertEqual(source.config(), {})
            self.assertEqual(source.descriptor(), {})

    def test_curves_concatenates_all_stage_versions_per_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_dir = root / "logs" / "teia" / "version_0"
            fit_dir.mkdir(parents=True)
            (fit_dir / "metrics.csv").write_text(
                "epoch,step,train/loss_epoch,lr-AdamW/pg1\n0,10,0.5,1e-4\n1,20,0.3,1e-5\n"
            )
            test_dir = root / "logs" / "teia" / "version_1"
            test_dir.mkdir(parents=True)
            (test_dir / "metrics.csv").write_text("epoch,step,test/loss\n0,0,0.1\n")

            source = RunEvalSource(root, split="test")
            rows = source.curves()["teia"]
            self.assertEqual(len(rows), 3)
            self.assertIn("lr-AdamW/pg1", rows[0])
            self.assertEqual(rows[0]["train/loss_epoch"], "0.5")
            self.assertEqual(rows[2]["test/loss"], "0.1")


if __name__ == "__main__":
    unittest.main()
