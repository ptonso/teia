"""Tests for val-metric unification: the streaming eval-graph
driver (``teia.core.eval.streaming``), each aligned metric's ``update``/``compute``/``reset``
streaming parity against its own ``from_source`` (same formula, two drivers), and the
capture callback's per-step atoms (``CaptureMap``).

Full callback-integration coverage (streaming through ``ImageCaptureCallback`` end to end,
per task) lives in ``test_vision_report_capture.py``'s ``..._validation_streams_..._fitness_...``
tests — this file is the unit layer underneath that.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from teia.base.eval import Metric
from teia.core.eval.graph import build_eval_graph, discover_node_entries
from teia.core.eval.streaming import LiveFrameSource, StreamingMonitorRunner, build_streaming_runner, monitor_closure
from teia.node.eval.metric.aligned.concordance import Concordance
from teia.node.eval.metric.aligned.confusion import Confusion
from teia.node.eval.metric.aligned.horizon_error import HorizonError
from teia.node.eval.metric.aligned.residuals import Residuals
from teia.node.eval.metric.aligned.semantic_confusion import SemanticConfusion

# -- fake components for StreamingMonitorRunner structural tests, mirroring test_eval_graph.py's
# style: module-level classes referenced by `_target_: f"{__name__}.ClassName"`. -------------------


class RootMetric(Metric):
    """A pure leaf: `in` is capture/batch-only, so the streaming driver calls `update` every
    batch. `MONITORS` includes a non-"fitness" name to prove per-node namespacing."""

    MONITORS = ("count",)

    def __init__(self) -> None:
        self.reset()

    def update(self, *, values: Any = None, **_: Any) -> None:
        if values is not None:
            self._buf.extend(values)

    def compute(self) -> dict[str, Any]:
        return {"count": len(self._buf)}

    def reset(self) -> None:
        self._buf: list[Any] = []

    def from_source(self, source: Any, **inputs: Any) -> dict[str, Any]:
        del source
        return {"count": len(inputs.get("values") or [])}


class DependentMetric(Metric):
    """Depends on `RootMetric`'s `eval.*` output — only fed at `compute()` time via the
    streaming runner's synthetic epoch-end `update()` injection, never per batch."""

    MONITORS = ("doubled",)

    def __init__(self) -> None:
        self.reset()

    def update(self, *, upstream: Any = None, **_: Any) -> None:
        if upstream is not None:
            self._upstream = upstream

    def compute(self) -> dict[str, Any]:
        return {"doubled": (self._upstream or {}).get("count", 0) * 2}

    def reset(self) -> None:
        self._upstream: dict[str, Any] | None = None

    def from_source(self, source: Any, **inputs: Any) -> dict[str, Any]:
        del source
        return {"doubled": inputs["upstream"]["count"] * 2}


class FitnessMetric(Metric):
    """`MONITORS` includes literal `"fitness"` — always logged bare, unlike every other name."""

    MONITORS = ("fitness",)

    def __init__(self) -> None:
        self.reset()

    def update(self, *, values: Any = None, **_: Any) -> None:
        if values is not None:
            self._buf.extend(values)

    def compute(self) -> dict[str, Any]:
        return {"fitness": float(len(self._buf))}

    def reset(self) -> None:
        self._buf: list[Any] = []

    def from_source(self, source: Any, **inputs: Any) -> dict[str, Any]:
        del source
        return {"fitness": float(len(inputs.get("values") or []))}


def _graph(*, root_monitor: bool, dependent_monitor: bool, include_unreachable: bool = False) -> list:
    entries: dict[str, dict[str, Any]] = {
        "root": {
            "_target_": f"{__name__}.RootMetric",
            "in": ["batch.values"],
            "out": "eval.root",
            **({"monitor": True} if root_monitor else {}),
        },
        "dependent": {
            "_target_": f"{__name__}.DependentMetric",
            "in": {"upstream": "eval.root"},
            "out": "eval.dependent",
            **({"monitor": True} if dependent_monitor else {}),
        },
    }
    if include_unreachable:
        entries["unreachable"] = {"_target_": f"{__name__}.RootMetric", "in": ["batch.other"], "out": "eval.unreachable"}
    return build_eval_graph(discover_node_entries(entries))


class MonitorClosureTests(unittest.TestCase):
    def test_includes_transitive_dependency_chain(self) -> None:
        records = _graph(root_monitor=False, dependent_monitor=True)
        self.assertEqual({r.name for r in monitor_closure(records)}, {"root", "dependent"})

    def test_excludes_non_monitor_non_dependency_nodes(self) -> None:
        records = _graph(root_monitor=True, dependent_monitor=False)
        self.assertEqual({r.name for r in monitor_closure(records)}, {"root"})

    def test_unreachable_node_excluded_entirely(self) -> None:
        records = _graph(root_monitor=True, dependent_monitor=False, include_unreachable=True)
        self.assertEqual({r.name for r in monitor_closure(records)}, {"root"})

    def test_no_monitor_node_gives_empty_closure(self) -> None:
        records = _graph(root_monitor=False, dependent_monitor=False)
        self.assertEqual(monitor_closure(records), [])


class StreamingMonitorRunnerTests(unittest.TestCase):
    def test_chain_runs_per_batch_root_and_epoch_end_dependent(self) -> None:
        records = _graph(root_monitor=False, dependent_monitor=True)
        runner = StreamingMonitorRunner(records)
        runner.reset()
        runner.update_batch({"batch.values": [1, 2]})
        runner.update_batch({"batch.values": [3]})
        # root isn't monitor:true, so only "dependent"'s MONITORS surface, namespaced since its
        # own node name ("dependent") differs from the published key ("doubled").
        self.assertEqual(runner.compute(), {"dependent.doubled": 6.0})

    def test_fitness_is_never_namespaced(self) -> None:
        entries = {"a": {"_target_": f"{__name__}.FitnessMetric", "in": ["batch.values"], "out": "eval.a", "monitor": True}}
        records = build_eval_graph(discover_node_entries(entries))
        runner = StreamingMonitorRunner(records)
        runner.reset()
        runner.update_batch({"batch.values": [1, 2, 3]})
        self.assertEqual(runner.compute(), {"fitness": 3.0})

    def test_duplicate_fitness_across_nodes_raises(self) -> None:
        # Caught at build_eval_graph time now (teia/core/eval/graph.py::_validate_eval_contracts),
        # not lazily at StreamingMonitorRunner.compute() — see EvalContractsBuildTimeChecksTests
        # below for the dedicated pre-flight coverage.
        entries = {
            "a": {"_target_": f"{__name__}.FitnessMetric", "in": ["batch.values"], "out": "eval.a", "monitor": True},
            "b": {"_target_": f"{__name__}.FitnessMetric", "in": ["batch.other"], "out": "eval.b", "monitor": True},
        }
        with self.assertRaisesRegex(ValueError, "duplicate val-metric monitor name"):
            build_eval_graph(discover_node_entries(entries))

    def test_build_streaming_runner_returns_none_without_monitor_node(self) -> None:
        graph = {"root": {"_target_": f"{__name__}.RootMetric", "in": ["batch.values"], "out": "eval.root"}}
        self.assertIsNone(build_streaming_runner(graph))

    def test_build_streaming_runner_returns_active_runner(self) -> None:
        graph = {"root": {"_target_": f"{__name__}.RootMetric", "in": ["batch.values"], "out": "eval.root", "monitor": True}}
        runner = build_streaming_runner(graph)
        self.assertIsNotNone(runner)
        runner.update_batch({"batch.values": [1, 2]})
        self.assertEqual(runner.compute(), {"root.count": 2})


class LiveFrameSourceTests(unittest.TestCase):
    def test_column_reads_the_batch_frame(self) -> None:
        source = LiveFrameSource({"batch.cls": [0, 1]})
        self.assertEqual(source.column("batch.cls"), [0, 1])
        self.assertIsNone(source.column("batch.missing"))

    def test_meta_reads_the_constant_dict(self) -> None:
        source = LiveFrameSource({}, {"class_names": ["a", "b"]})
        self.assertEqual(source.meta("class_names"), ["a", "b"])
        self.assertEqual(source.meta("missing", "fallback"), "fallback")


# -- per-metric streaming (update/compute/reset) vs materialized (from_source) parity ---------------


class ConfusionStreamingTests(unittest.TestCase):
    def test_single_label_streaming_matches_from_source_and_sets_fitness(self) -> None:
        scores = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]
        cls = [0, 1, 0]  # third sample wrong
        class_names = ["a", "b", "c"]

        offline = Confusion().from_source(None, scores=scores, cls=cls, class_names=class_names)

        streaming = Confusion()
        streaming.update(scores=[scores[0]], cls=[cls[0]], class_names=class_names)
        streaming.update(scores=scores[1:], cls=cls[1:])
        streamed = streaming.compute()

        self.assertAlmostEqual(offline["accuracy"], streamed["accuracy"])
        self.assertAlmostEqual(offline["macro_f1"], streamed["macro_f1"])
        self.assertEqual(streamed["fitness"], streamed["accuracy"])

    def test_multi_label_fitness_is_macro_f1(self) -> None:
        streaming = Confusion(multi_label=True, threshold=0.5)
        streaming.update(scores=[[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], cls=[[1, 0, 1], [0, 1, 0]], class_names=["a", "b", "c"])
        out = streaming.compute()
        self.assertAlmostEqual(out["macro_f1"], 1.0)
        self.assertEqual(out["fitness"], out["macro_f1"])

    def test_reset_clears_buffers(self) -> None:
        streaming = Confusion()
        streaming.update(scores=[[0.9, 0.1]], cls=[0], class_names=["a", "b"])
        streaming.reset()
        self.assertEqual(streaming.compute(), {"n_samples": 0})


def _write_label_png(path: Path, values: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(values, dtype=np.uint8), mode="L").save(path)


class _StaticSource:
    def __init__(self, columns: dict[str, Any]) -> None:
        self._columns = columns

    def column(self, key: str) -> Any:
        return self._columns.get(key)


class SemanticConfusionStreamingTests(unittest.TestCase):
    def test_tensor_accumulation_matches_file_based_from_source(self) -> None:
        gt = [[0, 0], [1, 0]]
        pred = [[0, 0], [1, 1]]  # bg IoU 2/3, fg IoU 1/2
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path, pred_path = root / "gt.png", root / "pred.png"
            _write_label_png(gt_path, gt)
            _write_label_png(pred_path, pred)

            keys = {"pred": "capture.seg.mask", "gt": "batch.masks"}
            offline = SemanticConfusion(**keys).from_source(
                _StaticSource({"capture.seg.mask": [str(pred_path)], "batch.masks": [str(gt_path)]}),
                class_names=["bg", "fg"],
            )

            streaming = SemanticConfusion(**keys)
            live_source = LiveFrameSource({"capture.seg.mask": torch.tensor(pred), "batch.masks": torch.tensor(gt)})
            streaming.update(_source=live_source, class_names=["bg", "fg"])
            streamed = streaming.compute()

            self.assertAlmostEqual(offline["mean_iou"], streamed["mean_iou"], places=5)
            self.assertEqual(streamed["fitness"], streamed["mean_iou"])

    def test_update_without_source_is_a_noop(self) -> None:
        streaming = SemanticConfusion(pred="capture.seg.mask", gt="batch.masks")
        streaming.update(class_names=["bg", "fg"])
        self.assertEqual(streaming.compute(), {"n_samples": 0})


class RegressionFamilyStreamingTests(unittest.TestCase):
    def test_residuals_streaming_matches_from_source(self) -> None:
        pred = [[1.0], [2.0], [3.0]]
        target = [[1.5], [2.0], [2.5]]
        offline = Residuals().from_source(None, pred=pred, target=target)

        streaming = Residuals()
        streaming.update(pred=[pred[0]], target=[target[0]])
        streaming.update(pred=pred[1:], target=target[1:])
        streamed = streaming.compute()

        self.assertAlmostEqual(offline["rmse"], streamed["rmse"])
        self.assertAlmostEqual(offline["r2"], streamed["r2"])
        self.assertNotIn("fitness", Residuals().MONITORS)

    def test_horizon_error_publishes_top_level_mae_rmse_matching_overall(self) -> None:
        quantiles = [[[1.0]], [[2.0]]]
        target = [[1.0], [3.0]]
        streaming = HorizonError()
        streaming.update(quantiles=quantiles, target=target)
        out = streaming.compute()
        self.assertEqual(out["mae"], out["overall"]["mae"])
        self.assertEqual(out["rmse"], out["overall"]["rmse"])

    def test_concordance_streaming_matches_from_source(self) -> None:
        time = [5.0, 3.0, 8.0, 1.0]
        event = [1, 1, 0, 1]
        risk = [0.9, 0.5, 0.2, 0.99]
        offline = Concordance().from_source(None, time=time, event=event, risk=risk)

        streaming = Concordance()
        streaming.update(time=time[:2], event=event[:2], risk=risk[:2])
        streaming.update(time=time[2:], event=event[2:], risk=risk[2:])
        streamed = streaming.compute()

        self.assertAlmostEqual(offline["c_index"], streamed["c_index"])
        self.assertEqual(set(Concordance().MONITORS), {"c_index", "ipcw_c_index"})


class DetectionFamilyStreamingTests(unittest.TestCase):
    """InstanceMatch (per-batch matching, buffered rows) + RankedAveragePrecision (fed once at
    epoch end via the streaming runner's synthetic update injection) — the det/obb/pose chain."""

    def test_instance_match_and_ranked_ap_streaming_chain(self) -> None:
        entries = {
            "match": {
                "_target_": "teia.node.eval.metric.aligned.instance_match.InstanceMatch",
                "overlap_thresholds": [0.5],
                "similarity": {"_target_": "teia.node.eval.kernel.iou_xyxy.IouXyxy"},
                "pred": {"geometry": "capture.det.boxes", "category": "capture.det.category", "score": "capture.det.score", "index": "capture.det.sample_idx"},
                "gt": {"geometry": "batch.bboxes", "category": "batch.cls", "index": "batch.batch_idx"},
                "in": [
                    "capture.det.boxes", "capture.det.category", "capture.det.score", "capture.det.sample_idx",
                    "batch.bboxes", "batch.cls", "batch.batch_idx",
                ],
                "out": "eval.det.matched",
            },
            "ap": {
                "_target_": "teia.node.eval.metric.aligned.ranked_average_precision.RankedAveragePrecision",
                "is_fitness": True,
                "in": {"matched": "eval.det.matched"},
                "out": "eval.det.ap",
                "monitor": True,
            },
        }
        records = build_eval_graph(discover_node_entries(entries))
        runner = StreamingMonitorRunner(records)
        runner.reset()
        runner.update_batch(
            {
                "capture.det.boxes": [[0.5, 0.5, 0.2, 0.2]],
                "capture.det.category": [0],
                "capture.det.score": [0.9],
                "capture.det.sample_idx": [0],
                "batch.bboxes": [[0.5, 0.5, 0.2, 0.2]],
                "batch.cls": [0],
                "batch.batch_idx": [0],
            }
        )
        flat = runner.compute()
        self.assertGreater(flat["ap.mean_average_precision"], 0.9)
        self.assertEqual(flat["fitness"], flat["ap.mean_average_precision"])


if __name__ == "__main__":
    unittest.main()
