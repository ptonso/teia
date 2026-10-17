from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from teia.base.eval import EvalSource, Metric, View
from teia.core.capture.store import CaptureWriter
from teia.core.eval.runner import run_group_eval
from teia.core.runtime.group_descriptor import replica_dirs, write_group_descriptor
from teia.core.runtime.layout import resolve_runtime_layout
from teia.core.runtime.validate import validate_n_runs


class MeanScore(Metric):
    def from_source(self, source: EvalSource, **inputs: Any) -> dict[str, Any]:
        scores = inputs["scores"]
        return {"mean": float(scores.mean())}


class StdAwareView(View):
    written: list[dict[str, Any]] = []

    def render(self, out_dir: Path, *, value: float, value_std: float | None = None) -> list[Path]:
        StdAwareView.written.append({"value": value, "value_std": value_std})
        return []


def _replica(root: Path, scores: list[float]) -> None:
    writer = CaptureWriter(root / "artifacts" / "capture" / "test", split="test")
    writer.declare_numeric("capture.cls.scores", shape=(1,), dtype="float32")
    writer.append_numeric("capture.cls.scores", [[s] for s in scores])
    writer.declare_route("cls", scores="capture.cls.scores")
    writer.close()


GRAPH = {
    "m": {"_target_": f"{__name__}.MeanScore", "in": ["capture.cls.scores"], "out": "eval.cls.m"},
    "v": {"_target_": f"{__name__}.StdAwareView", "in": {"value": "eval.cls.m:mean"}},
}


def test_group_eval_runs_metrics_per_replica_and_views_once_over_the_reduction(tmp_path: Path) -> None:
    for i, scores in enumerate([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]):
        (tmp_path / f"rep_{i}").mkdir()
        _replica(tmp_path / f"rep_{i}", scores)
    write_group_descriptor(tmp_path, n_runs=3, base_seed=0, status="ok")
    StdAwareView.written.clear()
    run_group_eval(tmp_path, graph=GRAPH, split="test")
    assert StdAwareView.written == [{"value": pytest.approx(3.0), "value_std": pytest.approx(2.0)}]


def test_replica_dirs_requires_a_group_and_its_replicas(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a group"):
        replica_dirs(tmp_path)
    write_group_descriptor(tmp_path, n_runs=2, base_seed=0, status="ok")
    with pytest.raises(FileNotFoundError, match="rep_0"):
        replica_dirs(tmp_path)


def test_layout_under_a_parent_pins_the_replica_name(tmp_path: Path) -> None:
    layout = resolve_runtime_layout({"runs_root": "runs"}, tmp_path, "rep_1", parent_dir=tmp_path / "group")
    assert layout.run_dir == tmp_path / "group" / "rep_1"
    (tmp_path / "group" / "rep_1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        resolve_runtime_layout({"runs_root": "runs"}, tmp_path, "rep_1", parent_dir=tmp_path / "group")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3", None])
def test_n_runs_validation_rejects_non_positive_integers(bad: Any) -> None:
    with pytest.raises(ValueError, match="n_runs"):
        validate_n_runs({"runtime": {"n_runs": bad}})
    validate_n_runs({"runtime": {"n_runs": 3}})
    validate_n_runs({"runtime": {}})


def test_teia_eval_on_a_group_dir_writes_the_aggregate_into_the_group(tmp_path: Path) -> None:
    from argparse import Namespace

    import yaml

    from teia.core.commands.eval import run as teia_eval_run

    for i, scores in enumerate([[0.0, 2.0], [2.0, 4.0]]):
        rep = tmp_path / f"rep_{i}"
        (rep / "config").mkdir(parents=True)
        (rep / "config" / "composed.yaml").write_text(yaml.safe_dump({"evalmodule": GRAPH}))
        _replica(rep, scores)
    write_group_descriptor(tmp_path, n_runs=2, base_seed=0, status="ok")
    StdAwareView.written.clear()
    rc = teia_eval_run(Namespace(run_dirs=[str(tmp_path)], split="test", output_dir=None, compare_path=None))
    assert rc == 0
    assert StdAwareView.written == [{"value": pytest.approx(2.0), "value_std": pytest.approx(2**0.5)}]
