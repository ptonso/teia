from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from teia.core.runtime import sweep
from teia.core.runtime.group_descriptor import write_group_descriptor
from teia.core.utils import read_yaml


def _run_with_series(run_dir: Path, values: list[float], column: str = "val/loss") -> None:
    logs = run_dir / "logs" / "csv" / "version_0"
    logs.mkdir(parents=True)
    rows = "\n".join([column, *map(str, values)])
    (logs / "metrics.csv").write_text(rows + "\n")


def test_split_overrides_separates_baseline_and_dimensions() -> None:
    baseline, dimensions = sweep.split_overrides(
        ["task=cls", "a=1,2,3", "b=range(0,6,2)", "c=interval(0.1,0.9)", "d=tag(log,interval(1e-5,1e-2))", "e=int(interval(2,8))"]
    )
    assert baseline == ["task=cls"]
    by_key = {d.key: d for d in dimensions}
    assert by_key["a"].values == ("1", "2", "3")
    assert by_key["b"].values == ("0", "2", "4")
    assert (by_key["c"].low, by_key["c"].high, by_key["c"].log, by_key["c"].integer) == (0.1, 0.9, False, False)
    assert by_key["d"].log and not by_key["d"].integer
    assert by_key["e"].integer and (by_key["e"].low, by_key["e"].high) == (2, 8)


def test_grid_plan_is_the_full_product_and_ignores_n_trials() -> None:
    _, dimensions = sweep.split_overrides(["a=1,2", "b=x,y,z"])
    plan = sweep.plan_trials(dimensions, n_trials=None, seed=0)
    assert len(plan) == 6
    assert plan[0] == ["a=1", "b=x"] and plan[-1] == ["a=2", "b=z"]


def test_random_plan_is_seeded_and_respects_bounds() -> None:
    _, dimensions = sweep.split_overrides(["lr=tag(log,interval(1e-5,1e-2))", "depth=int(interval(2,8))", "act=relu,gelu"])
    plan = sweep.plan_trials(dimensions, n_trials=50, seed=3)
    assert plan == sweep.plan_trials(dimensions, n_trials=50, seed=3)
    assert plan != sweep.plan_trials(dimensions, n_trials=50, seed=4)
    for lr, depth, act in plan:
        assert 1e-5 <= float(lr.split("=")[1]) <= 1e-2
        assert 2 <= int(depth.split("=")[1]) <= 8
        assert act in ("act=relu", "act=gelu")


@pytest.mark.parametrize("n_trials", [None, 0, True, 1.5])
def test_continuous_dimension_requires_a_positive_integer_budget(n_trials: Any) -> None:
    _, dimensions = sweep.split_overrides(["lr=interval(0,1)"])
    with pytest.raises(ValueError, match="n_trials"):
        sweep.plan_trials(dimensions, n_trials=n_trials, seed=0)


def test_log_dimension_needs_a_positive_lower_bound() -> None:
    _, dimensions = sweep.split_overrides(["lr=tag(log,interval(0,1))"])
    with pytest.raises(ValueError, match="positive lower bound"):
        sweep.plan_trials(dimensions, n_trials=2, seed=0)


def test_score_is_the_best_epoch_in_both_modes_and_the_mean_over_replicas(tmp_path: Path) -> None:
    single = tmp_path / "single"
    _run_with_series(single, [0.9, 0.4, 0.6])
    assert sweep.score_trial(single, "val/loss", "min") == pytest.approx(0.4)
    assert sweep.score_trial(single, "val/loss", "max") == pytest.approx(0.9)

    group = tmp_path / "group"
    _run_with_series(group / "rep_0", [0.5, 0.3])
    _run_with_series(group / "rep_1", [0.4, 0.1])
    write_group_descriptor(group, n_runs=2, base_seed=0, status="ok")
    assert sweep.score_trial(group, "val/loss", "min") == pytest.approx(0.2)


def test_score_raises_naming_the_run_when_the_series_is_missing(tmp_path: Path) -> None:
    _run_with_series(tmp_path / "run", [0.1], column="train/loss")
    with pytest.raises(KeyError, match="val/loss"):
        sweep.score_trial(tmp_path / "run", "val/loss", "min")


def _sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: list[str]) -> tuple[Path, list[str], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_train(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        lr = next(o for o in kwargs["overrides"] if o.startswith("trainer.accumulate_grad_batches=")).split("=")[1]
        _run_with_series(kwargs["parent_dir"] / kwargs["run_name"], [float(lr), float(lr) + 1])
        return {}

    monkeypatch.setattr(sweep, "train_project", fake_train)
    sweep_dir, best = sweep.run_sweep(project_dir=tmp_path, config_dir=None, overrides=overrides, run_name="s")
    return sweep_dir, best, calls


def test_run_sweep_trains_each_trial_without_test_and_records_the_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweep_dir, best, calls = _sweep(
        tmp_path, monkeypatch, ["data_root=.", "sweep.objective=val/loss", "sweep.mode=min", "trainer.accumulate_grad_batches=3,1,2"]
    )
    assert best == ["trainer.accumulate_grad_batches=1"]
    assert [c["run_name"] for c in calls] == ["trial_0", "trial_1", "trial_2"]
    assert all(c["skip_post_test"] and c["skip_report"] and c["skip_export"] for c in calls)
    descriptor = read_yaml(sweep_dir / "sweep.yaml")
    assert descriptor["status"] == "ok" and descriptor["best"] == "trial_1"
    assert [t["score"] for t in descriptor["trials"]] == [3.0, 1.0, 2.0]
    assert descriptor["dimensions"] == {"trainer.accumulate_grad_batches": "3,1,2"}


def test_run_sweep_maximizes_when_mode_is_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, best, _ = _sweep(tmp_path, monkeypatch, ["data_root=.", "sweep.objective=val/loss", "sweep.mode=max", "trainer.accumulate_grad_batches=3,1,2"])
    assert best == ["trainer.accumulate_grad_batches=3"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["data_root=.", "sweep.objective=val/loss", "sweep.mode=min", "lr=1"], "at least one sweep expression"),
        (["data_root=.", "sweep.mode=min", "lr=1,2"], "sweep.objective"),
        (["data_root=.", "sweep.objective=val/loss", "sweep.mode=avg", "lr=1,2"], "sweep.mode"),
        (["data_root=.", "sweep.objective=val/loss", "sweep.mode=min", "lr=interval(0,1)"], "n_trials"),
    ],
)
def test_run_sweep_fails_before_any_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: list[str], message: str) -> None:
    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("no trial may start")

    monkeypatch.setattr(sweep, "train_project", forbidden)
    with pytest.raises(ValueError, match=message):
        sweep.run_sweep(project_dir=tmp_path, config_dir=None, overrides=overrides, run_name="s")
