"""``teia sweep``: override-driven hyperparameter search. See specs/core/sweep.md."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser

from teia.core.eval.runner import RunEvalSource
from teia.core.runtime.composer import compose_overlay_config
from teia.core.runtime.engine import train_project
from teia.core.runtime.group import train_group
from teia.core.runtime.group_descriptor import is_group_dir, replica_dirs
from teia.core.runtime.layout import resolve_runtime_layout
from teia.core.runtime.validate import validate_n_runs
from teia.core.utils import ensure_dir, write_yaml

__all__ = ["Dimension", "split_overrides", "plan_trials", "score_trial", "run_sweep"]


@dataclass(frozen=True)
class Dimension:
    """One sweep override. ``values`` is set for a discrete dimension, ``low``/``high`` for a continuous one."""

    key: str
    expression: str
    values: tuple[str, ...] | None = None
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    integer: bool = False


def split_overrides(overrides: list[str]) -> tuple[list[str], list[Dimension]]:
    """The baseline (non-sweep overrides, verbatim) and the sweep dimensions, as Hydra's parser sees them."""
    baseline: list[str] = []
    dimensions: list[Dimension] = []
    for line, override in zip(overrides, OverridesParser.create().parse_overrides(overrides), strict=True):
        if not override.is_sweep_override():
            baseline.append(line)
            continue
        key, expression = override.get_key_element(), line.partition("=")[2]
        if override.is_interval_sweep():
            value = override.value()
            dimensions.append(
                Dimension(key, expression, low=value.start, high=value.end, log="log" in value.tags, integer=isinstance(value.start, int))
            )
        else:
            dimensions.append(Dimension(key, expression, values=tuple(override.sweep_string_iterator())))
    return baseline, dimensions


def plan_trials(dimensions: list[Dimension], n_trials: int | None, seed: int) -> list[list[str]]:
    """Per trial, the ``key=value`` overrides of every dimension: the full grid when all are discrete,
    else ``n_trials`` independent samples from a generator seeded with ``seed``."""
    if all(dimension.values is not None for dimension in dimensions):
        grids = [[f"{dimension.key}={value}" for value in dimension.values] for dimension in dimensions]
        return [list(trial) for trial in itertools.product(*grids)]
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"sweep.n_trials must be an integer of at least 1 when a dimension is continuous, got {n_trials!r}.")
    rng = random.Random(seed)
    return [[f"{dimension.key}={_sample(dimension, rng)}" for dimension in dimensions] for _ in range(n_trials)]


def _sample(dimension: Dimension, rng: random.Random) -> Any:
    if dimension.values is not None:
        return rng.choice(dimension.values)
    low, high = dimension.low, dimension.high
    if dimension.log:
        if low <= 0:
            raise ValueError(f"log sweep '{dimension.key}' needs a positive lower bound, got {low}.")
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
        return min(max(round(value), low), high) if dimension.integer else repr(value)
    return rng.randint(low, high) if dimension.integer else repr(rng.uniform(low, high))


def score_trial(trial_dir: Path, objective: str, mode: str) -> float:
    """Mean over the trial's replicas of the best value of the logged ``objective`` series."""
    pick = {"min": min, "max": max}[mode]
    dirs = replica_dirs(trial_dir) if is_group_dir(trial_dir) else [trial_dir]
    bests = [pick(RunEvalSource(run_dir).log(objective)) for run_dir in dirs]
    return sum(bests) / len(bests)


def run_sweep(*, project_dir: Path | None, config_dir: Path | None, overrides: list[str], run_name: str | None) -> tuple[Path, list[str]]:
    """Run every trial and return the sweep directory and the best trial's overrides."""
    baseline, dimensions = split_overrides(overrides)
    if not dimensions:
        raise ValueError("teia sweep needs at least one sweep expression in the overrides, e.g. optim.lr=1e-3,1e-4.")
    resolved_project_dir = Path(project_dir or ".").resolve()
    config, _ = compose_overlay_config(project_dir=resolved_project_dir, config_dir=config_dir, overrides=baseline)
    objective, mode = config["sweep"]["objective"], config["sweep"]["mode"]
    if not objective or mode not in ("min", "max"):
        raise ValueError(f"sweep.objective must be set and sweep.mode must be min or max, got {objective!r} and {mode!r}.")
    plan = plan_trials(dimensions, config["sweep"]["n_trials"], int(config["runtime"]["seed"]))

    sweep_dir = ensure_dir(resolve_runtime_layout(config, resolved_project_dir, run_name).run_dir)
    descriptor: dict[str, Any] = {
        "objective": objective,
        "mode": mode,
        "baseline": baseline,
        "dimensions": {dimension.key: dimension.expression for dimension in dimensions},
        "trials": [],
        "best": None,
        "status": "incomplete",
    }
    for index, sampled in enumerate(plan):
        name = f"trial_{index}"
        _train_trial(resolved_project_dir, config_dir, [*baseline, *sampled], name, sweep_dir)
        descriptor["trials"].append({"name": name, "overrides": sampled, "score": score_trial(sweep_dir / name, objective, mode)})
        scored = [trial["score"] for trial in descriptor["trials"]]
        descriptor["best"] = descriptor["trials"][scored.index({"min": min, "max": max}[mode](scored))]["name"]
        write_yaml(sweep_dir / "sweep.yaml", descriptor)
    descriptor["status"] = "ok"
    write_yaml(sweep_dir / "sweep.yaml", descriptor)
    best = next(trial for trial in descriptor["trials"] if trial["name"] == descriptor["best"])
    return sweep_dir, best["overrides"]


def _train_trial(project_dir: Path, config_dir: Path | None, overrides: list[str], name: str, sweep_dir: Path) -> None:
    config, _ = compose_overlay_config(project_dir=project_dir, config_dir=config_dir, overrides=overrides)
    validate_n_runs(config)
    common: dict[str, Any] = dict(
        project_dir=project_dir,
        config_dir=config_dir,
        overrides=overrides,
        run_name=name,
        checkpoint=None,
        skip_post_test=True,
        skip_report=True,
        skip_export=True,
        parent_dir=sweep_dir,
    )
    if int(config["runtime"]["n_runs"]) > 1:
        train_group(**common)
    else:
        train_project(**common)
