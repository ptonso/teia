"""Replicated train: ``runtime.n_runs`` runs of the same config in one group directory.

See specs/core/runtime.md.
"""

from __future__ import annotations

from pathlib import Path

from teia.core.eval.runner import run_group_eval, split_evalmodule
from teia.core.runtime.composer import compose_overlay_config
from teia.core.runtime.engine import train_project
from teia.core.runtime.group_descriptor import write_group_descriptor
from teia.core.runtime.layout import resolve_runtime_layout
from teia.core.runtime.validate import validate_n_runs
from teia.core.utils import ensure_dir

__all__ = ["train_group"]


def train_group(
    *,
    project_dir: Path | None,
    config_dir: Path | None,
    overrides: list[str],
    run_name: str | None,
    checkpoint: str | None,
    skip_post_test: bool,
    skip_report: bool,
    skip_export: bool | None,
    parent_dir: Path | None = None,
) -> Path:
    """Train replica ``i`` into ``<group>/rep_<i>`` with ``runtime.seed`` set to the base seed plus
    ``i``, then write the aggregate report. A replica failure propagates and leaves the group
    ``incomplete``."""
    resolved_project_dir = Path(project_dir or ".").resolve()
    config, _ = compose_overlay_config(project_dir=resolved_project_dir, config_dir=config_dir, overrides=overrides)
    validate_n_runs(config)
    n_runs = int(config["runtime"]["n_runs"])
    base_seed = int(config["runtime"]["seed"])

    group_dir = ensure_dir(resolve_runtime_layout(config, resolved_project_dir, run_name, parent_dir=parent_dir).run_dir)
    write_group_descriptor(group_dir, n_runs=n_runs, base_seed=base_seed, status="incomplete")
    for index in range(n_runs):
        train_project(
            project_dir=resolved_project_dir,
            config_dir=config_dir,
            overrides=[*overrides, f"runtime.seed={base_seed + index}"],
            run_name=f"rep_{index}",
            checkpoint=checkpoint,
            skip_post_test=skip_post_test,
            skip_report=skip_report,
            skip_export=skip_export,
            parent_dir=group_dir,
        )

    settings, graph = split_evalmodule(config.get("evalmodule"))
    if graph and settings.enabled and not skip_report and not skip_post_test:
        run_group_eval(group_dir, graph=graph, split=settings.split)
    write_group_descriptor(group_dir, n_runs=n_runs, base_seed=base_seed, status="ok")
    return group_dir
