"""
Shared helpers for ``teia.node.eval.compare.*``. Private infrastructure, not a component: no.

Source: common knowledge

Description:
  ``_target_``, no ``conf/eval/compare/*.yaml`` leaf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from teia.base.eval import EvalSource

__all__ = ["run_id", "flat_metrics", "override_map", "SEED_KEYS"]

#: Override keys treated as "seed, not a factor worth comparing on" by both `seed_aggregate`
#: (excluded from its grouping key) and `factor_diff` (excluded from auto-derivation, since a
#: seed sweep almost always coexists with the actual factor someone cares about).
SEED_KEYS = {"runtime.seed", "seed"}


def run_id(source: EvalSource) -> str:
    """``artifacts/run.yaml``'s ``run_id`` if present, else the run directory's own name."""
    descriptor: Any = source.descriptor()
    candidate = descriptor.get("run_id") if isinstance(descriptor, dict) else None
    return str(candidate) if candidate else Path(source.run_dir).name


def flat_metrics(source: EvalSource) -> dict[str, Any]:
    """The run's ``eval/summary.yaml`` (the ``summary`` view's output): the flat scalar map every
    comparison joins on. Read from the eval artifact, not the capture store, so comparison keeps
    working against a run whose capture store was pruned after the eval pass."""
    path = Path(source.run_dir) / "eval" / "summary.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return dict(payload) if isinstance(payload, dict) else {}


def override_map(overrides: list[str]) -> dict[str, str]:
    """``["trainer.max_epochs=10", "runtime.seed=1"]`` -> ``{"trainer.max_epochs": "10",
    "runtime.seed": "1"}``. A malformed entry (no ``=``) is skipped rather than raising, since
    an override list is free-form CLI text, not a validated schema."""
    parsed: dict[str, str] = {}
    for entry in overrides:
        key, sep, value = str(entry).partition("=")
        if sep:
            parsed[key.strip()] = value.strip()
    return parsed
