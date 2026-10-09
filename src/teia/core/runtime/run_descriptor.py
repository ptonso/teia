"""``artifacts/run.yaml``: the machine-readable identity of a run.

Written at run start (``status: incomplete``, no stages yet) and updated whenever a stage
completes successfully (``status: ok``, that stage appended). See
packages/teia/specs/core/eval.md.

A run that crashes, or trains but is never tested, is never advanced past its last successful
write. A comparison reading ``run.yaml`` then sees ``status: incomplete`` and a short
``stages`` list, rather than a missing file or a stale/absent ``eval/<route>/summary.yaml`` mistaken
for "never ran". This is not full exception-based failure detection (which would mean wrapping
every stage body in ``engine.py``); ``incomplete`` already answers the only question a
comparison has: did this run reach the stage whose numbers it is about to read.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from teia.core.utils import ensure_dir, read_yaml, write_yaml

__all__ = ["run_descriptor_path", "start_run_descriptor", "finish_run_stage"]

_FILENAME = "run.yaml"
_SCHEMA_VERSION = 2


def run_descriptor_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "artifacts" / _FILENAME


def start_run_descriptor(*, run_dir: str | Path, config: dict[str, Any]) -> None:
    """Idempotent: a later stage (e.g. ``test`` on an already-trained run) must not overwrite the
    identity (``run_id``/``seed``/``git_rev``) a prior stage already established."""
    path = run_descriptor_path(run_dir)
    if path.exists():
        return
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": Path(run_dir).resolve().name,
        "group": str(config.get("experiment_name") or ""),
        "seed": int((config.get("runtime") or {}).get("seed") or 0),
        "git_rev": _git_rev(),
        # Run artifacts must not contain absolute paths: data_root is recorded as given in
        # config (already project-relative in every composed config this repo produces), never
        # resolved to an absolute filesystem path.
        "data_root": str(config.get("data_root") or ""),
        "dataset_id": None,
        "task": (config.get("task") or {}).get("_target_"),
        "routes": {},
        "status": "incomplete",
        "stages": [],
    }
    ensure_dir(path.parent)
    write_yaml(path, payload)


def finish_run_stage(*, run_dir: str | Path, stage: str, routes: dict[str, list[str]] | None = None) -> None:
    """Append ``stage`` to the descriptor and mark ``status: ok``. Tolerates a missing descriptor
    (``start_run_descriptor`` should always have run first; this never fabricates run identity)."""
    path = run_descriptor_path(run_dir)
    if not path.exists():
        return
    payload = read_yaml(path) or {}
    stages = list(payload.get("stages") or [])
    if stage not in stages:
        stages.append(stage)
    payload["stages"] = stages
    payload["status"] = "ok"
    if routes:
        payload["routes"] = {**dict(payload.get("routes") or {}), **{str(k): [str(a) for a in v] for k, v in routes.items()}}
    write_yaml(path, payload)


def _git_rev() -> str | None:
    try:
        result = subprocess.run(  # noqa: S603, S607 - fixed argv, no shell, best-effort provenance
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
