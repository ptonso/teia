from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teia.core.utils import get_dot_path, slugify, timestamp_slug


PART_ALIASES = {
    "project": "project_name",
    "experiment": "experiment_name",
    "run": "run_name",
}
DEFAULT_RUN_PARTS = ["project", "experiment", "run"]
RESERVED_DIR_NAMES = ("config", "logs", "eval", "infer", "export", "artifacts")


@dataclass(slots=True)
class RuntimeLayout:
    project_dir: Path
    runs_root: Path
    cache_root: Path
    run_name: str
    run_dir: Path
    config_dir: Path
    logs_dir: Path
    eval_dir: Path
    infer_dir: Path
    export_dir: Path
    artifacts_dir: Path


def resolve_runtime_layout(config: dict[str, Any], project_dir: Path, cli_run_name: str | None = None) -> RuntimeLayout:
    project_path = Path(project_dir).resolve()
    runs_root = _resolve_root_path(project_path, config.get("runs_root", "runs"))
    cache_root = _resolve_root_path(project_path, config.get("cache_root", ".teia-cache"))

    resolved_run_name = slugify(str(cli_run_name or config.get("run_name") or timestamp_slug()))
    parts = _resolve_run_parts(config=config, resolved_run_name=resolved_run_name)
    run_dir = runs_root.joinpath(*parts) if parts else runs_root / resolved_run_name
    run_dir, resolved_run_name = _resolve_unique_run_dir(run_dir, resolved_run_name)

    return RuntimeLayout(
        project_dir=project_path,
        runs_root=runs_root,
        cache_root=cache_root,
        run_name=resolved_run_name,
        run_dir=run_dir,
        config_dir=run_dir / "config",
        logs_dir=run_dir / "logs",
        eval_dir=run_dir / "eval",
        infer_dir=run_dir / "infer",
        export_dir=run_dir / "export",
        artifacts_dir=run_dir / "artifacts",
    )


def _resolve_root_path(project_dir: Path, raw_value: Any) -> Path:
    path = Path(str(raw_value or "."))
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def _resolve_run_parts(config: dict[str, Any], resolved_run_name: str) -> list[str]:
    raw_parts = config.get("run_parts", DEFAULT_RUN_PARTS)
    if raw_parts is None:
        raw_parts = DEFAULT_RUN_PARTS
    if not isinstance(raw_parts, list):
        raise ValueError("Config run_parts must be a list of path segments.")

    normalized = dict(config)
    normalized["run_name"] = resolved_run_name

    parts: list[str] = []
    for entry in raw_parts:
        if entry is None:
            continue
        if not isinstance(entry, str):
            raise ValueError("Config run_parts entries must be strings.")
        value = _resolve_run_part_entry(normalized, entry)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        parts.append(slugify(text))
    return parts


def _resolve_unique_run_dir(run_dir: Path, run_name: str) -> tuple[Path, str]:
    if not run_dir.exists():
        return run_dir, run_name
    counter = 2
    while True:
        candidate_name = f"{run_name}_{counter}"
        candidate_dir = run_dir.parent / candidate_name
        if not candidate_dir.exists():
            return candidate_dir, candidate_name
        counter += 1


def _resolve_run_part_entry(config: dict[str, Any], entry: str) -> Any:
    lookup = PART_ALIASES.get(entry, entry)
    if lookup in PART_ALIASES.values() or "." in lookup:
        try:
            return get_dot_path(config, lookup)
        except KeyError as exc:
            raise ValueError(f"Config run_parts entry {entry!r} could not resolve config path {lookup!r}.") from exc
    return entry
