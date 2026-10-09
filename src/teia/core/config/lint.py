from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from teia.core.config.layers import ConfLayout, resolve_conf_layout

_PACKAGE_DIRECTIVE_RE = re.compile(r"^#\s*@package\s")
_PACKAGE_ADDRESS_RE = re.compile(r"^#\s*@package\s+(\S+)\s*$")

#: Config yaml carries wiring, not prose. A terse one-liner where a value is genuinely
#: non-obvious is allowed; a rationale essay belongs in ``specs/``.
_MAX_COMMENT_LINES = 2

_PRESET_SCALARS = frozenset({"seed", "data_root", "epochs", "project", "experiment"})
_SYSTEM_GROUPS = {"trainer", "runtime", "callbacks", "loggers", "train", "test", "infer", "export"}
_MODULE_GROUPS = ("datamodule", "netmodule", "evalmodule")
#: Module folders that are not tasks: shared fragments, task-less examples, ``config init`` output.
_NON_TASK_FOLDERS = {"_shared", "demo", "mine"}


@dataclass(frozen=True, slots=True)
class LintFinding:
    path: Path
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.rule}] {self.message}"


def _is_deprecated(path: Path, layer_root: Path) -> bool:
    return "deprecated" in path.relative_to(layer_root).parts


def _layer_files(layout: ConfLayout, group: str) -> list[tuple[Path, Path]]:
    """``(file, layer_root)`` for every ``<group>/**/*.yaml`` across all real layers, skipping deprecated/**."""
    found: list[tuple[Path, Path]] = []
    for layer in layout.ordered:
        base = layer.path / group
        if base.is_dir():
            found.extend((path, layer.path) for path in sorted(base.rglob("*.yaml")) if not _is_deprecated(path, layer.path))
    return found


def package_address(path: Path) -> str | None:
    """The ``# @package`` address declared on the file's first non-blank line, or ``None``."""
    first_line = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
    match = _PACKAGE_ADDRESS_RE.match(first_line.strip())
    return match.group(1) if match else None


def check_node_leaf(path: Path) -> LintFinding | None:
    """A ``node/**`` leaf is a ``_target_`` plus params: no ``# @package`` header, no ``in``/``out`` wiring."""
    if package_address(path) is not None:
        return LintFinding(path, "node-leaf", "node leaves carry no `# @package` header; the mounting module sets the address.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    wiring = sorted({"in", "out"} & set(raw))
    if wiring:
        return LintFinding(path, "node-leaf", f"node leaves carry no wiring; move {wiring} into the module that mounts it.")
    return None


_HYDRA_PACKAGE_KEYWORDS = ("_group_", "_name_")


def check_module(path: Path) -> LintFinding | None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "_target_" in raw:
        return LintFinding(path, "module", "module presets carry no `_target_`; the engine instantiates the executor.")
    clashing = sorted(alias for alias in raw if any(word in str(alias) for word in _HYDRA_PACKAGE_KEYWORDS))
    if clashing:
        return LintFinding(path, "module", f"aliases {clashing} contain a Hydra package keyword (`_group_`/`_name_`), which mounts rewrite.")
    return None


def _contract_tasks() -> set[str]:
    """Task names with a contract in ``teia.task`` (``vision_cls`` is the task ``vision-cls``)."""
    import pkgutil

    import teia.task

    return {name.replace("_", "-") for _, name, _ in pkgutil.iter_modules(teia.task.__path__) if not name.startswith("_")}


def check_module_folder(path: Path, layer_root: Path, tasks: set[str]) -> LintFinding | None:
    """Modules live under ``<graph>module/<task>/`` for a task with a preset or a contract (or a non-task folder)."""
    parts = path.relative_to(layer_root).parts
    if len(parts) < 3:
        return LintFinding(path, "module-folder", "modules live under `<graph>module/<task>/`.")
    if parts[1] in _NON_TASK_FOLDERS or parts[1] in tasks:
        return None
    return LintFinding(path, "module-folder", f"module folder `{parts[1]}` names no task preset or contract.")


def check_task_preset(path: Path) -> LintFinding | None:
    """A task preset is ``_global_``, sets ``task._target_``, overrides every module group, and defines no nodes."""
    if package_address(path) != "_global_":
        return LintFinding(path, "task-preset", "a task preset must be `# @package _global_`.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not (raw.get("task") or {}).get("_target_"):
        return LintFinding(path, "task-preset", "a task preset must set `task._target_` (the contract class).")
    overridden = {str(entry).split(":")[0].replace("override /", "").strip() for d in raw.get("defaults") or [] for entry in (d if isinstance(d, dict) else {d: None})}
    missing = [group for group in _MODULE_GROUPS if group not in overridden]
    if missing:
        return LintFinding(path, "task-preset", f"a task preset must select every module group; missing {missing}.")
    stray = sorted(set(raw) - {"defaults", "task", *_PRESET_SCALARS} - _SYSTEM_GROUPS)
    if stray:
        return LintFinding(path, "task-preset", f"a task preset sets only system groups and the scalars {sorted(_PRESET_SCALARS)}; found {stray}.")
    return None


def iter_all_conf_files(layout: ConfLayout) -> list[Path]:
    """Every ``*.yaml`` across all real layers, skipping ``deprecated/**`` and the root
    ``config.yaml`` skeleton (which carries the primary-slot section dividers)."""
    seen: set[Path] = set()
    found: list[Path] = []
    for layer in layout.ordered:
        if not layer.path.is_dir():
            continue
        for path in sorted(layer.path.rglob("*.yaml")):
            if _is_deprecated(path, layer.path) or path.parent == layer.path:
                continue
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def check_comment_budget(path: Path) -> LintFinding | None:
    """Config yaml comments must be minimal. At most two comment lines per file and no
    multi-line comment block; ``# @package`` directives are functional and never counted."""
    is_comment = [
        stripped.startswith("#") and not _PACKAGE_DIRECTIVE_RE.match(stripped)
        for stripped in (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    ]
    total = sum(is_comment)
    runs: list[int] = []
    run = 0
    for flag in is_comment:
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    if total > _MAX_COMMENT_LINES:
        return LintFinding(
            path=path,
            rule="comment-budget",
            message=f"{total} comment lines (max {_MAX_COMMENT_LINES}); move rationale to specs/.",
        )
    if runs and max(runs) > 1:
        return LintFinding(
            path=path,
            rule="comment-budget",
            message=f"multi-line comment block ({max(runs)} lines); config comments must be single-line.",
        )
    return None


def run_lint(project_dir: Path) -> list[LintFinding]:
    layout = resolve_conf_layout(project_dir)
    findings: list[LintFinding | None] = [check_comment_budget(path) for path in iter_all_conf_files(layout)]
    findings += [check_node_leaf(path) for path, _ in _layer_files(layout, "node")]
    tasks = {path.stem for path, _ in _layer_files(layout, "task")} | _contract_tasks()
    findings += [check_task_preset(path) for path, _ in _layer_files(layout, "task")]
    project = layout.project.path if layout.project else None
    for group in _MODULE_GROUPS:
        for path, root in _layer_files(layout, group):
            if root != project:  # a project module may name a custom executor and live anywhere
                findings += [check_module(path), check_module_folder(path, root, tasks)]
    return [finding for finding in findings if finding is not None]
