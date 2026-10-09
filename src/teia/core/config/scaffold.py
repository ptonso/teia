from __future__ import annotations

from pathlib import Path

import yaml

from teia.core.utils import ensure_dir


def write_overlay_scaffold(root: Path, force: bool = False) -> list[Path]:
    """Node-leaf stubs for a project overlay (teia:core/config.md#teia-config-init)."""
    conf_root = ensure_dir(root / "conf")
    files = {
        conf_root / "node" / "net" / "head" / "custom.yaml": "_target_: your_package.CustomHead\n",
        conf_root / "node" / "net" / "loss" / "custom.yaml": "_target_: your_package.CustomLoss\n",
        conf_root / "node" / "data" / "reader" / "custom.yaml": "_target_: your_package.CustomReader\n",
    }
    written: list[Path] = []
    for path, content in files.items():
        if path.exists() and not force:
            continue
        ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def write_resolved_task_scaffold(root: Path, task: str, force: bool = False) -> list[Path]:
    """``teia config init --from task=<task>``: the task's three modules, fully resolved.

    Writes ``conf/<graph>module/mine/<task>.yaml`` — real ``_target_``s and params, no ``defaults:``
    list — selectable as ``<graph>module=mine/<task>`` with no ``task`` (a level-1 run the user owns).
    """
    from teia.core.runtime.composer import compose_overlay_config

    composed, _ = compose_overlay_config(project_dir=root, overrides=[f"task={task}"])
    written: list[Path] = []
    for group in ("datamodule", "netmodule", "evalmodule"):
        path = ensure_dir(root / "conf" / group / "mine") / f"{task}.yaml"
        if path.exists() and not force:
            continue
        path.write_text(yaml.safe_dump(composed.get(group) or {}, sort_keys=False), encoding="utf-8")
        written.append(path)
    return written
