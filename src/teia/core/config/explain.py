from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.core.runtime.composer import compose_overlay_config


def compose_choices(project_dir: Path, overrides: list[str]) -> dict[str, Any]:
    """Compose ``overrides`` and return the flat ``group -> selected option`` map.

    This is ``hydra.runtime.choices`` — every config group Hydra knows about, mapped to the
    option name it resolved to (``None`` for an ``optional ...: null`` slot nobody selected).
    """
    _, hydra_payload = compose_overlay_config(project_dir=project_dir, overrides=overrides)
    choices = hydra_payload.get("runtime", {}).get("choices", {})
    return dict(choices) if isinstance(choices, dict) else {}


def diff_choices(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Groups whose selected option differs between two ``hydra.runtime.choices`` maps."""
    keys = set(base) | set(variant)
    changed = {key: (base.get(key), variant.get(key)) for key in keys if base.get(key) != variant.get(key)}
    return changed


def _is_bundle_override(override: str) -> bool:
    body = override.lstrip("+~")
    key = body.split("=", 1)[0]
    group = key.split("@", 1)[0]
    return group in {"task", "datamodule", "netmodule", "evalmodule"}


def _slot_qualified(override: str) -> bool:
    return "@" in override.split("=", 1)[0]


def explain_overrides(project_dir: Path, overrides: list[str]) -> list[tuple[str, str, str]]:
    """Attribute every changed group's final option to the override (or bundle chain) that set it.

    Composes once with every override, then once more per bundle-level override with that single
    override removed — a group whose choice reverts when a specific bundle override is dropped was
    set (transitively) by that bundle. A mount swap (``node/net/encoder@netmodule.encoder=x``)
    only composes on top of the module that mounts it, so it is dropped from those recompositions
    and its group is attributed to ``CLI`` first; whatever else is left changed only by the full
    override set is attributed to ``CLI`` too.
    """
    full_choices = compose_choices(project_dir, overrides)
    attribution: dict[str, str] = {}

    for override in overrides:
        if _slot_qualified(override) and not _is_bundle_override(override):
            group = override.lstrip("+~").split("=", 1)[0]
            attribution[group] = "CLI"

    bundle_overrides = [o for o in overrides if _is_bundle_override(o)]
    for bundle_override in bundle_overrides:
        remaining = [o for o in overrides if o != bundle_override and not _slot_qualified(o)]
        without_choices = compose_choices(project_dir, remaining)
        changed = diff_choices(without_choices, full_choices)
        label = bundle_override.split("=", 1)[0]
        for group in changed:
            attribution.setdefault(group, label)

    baseline_choices = compose_choices(project_dir, [])
    changed_from_baseline = diff_choices(baseline_choices, full_choices)
    for group in changed_from_baseline:
        attribution.setdefault(group, "CLI")

    rows: list[tuple[str, str, str]] = []
    for group in sorted(changed_from_baseline):
        option = full_choices.get(group)
        rows.append((group, str(option), attribution.get(group, "CLI")))
    return rows
