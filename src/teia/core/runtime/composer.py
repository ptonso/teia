from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.core.config.layers import resolve_conf_layout
from teia.core.utils import get_dot_path, parse_override, read_yaml, slugify


def resolve_resumable_run_dir(run_dir: Path) -> Path:
    """Resolve a --from-run-dir path, slugifying components to match how runs are written on disk."""
    resolved = Path(run_dir).resolve()
    if (resolved / "config" / "overrides.yaml").exists():
        return resolved
    slugged = Path(resolved.anchor, *(slugify(part) for part in resolved.relative_to(resolved.anchor).parts))
    if (slugged / "config" / "overrides.yaml").exists():
        return slugged
    raise FileNotFoundError(
        f"Run dir is not a resumable teia run (missing config/overrides.yaml): {resolved}. "
        "Check the path — run-dir parts are slugified (lowercased) when written."
    )


def compose_project_config(
    project_dir: Path,
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    composed, _ = compose_overlay_config(project_dir=project_dir, config_dir=config_dir, overrides=overrides or [])
    return composed


def compose_run_config(run_dir: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    composed, _, _, _, _ = compose_run_config_tree(run_dir=run_dir, overrides=overrides or [])
    return composed


def compose_run_config_tree(
    run_dir: Path,
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str], Path]:
    resolved_run_dir = resolve_resumable_run_dir(run_dir)
    snapshot_root = resolved_run_dir / "config" / "conf"
    saved_overrides = load_saved_overrides(resolved_run_dir)
    composed, hydra_payload = compose_overlay_config(
        project_dir=resolved_run_dir / "config",
        config_dir=snapshot_root,
        overrides=[*saved_overrides, *(overrides or [])],
    )
    saved = load_saved_composed_config(resolved_run_dir)
    project_dir = saved.get("__project_dir__")
    if project_dir:
        composed["__project_dir__"] = str(project_dir)
    composed["__run_dir__"] = str(resolved_run_dir)
    return composed, hydra_payload, saved, saved_overrides, snapshot_root


def compose_overlay_config(
    project_dir: Path,
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = resolve_conf_layout(Path(project_dir), config_dir)
    overrides = list(overrides or [])

    _raise_if_stray_plus_override(overrides)
    _raise_if_data_path_override(overrides)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    compose_overrides = list(overrides)
    if layout.searchpath:
        searchpath = "[" + ",".join(layer.uri for layer in layout.searchpath) + "]"
        compose_overrides = [f"hydra.searchpath={searchpath}", *compose_overrides]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(layout.primary.path), version_base="1.3"):
            config = compose(config_name="config", overrides=compose_overrides)
            hydra_config = compose(config_name="config", overrides=compose_overrides, return_hydra_config=True)
    finally:
        GlobalHydra.instance().clear()

    composed = OmegaConf.to_container(config, resolve=True)
    if not isinstance(composed, dict):
        raise TypeError("Hydra compose did not produce a mapping config.")

    _warn_unknown_overrides(overrides, composed)

    hydra_payload: dict[str, Any] = {}
    hydra_node = hydra_config.get("hydra") if hasattr(hydra_config, "get") else None
    if hydra_node is not None:
        resolved_hydra = OmegaConf.to_container(hydra_node, resolve=False)
        if isinstance(resolved_hydra, dict):
            hydra_payload = resolved_hydra
    return composed, hydra_payload


def load_saved_composed_config(run_dir: Path) -> dict[str, Any]:
    composed_path = Path(run_dir) / "config" / "composed.yaml"
    if not composed_path.exists():
        raise FileNotFoundError(f"Saved composed config not found: {composed_path}")
    return read_yaml(composed_path)


def load_saved_overrides(run_dir: Path) -> list[str]:
    overrides_path = Path(run_dir) / "config" / "overrides.yaml"
    if not overrides_path.exists():
        return []
    payload = read_yaml(overrides_path)
    overrides = payload.get("overrides") if isinstance(payload, dict) else []
    if not isinstance(overrides, list):
        return []
    return [str(item) for item in overrides]


def _warn_unknown_overrides(overrides: list[str], composed: dict[str, Any]) -> None:
    """Warn on dot-path overrides absent from the composed config (a likely typo).

    Skips Hydra config-group/package selectors (`group/subgroup@alias=value`) and a bare `task=<name>`
    selection, a `# @package _global_` preset that merges into the root rather than appearing at its
    own group key.
    """
    import warnings

    for raw in overrides:
        stripped = raw.lstrip("+~")
        if "=" not in stripped:
            continue
        try:
            key, _ = parse_override(stripped)
        except ValueError:
            continue
        if "/" in key or "@" in key or key == "task":
            continue
        try:
            get_dot_path(composed, key)
        except KeyError:
            warnings.warn(
                f"Override key {key!r} not found in composed config — possible typo.",
                stacklevel=4,
            )


_MODULE_GROUPS = ("datamodule", "netmodule", "evalmodule", "task")


def _raise_if_data_path_override(overrides: list[str]) -> None:
    """Reject ``data=``/``data_dir=`` and path-valued module selections before Hydra sees them.

    Users type ``data=coco.yaml``; Hydra's own error would suggest ``+data=``,
    which silently does the wrong thing. The dataset location is ``data_root=`` only
    (teia:core/config.md#data_root)."""
    for override in overrides:
        key, _, value = override.lstrip("+~").partition("=")
        is_path = value.endswith((".yaml", ".yml")) or value.startswith(("/", "./", "~"))
        if key in {"data", "data_dir"} or (key in _MODULE_GROUPS and is_path):
            raise ValueError(f"Override {override!r}: the dataset location is set with `data_root=<path>`.")


def _raise_if_stray_plus_override(overrides: list[str]) -> None:
    """Reject bare ``+key=value`` overrides.

    Hydra's ``+`` bypasses any schema unconditionally with no native strictness escape hatch
   , so teia enforces it here instead: every real,
    component, or bundle slot is already reachable through ``optional`` defaults entries, so a
    stray ``+`` almost always means a typo'd group path rather than an intentional new key.
    ``++`` (force-override, not add) is unaffected.
    """
    for override in overrides:
        if override.startswith("++"):
            continue
        if override.startswith("+"):
            raise ValueError(
                f"Override {override!r} uses the `+` (add-key) prefix, which is not supported — "
                "every real/component/bundle slot is already declared as an `optional` default. "
                "Drop the leading `+`, or use `++` to force-override an existing key."
            )
