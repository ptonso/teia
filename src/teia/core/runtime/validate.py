from __future__ import annotations

from typing import Any

from teia.core.config.schema import TOP_LEVEL_GROUPS
from teia.core.deps import check_missing
from teia.core.eval.graph import _resolve_component_class, build_eval_graph, discover_node_entries
from teia.core.eval.runner import split_evalmodule
from teia.core.instantiate import instantiate
from teia.core.module.net import _discover_alias_entries, _parse_node_record
from teia.core.runtime.determinism import determinism_mode


def validate_composed_config(config: dict[str, Any]) -> None:
    _reject_legacy_data_contract(config)
    missing = [group for group in TOP_LEVEL_GROUPS if group not in config]
    if missing:
        raise ValueError(f"Composed config is missing required top-level groups: {', '.join(missing)}")

    # Streaming datamodules (interactive collects transitions from the environment) carry no dataset
    # and declare `requires_data_root = False` on their class; everything else needs a root.
    if _datamodule_requires_data_root(config) and config.get("data_root") in (None, "???"):
        raise ValueError("Config data_root is required; configure dataset location at the config root.")

    _validate_datamodule_contract(config)
    _validate_determinism_contract(config)
    _validate_task_contract(config)
    _validate_eval_deps(config)
    _validate_eval_wiring(config)
    _validate_eval_callbacks(config)


def _datamodule_requires_data_root(config: dict[str, Any]) -> bool:
    """Whether the datamodule needs a dataset root.

    A live env stream source (``datamodule.env``, the interactive regime) collects transitions from
    the environment and carries no dataset; the datamodule class' ``requires_data_root`` (default
    True) governs every other (offline) datamodule.
    """
    return (config.get("datamodule") or {}).get("env") is None


def _reject_legacy_data_contract(config: dict[str, Any]) -> None:
    if isinstance(config.get("data"), dict):
        raise ValueError("Legacy top-level config group 'data' is no longer supported; use 'datamodule'.")


def _validate_datamodule_contract(config: dict[str, Any]) -> None:
    datamodule_cfg = config.get("datamodule") or {}
    if not isinstance(datamodule_cfg, dict):
        return

    if "root" in datamodule_cfg:
        raise ValueError(
            "Config datamodule.root is not supported; configure dataset location with top-level data_root."
        )
    if "data_root" in datamodule_cfg:
        raise ValueError(
            "Config datamodule.data_root is not supported; configure dataset location with top-level data_root."
        )


def _validate_task_contract(config: dict[str, Any]) -> None:
    """Pre-flight contract gate (teia:base/task.md): the netmodule reads only boundary-A keys and maps
    every capture atom, and every evalmodule node reads only boundary-B keys. ``check_data`` runs
    later, once the datamodule has planned."""
    if not config.get("task"):
        return
    contract = instantiate(config["task"])
    records = [_parse_node_record(name, entry) for name, entry in _discover_alias_entries(dict(config["netmodule"]))]
    in_keys = {key for rec in records for key in rec.in_key}
    tokens = {token for rec in records if not rec.is_loss for shape in (rec.out_shape or []) for token in _dim_ref_tokens(shape)}
    in_keys |= {f"meta.{token}" for token in tokens if token not in contract.batch}  # a token may also name a batch field
    contract.check_net(in_keys, dict(config["netmodule"].get("capture") or {}))
    _, graph = split_evalmodule(config.get("evalmodule"))
    contract.check_eval({key for record in build_eval_graph(discover_node_entries(graph)) for key in record.in_key})


def _dim_ref_tokens(shape: Any) -> list[str]:
    """String tokens of an ``out`` shape (``[num_classes]`` → ``["num_classes"]``)."""
    if isinstance(shape, str):
        return [shape]
    return [str(dim) for dim in (shape or []) if isinstance(dim, str)]


def _validate_eval_deps(config: dict[str, Any]) -> None:
    """Pre-flight gate: fail on a missing eval-component dependency here, before `trainer.fit`,
    rather than inside `run_report` after training has already completed."""
    _, graph = split_evalmodule(config.get("evalmodule"))
    if not graph:
        return
    check_missing({"evalmodule": graph}, feature="Eval graph")


def _validate_eval_wiring(config: dict[str, Any]) -> None:
    """Pre-flight gate: every eval node's `in` keys must resolve to a producer. Reuses
    `build_eval_graph`'s own cycle/missing-producer/duplicate-`out` checks rather than a second
    wiring validator. Runs after the dependency gate, since an uninstalled component's `_target_`
    cannot be resolved to infer its Metric/View/Comparison kind."""
    _, graph = split_evalmodule(config.get("evalmodule"))
    if not graph:
        return
    try:
        build_eval_graph(discover_node_entries(graph))
    except ValueError as exc:
        raise ValueError(f"Composed evalmodule is invalid: {exc}") from exc


def _validate_eval_callbacks(config: dict[str, Any]) -> None:
    """Pre-flight gate: a metric declaring ``REQUIRES_CALLBACK`` must find every named callback's
    ``_target_`` present in ``callbacks.items``, or its output silently degrades (e.g. an empty
    learning-rate-schedule plot) with no error anywhere. Runs after ``_validate_eval_wiring``, so
    the graph is already known to build cleanly by the time this re-builds it to inspect node
    classes."""
    _, graph = split_evalmodule(config.get("evalmodule"))
    if not graph:
        return
    records = build_eval_graph(discover_node_entries(graph))
    items = ((config.get("callbacks") or {}).get("items")) or []
    present = {str(item.get("_target_")) for item in items if isinstance(item, dict) and item.get("_target_")}

    for record in records:
        if record.kind != "metric":
            continue
        target = record.component.get("_target_")
        if not target:
            continue
        cls = _resolve_component_class(str(target), name=record.name)
        missing = [callback for callback in getattr(cls, "REQUIRES_CALLBACK", ()) if callback not in present]
        if missing:
            raise ValueError(
                f"eval node '{record.name}' ({target}) requires callback(s) {missing} in "
                "`callbacks.items`, but none is present. Add it, or this node's output will "
                "silently be empty."
            )


def _validate_determinism_contract(config: dict[str, Any]) -> None:
    trainer_cfg = config.get("trainer") or {}
    if "deterministic" in trainer_cfg:
        raise ValueError(
            "Config trainer.deterministic is not supported; use runtime.determinism.mode=force|prefer|stochastic."
        )
    mode = determinism_mode(config)
    if bool(trainer_cfg.get("benchmark")) and mode in {"force", "prefer"}:
        raise ValueError(f"Config trainer.benchmark=true conflicts with runtime.determinism.mode={mode}.")
