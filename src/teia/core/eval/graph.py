"""Eval-graph envelope parsing and topological sort.

The eval-plane counterpart of ``teia.core.datamodule.graph`` (data graph) and ``teia.core.module.net``
(net graph). An ``eval.graph`` config is a flat map of named eval-node envelopes wired by ``in``/``out``
keys, exactly like the data graph's; this module turns that map into a validated, topologically
ordered list of ``EvalNodeRecord``.

The data graph's ``producer_map`` and ``_toposort`` are reused directly. Both touch only
``.name``/``.in_key``/``.out_key``, so an ``EvalNodeRecord`` satisfies them structurally with no
inheritance from ``IoNodeRecord``. Dependency resolution differs in one respect: the data graph
exempts ``ctx.``-prefixed keys from producer resolution (they are bound externally via
``TeiaDataModule.bind_context``); the eval graph's externally-bound prefixes are ``capture.``,
``batch.`` and ``meta.``, read from the ``EvalSource`` and never produced by a graph node, so
``_resolve_eval_deps`` is written here rather than imported. The data graph's ``phase``/``stage``
and collate-field validation has no eval analog and is not reused.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from teia.base.envelope import check_envelope_collision, component_config
from teia.base.eval import Comparison, Metric, View
from teia.core.datamodule.graph import _toposort, producer_map

EvalKind = Literal["metric", "view", "comparison"]

#: Kind inferred from the ``_target_`` class's ABC, mirroring the data graph's ``_KIND_BY_BASE``.
_KIND_BY_BASE: list[tuple[type, EvalKind]] = [
    (Metric, "metric"),
    (View, "view"),
    (Comparison, "comparison"),
]

EVAL_RESERVED_KEYS = frozenset({"_target_", "_recursive_"})

#: Eval-only envelope keys, not part of the shared data/net ``ENVELOPE_KEYS`` vocabulary. Stripped
#: from the instantiate kwargs here rather than added to that shared constant.
_EVAL_META_KEYS = frozenset({"monitor", "split"})


@dataclass
class EvalNodeRecord:
    """A parsed eval-node envelope: ``{_target_, in, out, **params}``.

    Structurally compatible with ``teia.core.datamodule.graph``'s ``producer_map``/
    ``_resolve_deps``/``_toposort``, which only read ``.name``/``.in_key``/``.out_key``.

    ``in_key`` is always the flat list of source keys a node depends on, which is what producer
    and toposort resolution need. ``in_fields`` is set only when the node's ``in`` was written in
    mapping form (see ``_parse_in``); it carries the per-kwarg field-path info that ``in_key``
    alone cannot, and ``_resolve_inputs`` reads it to build each node's call kwargs.
    """

    name: str
    kind: EvalKind
    component: dict[str, Any]
    in_key: list[str]
    out_key: list[str]
    in_fields: dict[str, tuple[str, str | None]] | None = None
    #: A Metric node with ``monitor: true`` is also driven by the streaming path
    #: (``core/eval/streaming.py``) to publish a ``val/*`` monitor key during fit; same
    #: ``in``/``out``/``_target_``, one extra flag. Not an ``ENVELOPE_KEYS`` member (that
    #: vocabulary is shared with data/net, which have no monitor concept), so it is stripped from
    #: the component's instantiate kwargs below.
    monitor: bool = False
    #: Capture split this node reads its ``in`` keys from; ``None`` means the evalmodule's split.
    split: str | None = None


def _normalize_keys(raw: Any, *, name: str, which: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Mapping):
        return [str(k) for k in raw.keys()]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    raise ValueError(f"eval node '{name}': invalid {which}: expected string, list, or mapping, got {raw!r}.")


def _is_node_entry(name: str, value: Any) -> bool:
    if name in EVAL_RESERVED_KEYS or name.startswith("__"):
        return False
    return isinstance(value, Mapping) and "_target_" in value


def _resolve_component_class(target: str, *, name: str) -> type:
    module_path, _, attr = str(target).rpartition(".")
    if not module_path:
        raise ValueError(f"eval node '{name}': invalid `_target_` {target!r}.")
    try:
        cls = getattr(importlib.import_module(module_path), attr)
    except Exception as exc:
        raise ValueError(f"eval node '{name}': cannot resolve `_target_` {target!r}: {exc}") from exc
    if not isinstance(cls, type):
        raise ValueError(f"eval node '{name}': `_target_` {target!r} does not resolve to a class.")
    return cls


def resolve_component_class(record: EvalNodeRecord) -> type:
    """The class a record's ``_target_`` names."""
    return _resolve_component_class(str(record.component["_target_"]), name=record.name)


def _infer_kind(cls: type, *, name: str) -> EvalKind:
    for base, kind in _KIND_BY_BASE:
        if issubclass(cls, base):
            return kind
    raise ValueError(
        f"eval node '{name}': {cls.__module__}.{cls.__qualname__} is not a Metric/View/Comparison subclass."
    )


def _parse_in(raw: Any, *, name: str) -> tuple[list[str], dict[str, tuple[str, str | None]] | None]:
    """``in`` accepts two forms.

    List form (``[capture.cls.scores, batch.cls]``) is the common case: each kwarg is named by
    the key's trailing dot-segment (see ``runner._atom_name``), and the whole value at that key
    is passed through unchanged.

    Mapping form (``{kwarg_name: "source_key"}`` or ``{kwarg_name: "source_key:field.path"}``)
    is used when a consumer wants a named sub-field of a producer's result. A metric publishes
    one result dict per ``out_key``; a view's ``render(**inputs)`` often wants specific arrays
    out of it (``labels``, ``values``) rather than the whole dict under one name. ``source_key``
    resolves as in the list form; the optional ``:field.path`` then walks dot-separated lookups
    into that value. The colon keeps "which graph key" and "which field inside it" explicit
    rather than guessing a prefix split.

    Returns ``(in_key, in_fields)``. ``in_key`` is always the flat list of source keys that
    dependency and toposort resolution need (mapping form: deduplicated, field-path stripped).
    ``in_fields`` is ``None`` for list form, the parsed per-kwarg mapping otherwise.
    """
    if raw is None:
        return [], None
    if isinstance(raw, str):
        return [raw], None
    if isinstance(raw, Mapping):
        in_fields: dict[str, tuple[str, str | None]] = {}
        for kwarg_name, spec in raw.items():
            spec_str = str(spec)
            source_key, _, field_path = spec_str.partition(":")
            in_fields[str(kwarg_name)] = (source_key, field_path or None)
        in_key = sorted({source_key for source_key, _ in in_fields.values()})
        return in_key, in_fields
    if isinstance(raw, Sequence):
        return [str(item) for item in raw], None
    raise ValueError(f"eval node '{name}': invalid in: expected string, list, or mapping, got {raw!r}.")


def parse_eval_node(name: str, entry: Mapping[str, Any]) -> EvalNodeRecord:
    """Parse one flat envelope ``{_target_, in, out, **params}``. ``kind`` is inferred from the
    ``_target_`` class's base, never declared."""
    target = entry.get("_target_")
    if not target:
        raise ValueError(f"eval node '{name}' is missing `_target_`.")
    cls = _resolve_component_class(str(target), name=name)
    kind = _infer_kind(cls, name=name)
    check_envelope_collision(cls, entry, name=name)
    component = {key: value for key, value in component_config(entry).items() if key not in _EVAL_META_KEYS}

    out_key = _normalize_keys(entry.get("out"), name=name, which="out")
    if kind == "view" and out_key:
        raise ValueError(f"eval node '{name}': a View is terminal and must declare no `out` key.")

    monitor = bool(entry.get("monitor", False))
    if monitor and kind != "metric":
        raise ValueError(f"eval node '{name}': `monitor: true` is only valid on a Metric node.")

    in_key, in_fields = _parse_in(entry.get("in"), name=name)
    return EvalNodeRecord(
        name=str(name).replace("/", "_").replace("-", "_"),
        kind=kind,
        component=component,
        in_key=in_key,
        out_key=out_key,
        in_fields=in_fields,
        monitor=monitor,
        split=None if entry.get("split") is None else str(entry["split"]),
    )


def _resolve_eval_deps(records: list[EvalNodeRecord]) -> dict[str, set[str]]:
    """Per-node dependency sets, built from ``producer_map``.

    ``capture.*``/``batch.*``/``meta.*``/``log.*`` keys are exempt from producer resolution: they are read
    from the ``EvalSource``, not produced by any graph node. This mirrors the data graph's
    ``ctx.*`` exemption in ``teia.core.datamodule.graph._resolve_deps``.
    """
    producers = producer_map(records)
    deps: dict[str, set[str]] = {rec.name: set() for rec in records}
    for rec in records:
        for key in rec.in_key:
            if key.startswith(("capture.", "batch.", "meta.", "log.")):
                continue
            if key not in producers.producers_by_key:
                raise ValueError(f"eval node '{rec.name}': in_key '{key}' is not produced by any node's out_key.")
            deps[rec.name].add(producers.producer_of(rec.name, key))
    return deps


def _validate_eval_contracts(records: list[EvalNodeRecord]) -> None:
    """Eval-specific build-time checks.

    Cycles, missing producers and duplicate ``out`` keys are already rejected by
    ``producer_map``/``_toposort``. Two more checks are promoted here from "caught lazily by the
    streaming runner mid-training" to build time:

    1. **A metric that cannot stream cannot be a val metric.** Absence of ``update`` is the
       declaration (``base/eval/nodes.md`` §3/§5); a ``monitor: true`` node whose class never
       overrides it would otherwise only fail once ``StreamingMonitorRunner`` first calls it.
    2. **Monitor names MUST be unique per split** (``core/eval.md`` §5). Replays
       ``streaming.monitor_closure``/``_flat_monitor_name``'s naming logic against a skeleton
       instantiation of every monitor-reachable node (``MONITORS`` is often set in ``__init__``,
       e.g. ``Confusion``, so a class-level check cannot see it) — cheap, since every current
       ``monitor: true`` node is a lightweight aligned metric.
    """
    for record in records:
        if not record.monitor:
            continue
        cls = _resolve_component_class(str(record.component.get("_target_")), name=record.name)
        if cls.update is Metric.update:
            raise ValueError(
                f"eval node '{record.name}': `monitor: true` requires a streaming driver "
                f"(override `update`/`compute`), but {cls.__module__}.{cls.__qualname__} has none."
            )

    from teia.core.eval.streaming import _flat_monitor_name, monitor_closure
    from teia.core.instantiate import instantiate

    owner: dict[str, str] = {}
    for record in monitor_closure(records):
        if not record.monitor:
            continue
        component = instantiate(record.component)
        for key in getattr(component, "MONITORS", ()):
            flat_name = _flat_monitor_name(record.name, key)
            if flat_name in owner and owner[flat_name] != record.name:
                raise ValueError(
                    f"duplicate val-metric monitor name 'val/{flat_name}': both node "
                    f"'{owner[flat_name]}' and '{record.name}' publish it. Rename one node, "
                    "or (for 'fitness') set is_fitness=False on all but the route that should own it."
                )
            owner[flat_name] = record.name


def build_eval_graph(node_entries: list[tuple[str, Mapping[str, Any]]]) -> list[EvalNodeRecord]:
    """Parse -> toposort (dup/missing/cycle, via the shared data-graph resolver) -> validate."""
    records = [parse_eval_node(name, entry) for name, entry in node_entries]
    deps = _resolve_eval_deps(records)
    ordered = _toposort(records, deps)
    _validate_eval_contracts(ordered)
    return ordered


def discover_node_entries(graph: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Pick the eval-node envelopes out of an ``eval.graph`` mapping."""
    return [(name, dict(value)) for name, value in graph.items() if _is_node_entry(name, value)]


__all__ = [
    "EvalKind",
    "EvalNodeRecord",
    "parse_eval_node",
    "resolve_component_class",
    "build_eval_graph",
    "discover_node_entries",
    "producer_map",
]
