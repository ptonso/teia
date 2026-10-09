"""IO-graph envelope parsing, topological sort, and validation.

The data-plane analog of ``teia.core.module.net`` (parse + toposort) and ``…module.codegen``
(validate). A ``datamodule`` config is a flat map of named data-node envelopes wired by ``in``/``out``
keys; this module turns that map into a validated, topologically ordered list of ``IoNodeRecord``. See
teia:core/data_graph.md and teia:core/datamodule.md.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from teia.base.envelope import check_envelope_collision, component_config
from teia.base.fields import FIELD_VOCAB
from teia.base.data import (
    PHASES,
    STAGE_RANK,
    STAGES,
    Collate,
    Join,
    Kind,
    Phase,
    Reader,
    Reshape,
    Stage,
    Transform,
    Writer,
    default_phase,
    default_stage,
)

#: Kind inferred from the ``_target_`` class's ABC — checked in this order (each concrete data node
#: inherits exactly one of these; ``Reshape`` covers both ``PlanReshape``/``StreamReshape``).
_KIND_BY_BASE: list[tuple[type, Kind]] = [
    (Collate, "collate"),
    (Writer, "write"),
    (Reader, "read"),
    (Join, "join"),
    (Reshape, "reshape"),
    (Transform, "transform"),
]

DATAMODULE_RESERVED_KEYS = frozenset({
    "_target_",
    "_recursive_",
    "batch_size",
    "num_workers",
    "val_num_workers",
    "pin_memory",
    "persistent_workers",
    "drop_last",
    "data_root",
    "project_dir",
    "split",
})


@dataclass
class IoNodeRecord:
    """A parsed data-node envelope.

    ``component`` is the ``{_target_: ..., **params}`` mapping the executor instantiates; ``field`` is set
    only for collate nodes (the atomic ``batch.<field>`` name, derived from ``out_key``).
    """

    name: str
    kind: Kind
    component: dict[str, Any]
    in_key: list[str]
    out_key: list[str]
    phase: Phase
    stage: Stage
    field: str | None = None


def _normalize_keys(raw: Any, *, name: str, which: str) -> list[str]:
    """Coerce an ``in``/``out`` value to a key list. A ``{key: type}`` mapping keeps its keys; the type
    values are structural hints consumed elsewhere, not here."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Mapping):
        return [str(k) for k in raw.keys()]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    raise ValueError(f"data node '{name}': invalid {which}: expected string, list, or mapping, got {raw!r}.")


def _is_node_entry(name: str, value: Any) -> bool:
    if name in DATAMODULE_RESERVED_KEYS or name.startswith("__"):
        return False
    return isinstance(value, Mapping) and "_target_" in value


def _resolve_component_class(target: str, *, name: str) -> type:
    module_path, _, attr = str(target).rpartition(".")
    if not module_path:
        raise ValueError(f"data node '{name}': invalid `_target_` {target!r}.")
    try:
        cls = getattr(importlib.import_module(module_path), attr)
    except Exception as exc:
        raise ValueError(f"data node '{name}': cannot resolve `_target_` {target!r}: {exc}") from exc
    if not isinstance(cls, type):
        raise ValueError(f"data node '{name}': `_target_` {target!r} does not resolve to a class.")
    return cls


def _infer_kind(cls: type, *, name: str) -> Kind:
    for base, kind in _KIND_BY_BASE:
        if issubclass(cls, base):
            return kind
    raise ValueError(
        f"data node '{name}': {cls.__module__}.{cls.__qualname__} is not a "
        "Reader/Transform/Join/Reshape/Collate/Writer subclass."
    )


def parse_node(name: str, entry: Mapping[str, Any]) -> IoNodeRecord:
    """Parse one flat envelope ``{_target_, in, out, phase, stage, **params}``.

    ``kind`` is inferred from the ``_target_`` class's base (a ``Reader`` subclass is a reader, a
    ``Transform`` is a transform, …), never declared. ``in``/``out``/``phase``/``stage`` are graph
    metadata; every other key is merged into the component config as a param. ``phase``/``stage``
    default from the inferred kind when omitted.
    """
    target = entry.get("_target_")
    if not target:
        raise ValueError(f"data node '{name}' is missing `_target_`.")
    cls = _resolve_component_class(str(target), name=name)
    kind: Kind = _infer_kind(cls, name=name)
    check_envelope_collision(cls, entry, name=name)
    component = component_config(entry)

    phase = str(entry.get("phase") or default_phase(kind))
    stage = str(entry.get("stage") or default_stage(kind))
    if phase not in PHASES:
        raise ValueError(f"data node '{name}': unknown phase {phase!r}; expected one of {sorted(PHASES)}.")
    if stage not in STAGES:
        raise ValueError(f"data node '{name}': unknown stage {stage!r}; expected one of {sorted(STAGES)}.")

    out_key = _normalize_keys(entry.get("out"), name=name, which="out")
    field: str | None = None
    if kind == "collate":
        if len(out_key) != 1 or not out_key[0].startswith("batch."):
            raise ValueError(f"data node '{name}': a collate must emit exactly one `batch.<field>` out key.")
        field = out_key[0][len("batch.") :]

    return IoNodeRecord(
        name=str(name).replace("/", "_").replace("-", "_"),
        kind=kind,
        component=component,
        in_key=_normalize_keys(entry.get("in"), name=name, which="in"),
        out_key=out_key,
        phase=phase,
        stage=stage,
        field=field,
    )


@dataclass
class ProducerMap:
    """Per-key producer resolution, including in-place rewrite chains (see ``producer_map``)."""

    producers_by_key: dict[str, list[str]]
    final_producer: dict[str, str]
    rewrite_prev: dict[tuple[str, str], str]

    def producer_of(self, consumer: str, key: str) -> str:
        """The node ``consumer`` binds to for ``key``: its rewrite predecessor if ``consumer``
        itself rewrites ``key``, else the key's final producer."""
        return self.rewrite_prev.get((consumer, key), self.final_producer[key])


def producer_map(records: list[IoNodeRecord]) -> ProducerMap:
    """Resolve each key's producer(s) with in-place rewrite support (free DAG ordering; cycles
    caught by toposort).

    A key normally has one producer. A node that *both consumes and re-produces* a key (``letterbox``
    ``out == in``) is a **rewrite**: it forms a chain rooted at the key's single non-consuming *base*
    producer, ordered by declaration among the rewrites. A non-rewrite consumer binds to the final
    version; each rewrite binds to its predecessor. Two independent (non-rewrite) producers of one key
    stay an error. Missing producers / cycles are still rejected here / by ``_toposort``.
    """
    order_index = {rec.name: idx for idx, rec in enumerate(records)}
    rec_by_name = {rec.name: rec for rec in records}
    producers_by_key: dict[str, list[str]] = {}
    for rec in records:
        for key in rec.out_key:
            producers_by_key.setdefault(key, []).append(rec.name)

    final_producer: dict[str, str] = {}
    rewrite_prev: dict[tuple[str, str], str] = {}
    for key, names in producers_by_key.items():
        bases = [n for n in names if key not in rec_by_name[n].in_key]
        if len(bases) > 1:
            raise ValueError(f"Duplicate out_key '{key}' in data graph.")
        chain = sorted(names, key=lambda n: (key in rec_by_name[n].in_key, order_index[n]))
        prev: str | None = None
        for name in chain:
            if key in rec_by_name[name].in_key:
                if prev is None:
                    raise ValueError(f"data node '{name}': in_key '{key}' is not produced by any node's out_key.")
                rewrite_prev[(name, key)] = prev
            prev = name
        final_producer[key] = chain[-1]

    return ProducerMap(producers_by_key=producers_by_key, final_producer=final_producer, rewrite_prev=rewrite_prev)


def join_input_name(stream_key: str) -> str:
    return stream_key.removeprefix("stream.").replace(".", "_")


def batch_field_names(records: list[IoNodeRecord], *, prefix: bool) -> dict[str, str]:
    """Exposed ``batch.<field>`` name per collate node. Under a multi-input join declaring ``prefix``,
    a collate whose upstream lineage reaches exactly one join input is prefixed ``<input>_<field>``;
    reaching several inputs is a hard error. Otherwise names stay unprefixed."""
    joins = [r for r in records if r.kind == "join"]
    names = {r.name: r.field for r in records if r.kind == "collate"}
    if not prefix or not joins or len(joins[0].in_key) < 2:
        return names
    join = joins[0]
    raw_index = {key: i for i, key in enumerate(join.out_key[: len(join.in_key)])}
    producers = producer_map(records).producers_by_key
    by_name = {r.name: r for r in records}

    def inputs_of(key: str, seen: frozenset[str] = frozenset()) -> set[int]:
        """Join inputs reaching ``key``; ``seen`` stops in-place rewrites (``in == out``) from recursing."""
        if key in raw_index:
            return {raw_index[key]}
        return {
            i
            for n in producers.get(key, [])
            if n != join.name and n not in seen
            for k in by_name[n].in_key
            for i in inputs_of(k, seen | {n})
        }

    for rec in records:
        if rec.kind != "collate":
            continue
        reached = {i for k in rec.in_key for i in inputs_of(k)}
        if len(reached) > 1:
            raise ValueError(f"data node '{rec.name}': collate mixes several join inputs; cannot prefix its field.")
        if reached:
            names[rec.name] = f"{join_input_name(join.in_key[reached.pop()])}_{rec.field}"
    return names


def _resolve_deps(records: list[IoNodeRecord]) -> dict[str, set[str]]:
    """Per-node dependency sets, built from ``producer_map``.

    A ``ctx.*`` key is exempt from producer resolution — it is bound externally via
    ``TeiaDataModule.bind_context``, not produced by any graph node. A writer's ``capture.*`` and
    ``meta.*`` inputs are handed to it at inference, not produced by the data graph.
    """
    producers = producer_map(records)
    deps: dict[str, set[str]] = {rec.name: set() for rec in records}
    for rec in records:
        for key in rec.in_key:
            if key.startswith("ctx.") or (rec.kind == "write" and key.startswith(("capture.", "meta."))):
                continue
            if key not in producers.producers_by_key:
                raise ValueError(f"data node '{rec.name}': in_key '{key}' is not produced by any node's out_key.")
            deps[rec.name].add(producers.producer_of(rec.name, key))
    return deps


def _toposort(records: list[IoNodeRecord], deps: dict[str, set[str]]) -> list[IoNodeRecord]:
    """Kahn sort over declaration-order dependencies; rejects duplicate produced keys, missing producers, cycles."""
    rec_by_name = {rec.name: rec for rec in records}
    order_index = {rec.name: idx for idx, rec in enumerate(records)}

    consumers: dict[str, set[str]] = {rec.name: set() for rec in records}
    for name, producers in deps.items():
        for producer in producers:
            consumers[producer].add(name)
    deps = {name: set(producers) for name, producers in deps.items()}

    ready = sorted((n for n, incoming in deps.items() if not incoming), key=order_index.get)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for consumer in sorted(consumers[current], key=order_index.get):
            deps[consumer].discard(current)
            if not deps[consumer]:
                ready.append(consumer)
        ready.sort(key=order_index.get)

    if len(ordered) != len(records):
        cyclic = sorted((n for n, incoming in deps.items() if incoming), key=order_index.get)
        raise ValueError(f"Cycle detected in data graph involving: {', '.join(cyclic)}")

    return [rec_by_name[name] for name in ordered]


def _validate(records: list[IoNodeRecord], deps: dict[str, set[str]]) -> None:
    """Enforce stage-monotonicity (no later-stage node feeds an earlier one) and the collate seam typing."""
    rec_by_name = {rec.name: rec for rec in records}
    for rec in records:
        for producer_name in deps[rec.name]:
            producer = rec_by_name[producer_name]
            if STAGE_RANK[producer.stage] > STAGE_RANK[rec.stage]:
                raise ValueError(
                    f"Phase violation: '{rec.name}' ({rec.stage}) consumes an output of "
                    f"'{producer.name}' ({producer.stage}); a later-stage node cannot feed an earlier one."
                )
        if rec.kind == "collate" and rec.field not in FIELD_VOCAB:
            raise ValueError(
                f"data node '{rec.name}': collate field '{rec.field}' is not in teia.base.fields.FIELD_VOCAB."
            )


def _component_class(target: str) -> type | None:
    """Resolve a ``_target_`` string to its class without instantiating (dynamic, no static zoo import).

    Reading the data-node contract attributes off the class is a build-time check; a target that fails to
    import is left uncontracted here (instantiation surfaces the real error moments later).
    """
    module_path, _, attr = target.rpartition(".")
    if not module_path:
        return None
    try:
        return getattr(importlib.import_module(module_path), attr, None)
    except Exception:
        return None


def _validate_contracts(records: list[IoNodeRecord]) -> None:
    """Enforce declared data-node data contracts before any IO (see base/data ``IoNode.requires_*``).

    A node that ``requires_schema`` needs a ``plan.schema`` producer (a ``ResolveSchema``); a node that
    ``requires_structure`` needs a collate in the graph emitting that structure — so a geometric collate
    dropped into a schema-only tabular graph is rejected at build instead of crashing mid-stream.
    """
    produces_schema = any("plan.schema" in rec.out_key for rec in records)
    graph_structures = {
        getattr(_component_class(rec.component["_target_"]), "structure", None)
        for rec in records
        if rec.kind == "collate"
    }
    for rec in records:
        cls = _component_class(rec.component["_target_"])
        if cls is None:
            continue
        if getattr(cls, "requires_schema", False) and not produces_schema:
            raise ValueError(
                f"data node '{rec.name}' requires a resolved schema but no node produces 'plan.schema'; "
                f"add a ResolveSchema plan node."
            )
        required = getattr(cls, "requires_structure", None)
        if required is not None and required not in graph_structures:
            raise ValueError(
                f"data node '{rec.name}' requires {required!r}-structured data, but no collate in the graph "
                f"emits structure={required!r} — it cannot apply to this dataset."
            )


def build_graph(node_entries: list[tuple[str, Mapping[str, Any]]]) -> list[IoNodeRecord]:
    """Parse → toposort (dup/missing/cycle) → validate (phase-monotonicity, collate seam, data contracts)."""
    records = [parse_node(name, entry) for name, entry in node_entries]
    deps = _resolve_deps(records)
    ordered = _toposort(records, deps)
    _validate(ordered, deps)
    _validate_contracts(ordered)
    return ordered


def discover_node_entries(kwargs: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Pick the data-node envelopes out of the datamodule kwargs (mirrors module alias discovery)."""
    return [(name, dict(value)) for name, value in kwargs.items() if _is_node_entry(name, value)]
