"""Streaming eval-graph execution: the val-metric driver, counterpart to ``run_eval_graph``.

A val metric is an ordinary metric with a streaming driver, wired like any other node. A node
opts in by adding ``monitor: true`` to its envelope; its ``_target_``, ``in`` and ``out`` are
identical to the offline/report wiring that ``run_eval_graph`` drives, so one bundle config
feeds both drivers.

Two drivers, two timings, one wiring
------------------------------------
``run_eval_graph`` calls each node's ``from_source`` once, against a finished capture store.
This module calls each monitor-reachable Metric node's ``update`` once per validation batch,
against a live per-batch frame the caller builds (no capture store exists during fit; capture
happens later, at test/report time), then ``compute`` once at epoch end.

A node whose ``in`` references another node's ``eval.*`` output (det-family's
``RankedAveragePrecision`` consuming ``InstanceMatch``'s ``matched`` rows, for example) cannot
be driven per batch, because ``update`` returns nothing and there is no per-batch value to hand
it. Such a node is skipped during ``update_batch``; at ``compute`` time it is fed its
upstream's finished result through one synthetic ``update`` call (reusing ``runner``'s
``_resolve_inputs``/``_assign_result`` so resolution semantics are defined once) immediately
before its own ``compute``. This assumes a node with an ``eval.*`` dependency is itself cheap
at compute time, which holds for every current bundle: at most one metric feeds another before
a view.

``capture.*``/``batch.*`` resolution during streaming
----------------------------------------------------
:class:`LiveFrameSource` stands in for the offline driver's ``RunEvalSource``. ``column(key)``
reads a plain dict the caller rebuilds every batch, holding this batch's raw values in whatever
shape the target metric's ``update`` expects (usually parallel arrays). ``meta(key)`` reads a
constant dict set once at construction (``class_names`` and the like do not change batch to
batch).

The ``_source`` kwarg
---------------------
A metric that reads columns directly inside ``from_source`` (``InstanceMatch``,
``SemanticConfusion`` at report time) cannot be fed through ``_resolve_inputs``'s per-key kwarg
resolution. The streaming driver therefore passes every ``update`` call an extra
``_source=<LiveFrameSource>`` kwarg alongside the resolved ones. A metric that wants direct
column access takes it (``InstanceMatch.update`` does); simple buffered-accumulator metrics
ignore it via ``**_``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from teia.core.eval.graph import EvalNodeRecord, build_eval_graph, discover_node_entries, producer_map
from teia.core.eval.runner import _assign_result, _resolve_inputs
from teia.core.instantiate import instantiate

__all__ = ["LiveFrameSource", "StreamingMonitorRunner", "build_streaming_runner", "monitor_closure"]


class LiveFrameSource:
    """Per-batch stand-in for an ``EvalSource``. See module docstring."""

    def __init__(self, frame: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> None:
        self._frame = frame
        self._meta = meta or {}

    def column(self, key: str) -> Any:
        return self._frame.get(key)

    def meta(self, key: str, default: Any = None) -> Any:
        return self._meta.get(key, default)


def _has_eval_dep(record: EvalNodeRecord) -> bool:
    return any(key.startswith("eval.") for key in record.in_key)


def monitor_closure(records: list[EvalNodeRecord]) -> list[EvalNodeRecord]:
    """Metric nodes reachable (via producer deps) from any ``monitor: true`` node, preserving
    ``records``'s topo order. Views/Comparisons are never monitor-reachable (only Metric nodes
    declare ``out``, so nothing else can be a producer)."""
    metric_records = [record for record in records if record.kind == "metric"]
    by_name = {record.name: record for record in metric_records}
    producers = producer_map(metric_records)
    monitored = [record.name for record in metric_records if record.monitor]
    if not monitored:
        return []

    needed: set[str] = set()

    def visit(name: str) -> None:
        if name in needed:
            return
        needed.add(name)
        record = by_name[name]
        for key in record.in_key:
            if key.startswith("capture.") or key.startswith("batch.") or key.startswith("meta."):
                continue
            visit(producers.producer_of(record.name, key))

    for name in monitored:
        visit(name)
    return [record for record in metric_records if record.name in needed]


def _flat_monitor_name(record_name: str, key: str) -> str:
    """The flat monitor key a node publishes, before the ``val/`` prefix.

    ``fitness`` is always published bare (``val/fitness``): the vision callback configs
    (``vision_fitness.yaml``, ``vision_detection.yaml``) hardcode ``monitor: val/fitness``, and
    that wiring is preserved. Every other name is namespaced by its owning node's name unless
    the node is already named exactly that quantity (the common single-purpose case, a node
    literally named ``macro_f1``). Self-namespacing is applied uniformly, not only on a detected
    collision, so a graph always logs under the same key regardless of what else is wired
    alongside it.
    """
    if key == "fitness":
        return "fitness"
    return key if record_name == key else f"{record_name}.{key}"


class StreamingMonitorRunner:
    """Drives the monitor-reachable subgraph of an ``eval.graph`` across a validation epoch's
    batches. One instance per epoch: construct via :func:`build_streaming_runner`, call
    :meth:`update_batch` once per batch, :meth:`compute` once at epoch end."""

    def __init__(self, records: list[EvalNodeRecord], *, meta: Mapping[str, Any] | None = None) -> None:
        self._records = monitor_closure(records)
        self._monitor_names = {record.name for record in self._records if record.monitor}
        self._meta = dict(meta or {})
        self._components: dict[str, Any] = {}
        self._results: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return bool(self._records)

    def reset(self) -> None:
        self._components = {}
        self._results = {}
        for record in self._records:
            component = instantiate(record.component)
            component.in_key = list(record.in_key)
            component.out_key = list(record.out_key)
            component.reset()
            self._components[record.name] = component

    def update_batch(self, live_frame: Mapping[str, Any]) -> None:
        source = LiveFrameSource(live_frame, self._meta)
        for record in self._records:
            if _has_eval_dep(record):
                continue  # fed at compute() time instead; see module docstring
            component = self._components[record.name]
            inputs = _resolve_inputs(record, source=source, results={})
            inputs["_source"] = source
            component.update(**inputs)

    def compute(self) -> dict[str, float]:
        source = LiveFrameSource({}, self._meta)
        flat: dict[str, float] = {}
        owner: dict[str, str] = {}
        for record in self._records:
            component = self._components[record.name]
            if _has_eval_dep(record):
                inputs = _resolve_inputs(record, source=source, results=self._results)
                inputs["_source"] = source
                component.update(**inputs)
            outputs = component.compute()
            _assign_result(self._results, record.out_key, outputs)
            if record.name not in self._monitor_names:
                continue
            for key in getattr(component, "MONITORS", ()):
                if key not in outputs:
                    continue
                flat_name = _flat_monitor_name(record.name, key)
                if flat_name in owner and owner[flat_name] != record.name:
                    raise ValueError(
                        f"duplicate val-metric monitor name 'val/{flat_name}': both node "
                        f"'{owner[flat_name]}' and '{record.name}' publish it. Rename one node, "
                        "or (for 'fitness') set is_fitness=False on all but the route that should own it."
                    )
                owner[flat_name] = record.name
                flat[flat_name] = float(outputs[key])
        return flat


def build_streaming_runner(graph: Mapping[str, Any], *, meta: Mapping[str, Any] | None = None) -> StreamingMonitorRunner | None:
    """Build the streaming runner for a composed ``eval.graph`` config, or ``None`` if it wires
    no ``monitor: true`` node (nothing to drive during fit-validation)."""
    records = build_eval_graph(discover_node_entries(graph))
    runner = StreamingMonitorRunner(records, meta=meta)
    if not runner.active:
        return None
    runner.reset()
    return runner
