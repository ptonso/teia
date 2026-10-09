"""Eval-graph execution: build, toposort, run once over a finished run's capture store.

The eval-plane counterpart of the data/net graph executors, but simpler. The graph runs once per
pass (offline ``test`` / ``teia eval``), not per batch, so this is a plain interpreted loop
rather than the net graph's generated-forward-function approach (which exists to amortize a
per-batch codegen cost the eval graph never pays). Intermediate ``eval.*`` results live in
memory for the pass only; re-running a view re-runs its upstream metric.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from teia.base.eval import EvalSource
from teia.core.eval.graph import EvalNodeRecord, build_eval_graph, discover_node_entries, resolve_component_class
from teia.core.instantiate import instantiate
from teia.core.capture.store import CaptureStore, store_dir
from teia.core.utils import ensure_dir, teia_log, read_yaml


@dataclass(frozen=True)
class EvalSettings:
    """The evalmodule's non-node keys (teia:core/eval.md#evalmodule-settings); unknown keys fail fast."""

    enabled: bool = True
    split: str = "test"
    max_rows: int | None = None
    ctx: dict[str, Any] = field(default_factory=dict)
    media: dict[str, Any] = field(default_factory=dict)
    eval_episodes: int | None = None


def split_evalmodule(evalmodule: Mapping[str, Any] | None) -> tuple[EvalSettings, dict[str, Any]]:
    """``(settings, graph)`` from a composed ``evalmodule``: nodes are the mappings carrying ``_target_``."""
    evalmodule = dict(evalmodule or {})
    graph = dict(discover_node_entries(evalmodule))
    settings = {key: value for key, value in evalmodule.items() if key not in graph}
    try:
        return EvalSettings(**settings), graph
    except TypeError as exc:
        raise ValueError(f"evalmodule has unknown settings {sorted(set(settings) - set(EvalSettings.__dataclass_fields__))}.") from exc


class RunEvalSource(EvalSource):
    """Reads one finished run directory: the route-keyed capture store, the composed config
    snapshot, and the per-epoch logger CSVs. Never touches a live module/datamodule/trainer."""

    def __init__(self, run_dir: str | Path, *, split: str = "test") -> None:
        self.run_dir = Path(run_dir)
        self.split = str(split)
        self._store: CaptureStore | None = None

    @property
    def _capture(self) -> CaptureStore:
        if self._store is None:
            self._store = CaptureStore(store_dir(self.run_dir, self.split))
        return self._store

    def routes(self) -> list[str]:
        return self._capture.routes()

    def has(self, key: str) -> bool:
        return self._capture.has(key)

    def column(self, key: str) -> Any:
        return self._capture.column(key)

    def config(self) -> Mapping[str, Any]:
        path = self.run_dir / "config" / "composed.yaml"
        return read_yaml(path) if path.exists() else {}

    def curves(self) -> Any:
        """``{logger_name: [{column: value}, ...]}``, all per-epoch rows from each logger's CSV,
        across every stage.

        A Lightning ``CSVLogger`` nests two levels under ``logs/``:
        ``logs/<logger_name>/version_<n>/metrics.csv``, where ``version_<n>`` auto-increments
        once per Trainer construction within the run directory — a ``fit`` followed by a
        standalone ``test`` leaves both ``version_0`` (train/val/lr history) and ``version_1``
        (test-only columns) under the same logger name. Both are stages of one run, not
        competing attempts, so rows from every version are concatenated in version order rather
        than keeping only the last.
        """
        by_logger: dict[str, list[tuple[int, Path]]] = {}
        for csv_path in self.run_dir.glob("logs/*/version_*/metrics.csv"):
            logger_name = csv_path.parent.parent.name
            by_logger.setdefault(logger_name, []).append((_version_number(csv_path.parent.name), csv_path))

        curves: dict[str, list[dict[str, str]]] = {}
        for logger_name, versions in sorted(by_logger.items()):
            rows: list[dict[str, str]] = []
            for _, csv_path in sorted(versions):
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows.extend(csv.DictReader(handle))
            curves[logger_name] = rows
        return curves

    def log(self, column: str) -> list[float]:
        """One per-epoch logger series (``log.<column>``): the column's non-empty values, in epoch order."""
        rows = [row for logger_rows in self.curves().values() for row in logger_rows]
        values = [float(row[column]) for row in rows if row.get(column) not in (None, "")]
        if not values:
            raise KeyError(f"no logger series '{column}' in {self.run_dir / 'logs'}.")
        return values

    def descriptor(self) -> Mapping[str, Any]:
        path = self.run_dir / "artifacts" / "run.yaml"
        return read_yaml(path) if path.exists() else {}

    def overrides(self) -> list[str]:
        path = self.run_dir / "config" / "overrides.yaml"
        if not path.exists():
            return []
        payload = read_yaml(path) or {}
        return [str(item) for item in (payload.get("overrides") or [])]

    def meta(self, key: str, default: Any = None) -> Any:
        """Manifest-level metadata a capture writer set via ``CaptureWriter.set_meta``
        (``class_names``, ``task``, ...). Not part of the ``EvalSource`` ABC, since it is a
        route-keyed capture-store detail rather than something every eval source has, so nodes
        reach it through the ``meta.*`` key prefix (see ``_resolve_source_key``)."""
        return self._capture.meta(key, default)


def _version_number(dirname: str) -> int:
    suffix = dirname.rpartition("_")[2]
    try:
        return int(suffix)
    except ValueError:
        return -1


def _atom_name(key: str) -> str:
    """The kwarg name a node's ``from_source``/``render`` sees for one ``in_key`` entry: the
    key's trailing atom (``capture.det.boxes`` -> ``boxes``, ``batch.cls`` -> ``cls``,
    ``eval.det.matched`` -> ``matched``). The base contract only specifies ``**inputs``; this
    is the concrete naming convention."""
    return key.rpartition(".")[2] or key


def _resolve_source_key(key: str, *, source: EvalSource, results: dict[str, Any]) -> Any:
    if key.startswith("eval."):
        return results[key]
    if key.startswith("meta."):
        return source.meta(key[len("meta.") :])
    if key.startswith("log."):
        return source.log(key[len("log.") :])  # type: ignore[attr-defined]
    return source.column(key)


def _walk_field_path(value: Any, field_path: str | None) -> Any:
    """Dot-separated field extraction, e.g. ``"reliability_diagram.mean_confidence"``.

    A segment ending in ``[]`` (``"class_rows[].f1"``) plucks one field across a list of dicts
    into a flat list. This is the row-to-column pivot that a view's parallel-array kwargs
    (``labels``, ``values``, ``support``, ...) need from a metric's row-oriented
    ``class_rows``/``per_category`` output, so no metric has to publish pre-pivoted duplicate
    arrays alongside its rows.
    """
    if not field_path:
        return value
    segments = field_path.split(".")
    for index, segment in enumerate(segments):
        if segment.endswith("[]"):
            key = segment[:-2]
            items = value[key] if key else value
            remainder = ".".join(segments[index + 1 :])
            return [_walk_field_path(item, remainder) for item in items]
        value = value[int(segment)] if isinstance(value, (list, tuple)) else value[segment]
    return value


def _resolve_inputs(record: EvalNodeRecord, *, source: EvalSource, results: dict[str, Any]) -> dict[str, Any]:
    if record.in_fields is not None:
        return {
            kwarg_name: _walk_field_path(_resolve_source_key(source_key, source=source, results=results), field_path)
            for kwarg_name, (source_key, field_path) in record.in_fields.items()
        }
    resolved: dict[str, Any] = {}
    for key in record.in_key:
        resolved[_atom_name(key)] = _resolve_source_key(key, source=source, results=results)
    return resolved


def _assign_result(results: dict[str, Any], out_key: list[str], outputs: dict[str, Any]) -> None:
    """Publish one metric's return value under its declared ``out_key``(s); the eval-plane
    counterpart of the data graph's ``executor._assign``.

    With one ``out_key`` (the common case, ``out: eval.det.ap``) the metric's whole result dict
    is published under that key, so a downstream node consuming ``eval.det.ap`` receives the
    dict as its ``ap`` kwarg and reads whichever fields it needs. With several ``out_key``s the
    metric's return value is itself the multi-key mapping, and each declared key must be present
    in it.

    Downstream nodes resolve inputs by the literal ``out_key`` string (see ``_resolve_inputs``),
    so the metric's own field names must never leak into ``results`` as top-level keys.
    """
    if not out_key:
        return
    if len(out_key) == 1:
        results[out_key[0]] = outputs
        return
    for key in out_key:
        results[key] = outputs[key]


class EvalPass:
    """The handle a metric's ``after_pass`` receives: read the finished base pass and rerun part of it.

    ``rerun`` executes the named nodes again, in graph order, on a copy of the base results, so
    every other result is reused and the base ``results`` stay untouched. Reruns never call
    ``after_pass``.
    """

    def __init__(self, records: list[EvalNodeRecord], *, results: dict[str, Any], out_dir: Path, source_for: Callable[[str | None], EvalSource]) -> None:
        self.records = tuple(records)
        self.results = results
        self.out_dir = out_dir
        self._source_for = source_for

    def component_class(self, name: str) -> type:
        return resolve_component_class(self._record(name))

    def downstream(self, names: Sequence[str]) -> list[str]:
        """Nodes that transitively consume the ``eval.*`` outputs of ``names``, in graph order."""
        reached = {self._record(name).name for name in names}
        keys = {key for record in self.records if record.name in reached for key in record.out_key}
        found: list[str] = []
        for record in self.records:
            if record.name not in reached and keys.intersection(record.in_key):
                reached.add(record.name)
                keys.update(record.out_key)
                found.append(record.name)
        return found

    def rerun(self, names: Sequence[str], *, out_dir: Path, extra: Mapping[str, Mapping[str, Any]]) -> dict[str, list[Path]]:
        wanted = {self._record(name).name for name in names}
        unknown = sorted(set(extra) - wanted)
        if unknown:
            raise ValueError(f"rerun extra kwargs given for nodes outside the rerun set: {unknown}.")
        subset = [record for record in self.records if record.name in wanted]
        written, _ = _run_records(subset, source_for=self._source_for, results=dict(self.results), out_dir=ensure_dir(Path(out_dir)), extra=extra)
        return written

    def _record(self, name: str) -> EvalNodeRecord:
        normalized = str(name).replace("/", "_").replace("-", "_")
        for record in self.records:
            if record.name == normalized:
                return record
        raise KeyError(f"no eval node named '{name}'.")


def _run_records(
    records: Sequence[EvalNodeRecord],
    *,
    source_for: Callable[[str | None], EvalSource],
    results: dict[str, Any],
    out_dir: Path,
    extra: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    """Instantiate and run each node once, in the given order. Metrics publish into ``results``,
    views write under ``out_dir``. ``extra`` adds per-node kwargs to a metric's ``from_source``."""
    written: dict[str, list[Path]] = {}
    components: dict[str, Any] = {}
    for record in records:
        component = instantiate(record.component)
        component.in_key = list(record.in_key)
        component.out_key = list(record.out_key)
        source = source_for(record.split)
        inputs = _resolve_inputs(record, source=source, results=results)
        if record.kind == "metric":
            outputs = component.from_source(source, **inputs, **extra.get(record.name, {}))
            _assign_result(results, record.out_key, outputs)
        elif record.kind == "view":
            written[record.name] = component.render(out_dir, **inputs)
        else:
            raise ValueError(f"eval node '{record.name}': comparison nodes cannot run in a single-source pass.")
        components[record.name] = component
    return written, components


def run_eval_graph(
    records: list[EvalNodeRecord],
    *,
    source: EvalSource,
    out_dir: Path,
    source_for: Callable[[str], EvalSource] | None = None,
) -> dict[str, list[Path]]:
    """Run each toposorted node once, then give every metric its ``after_pass`` turn. A node with a
    ``split`` reads from ``source_for(split)``, all others from ``source``. Comparison nodes take
    multiple sources and are driven by ``run_compare``, not from here."""

    def pick(split: str | None) -> EvalSource:
        return source if split is None else source_for(split)

    results: dict[str, Any] = {}
    written, components = _run_records(records, source_for=pick, results=results, out_dir=out_dir, extra={})
    handle = EvalPass(records, results=results, out_dir=out_dir, source_for=pick)
    for record in records:
        if record.kind == "metric":
            components[record.name].after_pass(handle)
    return written


def required_splits(evalmodule: Mapping[str, Any] | None) -> set[str]:
    """The capture splits an evalmodule reads: its own ``split`` plus every node's ``split``."""
    settings, graph = split_evalmodule(evalmodule)
    return {settings.split} | {str(entry["split"]) for entry in graph.values() if entry.get("split") is not None}


def run_eval(run_dir: str | Path, *, graph: Mapping[str, Any], split: str = "test", out_dir: Path | None = None) -> dict[str, list[Path]]:
    """Build the eval graph from a composed evalmodule's nodes and run it once, offline,
    against ``run_dir``. No model, datamodule, or trainer is constructed."""
    records = build_eval_graph(discover_node_entries(graph))
    for needed in sorted({split, *(record.split for record in records if record.split is not None)}):
        path = store_dir(Path(run_dir), needed)
        if not path.exists():
            raise FileNotFoundError(f"eval needs the '{needed}' capture at {path}; run `teia test --run-dir {run_dir}` to create it.")
    sources: dict[str, RunEvalSource] = {}

    def source_for(name: str) -> RunEvalSource:
        return sources.setdefault(name, RunEvalSource(run_dir, split=name))

    resolved_out_dir = Path(out_dir) if out_dir is not None else Path(run_dir) / "eval"
    return run_eval_graph(records, source=source_for(split), out_dir=resolved_out_dir, source_for=source_for)


def run_offline_eval(config: dict[str, Any], context: dict[str, Any]) -> list[Path]:
    """The post-train / post-test eval pass: run the composed evalmodule against the run's capture
    store and write every view flat under ``<run_dir>/eval/``."""
    settings, graph = split_evalmodule(config.get("evalmodule"))
    if not settings.enabled or not graph:
        return []
    run_dir = Path(context["run_dir"])
    out_dir = ensure_dir(Path(context.get("eval_dir") or (run_dir / "eval")))
    teia_log("eval start", stage=context.get("stage"), run_dir=str(run_dir), out_dir=str(out_dir))
    written = run_eval(run_dir, graph=graph, split=settings.split, out_dir=out_dir)
    paths = [path for group in written.values() for path in group]
    context["report_outputs"] = [str(path) for path in paths]
    teia_log("eval complete", stage=context.get("stage"), run_dir=str(run_dir), outputs=len(paths))
    return paths


__all__ = ["EvalPass", "EvalSettings", "RunEvalSource", "required_splits", "run_eval_graph", "run_eval", "run_offline_eval", "split_evalmodule"]
