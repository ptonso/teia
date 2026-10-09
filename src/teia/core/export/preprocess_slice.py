"""Slice the data graph's preprocess subgraph per model-input field for export.

Mirrors decode collection on the activation side (``assemble.py``): walks a model-input
``batch.<field>`` backward through ``preprocess``-phase nodes — crossing the one ``Collate``
seam between item and batch space — to its data-loading root (a ``read``/``join`` node, or a
``target``/``source``-phase transform that supplies an input field, e.g. ``ParseLabelme`` yielding
``item.img_path``), collecting each node's ``kernel``/``params()``. ``augment``-phase rewrites on
the path are transparent: the
walk steps past them to their rewrite predecessor without emitting a step, matching the runtime
executor's train-only skip (an augment rewrite leaves the prior value in place off the train
split). the interactive regime's stream-stage reader is an ordinary ``read`` node in this same graph, so a live
datamodule's obs-normalize chain walks identically to any offline preprocess chain. See
teia:core/export.md and teia:base/data/nodes.md §"Phase enables export/inference slicing".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from teia.core.datamodule.graph import IoNodeRecord, producer_map

ExportKernel = Callable[[Any, Mapping[str, Any]], tuple[Any, dict[str, Any]]]


@dataclass(slots=True)
class PreprocessStep:
    kernel: ExportKernel
    params: dict[str, Any]
    owner: str | None = None
    row_mapped: bool = False
    collate: bool = False


_VALUE_PAYLOAD: dict[str, Any] = {"kind": "value"}


def resolve_data_graph_chain(datamodule: Any, field: str) -> tuple[list[PreprocessStep], str, dict[str, Any]]:
    """Chain of ``PreprocessStep``s (root-to-field order), the payload key, and the payload
    spec (``{"kind": "value"}`` or ``{"kind": "window", **reshape.payload_spec()}``) the
    exported bundle expects for this ``batch.<field>`` input, derived from the chain's
    reader/join/reshape root."""
    records: list[IoNodeRecord] = datamodule._records
    nodes: dict[str, Any] = datamodule._nodes
    records_by_name = {rec.name: rec for rec in records}
    producers = producer_map(records)

    key = f"batch.{field}"
    if key not in producers.producers_by_key:
        raise ValueError(f"No node in this data graph produces {key!r}.")

    steps: list[PreprocessStep] = []
    current = producers.final_producer[key]
    while True:
        rec = records_by_name[current]
        if rec.kind in ("read", "join"):
            payload_key = key.split(".", 1)[1] if "." in key else key
            steps.reverse()
            return steps, payload_key, dict(_VALUE_PAYLOAD)
        if rec.kind == "reshape":
            payload_key = key.split(".", 1)[1] if "." in key else key
            # steps[0] is always the collate seam (the first node appended, walking backward
            # from batch.<field>) — it applies once over the whole row list; every step above
            # it is a pre-seam, per-item transform that must loop over each row.
            for step in steps[1:]:
                step.row_mapped = True
            steps.reverse()
            payload = {"kind": "window", **nodes[current].payload_spec()}
            return steps, payload_key, payload
        if rec.kind not in ("transform", "collate"):
            raise ValueError(f"data node {current!r} ({rec.kind}) cannot sit on the export path for field {field!r}.")
        if rec.kind == "transform" and rec.phase == "augment":
            prev = producers.rewrite_prev.get((current, key))
            if prev is None:
                raise ValueError(
                    f"data node {current!r} has phase 'augment' but does not rewrite {key!r} in place; "
                    "augment-phase transforms must be rewrites to be skippable on the export path."
                )
            current = prev
            continue
        if rec.kind == "transform" and rec.phase in ("target", "source"):
            payload_key = key.split(".", 1)[1] if "." in key else key
            steps.reverse()
            return steps, payload_key, dict(_VALUE_PAYLOAD)
        if rec.kind == "transform" and rec.phase != "preprocess":
            raise ValueError(
                f"data node {current!r} has phase {rec.phase!r}; only preprocess-phase transforms may feed "
                f"model input {field!r}."
            )
        node = nodes[current]
        kernel = type(node).kernel
        if kernel is None:
            raise ValueError(f"data node {current!r} feeds model input {field!r} but declares no kernel.")
        steps.append(PreprocessStep(kernel=kernel, params=node.params(), owner=type(node).__name__, collate=rec.kind == "collate"))

        if key in rec.in_key:
            upstream_key = key
        elif len(rec.in_key) == 1:
            upstream_key = rec.in_key[0]
        else:
            raise ValueError(
                f"data node {current!r} has {len(rec.in_key)} input keys and does not rewrite {key!r}; "
                "export chains support single-input preprocess nodes (or in-place rewrites) only."
            )
        current = producers.producer_of(current, key) if upstream_key == key else producers.final_producer[upstream_key]
        key = upstream_key


def resolve_forward_batch_fields(model: Any) -> list[str]:
    """``batch.*`` fields the model's forward graph actually consumes (excludes loss-only fields,
    which never sit on an export chain)."""
    records = getattr(model, "_pipeline_records", None)
    if not isinstance(records, list):
        return []
    fields: set[str] = set()
    for record in records:
        for key in getattr(record, "in_key", None) or []:
            if isinstance(key, str) and key.startswith("batch."):
                fields.add(key.split(".", 1)[1])
    return sorted(fields)


def resolve_chain_for(datamodule: Any) -> Callable[[Any, str], tuple[list[PreprocessStep], str, dict[str, Any]]]:
    return resolve_data_graph_chain


def validate_export_kernels(datamodule: Any, model: Any) -> None:
    """Fail fast at dry-run (train/test start) if a live forward-chain preprocess node lacks a
    kernel, instead of only surfacing at ``teia export`` time. Reuses ``resolve_data_graph_chain``
    itself (including windowed/reshape-rooted chains), so there's exactly one place that knows
    what's exportable.

    Skips datamodules with no data graph at all (``_records`` unset)."""
    records = getattr(datamodule, "_records", None)
    if records is None:
        return
    resolve_chain = resolve_chain_for(datamodule)
    for field in resolve_forward_batch_fields(model):
        resolve_chain(datamodule, field)
