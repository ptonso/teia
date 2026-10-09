# Batch Spec

Related: **Read first** [structure](structure.md). **See also** [data/nodes](data/nodes.md), [core/module/user_contracts](../core/module/user_contracts.md), [core/module/dry_run](../core/module/dry_run.md), [core/data_graph](../core/data_graph.md).

## Overview

The batch is the seam between the data graph's collate nodes and the net graph. It is an atomic, structure-typed field vocabulary plus a composed `typing.NamedTuple` assembled at build time from the union of fields the active data graph emits. Every field is one dotted key `batch.<field>`.

One datamodule therefore serves every task and every multi-input composition, and there is no per-task Batch class to maintain.

Walk-through: collate nodes each declare the atomic fields they produce. At build time the datamodule composes those into one flat NamedTuple (`batch_type()`) and reports field shapes (`batch_meta()`). An objective declares the `batch.*` fields it consumes. The dry run checks producers against consumers and builds synthetic tensors from `batch_meta()`.

Scenario: a joined camera + table input exposes `batch.cam_image` and `batch.table_numerical`, plus a `cls` target. The classification objective reads `batch.cls`; a missing producer is a hard error at dry run.

## Language

- **atomic field**: one named, structure-typed tensor slot (`image`, `text`, `numerical`, `cls`, `bboxes`, …), independent of any task.
- **composed batch**: the flat NamedTuple built from the union of atomic fields the data graph produces.
- **batch_meta()**: the field-shape map the dry run uses to build synthetic tensors.
- **prefixing**: under a multi-input join, a child field is re-exposed as `<input>_<field>`.

Field names are content-named, never positional; see [structure](structure.md) for the structural types.

## Map

- `teia.base.fields`: `FIELD_VOCAB`, the authoritative field inventory.
- `teia.base.protocols`: `DatamoduleProtocol`.
- Producers: collate nodes in `teia.node.data.collate.*`. Consumers: objectives and net nodes, checked by the [dry run](../core/module/dry_run.md).

## Contracts

### Batch type

A valid batch is a flat `typing.NamedTuple` with attribute access (`batch.field_name`) and typed fields (`Tensor`, `Optional[Tensor]`, `int`, `float`, `str`, `bool`, or `list`/`tuple` of those). `Any`, nesting and a `meta: dict` field are not allowed. Metadata is promoted to typed fields or handled outside the nn boundary.

### Atomic field vocabulary

Representative; the registered `FIELD_VOCAB` is authoritative.

| field | tensor | structure / value-type |
|---|---|---|
| `image` | `[B, C, H, W]` | grid |
| `tokens` | `[B, L]` int64 | sequence, categorical |
| `lengths` | `[B]` int64 | variable-length bookkeeping |
| `numerical` | `[B, F]` float | set, continuous |
| `categorical` | `[B, F]` int64 | set, categorical |
| `cls` | `[B]` or `[B, K]` | classification target (single or multi-hot) |
| `bboxes` | `[sum Ni, D]`, D = 4 or 5 (with `batch_idx`) | detection target, ragged across samples |
| `masks` | `[B, …]` | dense spatial target |
| `target` | `[B, …]` | regression target |
| `keypoints`, `sem_masks`, `waveform`, `count`, `temporal`, `boolean`, `static` | tensors | registered in `FIELD_VOCAB` |

### Datamodule protocol

Both methods are required by `DatamoduleProtocol`, and the dry run calls them unguarded:

```python
def batch_type(self) -> type: ...                       # composed NamedTuple of emitted fields
def batch_meta(self) -> dict[str, tuple[int, ...]]: ... # shapes excluding batch dim; {} = key check only
```

### Composition and prefixing

A join declaring `prefix: true` (a multi-input fusion such as camera plus table) re-exposes each child field as `<input>_<field>` (`cam_image`, `table_numerical`). A pairing join whose inputs are parts of one sample (image with its label file) leaves fields unprefixed, which is the default, and so does a single input. `<input>_present` bool masks appear only under an `outer` join, and `batch_meta()` returns the prefixed union.

### Transfer boundary

GPU transforms and dtype conversion happen on data-graph `batch`-phase nodes. Nodes expect `float32` normalized to `[0,1]` or `[-1,1]` unless a trunk documents otherwise.

## Extending

To add a field, register it in `FIELD_VOCAB` with its structural type, then emit it from a collate node under `teia.node.data.collate.*`. Reuse an existing field before adding one.

## Constraints

- A field name MUST be content-named, never positional.
- A `batch.*` reference with no producer MUST fail fast: `ValueError` when the module is built against the datamodule (`batch in_key … missing from datamodule batch_meta()`), or `AttributeError` on the first batch.
- `teia.base` MUST own only the field vocabulary and structural typing. Collate nodes live in `teia.node.data.collate.*`.
- `list`/`str` fields (`path`, `ori_shape`, `ratio_pad`) are not JIT-compatible. A batch containing them cannot use `torch.compile fullgraph=True` or `torch.jit.trace`. Only detection-family fields are exempt.

