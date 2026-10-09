# TeiaDataModule Spec

Related: **Read first** [data_graph](data_graph.md), [base/data/nodes](../base/data/nodes.md). **See also** [base/batch](../base/batch.md), [base/task](../base/task.md), [runtime](runtime.md), [task](task.md), [module/dry_run](module/dry_run.md).

## Overview

`TeiaDataModule` is the single core `LightningDataModule` that executes an data graph. It builds the graph's nodes, runs the three-stage lifecycle (plan, item, batch), and exposes the `batch_type()` and `batch_meta()` protocol the module dry run consumes. Datasets, formats, joins and augmentation are data-graph config, never subclasses.

A graph containing any `stage="stream"` node (a live reader, a buffer) selects the generator-driven loop instead of `Dataset` plus `DataLoader`. The choice is read from the graph itself, computed once at build time as `self._live`. Live collection uses this path with ordinary data nodes and no separate datamodule class.

Walk-through: `setup` runs `plan` nodes once (indices, splits, fitting stateful transforms on train). DataLoader workers run `item` nodes per sample. `on_after_batch_transfer` runs `batch` nodes (GPU transforms, float conversion) and yields the composed batch. After plan, the executor collects each node's published dims and hands them back.

## Language

- **data graph**: the config-declared data-plane graph ([data_graph](data_graph.md)).
- **stage runner**: the execution site for a stage group, `plan`, `item`, `stream` or `batch`.
- **workspace**: `stream.*` → `item.*` → `batch.*`, plus `ctx.*` (externally bound, no graph producer).
- **meta**: dataset-derived facts (`num_classes`, `class_names`, `kpt_shape`, …) published by nodes as `runtime_dims()`, exposed by `meta()`, and read by net `out` dim-refs and the task contract.
- **partial workspace**: the workspace built up to (excluding) a fitting node, used to fit a stateful transform whose input is produced at item stage.

## Map

- `teia.core.datamodule`: `executor.TeiaDataModule`, `graph` (shared graph machinery), and `interactive/` (`regime`, `policy`, `runner_base`, the collection runtime).
- Contracts implemented: [base/data/nodes](../base/data/nodes.md), [base/batch](../base/batch.md). Graph DSL: [data_graph](data_graph.md).
- Consumers of `batch_type()` and `batch_meta()`: [module/dry_run](module/dry_run.md).

## Contracts

### Config interface

`datamodule` is a module from `datamodule/<task>/<format>` ([task](task.md)): a wired graph of `node/data/*` leaves plus the executor settings `batch_size`, `num_workers`, `val_num_workers`, `pin_memory`, `persistent_workers`, `drop_last`, `image_size`, and split and cache controls. The executor's defaults live in its constructor, and the module carries no `_target_` ([config](config.md)). By phase, the graph's nodes are:

- `reader`, `join` and `target`/`source` parses → raw `item.*` (read and parse only, deterministic, sources resolve under `data_root`).
- `preprocess` transforms: deterministic item transforms.
- `augment` transforms: stochastic and train-only.
- `collate`: `item.* → batch.*`, then `batch`-stage transforms.
- `writer`: inference-only, reading `capture.*` and `meta.*` ([infer](infer.md)).

Emitted fields are content-named, never positional. `drop_last` (default `false`) applies to the train dataloader only. Val, test and predict keep the last partial batch and use `val_num_workers`. A net-owned preference is interpolated from the netmodule (`image_size: ${netmodule.input.image_size}`).

### Stage lifecycle

```text
setup(stage)              plan nodes: resolve reader indices and splits; fit stateful transforms on train;
                          plan joins and reshapes; persist fitted state to run artifacts.
DataLoader worker         item nodes: read → parse/transform (preprocess + train-only augment) per sample.
generator loop            stream nodes (only when self._live): live reader → item nodes → buffer, unbounded.
on_after_batch_transfer   batch nodes: GPU transforms + float conversion → the composed batch.
```

Streaming and map sourcing share one path. A map dataset iterates indexed streams and joins by key, while a live reader emits a stream of pre-joined items.

### Batch protocol

- `batch_type() -> type` returns the composed NamedTuple.
- `batch_meta() -> dict[str, tuple[int, ...]]` returns field shapes excluding the batch dim.
- `meta() -> dict[str, Any]` returns the dataset facts published by nodes after plan (`num_classes`, `class_names`, `num_targets`, `kpt_shape`, …).

Structure is declared on the collate node and `FIELD_VOCAB`, not returned. The executor collects each node's `runtime_dims()` into `meta()` and calls `bind_dims(**meta)` back on every node. Net `out` dim-refs and the task contract's `check_data` read `meta()` ([base/task](../base/task.md)). The label list is always published as `class_names`.

### Augment slice

The executor runs the toposorted item nodes and skips `augment` nodes off the train path. Because augments are in-place rewrites, the deterministic letterbox values remain for downstream consumers, which keeps the eval and export slice deterministic without a second graph. A cross-sample augment declaring `needs_siblings` is handed `siblings(i)` and `siblings_len` over the split's other deterministic items before the train `item` stage. The mosaic close-window schedule is owned by the node and pushed in by an `on_train_epoch_start` callback.

### ctx resolution

`bind_context(**live)` registers externally produced values (`bind_context(model=module)`), and `resolve_context(name)` resolves a node's `ctx.<name>` at call time. An explicit binding wins. `policy` additionally falls back to `trainer.lightning_module`, and every other name must be explicitly bound.

### Ground truth

The datamodule has no ground-truth path of its own. Captured targets are the contract's `target=True` fields of the eval batch ([core/eval](eval.md#ground-truth)), so the deterministic eval slice is the single source of both model input and ground truth.

### Artifacts

Fitted transform state (normalization, vocab, schema, obs-norm) is saved with `save_artifacts(dir)` after fit and reloaded with `load_artifacts(dir, *, required)` for test, infer and export. There is one pickled `<node>.state.pkl` per stateful `Transform`, and nodes may add JSON extras through `IoNode.export_artifacts()`. This is the single source of truth for [export](export.md) and [infer](infer.md).

### Post-transfer helper

All post-transfer preparation routes through one datamodule helper, so GPU transforms are never bypassed by manual prediction or capture materialization.

## Constraints

- Core owns `TeiaDataModule` and the stage runners, and concrete data nodes live in `teia.node.data.*`. No modality-specific datamodule subclass MAY be introduced, and live collection is no exception.
- The executor MUST NOT hold a task string or branch on one.
- The graph MUST be validated at build (topological sort, missing producer, cycle, duplicate key, phase monotonicity, structural type at the collate seam) so failures are config errors, not runtime crashes.
- No stateful transform MAY fit on a split other than train. Fitted state is persisted and reloaded, never re-fit.
- All post-transfer batch preparation MUST go through the datamodule-owned helper.
- Stream readers requiring live model weights MUST use `num_workers=0`. Vectorization of a live source is internal to the reader.

