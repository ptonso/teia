# Data Node Spec

Related: **Read first** [structure](../structure.md), [batch](../batch.md). **See also** [net/nodes](../net/nodes.md), [core/datamodule](../../core/datamodule.md), [core/data_graph](../../core/data_graph.md).

## Overview

The data graph is the data-plane component graph that turns raw bytes into typed `batch.*` fields and, in reverse, predictions back into bytes. This spec defines the `teia.base` contract types for it: small ABCs that concrete components in `teia.node.data.*` implement, wired by `in_key`/`out_key` over one workspace. `teia.base` owns the language and ships no concrete reader, transform or format.

There are six node kinds: `Reader`, `Transform`, `Join`, `Reshape`, `Collate` and `Writer`. Each is executed in topological order like an nn `TeiaNode`, with free-form wiring typed by kind and phase and no rigid pipeline skeleton.

Walk-through: a reader opens a source and emits an indexed stream `{key → item}`. Transforms rewrite items (stateful ones are fit on the train split at `plan` time), joins align streams by key, reshapes redefine item granularity, and collate nodes assemble `batch.<field>` tensors, which is the seam to the net graph. A writer is the mirror of a reader and materializes `pred.*` or `act.*` back to a container.

Scenario: a class-directory reader yields one item per file, a resize-and-normalize transform runs in `preprocess`, an augment runs on train only, a `class from path` transform runs in `target`, and a stack collate emits `batch.image` and `batch.cls`.

## Language

- **data node**: one atomic component in the data graph.
- **workspace**: the io key namespaces. `stream.*` (indexed streams), `item.*` (per-sample values), `batch.*` (assembled fields), `ctx.*` (externally bound, no graph producer). Predictions re-enter as `pred.*` and `act.*` for writers.
- **key**: the per-item identity a reader assigns (filename stem, row id, timestamp, index), used by joins and provenance.
- **phase**: the execution slot of a node: `source`, `preprocess`, `augment`, `target`, `collate`, `writer`. Phases let the engine slice the graph for export and inference.
- **stage**: where a node runs: `plan` (once, at setup), `item` (per-sample, in workers), `stream` (live, unbounded, co-equal with `item`), `batch` (per-batch, on transfer).
- **kernel**: the pure function holding a node's math, shared by the torch path and export.

## Map

- `teia.base` (io ABCs, phase and stage enums, key and structure typing): the contracts here.
- `teia.node.data.{reader,transform,join,reshape,collate,writer}`: concrete components by kind. Each kind folder is one node ABC.
- Execution: [core/data_graph](../../core/data_graph.md) (graph build, toposort, phase slicing) and [core/datamodule](../../core/datamodule.md) (the executor).
- Export of the `preprocess` subgraph: [core/export](../../core/export.md).

## Contracts

### Base interface

Every data node is constructed from flat kwargs. `in_key`, `out_key`, `phase` and `stage` come from the graph envelope, mirroring the nn `node` envelope. Each node implements `__call__(*inputs)` returning its declared outputs. Two optional hooks publish and consume dataset-derived sizes: `runtime_dims() -> {name: value}` (for example `{"num_classes": 80}`), surfaced by the executor as datamodule attributes, and `bind_dims(**dims)`, which receives the published dims so a node can size itself.

### Phases

`source` (readers), `preprocess` (deterministic eval transforms: decode, resize, normalize, mel, tokenize), `augment` (train-only, stochastic), `target` (label parse, join, class from path), `collate`, `writer`. Export extracts the `preprocess` subgraph, inference drops `target` and `augment`, and `augment` is train-only. An `augment` transform is an in-place rewrite (`out_key` ⊆ `in_key`), so skipping it leaves the deterministic value in place.

### Reader

`iter_split(split) -> Iterable[(key, raw)]`. It declares `key_fn` and split resolution, may expose an index for caching, and emits `stream.*`. Physical layout is internal: file-per-item and container-of-items satisfy the same contract. A reader over a live source (an interactive environment) declares `stage: stream` instead of `plan`.

### Transform

`__call__(*item_inputs) -> item_outputs`, with `phase ∈ {preprocess, augment, target}` and `stage ∈ {item, batch}`. A stateful transform adds `fit(train_stream) -> state` at `plan` and (de)serializes `state` to run artifacts. `fit_all_splits` opts a dataset-metadata vocabulary (class list, label map) into fitting across every split, since no fitted statistic crosses the split boundary. A cross-sample augment (mosaic) sets `needs_siblings = True`, and the executor injects a `siblings(i)` provider over the split's other items on the train `item` stage, and `None` elsewhere.

### Join

`__call__(*streams) -> stream`, aligning by key with `how ∈ {inner, outer, left}` and `on: key`. Under `outer` it emits per-input `present` masks.

### Reshape

A node that redefines item granularity. The shared base declares `payload_spec() -> dict`, an export chain root stating what an export client must send (for example `{"lookback": ..., "horizon": ..., "group": ..., "order": ...}`; default none). Two concrete contracts:

- **`PlanReshape`**: `plan`-stage, single-shot and stateless (`window`, `group`, `explode`), `__call__(*inputs) -> outputs` over a materialized list. A `window` node groups, orders and slices into lookback/horizon items.
- **`StreamReshape`**: `stream`-stage, incremental, stateful, with variable-cardinality output (experience buffers). It exposes `configure(**kwargs)` (bind run-time sizing and the live collater once per fit), `add(item)`, `ready() -> bool`, `sample_iter()` (item-shaped records for one ready cycle, each fed through the ordinary collate fn), `on_consumed()` (post-cycle hook: a consuming buffer clears, a persistent one is a no-op) and `metrics() -> dict`.

### Collate

`__call__(item_field_batch) -> batch.<field>`. It assembles a field over the batch (`stack`, `pad`, `ragged`, `multi_hot`) and declares the atomic field and structure it emits. This is the data↔net seam.

### Writer

`__call__(pred_or_act, keys) -> materialized`. The inverse of a reader, in either the file-per-item or the container shape (class-dir tree, coco json, csv).

### Kernel and params

Every computational node (a `preprocess` `Transform` or a `Collate`) has one implementation: a pure `kernel: ClassVar[(value, params) -> (value, restore)]` declared as a `@staticmethod` on the class. `params(self) -> dict` bakes this instance's literals (resolved sizes, fitted means, stds and vocabularies) at export. `kernel = None` marks a boundary node (a reader, a `target` parse, an augment) that export never collects.

The base class supplies the `__call__` adapter for a plain per-value `Transform` and a plain `Collate`, so a concrete class with a `kernel` needs no `__call__`. Structure nodes (multi-lane rewrites such as letterbox, or a schema-bound family fanning one row into several fields) override `__call__` to compose the kernel with glue but hold no math. A `Collate` kernel receives the list of item values for the field, so export can call it with a singleton list.

### `ctx.*` values

A node that needs something no graph node produces (a live object bound mid-collection) declares a `ctx.<name>` key in `in_key`, with no `needs_*` flag and no bespoke injection. `ctx.*` is distinguished by origin. The graph build exempts it from producer validation, and the executor resolves it at call time through `TeiaDataModule.bind_context(**live)` and `resolve_context(name)`, described in [core/datamodule](../../core/datamodule.md).

### `stream` stage

Two roles need `stage: stream`: a live reader and a `StreamReshape` buffer. `stream` and `item` are co-equal in rank, so a `stream` node may feed or be fed by an `item` node. Neither may be fed by `batch`.

## Extending

To add a component, subclass the ABC for its kind under the matching `teia.node.data.<kind>` folder, declare phase and stage, and put the math in a `kernel` with `params()`. Imitate an existing exemplar of the same kind, and reuse an existing collate and field before adding one. A stateful transform needs `fit` plus state (de)serialization.

## Constraints

- `teia.base` MUST NOT import project packages above `teia.base`. It owns the ABCs, phase and stage enums, and key and structure typing only.
- Every data node MUST declare a phase and a stage, and the dry run rejects violations. Phase is monotonic: a `batch`-stage node never feeds an `item`-stage node, and `augment` never sits on the export path.
- A stateful transform MUST fit on the train split only. Fitting on val, test or infer is a hard error.
- A `Collate` node's declared field and structure MUST match what the net nodes consume.
- A computational node MUST NOT duplicate its math outside `kernel`.

