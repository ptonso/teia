# TeiaDataModule Spec

Related: **Read first** [data_graph](data_graph.md), [base/data/nodes](../base/data/nodes.md). **See also** [base/batch](../base/batch.md), [base/task](../base/task.md), [runtime](runtime.md), [task](task.md), [module/dry_run](module/dry_run.md).

## Overview

`TeiaDataModule` is the single core `LightningDataModule` that executes an data graph. It builds the graph's nodes, runs the three-stage lifecycle (plan, item, batch), and exposes the `batch_type()` and `batch_meta()` protocol the module dry run consumes. Datasets, formats, joins and augmentation are data-graph config, never subclasses.

A graph containing an engine node ([base/data/nodes](../base/data/nodes.md#engine)) is in **stream mode**: a runner drives the engine and the item nodes per environment step, fills a `StreamReshape` buffer, and yields minibatches instead of `Dataset` plus `DataLoader`. The mode is read from the graph itself, computed once at build time. Stream mode uses ordinary data nodes and no separate datamodule class, and core names no field of it: every key comes from the engine's `out`, the graph's wiring, or the task contract.

Walk-through: `setup` runs `plan` nodes once (indices, splits, fitting stateful transforms on train). DataLoader workers run `item` nodes per sample. `on_after_batch_transfer` runs `batch` nodes (GPU transforms, float conversion) and yields the composed batch. After plan, the executor collects each node's published dims and hands them back.

## Language

- **data graph**: the config-declared data-plane graph ([data_graph](data_graph.md)).
- **stage runner**: the execution site for a stage group, `plan`, `item`, `stream` or `batch`.
- **workspace**: `stream.*` → `item.*` → `batch.*`, plus `ctx.*` (externally bound, no graph producer).
- **meta**: dataset-derived facts (`num_classes`, `class_names`, `kpt_shape`, …) published by nodes as `runtime_dims()`, exposed by `meta()`, and read by net `out` dim-refs and the task contract.
- **partial workspace**: the workspace built up to (excluding) a fitting node, used to fit a stateful transform whose input is produced at item stage.

## Map

- `teia.core.datamodule`: `executor.TeiaDataModule`, `graph` (shared graph machinery), and `interactive/` (the stream-mode runtime: `sync.SyncRunner`, `async_runner.AsyncRunner`, `regime` for eval and `batch_meta`, the replay and rollout buffers).
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
runner loop               stream mode only: engine → item passes → act → buffer, unbounded (see Stream mode).
on_after_batch_transfer   batch nodes: GPU transforms + float conversion → the composed batch.
```

Map sourcing iterates indexed streams and joins by key. Stream mode replaces the reader and the dataloader with an engine and a runner, and keeps the item, reshape and collate nodes.

### Batch protocol

- `batch_type() -> type` returns the composed NamedTuple.
- `batch_meta() -> dict[str, tuple[int, ...]]` returns field shapes excluding the batch dim.
- `meta() -> dict[str, Any]` returns the dataset facts published by nodes after plan (`num_classes`, `class_names`, `num_targets`, `kpt_shape`, …).

Structure is declared on the collate node and `FIELD_VOCAB`, not returned. The executor collects each node's `runtime_dims()` into `meta()` and calls `bind_dims(**meta)` back on every node. Net `out` dim-refs and the task contract's `check_data` read `meta()` ([base/task](../base/task.md)). The label list is always published as `class_names`.

### Augment slice

The executor runs the toposorted item nodes and skips `augment` nodes off the train path. Because augments are in-place rewrites, the deterministic letterbox values remain for downstream consumers, which keeps the eval and export slice deterministic without a second graph. A cross-sample augment declaring `needs_siblings` is handed `siblings(i)` and `siblings_len` over the split's other deterministic items before the train `item` stage. The mosaic close-window schedule is owned by the node and pushed in by an `on_train_epoch_start` callback.

### ctx resolution

`bind_context(**live)` registers externally produced values (`bind_context(model=module)`), and `resolve_context(name)` resolves a node's `ctx.<name>` at call time. Every name must be explicitly bound. Stream mode binds nothing: the data graph never calls the network (see Stream mode).

### Stream mode

A runner (`datamodule.runner`, `SyncRunner` by default, `AsyncRunner` for replay buffers) owns the loop. The engine only turns actions into signals. The graph turns signals into observations, rewards and termination flags, and the runner resets the finished environments. Per environment step, for all environments at once:

```text
1. engine.step(action_t)    -> signals after the transition, nothing reset
2. item pass on `end`       preprocess + target nodes, seeded with every signal and the feedback items of step t
                            -> targets of transition t (reward, terminated, ...) and successor features
3. boundary                 OR of the items named by runner.boundary; accounting and episode ids
4. engine.reset(mask)       the boundary mask under per_env (all environments once all have ended under all_done)
5. episodic hooks           reset(mask) then advance() on every command and episodic transform
6. item pass on `next`      preprocess nodes only, seeded with the state signals of the reset engine -> features to act from at t+1
7. act batch                the collates over step 6's items, the environment axis as batch axis
8. act                      the netmodule's capture slice, no grad (module/capture_map#capture-slice)
                            -> capture atoms; the contract's feedback map names those written back as item.<name>
9. buffer.add(row_t)        row_t = step-t act-pass items + feedback items of step t + every step-2 item the act pass lacks
                            (outcomes, targets) + every act-pass key K stored again as item.next_<K> from step 2
10. engine.step(capture.<engine_atom>)
```

- **Item passes.** Both passes run the ordinary toposorted item subgraph with a phase filter, and a node runs when all its inputs are present. The `next` pass is seeded with the engine's `state` signals only and runs `preprocess` (plus `augment` while training); the `end` pass is seeded with every signal and the feedback items and runs `preprocess` and `target`. A node whose inputs include an `outcome` key therefore runs on the `end` pass only. Commands only read their state in both passes, so the goal the policy acted on is the goal its reward is scored against.
- **Successors.** `item.next_<K>` is the generic time-successor of a `preprocess` output `K`, read from the `end` pass, so the successor of a finished episode is its final state and never the reset state. A collate wires it like any key (`item.next_obs_state -> batch.next_obs_state`).
- **Boundary.** `runner.boundary` lists the item keys whose OR ends an episode (`[item.terminated, item.truncated]`). The runner derives episode accounting and ids from the mask, writes ids under the key named by `runner.episode_id`, and passes the mask to a stateful policy adapter.
- **Reset policy.** `runner.reset_policy: per_env | all_done`. Under `all_done` a finished environment idles until every environment has finished, then all reset together; the item named by `runner.valid` marks the live environments and rows of idle environments are not stored.
- **Runner-supplied keys.** The graph may read keys no node produces when the runner supplies them: the contract's fed-back fields, the `item.next_<K>` successors and the `episode_id` key. `bind_contract` checks this and fails naming any other unproduced key.
- **Episode sums.** `runner.episode_sums` lists item keys whose per-step values the runner adds up per environment while an episode runs. When an episode ends the runner emits one record: the sum of each listed key, the episode length and which `boundary` key ended it. Training and eval share this accounting, and the online-metrics callback and the eval metrics read these records.
- **Runner config.** `total_env_steps` (the authoritative training budget), `eval_episodes`, `boundary`, `reset_policy`, `valid`, `episode_id`, `episode_sums` and for `AsyncRunner` `parallel: inline | thread`. Buffer sizing lives on the buffer node. A stream-mode datamodule declares its runner and a buffer; there is no bufferless default.
- **Budget.** The train iterable stops at `total_env_steps`; Lightning's `global_step` counts gradient steps.
- **Eval.** Eval runs the same cycle on the already-open engine, deterministic, after resetting it, and resets it again afterwards. No second simulator is built. The eval graph reads the ended-episode records of the cycle.
- **`batch_meta()`.** Probed with frozen item nodes: the engine reset, a zero action from the action spec and zero scalars for the other fed-back atoms, through both passes and the env-axis collates. A field no node produces before training (a buffer output) is a per-sample scalar. No field list is hardcoded.
- **Live view.** The live-view callback renders the open engine (`render()`), one act per step, and is disabled for engines without `render`.

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
- Stream mode MUST use `num_workers=0`, and vectorization of a live source is internal to the engine.
- In stream mode core MUST NOT write, read or default any `item.*` or `batch.*` name of its own. The only keys the runner writes are the contract's `feedback` names, the `item.next_<K>` successors and the configured `episode_id` key; the only keys it reads are the configured `boundary`, `valid` and `episode_sums` items.
- The data graph MUST NOT call the network; acting goes through the capture slice only.
- The runner MUST reset an engine only after the `end` pass of the step, and MUST call the episodic hooks after the engine reset and before the `next` pass.
