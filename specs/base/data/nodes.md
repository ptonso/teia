# Data Node Spec

Related: **Read first** [structure](../structure.md), [batch](../batch.md). **See also** [net/nodes](../net/nodes.md), [core/datamodule](../../core/datamodule.md), [core/data_graph](../../core/data_graph.md).

## Overview

The data graph is the data-plane component graph that turns raw bytes into typed `batch.*` fields and, in reverse, predictions back into bytes. This spec defines the `teia.base` contract types for it: small ABCs that concrete components in `teia.node.data.*` implement, wired by `in_key`/`out_key` over one workspace. `teia.base` owns the language and ships no concrete reader, transform or format.

There are eight node kinds: `Reader`, `Engine`, `Command`, `Transform`, `Join`, `Reshape`, `Collate` and `Writer`. Each is executed in topological order like an nn `TeiaNode`, with free-form wiring typed by kind and phase and no rigid pipeline skeleton.

Walk-through: a reader opens a source and emits an indexed stream `{key → item}`. In stream mode an engine takes its place: it owns a live simulator, is stepped by the runner with actions, and emits named signals as `item.*` values per step. Commands add per-environment targets, and transforms turn signals into observations, rewards and termination flags. Transforms rewrite items (stateful ones are fit on the train split at `plan` time), joins align streams by key, reshapes redefine item granularity, and collate nodes assemble `batch.<field>` tensors, which is the seam to the net graph. A writer is the mirror of a reader and materializes `pred.*` or `act.*` back to a container.

Scenario: a Spot engine emits `item.joint_pos`, `item.lin_vel_b` and `item.foot_force`, a command draws `item.goal` per environment, transforms build `item.obs_state`, `item.reward` and `item.terminated` from them, and collates emit the `batch.*` fields the netmodule trains on.

Scenario: a class-directory reader yields one item per file, a resize-and-normalize transform runs in `preprocess`, an augment runs on train only, a `class from path` transform runs in `target`, and a stack collate emits `batch.image` and `batch.cls`.

## Language

- **data node**: one atomic component in the data graph.
- **workspace**: the io key namespaces. `stream.*` (indexed streams), `item.*` (per-sample values), `batch.*` (assembled fields), `ctx.*` (externally bound, no graph producer). Predictions re-enter as `pred.*` and `act.*` for writers.
- **key**: the per-item identity a reader assigns (filename stem, row id, timestamp, index), used by joins and provenance.
- **phase**: the execution slot of a node: `source`, `preprocess`, `augment`, `target`, `collate`, `writer`. Phases let the engine slice the graph for export and inference.
- **stage**: where a node runs: `plan` (once, at setup), `item` (per-sample, in workers), `stream` (live, unbounded, co-equal with `item`), `batch` (per-batch, on transfer).
- **engine**: the output-only source of a stream-mode datamodule, owning a live simulator and stepped with actions by the runner.
- **boundary**: the structural per-environment mask an engine returns for environments whose episode ended at this step. It is never a named field.
- **reset policy**: when an engine restarts finished environments, `per_env` (immediately) or `all_done` (when every environment has finished).
- **kernel**: the pure function holding a node's math, shared by the torch path and export.

## Map

- `teia.base` (io ABCs, phase and stage enums, key and structure typing): the contracts here.
- `teia.node.data.{reader,engine,transform,join,reshape,collate,writer}`: concrete components by kind. Each kind folder is one node ABC.
- Execution: [core/data_graph](../../core/data_graph.md) (graph build, toposort, phase slicing) and [core/datamodule](../../core/datamodule.md) (the executor).
- Export of the `preprocess` subgraph: [core/export](../../core/export.md).

## Contracts

### Base interface

Every data node is constructed from flat kwargs. `in_key`, `out_key`, `phase` and `stage` come from the graph envelope, mirroring the nn `node` envelope. Each node implements `__call__(*inputs)` returning its declared outputs. Two optional hooks publish and consume dataset-derived sizes: `runtime_dims() -> {name: value}` (for example `{"num_classes": 80}`), surfaced by the executor as datamodule attributes, and `bind_dims(**dims)`, which receives the published dims so a node can size itself.

### Phases

`source` (readers and engines), `preprocess` (deterministic eval transforms: decode, resize, normalize, mel, tokenize), `augment` (train-only, stochastic), `target` (label parse, join, class from path), `collate`, `writer`. Export extracts the `preprocess` subgraph, inference drops `target` and `augment`, and `augment` is train-only. An `augment` transform is an in-place rewrite (`out_key` ⊆ `in_key`), so skipping it leaves the deterministic value in place.

### Reader

`iter_split(split) -> Iterable[(key, raw)]`. It declares `key_fn` and split resolution, may expose an index for caching, and emits `stream.*`. Physical layout is internal: file-per-item and container-of-items satisfy the same contract. A reader never serves a live source; that is an engine.

### Engine

The output-only source of a stream-mode datamodule. It owns one live simulator (a physics engine, a Gymnasium vector env, an API session) for the process, and is driven by the runner, never by graph wiring. It takes no `in`. It turns an action into signals: raw tensors of the simulated world, plus the pieces its simulator computes by name (an observation group, a reward, a termination flag). The graph decides which signals exist as items and what they mean.

Its `out` is a mapping `{item key: signal name}`, one line per signal. The engine validates every signal name at `open` and fails fast on an unknown one. The world comes from `data_root`: a path to an assets folder, or `<scheme>:<id>` for a registered environment (`isaaclab:Isaac-Cartpole-v0`, `gym:CartPole-v1`); the datamodule hands it to the engine as the shared `data_root` attribute and the engine rejects a scheme or id it does not know.

```python
class Engine(DataNode):                                  # stage: stream, phase: source
    def open(self, seed: int) -> None: ...               # lazy, idempotent: build the simulator once per process (the datamodule passes the engine's `seed` attribute)
    def close(self) -> None: ...
    def spec(self) -> EngineSpec: ...                     # per out key: shape (no env axis), dtype and role; action shape, dtype and the simulator's bounds
    def reset(self, mask: Tensor | None = None) -> tuple[Tensor, ...]: ...   # reset every environment, or those in the bool mask [num_envs]; one tensor per out key
    def step(self, action: Tensor) -> tuple[Tensor, ...]: ...                # advance every environment; one tensor per out key, no reset
    def render(self) -> Tensor | None: ...                # optional frames of the open simulator
```

- **No reset inside `step`.** `step` returns the signals at the end of the transition, and nothing resets until the runner calls `reset(mask)`. The runner decides the mask from the termination items of the graph ([core/datamodule](../../core/datamodule.md#stream-mode)), so the end-of-transition values are always the true final values and no simulator-specific "final observation" mechanism exists.
- **Output roles.** `spec()` gives each out key a role: `state` (describes the world at a time; the policy acts on it and its successor is the next state) or `outcome` (describes the transition just taken: a reward, a termination flag). The runner seeds the acting pass with state keys only and takes outcome keys from the step ([core/datamodule](../../core/datamodule.md#stream-mode)).
- **Stock pieces.** A simulator that ships an environment definition (observation groups, reward terms, termination terms) exposes them as named signals, so a recipe can reproduce the published task by naming them. The engine computes a stock piece only when `out` requests it.
- **Logs.** The datamodule injects `log_path` (`<run_dir>/engine/<node>.log`) when it knows its run directory. An engine routes the simulator's own console output there, so the console holds Teia's logs and the run directory holds the engine's.
- **Devices.** Tensors stay on the simulator's device. The action arrives on the policy's device and the engine moves it once.
- **Actions.** `spec()` states the action shape, dtype and the simulator's bounds. The engine passes the action through unchanged and fails fast when a finite bound is violated. The network produces valid actions by itself (its activation owns the range), so a trained policy runs standalone.
- **Dims.** `runtime_dims()` publishes, for each out key `item.<k>`, `<k>_shape` and, for rank-1 values, `<k>_dim`, plus `action_dim` (continuous) or `num_actions` (discrete).

### Command

A source of per-environment targets (a goal velocity, a goal pose, a task selector), drawn by Teia and read by transforms. A command holds state per environment, is seeded at `open(seed)`, may take an `in`, writes its declared `out` and runs in the `preprocess` phase. Inside a pass it only reads its state, so the value the policy acted on is the value the reward is scored against. The cycle changes the state between passes through the `Episodic` hooks.

```python
class Episodic(Protocol):
    def reset(self, mask: Tensor) -> None: ...           # redraw the state of the environments in the bool mask [num_envs]
    def advance(self) -> None: ...                       # one environment step passed (timers tick, resampling happens)

class Command(DataNode, Episodic):                       # stage: stream, phase: preprocess
    def open(self, seed: int, num_envs: int, device: torch.device) -> None: ...   # device: the engine's, so values mix with its signals
    def __call__(self, *item_inputs) -> Any: ...         # the current value per environment, leading axis num_envs
```

A transform that implements `Episodic` receives the same hooks, so per-environment history (the previous action, a frame stack) shares the mechanism. The cycle calls `reset(mask)` with the boundary mask after the engine reset and `advance()` once per step, in this order, before the acting pass of the next step. A node may set `engine: <engine node name>`; the datamodule injects that engine instance after build, for nodes that read the live simulator (a command wrapping a simulator-native command generator).

### Transform

`__call__(*item_inputs) -> item_outputs`, with `phase ∈ {preprocess, augment, target}` and `stage ∈ {item, batch}`. A stateful transform adds `fit(train_stream) -> state` at `plan` and (de)serializes `state` to run artifacts. `fit_all_splits` opts a dataset-metadata vocabulary (class list, label map) into fitting across every split, since no fitted statistic crosses the split boundary. A cross-sample augment (mosaic) sets `needs_siblings = True`, and the executor injects a `siblings(i)` provider over the split's other items on the train `item` stage, and `None` elsewhere.

### Join

`__call__(*streams) -> stream`, aligning by key with `how ∈ {inner, outer, left}` and `on: key`. Under `outer` it emits per-input `present` masks.

### Reshape

A node that redefines item granularity. The shared base declares `payload_spec() -> dict`, an export chain root stating what an export client must send (for example `{"lookback": ..., "horizon": ..., "group": ..., "order": ...}`; default none). Two concrete contracts:

- **`PlanReshape`**: `plan`-stage, single-shot and stateless (`window`, `group`, `explode`), `__call__(*inputs) -> outputs` over a materialized list. A `window` node groups, orders and slices into lookback/horizon items.
- **`StreamReshape`**: `stream`-stage, incremental, stateful, with variable-cardinality output (experience buffers). It exposes `configure(**kwargs)` (bind run-time sizing and the live collater once per fit), `add(row)` (one stored row per environment step, leading axis `num_envs`), `ready() -> bool`, `sample_iter()` (item-shaped records for one ready cycle, each fed through the ordinary collate fn), `on_consumed()` (post-cycle hook: a consuming buffer clears, a persistent one is a no-op) and `metrics() -> dict`. The keys a buffer interprets are bound by name through a mapping-form `in` (`in: {reward: item.reward, terminated: item.terminated}`), and the keys it writes are its declared `out`. A buffer never reads an `item.*` name it was not given.

### Collate

`__call__(item_field_batch) -> batch.<field>`. It assembles a field over the batch (`stack`, `pad`, `ragged`, `multi_hot`) and declares the atomic field and structure it emits. This is the data↔net seam. A collate emits one field. A collate written with several `in` and `out` keys is a fan: the graph pairs them by position and builds one collate per pair, named `<node>_<field>`, with the node's params ([core/data_graph](../../core/data_graph.md#collate-fan)).

### Writer

`__call__(pred_or_act, keys) -> materialized`. The inverse of a reader, in either the file-per-item or the container shape (class-dir tree, coco json, csv).

### Kernel and params

Every computational node (a `preprocess` `Transform` or a `Collate`) has one implementation: a pure `kernel: ClassVar[(value, params) -> (value, restore)]` declared as a `@staticmethod` on the class. `params(self) -> dict` bakes this instance's literals (resolved sizes, fitted means, stds and vocabularies) at export. `kernel = None` marks a boundary node (a reader, a `target` parse, an augment) that export never collects.

The base class supplies the `__call__` adapter for a plain per-value `Transform` and a plain `Collate`, so a concrete class with a `kernel` needs no `__call__`. Structure nodes (multi-lane rewrites such as letterbox, or a schema-bound family fanning one row into several fields) override `__call__` to compose the kernel with glue but hold no math. A `Collate` kernel receives the list of item values for the field, so export can call it with a singleton list.

### `ctx.*` values

A node that needs something no graph node produces (a live object bound mid-collection) declares a `ctx.<name>` key in `in_key`, with no `needs_*` flag and no bespoke injection. `ctx.*` is distinguished by origin. The graph build exempts it from producer validation, and the executor resolves it at call time through `TeiaDataModule.bind_context(**live)` and `resolve_context(name)`, described in [core/datamodule](../../core/datamodule.md).

### `stream` stage

Three roles need `stage: stream`: an engine, a command and a `StreamReshape` buffer. `stream` and `item` are co-equal in rank, so a `stream` node may feed or be fed by an `item` node. Neither may be fed by `batch`.

## Extending

To add a component, subclass the ABC for its kind under the matching `teia.node.data.<kind>` folder, declare phase and stage, and put the math in a `kernel` with `params()`. Imitate an existing exemplar of the same kind, and reuse an existing collate and field before adding one. A stateful transform needs `fit` plus state (de)serialization.

## Constraints

- `teia.base` MUST NOT import project packages above `teia.base`. It owns the ABCs, phase and stage enums, and key and structure typing only.
- Every data node MUST declare a phase and a stage, and the dry run rejects violations. Phase is monotonic: a `batch`-stage node never feeds an `item`-stage node, and `augment` never sits on the export path.
- A stateful transform MUST fit on the train split only. Fitting on val, test or infer is a hard error.
- A `Collate` node's declared field and structure MUST match what the net nodes consume.
- A computational node MUST NOT duplicate its math outside `kernel`.
- An engine MUST send a noisy simulator's console output to its `log_path` and MUST NOT redirect the process's output outside bounded calls (launch, reset, a few steps), since a process-wide redirect also swallows Teia's progress display.
- An engine MUST be output-only (no `in`), write exactly its declared `out` keys, never reset inside `step`, never rescale or clip an action, and keep tensors on its simulator's device.
- A command MUST read only its own state inside a pass and change it only in `reset` and `advance`.
- A `StreamReshape` MUST read only the keys bound in its mapping-form `in` and write only its declared `out`.
