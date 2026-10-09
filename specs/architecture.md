# Teia Architecture Spec

Related: **Read first** [terminology](terminology.md). **See also** [core/overview](core/overview.md), [base/structure](base/structure.md), [base/task](base/task.md), [core/deps](core/deps.md), [stability](stability.md), [ecosystem](ecosystem.md), [plugin](plugin.md), [_template](_template.md).

## Overview

Teia is **component-oriented**. Three component graphs, each a flat map of named nodes wired by `in` and `out`, describe every run: a **data graph** (executed by `TeiaDataModule`) turns raw data into `batch.*` fields, a **net graph** (executed by `TeiaNetModule`) turns those into predictions and losses, and an **eval graph** (executed by the eval runner) turns a captured run into named results and artifact files. The data and net graphs meet at `batch.*`, and the eval graph reads what a run captured (`capture.*`, `batch.*`) and produces `eval.*` and files.

The distribution is published as `pyteia` and installs the `teia` import package and the `teia` command. There are no modality subpackages. "Vision", "audio" and "tabular" are config recipes over shared components. The distribution contributes to one `teia.*` import namespace through PEP 420, so other distributions can add components under identical paths without touching this one.

Walk-through: the user selects a task, or three modules directly. Hydra composes the data, net and eval graphs from component configs, and the engine executes them in the stage requested (`train`, `test`, `infer`, `export`, `eval`). Components live in `teia.node`, and the engine in `teia.core` executes whatever the config declares.

## Language

Terms shared across the whole spec set. Subsystem specs link here instead of redefining them. The vocabulary rules for domain words are in [terminology](terminology.md).

- **data graph / net graph / eval graph**: the three component graphs above.
- **node**: one component in a graph, wired by `in` and `out`.
- **structure**: the topology of a signal (`set`, `sequence`, `grid`, `graph`), the single axis for both data and compute ([base/structure](base/structure.md)).
- **objective**: what to optimize, an activation plus a loss.
- **task**: a contract typing the two graph boundaries (`batch.*`/`meta.*`, and `capture.*`), plus optional module folders and a default preset ([core/task](core/task.md)).
- **regime**: how data and optimization are coupled in time, offline (default) or live collection.
- **format**: how raw bytes are read or written (jpeg, wav, csv, json).
- **fan / assumes**: the two constraint axes of the net Python tree, io cardinality and input requirement ([net/layout](net/layout.md)).
- **capture**: the persisted per-run atoms the eval graph reads.
- **route**: the first segment of a capture key, the namespace a task contract files predictions under.

## Map

- `teia.base`: the torch-only contract floor ([base/structure](base/structure.md), [base/batch](base/batch.md), [base/task](base/task.md), the node ABCs).
- `teia.core`: the engine, CLI, Hydra composition, runtime, the three graph executors, capture, export and infer ([core/overview](core/overview.md)).
- `teia.node.data`: data components by node kind ([data/layout](data/layout.md)). `teia.node.net`: compute components by `fan × assumes` ([net/layout](net/layout.md)). `teia.node.eval`: evaluation components by ABC kind × pairing ([eval/layout](eval/layout.md)). `teia.task`: task contracts. `teia.callbacks`: training callbacks.
- `src/teia/conf/`: operational config, node leaves under `conf/node/{data,net,eval}`, modules, task presets and the demo recipes. `demo/`: two runnable scripts.
- No distribution owns `teia/__init__.py`, since `teia` and `teia.node` are namespace levels. Plugins add components and configs under their own package and attach their configs through auto-discovered Hydra search-path plugins ([plugin](plugin.md)).

## Contracts

### The four axes

Every run is described by four independent axes, each with one home.

| Axis | Question | Values | Home |
|---|---|---|---|
| **structure** (data ≡ compute) | domain topology of the signal | set, sequence, grid, graph (compositional) | batch-field typing and `teia.node.net.*` |
| **objective** (learn) | what to optimize? | cls, det, seg, similarity, recon, … | objective atoms ([core/module/objective](core/module/objective.md)) |
| **regime** (loop) | how are data and optimization coupled in time? | offline (default), live collection | engine loop property |
| **format** (bytes) | how are raw bytes read or written? | jpeg, wav, csv, json, … | `teia.node.data.reader` and `writer` |

"Modality" is not an axis. It is a bundle of `(structure × value-type × format)`, which is why it was ambiguous and drove duplication.

### Two organizing principles

- **Python trees organize by hard architectural constraint.** `teia.node.data` by ABC kind. `teia.node.net` by io cardinality (**fan**: `zero_to_one`, `one_to_one`, `many_to_one`, `one_to_many`, `many_to_many`) × input type (**assumes**: value domain `categorical`, `count`, `numerical`, or structure `sequence`, `grid`, `set`, `graph`, or the residual `agnostic`, `heterogeneous`). `teia.node.eval` by ABC kind × pairing. A module does not care whether its key is `batch.*` or `feat.*`.
- **Conf trees organize by purpose and interface**, what a user wires (`stem`, `encoder`, `head`, …). The same Python module is wired for different purposes by different conf files that differ only in `in` and `out`. The engine is a pure DAG that classifies producers by `out` namespace alone (`feat`, `pred`, `act`, `loss`) and never sees purpose labels.

### Namespace

- `teia.core`: the engine, `TeiaDataModule` (data-graph executor), `TeiaNetModule` (net-graph executor), the eval-graph runner and capture, stage runners, and regime-loop selection.
- `teia.node.data.*`: data components grouped by ABC kind (`reader/`, `transform/`, `join/`, `reshape/`, `collate/`, `writer/`).
- `teia.node.eval.*`: evaluation components grouped by ABC kind (`metric/` sub-grouped by pairing `aligned/` and `unpaired/`, `view/` by payload geometry, `compare/`), plus `kernel/` sub-nodes. No domain folder.
- `teia.node.net.*`: compute components `<fan>/<assumes>/`, with `activation/` and `loss/` flat. There is no Python `stem`, `encoder` or `head` folder, and a stem is just a `one_to_one` node.
- A graph folder (`teia.node.{net,data,eval}`, `conf/node/{net,data,eval}`) holds nodes only, meaning components with `in`/`out` wired by a module. Non-nodes are siblings named by kind: `teia.task` with `conf/task/`, `teia.callbacks`, and `conf/optim/`.
- Recipes are config: node leaves `conf/node/{data,net,eval}/*`, modules `conf/{datamodule,netmodule,evalmodule}/<task>/*`, and task presets `conf/task/*`. Task contracts are `teia.task.*`.

### The base contract floor

`teia.base.*` is the torch-only contract floor: the atomic batch-field vocabulary and structure typing, the `TeiaNode` base, the data-node ABCs (`Reader`, `Transform`, `Join`, `Reshape`, `Collate`, `Writer`) and the eval-node ABCs (`Metric`, `View`, `Comparison`, plus the `EvalSource` read contract). It owns the shared language and ships no concrete components.

### Config ownership

`teia` owns operational config in-package (`config.yaml` with the primary slot list, core runtime groups, trainer and callback defaults, infer and export adapters), `conf/node/eval/*` with the per-task `evalmodule/<task>/default` presets, the `conf/node/{data,net}` leaves of its components, and the shipped modules and presets. Its Hydra search-path plugin auto-registers the config, so `teia train task=<name>` composes after install. Module groups and `task` are primary-level `optional <name>: null` selections, and a task preset is written `# @package _global_` ([core/config](core/config.md)).

### Dependency direction

`teia` depends on torch and engine and runtime libraries only. The eval plane ships inside `teia` because training needs it. `teia.core` may instantiate `teia.node.*` from config strings but MUST NOT import `teia.node` statically in engine or executor code, the eval plane included. Third-party dependencies of components are per component and resolved per run ([core/deps](core/deps.md)).

## Constraints

- Core MUST NOT contain modality-specific code. There are no vision, tabular or other domain datamodule subclasses, and no `domain` field routes behavior.
- `teia.core` MUST be Lightning-only, and all three graphs are config-declared and executed generically.
- `teia.core` MUST NOT contain a metric, plot, table or comparison implementation, and MUST NOT interpret the meaning of a captured key ([core/eval](core/eval.md)).
- Structure is the single data ≡ compute axis, and a `batch.*` field is typed by structure, not modality.
- Fields MUST be content-named (`batch.image`, `batch.text`, `batch.numerical`, `feat.pooled`), never positional (`batch.x`).
- The Python taxonomy encodes hard constraints only. Purpose (`stem`, `encoder`, `head`) and application (`det`, `survival`) are conf concerns and MUST NOT become Python folders or engine-visible roles.
