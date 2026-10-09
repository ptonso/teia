# Core Overview Spec

Related: **Read first** [README](../README.md). **See also** [config](config.md), [runtime](runtime.md), [task](task.md), [eval](eval.md), [cli](cli.md).

## Overview

`teia` is a Hydra-first training runtime. It turns a composed configuration into executed `train`, `test`, `infer`, `export` and `eval` stages over a single Lightning module. This spec is the entry point to the `core` specs: the package surface, the user workflow and the layering between a generic core and the component libraries.

Core is application-agnostic and Lightning-only. It executes three config-declared component graphs (io, nn, eval) and ships no metric, plot or architecture of its own. Component libraries own the components and the recipe cookbook.

Walk-through: the user selects a run with Hydra overrides (`task=<name>`, or `datamodule=… netmodule=… evalmodule=…`). `teia train` composes from the live project `conf/`, snapshots it into `run_dir/config/`, and runs the stage. Every later stage (`test`, `infer`, `export`, `eval`) rehydrates from that immutable snapshot rather than the live tree.

Scenario: `teia train task=tabular-cls data_root=/data/adult` trains, captures the test split, and runs the eval graph. `teia export --run-dir <run>` then produces an ONNX bundle from the same snapshot.

## Language

- **core**: `teia.core`, the application-agnostic runtime (config, Lightning execution, capture, eval-graph execution, inference, export) plus the two executors `TeiaDataModule` and `TeiaNetModule`.
- **component library**: `teia.node.data.*` (data components by kind), `teia.node.net.*` (compute components by fan × assumes) and `teia.node.eval.*` (evaluation components by kind × pairing). There are no modality packs. `vision` and `tabular` survive only as task and dataset names.
- **task**: a contract plus one default per module group (`task=vision-det`), described in [task](task.md).
- **run**: one stage execution rooted at a `run_dir` with an immutable config snapshot.

## Map

- Configuration: [config](config.md), [task](task.md), [cli](cli.md), [deps](deps.md).
- Execution: [runtime](runtime.md).
- The three graphs: [data_graph](data_graph.md) and [datamodule](datamodule.md) (data), [module/overview](module/overview.md) (nn), [eval](eval.md) (eval).
- Stage outputs: [infer](infer.md), [export](export.md).
- Contracts core implements: the `base` specs, starting at [base/structure](../base/structure.md).

## Contracts

### User workflow

1. Install the runtime and component libraries. Optional dependency groups are resolved per run at the instantiate/engine boundary, not by a modality extra (see [deps](deps.md)).
2. Optionally `teia config init` to write thin project override stubs (`conf/node/net/head/custom.yaml`, `conf/node/data/reader/custom.yaml`).
3. Select a run with Hydra overrides and edit the stubs.
4. Run `teia train`, `teia test`, `teia infer`, `teia export` or `teia eval`. See [cli](cli.md).

### Config source of truth

Hydra and OmegaConf are the only configuration system. The installed package is the config source of truth, and `teia config init` writes only thin overrides. The project `conf/` tree is authoritative for `teia train`, a continued phase is authoritative on its parent run snapshot, and the immutable `run_dir/config/conf/` snapshot is authoritative for `test`, `infer`, `export` and `eval`.

### Run directory

`runs_root/<run_parts...>/` holds:

- `config/`: config snapshots (project overlay, `composed.yaml`, `overrides.yaml`, `hydra.yaml`).
- `logs/`: stage logs and the per-epoch logger CSV.
- `eval/`: eval-graph artifacts (`*.{png,csv,yaml}` including `summary.yaml`, plus `walltime.yaml`).
- `infer/`: inference outputs, or an explicit target.
- `export/`: export bundles.
- `artifacts/`: runtime metadata, the run descriptor `run.yaml`, and the capture store `capture/<split>/`.

### Extension interfaces

Custom `LightningDataModule` and `LightningModule` targets, custom config groups under the project `conf/`, custom eval, callback, logger, infer and export components, and component-library config trees that overlay through an auto-discovered `hydra.searchpath` plugin. A custom metric or plot is an ordinary component: subclass the ABC in [base/eval/nodes](../base/eval/nodes.md) and wire it in an evalmodule.

## Constraints

- Core MUST NOT contain application- or modality-specific code, meaning no `domain` dispatch, no task-name dispatch and no concrete task contract. Component libraries MUST NOT ship a Lightning module.
- Core MUST NOT contain a metric, plot, table or comparison implementation.
- Core MUST be Lightning-only, with no model-owned stage runners.
- `teia export` is ONNX-first. `teia infer` always writes a canonical prediction artifact and may also write io-specific outputs.

