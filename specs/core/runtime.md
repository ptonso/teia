# Runtime Spec

Related: **Read first** [overview](overview.md), [config](config.md). **See also** [deps](deps.md), [eval](eval.md), [infer](infer.md), [export](export.md), [datamodule](datamodule.md).

## Overview

The Lightning execution engine in `teia.core`. It turns a composed config or a saved run snapshot into an executed `train`, `test`, `infer`, `export` or `eval` stage, produces the run layout, and enforces the safety guards.

The runtime always bootstraps through Hydra: the installed package is the source of truth and the project `conf/` is a thin `searchpath` overlay. The public contract is the routed datamodule and module surface, not task-specific stage runners.

Walk-through: every stage composes or rehydrates a config, validates it, applies the RAM guard, builds the datamodule and module, loads a checkpoint, runs, and performs follow-on work. `train` snapshots only the project overlay. The other stages re-compose from that snapshot and replay the saved training overrides before any stage-specific overrides.

## Language

- **stage**: `train`, `test`, `infer`, `export` or `eval`.
- **run layout**: the directory fields a run resolves (`run_dir`, `config_dir`, `logs_dir`, `eval_dir`, `infer_dir`, `export_dir`, `artifacts_dir`) under `runs_root` and `cache_root`.
- **group**: the directory holding the `rep_i` replicas of one replicated train, with its own `group.yaml` and aggregate `eval/`.
- **replica**: one `rep_i` run of a group, a complete run directory.
- **ram_guard**: the host-RAM watchdog applied before datamodule and module instantiation.
- **determinism mode**: `runtime.determinism.mode`, the core-owned policy for deterministic algorithms.

## Map

- `teia.core.runtime.engine`: the stage entrypoints. `composer`, `validate`, `layout`, `memory`, `determinism`, `hooks` and `run_descriptor` sit beside it. `group` (`train_group`) and `group_descriptor` (`group.yaml`) hold replicated runs.
- Stage details: [infer](infer.md), [export](export.md), [eval](eval.md). Config inputs: [config](config.md).

## Contracts

### Entrypoints

`train_project(...)`, `test_project(...)`, `infer_project(...)` and `export_project(...)` in `teia.core.runtime.engine`. Composed config inputs are the root path fields and the `datamodule`, `netmodule`, `trainer`, `runtime`, `callbacks`, `loggers`, `eval`, `train`, `test`, `infer` and `export` groups.

### Stage flows

- **train**: fresh from `project_dir/conf`, or continued from `from_run_dir/config/conf/` (replaying parent overrides). Validate `data_root`, resolve a new child run, snapshot the overlay plus `composed.yaml`, `overrides.yaml` and `hydra.yaml`, build, load the parent checkpoint when continuing, fit, resolve the downstream checkpoint, then optionally test, run the eval graph and export.
- **test**: rehydrate the snapshot, rebuild, load the checkpoint, run the capture pass, run the eval graph.
- **eval**: rehydrate one or more snapshots and run the eval graph offline from each run's capture store, with no model, datamodule or trainer (see [cli](cli.md)).
- **infer**: rehydrate, rebuild the datamodule for the new input source, predict, and write `infer/predictions.jsonl` (or `<output-dir>/predictions.jsonl`) plus additive `infer/io/<io_name>/`.
- **export**: rehydrate, rebuild the module, load the checkpoint, extract the inference graph and runtime metadata, and write an ONNX-first bundle.

### Replicated runs

With `runtime.n_runs` above 1, `teia train` resolves one group directory through the usual run-parts rules, then trains replica `i` in `<group>/rep_<i>` with the override `runtime.seed=<base seed + i>`, a concrete integer. The base seed is the composed `runtime.seed`. Each replica is a complete run (its own snapshot, `run.yaml`, checkpoints, capture store and post-train report), so `test`, `infer`, `export` and `eval` accept a `rep_i` unchanged. After the last replica, the group writes `group.yaml` and its aggregate report ([eval](eval.md#group-reduction)), unless the report is skipped.

With `n_runs` equal to 1 no group exists and the run directory is the one described by the run layout.

```
<group>/
  group.yaml         n_runs, base seed, replica names, status
  rep_0/ ... rep_<n-1>/
  eval/              the aggregate report
```

### Checkpoint selection

A stage loads an explicit `--checkpoint`, else `<stage>.checkpoint`, else the `ModelCheckpoint` file named by `test.weights` (`core/conf/test/default.yaml`). The post-train test and export use the same resolution after fit. `test.weights: best` (default) takes `best`, then `last`. `test.weights: last` takes the final weights, and a run with checkpoints but no `last` one raises `FileNotFoundError` instead of falling back to `best`. It is for runs whose `best` is picked by a biased monitor, such as an online training return. Any other value raises `ValueError`.

### Pre-flight validation

`validate_composed_config` runs before any instantiation and also gates the eval plane. Every eval node's dependencies must be importable and its `in` keys must have a producer. Both checks run before `trainer.fit`, so a mis-wired or un-installable plot fails in seconds (see [eval](eval.md), [deps](deps.md)).

### Determinism

Deterministic behavior is runtime-wide. `runtime.determinism.mode` applies to all stages. `prefer` (default) maps to PyTorch `warn_only=True` and Lightning `deterministic="warn"`. `force` uses strict deterministic algorithms and enriches PyTorch errors with override guidance. `stochastic` disables enforcement and emits a terminal warning.

### RAM guard

`runtime.ram_guard` applies to all stages. The default (`core/conf/runtime/default.yaml`) is `total_percent` at `0.8` of `MemTotal` with `warn_fraction: 0.9`. `usable_percent` (of `MemAvailable`), `gb` and `none` are also supported. It warns every 10 s once RSS reaches `warn_fraction` of the limit and hard-exits once RSS crosses it. It polls process RSS on a thread and never uses `RLIMIT_AS`, which would break CUDA's virtual reservations.

## Constraints

- Relative path fields MUST resolve against `project_dir`.
- Continued `teia train` MUST reject a change to `datamodule._target_` or `netmodule._target_` (`ValueError` from `_validate_continued_train_compatibility`).
- `data_root` existence MUST be validated before any snapshot is written, and a missing path raises `FileNotFoundError`.
- Packs and project configs MUST NOT set `trainer.deterministic`. Teia derives it from `runtime.determinism.mode`.
- `trainer.benchmark=true` MUST be rejected when the determinism mode is `force` or `prefer`.
- The engine MUST NOT hold the module or datamodule alive for the eval pass. Eval nodes read the capture store, never live objects.
- Fallback checkpoint search MUST exclude `export/` and `infer/`.
- A group directory MUST hold only `group.yaml`, `eval/` and `rep_<i>/` directories, and a replica MUST NOT read another replica's files.
- Replica seeds MUST be distinct, and the seed of replica `i` MUST be the base seed plus `i`.
- A failed replica MUST stop the group and MUST NOT leave a `group.yaml` with status `ok`.
- Optional dependency checks are lazy: only the encountered non-core target closure is validated before stage build, and suggestions are enriched only after a dependency-shaped failure.
- Instantiation errors MUST include the dotted config path and the attempted `_target_`. A hook object that is neither callable nor exposes `run()` emits a `UserWarning` and is skipped.

## History

- [2026-10-17 replicated runs and a cross-run reducer](../logs/2026-10-17_replicated-runs-and-reducer.md): the group directory and per-replica seeds.
