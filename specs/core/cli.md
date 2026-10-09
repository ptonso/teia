# CLI Spec

Related: **Read first** [overview](overview.md), [config](config.md). **See also** [runtime](runtime.md), [infer](infer.md), [export](export.md), [eval](eval.md).

## Overview

The `teia` command surface is an `argparse` front end backed entirely by Hydra composition. Each stage rehydrates the right config source: the live project for a fresh train, a parent snapshot for a continued train, and the immutable run snapshot for everything else. The CLI is core-owned, even though component packages contribute config trees and runtime components through Hydra's `SearchPathPlugin` registry.

Walk-through: `teia train` composes, instantiates the datamodule and module, persists the overlay and composed config, trains, and optionally chains test, eval and export. `test`, `infer`, `export` and `eval` are independent commands that rehydrate from a run directory. Multi-phase training is several explicit `teia train` calls, not one in-process pipeline.

## Language

- **fresh train**: `teia train` composing from the installed package plus an optional project overlay.
- **continued train**: `teia train --from-run-dir`, composing from a parent run snapshot and its saved overrides, then new phase overrides, writing a new child run.
- **rehydrate**: re-compose a stage from `run_dir/config/conf/` against the installed package and replay the saved training overrides before stage-specific overrides.

## Map

- Command entry: `teia.core.cli`, with one module per command under `teia.core.commands`.
- Behavior behind each command: [runtime](runtime.md), [infer](infer.md), [export](export.md), [eval](eval.md).
- Override syntax: [config](config.md).

## Contracts

### Commands

`teia config` (`init`, `list`, `show`, `tree`, `explain`, `lint`), `teia train`, `teia test`, `teia infer`, `teia export`, `teia eval`, and `teia init` (`--project-root`, default `.`; creates `conf/` and links the skills of `teia` and of every installed plugin). `teia plugin init [path] --name` scaffolds a plugin and `teia plugin check [path] [--publish] [--namespace]` verifies one ([plugin](../plugin.md)). Every runtime command (`train`, `test`, `infer`, `export`) also accepts `-c/--cfg {job,hydra,all}` to print the composed config as YAML and exit. All runtime commands accept Hydra overrides.

### Flags

- `teia config init`: `--force`, `--from task=<name>` (materialize the task's fully resolved `datamodule`, `netmodule` and `evalmodule` instead of override stubs).
- `teia config show <group=option>`: `--resolved`, plus trailing Hydra overrides with `--resolved`.
- `teia config tree`: `--root <dotted>`, `--level <N>`, `--node-only`. No overrides.
- `teia config explain <overrides...>`: no flags.
- `teia config lint`: no flags.
- `teia train`: `--project-dir`, `--config-dir`, `--run-name`, `--from-run-dir`, `--from-run-weights`, `--checkpoint`, and the post-train toggles `--skip-post-test`, `--skip-report`, `--export`.
- `teia eval`: one or more `--run-dir` (repeatable), `--split` (capture split, default `test`), `--output-dir` (default `<run-dir>/eval`, or `./compare` for several runs), `--compare <yaml>` (a `compare:` node map, default `core/conf/compare/default.yaml`).
- `teia test`, `teia infer`, `teia export`: required `--run-dir`, plus `--checkpoint` and `--output-dir`. `test` also takes `--skip-report`. `infer` also takes `--src` (new input data) and `--dst` (prediction destination, alias of `--output-dir`, which `--dst` wins over).

### Behavior

- `config init` writes `conf/node/net/head/custom.yaml`, `conf/node/net/loss/custom.yaml` and `conf/node/data/reader/custom.yaml` stubs. With `--from task=<name>` it composes the task and writes fully resolved `conf/{datamodule,netmodule,evalmodule}/mine/<name>.yaml` (real `_target_`s, no `defaults:` list, nothing left null), selectable as `<graph>module=mine/<name>`. Neither mode copies the package tree, and the overlay enters `searchpath` automatically when `./conf` exists.
- `init` creates `<project-root>/conf/` and symlinks every packaged `teia/skills/<name>/` that holds a `SKILL.md` into `<project-root>/.claude/skills/` and `.agents/skills/`. The links are absolute into the installed package, so the package stays the single source of truth and `pip install -U teia` updates every project. It is idempotent, drops dangling links, never replaces a real file or directory, and appends the managed link paths to the project `.gitignore` (each teammate's `teia init` recreates them). Override stubs come from `config init`, never from `init`.
- `config list` lists the layers: core, then the project overlay if `./conf` exists, then each plugin package's conf tree in registration order. Hydra's real search order puts the project overlay after the plugin trees, so a project file that reuses the path of a packaged option is ignored: an override goes in a new file that a selector picks (`node/net/head@netmodule.head=mine`).
- `config show` prints the raw source file plus its layer, and `--resolved` prints the composed subtree at the file's `# @package` address (the group path when the file has none), or the whole composed config for a `# @package _global_` file such as a task preset.
- `config tree` prints the config-group directory tree merged across all layers, each entry tagged with the layers that provide it. `--node-only` drops options, `--level` caps depth and `--root` scopes a subtree.
- `config explain` composes once with all overrides and once per group-level override removed (`task=`, `<graph>module=`), then diffs `hydra.runtime.choices` to attribute each changed group to the override that set it. A mount swap (`node/net/encoder@netmodule.encoder=x`) only composes on top of the module that mounts it, so it is left out of those recompositions and attributed to `CLI`.
- `config lint` checks that every `node/*` leaf has no `# @package` header and no `in`/`out`, that no module carries a `_target_`, that every task preset follows [task](task.md#constraints), and that every module lives under a `<graph>module/<task>/` folder with a matching `task/<task>.yaml`.
- `train` composes (fresh or continued), persists the overlay plus `composed.yaml`, `overrides.yaml` and `hydra.yaml`, trains, optionally chains test and eval, and exports only when config or `--export` enables it.
- `eval` rebuilds one run's `eval/` or, with several runs, runs the `compare/` nodes across them. It reads each run's snapshot and capture store, and `evalmodule` overrides apply, so a new plot can be added to an old run without retraining.
- `test`, `infer` and `export` rehydrate from the snapshot, rebuild the module (and datamodule), load the checkpoint and run their stage.

### Examples

```text
teia train task=vision-cls data_root=/ds/pets
teia train task=vision-cls netmodule=vision-cls/mlp
teia train task=vision-cls node/net/encoder@netmodule.encoder=mlp_tabular
teia train --from-run-dir runs/.../head-warmup --from-run-weights best netmodule.encoder.freeze=false
teia infer --run-dir runs/... infer.output_dir=/tmp/preds
teia export --run-dir runs/... export.output_dir=/tmp/bundle
```

## Constraints

- The CLI MUST NOT bypass Hydra composition.
- `teia train` is the only stage that may boot from live project config or a parent snapshot. `--from-run-dir` MUST be mutually exclusive with `--project-dir` and `--config-dir`.
- `test`, `infer` and `export` MUST NOT compose from the live project tree when `--run-dir` is provided.
- `teia train` MUST run post-train test and the eval graph when a test split exists, unless disabled by config or CLI. Post-train export runs only when enabled by config or `--export`.

