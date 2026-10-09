# Teia Core

Teia core is the thin runtime layer around config composition, object instantiation, run layout, eval hooks, and export hooks.

The console entrypoint lives in `teia.core.cli`, and subcommands are discovered from `teia.core.commands`.
The installed package is the config source of truth; a project `conf/` is a thin optional overlay composed on top through Hydra `searchpath`.

## CLI
- `teia config init`: write project `conf/module/custom.yaml` and `conf/datamodule/custom.yaml` override stubs
- `teia config list`: list the config layers Teia composes from (core, active packs, project overlay)
- `teia config show <group>=<option> [--resolved]`: print a config group option's source file, or its composed value with `--resolved`
- `teia train`: compose config, build runtime objects, execute training, run post-train test/eval, and run export only when enabled (`--export` or `train.run_export_after_train=true`)
- `teia test`: compose config, run test stage, run the eval graph
- `teia export`: export an inference bundle

Install note:
- Core runtime/eval/export dependencies ship with `teia`; component-specific packages are suggested lazily when a configured run imports them.

## Export Contract
`teia export` now writes a bundle contract owned by core:
- `manifest.json`: bundle version, runtime family, entrypoint, and artifact/spec paths
- `metadata/task.json`: task id, exporter family, tensor names, dtypes, shapes, and dynamic axes
- `metadata/preprocess.json`: runtime adapter and payload contract
- `metadata/postprocess.json`: runtime decode adapter and output policy
- `metadata/labels.json`: resolved label names when the task exposes them
- `python/infer.py`: generated runnable inference entrypoint

Core owns the generic ONNX bundle/runtime shape. Branch packs such as `teia.vision` specialize that contract with domain-specific preprocess/postprocess adapters.

## Config Contract
Top-level groups:
- `task`
- `datamodule`
- `module`
- `eval`
- `trainer`
- `runtime`
- `callbacks`
- `loggers`
- `export`
- `train`
- `test`
- `infer`

Common root-level override keys:
- `project_name`
- `experiment_name`
- `runs_root`
- `cache_root`
- `run_name`
- `run_parts`

`task` is the main selector. In Teia-shaped projects, task selection implies the default `datamodule`, `module`, and `eval` groups. Users override only what differs from the default behavior.

The config source of truth is packaged YAML under `teia.core.config/conf/`, with active branch packs layering their own packaged `config/conf/` trees on top through manifest-driven discovery and Hydra `searchpath`. A project `conf/` (written by `teia config init`) is an optional overlay that adds option files such as `module/custom.yaml`; it is never a full copy.
For built-in Teia targets, the shipped `conf/` should spell out all defaulted parameters explicitly, including `auto` modes for dynamic defaults.

`runtime` now owns only execution controls such as seeding, strictness, matmul precision, and post-stage behavior flags.

## User Contract
- If you use only built-in task packs, `teia train task=<task-id> ...` should be enough for common cases.
- Common filesystem and naming overrides stay flat where they are runtime-wide, for example `teia train task=det data_root=/datasets project_name=teia experiment_name=det-v2`.
- Dataset location lives at top-level `data_root`.
- Built-in module tuning uses alias overrides such as `module.head.num_classes=10` or `module.encoder.hidden=128`.
- If you need custom behavior, point `datamodule` or `module` at your own `_target_`.
- Custom training code belongs in normal Lightning modules and datamodules; Teia owns the runtime around them.
- Custom datamodules should accept `data_root` in `__init__`. Teia injects `data_root` and `project_dir` when the target supports those kwargs.

## Run Contract
Runs are written under `runs_root/<run_parts...>/`, where the default `run_parts` is `[project, experiment, run]`.
Each run bundle includes:
- `config/`
- `logs/`
- `eval/`
- `export/`
- `artifacts/metadata.json`

Runtime defaults include `runtime.float32_matmul_precision=high` and `runtime.determinism.mode=prefer`,
which Teia applies through PyTorch and Lightning when available.
