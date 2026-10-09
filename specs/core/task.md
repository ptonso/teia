# Task Spec

Related: **Read first** [config](config.md), [base/task](../base/task.md). **See also** [datamodule](datamodule.md), [module/capture_map](module/capture_map.md), [eval](eval.md), [cli](cli.md).

## Overview

A task groups everything that speaks one contract. It has three parts:
- a contract class (the `_target_`, see [base/task](../base/task.md));
- a folder of interchangeable modules per graph: `datamodule/<task>/`, `netmodule/<task>/` and `evalmodule/<task>/`;
- at most one preset, `task/<task>.yaml`, holding the single best default for all of them. A distribution ships a preset only where it ships a default.

Selecting a task gives a working run. Swapping a module for another in the same task folder gives another working run.

Users work at three levels:
1. **Blind** (`task=<task>`): take the preset and choose nothing besides the data location, a size, or one backbone override.
2. **Within a task** (`task=<task> netmodule=<task>/<arch>`): swap modules inside the task's folders to test architectures, eval profiles or data formats. Every such swap is valid.
3. **Full control** (no `task`): wire any datamodule, netmodule and evalmodule. Only the in-graph key checks of the dry run apply. A custom contract is one class away.

Walk-through:
1. `task=vision-cls` selects `task/vision-cls.yaml`.
2. Its defaults `override` the three module groups and any system groups it tunes (trainer, callbacks, export), and set `task._target_`.
3. CLI selections and values apply on top.
4. Pre-flight validates the composed run against the contract.

Scenario: `teia train task=vision-cls netmodule=vision-cls/wide_mlp data_root=/ds/pets` trains a wider MLP on the preset's datamodule and evalmodule. Both netmodules map their activation onto `capture.cls.scores`, so the eval graph needs no change.

## Language

- **task**: a named contract plus its module folders and preset. The name (`vision-cls`, `tabular-cls`) is a config choice and lives in paths and the contract's module name only.
- **task preset**: `task/<task>.yaml`, the single best default for the task. It is updated when a better default is found, and never multiplied into variants.
- **module**: a wired graph preset, one of `datamodule/<task>/<name>`, `netmodule/<task>/<name>` or `evalmodule/<task>/<name>`.
- **conformance**: every module under a task's folders composes with the task's other defaults and passes the contract and dry-run checks.

## Map

- Presets `conf/task/*`, modules `conf/{datamodule,netmodule,evalmodule}/<task>/*`, and contracts `teia.task.*`, all in the component library. `teia` ships none of these. Its example runs are task-less.
- Composition rules: [config](config.md). The contract and its checks: [base/task](../base/task.md). Boundary B in the netmodule: [module/capture_map](module/capture_map.md).

## Contracts

### Task preset

```yaml
# conf/task/vision-cls.yaml
# @package _global_
defaults:
  - override /datamodule: vision-cls/class_dir
  - override /netmodule: vision-cls/mlp
  - override /evalmodule: vision-cls/default
  - override /trainer: vision
  - override /callbacks: vision_fitness
  - _self_
task:
  _target_: teia.task.vision_cls.VisionCls
```

A preset selects exactly one option per module group and sets `task._target_`. It may override system groups and scalar run settings the task needs (`seed`, `data_root`, `epochs`, `project`, `experiment`). It MUST NOT carry graph nodes, which belong to modules.

### Module folders

- `datamodule/<task>/<format>.yaml`: files or a live source in, the contract's `batch.*` and `meta.*` out. Named by the on-disk format or source it reads (`class_dir`, `csv`).
- `netmodule/<task>/<arch>.yaml`: the contract's `batch.*` in, `loss.*` and a `capture:` map covering the contract's atoms out. Named by architecture (`mlp`, `wide_mlp`).
- `evalmodule/<task>/<profile>.yaml`: the contract's boundary B in, files out. Named by profile (`default`, `light`, `full`).

A module mounts leaves from `node/{data,net,eval}/*` and wires them ([config](config.md)). Duplicating wiring across task folders is expected, while leaves are never duplicated.

### Cross-folder reuse

Validation is structural. A module from another task's folder is accepted whenever it satisfies the selected contract's boundaries. For example, `task=distill-cls datamodule=vision-cls/class_dir` is valid, because the distillation contract keeps `VisionCls`'s boundary A. Folders are the curated catalog, and the contract is the check.

### Levels on the CLI

```text
teia train task=vision-cls data_root=/ds/pets                                            # blind preset
teia train task=vision-cls netmodule=vision-cls/wide_mlp data_root=/ds/pets              # swap within the task
teia train task=vision-cls node/net/encoder@netmodule.encoder=mlp_image data_root=…  # swap one node
teia train datamodule=mine/x netmodule=mine/y evalmodule=mine/z data_root=…              # full control, no contract
```

## Extending

To add a task:
1. Write its contract ([base/task](../base/task.md#extending)).
2. Write at least one module per graph under `<graph>module/<task>/`, reusing leaves from `node/*`.
3. Write `task/<task>.yaml`.
4. Add a run script under the matching end-to-end pack in `tests/*-runs/run/`.

The conformance test discovers the new folders by path. To add an architecture to an existing task, add `netmodule/<task>/<arch>.yaml` with a `capture:` map that covers the contract.

## Constraints

- A task preset MUST be `# @package _global_`, MUST set `task._target_`, and MUST select exactly one option in each of `datamodule`, `netmodule` and `evalmodule`. It MUST NOT define graph nodes.
- Every module under `<graph>module/<task>/` MUST pass the contract checks and dry run when composed with that task's other defaults. This is enforced by `tests/test_task_conformance.py`.
- Every task preset MUST have at least one end-to-end run script that uses the preset alone.
- A module folder `<graph>module/<task>/` MUST name a task that has a contract in `teia.task` or a preset in `task/`. Config lint enforces it.
- A module MUST NOT select another module, and a task preset MUST NOT select another task.

