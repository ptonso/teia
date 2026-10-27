# Task Spec

Related: **Read first** [config](config.md), [base/task](../base/task.md). **See also** [datamodule](datamodule.md), [module/capture_map](module/capture_map.md), [eval](eval.md), [cli](cli.md), [ecosystem](../ecosystem.md).

## Overview

A task groups everything that speaks one contract. It has three parts:
- a contract class (the `_target_`, see [base/task](../base/task.md));
- a folder of interchangeable data and net modules per task: `datamodule/<task>/` and `netmodule/<task>/`;
- an evaluation, chosen from the folder of its capture family: `evalmodule/<family>/`, shared by every task that captures the same atoms;
- one preset, `task/<task>.yaml`, shipped by the package that owns the contract.

For every task it owns, `teia` ships the wiring the contract determines and nothing tuned: one format datamodule, the task's evaluation family, and a `linear_probe` netmodule that proves the contract can be satisfied ([ecosystem](../ecosystem.md#where-code-lives)). Selecting a task gives a working run that is meant to run and not to perform. A project replaces the probe with its own netmodule, written in the probe's image.

Users work at three levels:
1. **Probe** (`task=<task>`): the preset's datamodule, probe and evalmodule. It checks the data and the contract end to end.
2. **Within a task** (`task=<task> netmodule=<task>/<arch>`): a project's own modules in the task's folders. The contract check still applies.
3. **Full control** (no `task`): wire any datamodule, netmodule and evalmodule. Only the in-graph key checks of the dry run apply. A custom contract is one class away.

Walk-through:
1. `task=vision-cls` selects `task/vision-cls.yaml`.
2. Its defaults `override` the three module groups and set `task._target_`.
3. CLI selections and values apply on top, so `netmodule=vision-cls/<arch>` replaces the probe.
4. Pre-flight validates the composed run against the contract.

Scenario: `teia train task=vision-cls data_root=/ds/pets` trains the linear probe on the image-folder datamodule and scores it with the `cls` evalmodule. The project then writes `netmodule/vision-cls/resnet18_pets.yaml`, maps its activation onto `capture.cls.scores` and `capture.cls.label` exactly as the probe does, and runs `teia train task=vision-cls netmodule=vision-cls/resnet18_pets data_root=/ds/pets`. The datamodule and evalmodule need no change.

## Language

- **task**: a named contract plus its data and net module folders and preset. The name (`vision-cls`, `tabular-cls`) is a config choice and lives in paths and the contract's module name only.
- **task preset**: `task/<task>.yaml`, selecting the task's format datamodule, its probe and its evalmodule.
- **module**: a wired graph preset, one of `datamodule/<task>/<name>`, `netmodule/<task>/<name>` or `evalmodule/<family>/<profile>`.
- **evaluation family**: the capture schema an evalmodule scores, named `cls`, `multi-cls`, `det`, `obb`, `pose`, `inst-seg`, `sem-seg`, `reg`, `survival`, `forecast`, `gen` or `rl`. Tasks whose contracts capture the same atoms share one family.
- **format datamodule**: the datamodule that reads one standard on-disk format into the contract's boundary A, with default augmentation.
- **fused task**: a task whose contract subclasses the contract whose targets it predicts and adds input fields that collide with none of its keys and are never targets (`vision-graph-node-cls` is `graph-node-cls` with an image per node). It has its own folders, preset, format datamodule and probe, and the evaluation family of its parent. A fused datamodule reads stable on-disk formats, and how the inputs were derived (crops, a proximity graph) is the application's preprocessing.
- **linear probe**: the netmodule `netmodule/<task>/linear_probe`, a flatten-plus-linear body with the task's objective and capture map. It is the living example of the task's net-side contract.
- **shared fragment**: a config under `<graph>module/_shared/` mounted into a module with `@_here_`. It is never selected as a module.
- **conformance**: every module under a task's folders composes with the task's other defaults and passes the contract and dry-run checks.

## Map

- Presets `conf/task/*`, modules `conf/{datamodule,netmodule}/<task>/*` and `conf/evalmodule/<family>/*`, shared fragments `conf/{netmodule,evalmodule}/_shared/*`, and contracts `teia.task.*`.
- Composition rules: [config](config.md). The contract and its checks: [base/task](../base/task.md). Boundary B in the netmodule: [module/capture_map](module/capture_map.md).
- Demo scripts and their data preparation: `demo/` at the repository root.

## Contracts

### Task preset

```yaml
# conf/task/vision-cls.yaml
# @package _global_
defaults:
  - override /datamodule: vision-cls/image_folder
  - override /netmodule: vision-cls/linear_probe
  - override /evalmodule: cls/default
  - _self_
task:
  _target_: teia.task.vision_cls.VisionCls
```

A preset selects exactly one option per module group and sets `task._target_`. It may set scalar run settings the task needs (`seed`). It MUST NOT carry graph nodes or values under `datamodule`, `netmodule` or `evalmodule`, since those would also apply to any module a caller selects instead.

### Module folders

- `datamodule/<task>/<format>.yaml`: files or a live source in, the contract's `batch.*` and `meta.*` out. Named by the on-disk format it reads (`image_folder`, `csv`, `yolo_format`, `tu`).
- `netmodule/<task>/<arch>.yaml`: the contract's `batch.*` in, `loss.*` and a `capture:` map covering the contract's atoms out. Named by architecture. `teia` ships only `linear_probe`.
- `evalmodule/<family>/<profile>.yaml`: the contract's boundary B in, files out. Named by evaluation family and profile (`cls/default`, `gen/default`).

A module mounts leaves from `node/{data,net,eval}/*` and wires them ([config](config.md)). Duplicating wiring across task folders is expected, while leaves are never duplicated. Evaluations are the exception: they are shared by family. A module may mount a shared fragment from `_shared/` and MUST NOT select another module.

### Evaluation families

An evalmodule reads only captured atoms, target `batch.*` fields and `meta.*` keys, never an input field ([base/task](../base/task.md#taskcontract)), so it cannot depend on a modality. Two contracts with the same boundary B accept the same evalmodules, and the family folders record that: `cls/default` scores `vision-cls`, `tabular-cls`, `audio-cls`, `graph-node-cls` and every task that captures `cls.scores` and `cls.label`, with the same file.

A family is named after the capture schema, which is finer than the route. `cls` (single-label, `cls.label`) and `multi-cls` (multi-label, `cls.labels`) share the route `cls` and differ in atoms, and `det`, `obb`, `pose` and `inst-seg` share the route `det` and add different atoms. A task cross-selects an evalmodule with the same structural check as any module ([cross-folder reuse](#cross-folder-reuse)): `task=distill-cls evalmodule=cls/default` is valid when the distillation contract keeps `cls.scores`.

A plugin whose contract captures atoms no family scores adds a family folder in its own package. A plugin whose contract fits an existing family selects it in its preset and ships no evalmodule.

### Linear probe

The body is one shared fragment, and each task's probe extends it with what its contract requires:

```yaml
# conf/netmodule/_shared/linear_probe.yaml: the body every probe shares
defaults:
  - /node/net/stem@stem: flatten
  - /node/net/head@head: linear
  - /optim@_here_: adam_1e3
  - _self_
stem: {out: {feat.flat: null}}
head: {in: feat.flat}
```

```yaml
# conf/netmodule/vision-multi-cls/linear_probe.yaml
defaults:
  - /netmodule/_shared/linear_probe@_here_
  - /node/net/act@act: sigmoid
  - /node/net/loss@loss: bce
  - _self_
stem:    {in: batch.image}
head:    {out: {pred.logits: [num_classes]}}
act:     {in: pred.logits, out: {act.probs: null, act.labels: null}}
loss:    {in: [pred.logits, batch.cls], out: {loss.cls: null}}
capture: {cls.scores: act.probs, cls.labels: post.act.labels}
```

- The fragment holds only what every probe shares. The per-task file holds the input, the head output, the activation, the loss and the capture map. A Hydra merge cannot delete a key, so anything task-specific in the fragment would leak into other tasks.
- A task whose structure the shared body cannot serve writes its probe without the fragment, with the fewest nodes that reach its atoms. A graph probe applies a linear head per node and unpacks the rows with the node mask, since flattening would mix the nodes of a graph.
- A probe is never tuned. It changes only when its contract changes ([stability](../stability.md#layers)).

### Demo and fixtures

Every task `teia` ships has `demo/<task>.sh` at the repository root and a prepare script under `demo/prepare/`, named for the task or shared by the tasks that read one layout. The prepare script writes a tiny dataset in the format datamodule's layout, generated or fetched without credentials, and the shell script calls it and then runs `teia train task=<task>` on the result. A task whose world is a registered id (`data_root=gym:CartPole-v1`) has no data to prepare and no prepare script. `demo/` is never installed. The end-to-end test packs run the demo scripts with small settings, so the demos are also the fixtures that keep every preset running.

### Cross-folder reuse

Validation is structural. A module from another task's folder is accepted whenever it satisfies the selected contract's boundaries. For example, `task=distill-cls datamodule=vision-cls/image_folder` is valid, because the distillation contract keeps `VisionCls`'s boundary A. Folders are the curated catalog, and the contract is the check.

### Levels on the CLI

```text
teia train task=vision-cls data_root=/ds/pets                                               # probe
teia train task=vision-cls netmodule=vision-cls/resnet18_pets data_root=/ds/pets            # a project's netmodule
teia train task=vision-cls netmodule=vision-cls/resnet18_pets node/net/head@netmodule.head=linear data_root=…  # swap one node
teia train datamodule=mine/x netmodule=mine/y evalmodule=mine/z data_root=…                 # full control, no contract
```

## Extending

To add a task to `teia`:
1. Write its contract ([base/task](../base/task.md#extending)).
2. Write its format datamodule and `linear_probe` under `<graph>module/<task>/`, reusing leaves from `node/*`. A missing objective is a component with a reference leaf ([net/layout](../net/layout.md)).
3. Pick its evaluation family. When no family scores its captured atoms, write `evalmodule/<family>/default.yaml`.
4. Write `task/<task>.yaml` selecting the three.
5. Write `demo/<task>.sh` and its prepare script in `demo/prepare/`.

To add a task in a plugin, follow the same steps in the plugin ([plugin](../plugin.md#task-contracts)). The probe and demo are optional there.

## Constraints

- A task preset MUST be `# @package _global_`, MUST set `task._target_`, and MUST select exactly one option in each of `datamodule`, `netmodule` and `evalmodule`. It MUST NOT define graph nodes or values under a module group.
- Every task `teia` ships MUST have a format datamodule, a `linear_probe` netmodule, an evaluation family that passes its `check_eval`, a preset selecting the three, and a demo script.
- `teia` MUST NOT ship a netmodule other than `linear_probe` under any task folder, nor a tuned trainer, callback or optimizer config (the core system groups `trainer/default` and `callbacks/{default,fitness}` are machinery and stay). This is enforced by `tests/test_task_conformance.py`.
- Every module under `{data,net}module/<task>/` MUST pass the contract checks and dry run when composed with that task's other defaults, every evalmodule MUST pass `check_eval` for at least one shipped contract, and every shipped contract MUST pass `check_eval` for at least one evalmodule. This is enforced by `tests/test_task_conformance.py`.
- A module folder `{data,net}module/<task>/` MUST name a task that has a contract in `teia.task` or a preset in `task/`, and an evalmodule folder `evalmodule/<family>/` names a capture family. A folder named `_shared` holds fragments only. Config lint enforces the data and net rule and the shape of the evalmodule path.
- A module MUST NOT select another module, and a task preset MUST NOT select another task.
- A task MUST have at most one preset across all installed packages. A package that does not ship it adds modules to the task instead.
