# Config Spec

Related: **Read first** [overview](overview.md). **See also** [task](task.md), [base/task](../base/task.md), [cli](cli.md), [runtime](runtime.md), [datamodule](datamodule.md), [module/overview](module/overview.md), [eval](eval.md), [plugin](../plugin.md).

## Overview

Configuration is pure Hydra/OmegaConf composition rooted at a core-owned `config.yaml`. A run is wired from four things:
1. **System groups**: `trainer`, `runtime`, `callbacks`, `loggers`, `train`, `test`, `infer` and `export`.
2. **Three module groups**, one per graph: `datamodule`, `netmodule` and `evalmodule`.
3. **Leaf nodes** those modules mount: `node/data/*`, `node/net/*` and `node/eval/*`.
4. **An optional task**: `task/<task>`, a contract plus one default per group ([task](task.md)).

The same three words name everything. A `datamodule` is a wired graph of `node/data` leaves, a `netmodule` of `node/net` leaves, and an `evalmodule` of `node/eval` leaves.

Every group is a primary entry of the root defaults list, so every selection, swap and value override uses native Hydra syntax, and there is no custom composition code.

Walk-through:
1. Hydra composes the root defaults.
2. A selected task, which is a `_global_` preset, overrides the module and system groups it fills.
3. CLI selections and values apply last.
4. The result is one flat `datamodule` graph, one flat `netmodule` graph and one flat `evalmodule` graph, plus `task._target_` and the system groups.

Scenario: `teia train task=vision-cls netmodule=vision-cls/wide_mlp node/net/encoder@netmodule.encoder=mlp_image data_root=/ds/pets netmodule.lr=1e-3` selects a task, swaps its architecture, swaps one node inside it, and overrides a value.

## Language

- **group**: a Hydra config group listed as a primary `optional … : null` (or `: default`) entry in the root `config.yaml`.
- **node leaf**: one component config, a `_target_` plus parameters with no wiring, under `node/<graph>/<kind>/`.
- **module**: a wired graph preset under `<graph>module/<task>/<name>.yaml` that mounts node leaves and sets their `in` and `out`.
- **mount**: a module's defaults entry `/node/<graph>/<kind>@<alias>: <leaf>`, placing a leaf at `<graph>module.<alias>`.
- **task preset**: `task/<task>.yaml`, a `# @package _global_` overrider ([task](task.md)).
- **dim-ref**: an `out` shape token resolved against the datamodule's `meta()` at build time (`pred.logits: [num_classes]`).
- **data_root**: the single dataset-location knob.

## Map

- The root `config.yaml` and the system defaults: `teia/core/conf/` (`trainer`, `runtime`, `callbacks`, `loggers`, `train`, `test`, `infer`, `export`).
- Node leaves: `conf/node/data/*` and `conf/node/net/*` in the component library, and `conf/node/eval/*` in `teia`.
- Modules and tasks: `conf/{datamodule,netmodule,evalmodule}/<task>/*` and `conf/task/*` in the component library. `teia` ships task-less examples and the shared eval fragment `evalmodule/_shared/run.yaml`.
- Composition and validation code: `teia.core.config`, `teia.core.runtime.composer`, `teia.core.runtime.validate`.

## Contracts

### Root skeleton

```yaml
defaults:
  - trainer: default        # + runtime, callbacks, loggers, train, test, infer, export
  - optional datamodule: null
  - optional netmodule: null
  - optional evalmodule: null
  - optional task: null
  - _self_
```

`task` comes after the module groups, so a preset's `override /<graph>module` entries apply. CLI selections always win over the preset.

### Node leaves

A leaf holds a `_target_` and its parameters. It has no `# @package` header and no `in` or `out` wiring, because the mounting module supplies both.

```yaml
# conf/node/net/encoder/mlp_image.yaml
_target_: teia.node.net.one_to_one.agnostic.mlp.MLP
hidden: 128
pretrained: true
freeze: true
```

Leaf kinds are by purpose:
- `node/data/{reader,join,transform,augment,reshape,collate,writer}`
- `node/net/{stem,embedder,encoder,neck,fusion,branch,decoder,head,act,loss,optim}`
- `node/eval/{metric,view,kernel}`

The purpose folder is an interface label. The `_target_` names the Python component, and one component may be wired for several purposes.

### Modules

A module is packaged at its group (`datamodule`, `netmodule`, `evalmodule`). It mounts leaves under aliases and wires them:

```yaml
# conf/netmodule/vision-cls/mlp.yaml
defaults:
  - /node/net/encoder@encoder: mlp_image
  - /node/net/head@head: linear
  - /node/net/act@act: softmax
  - /node/net/loss@loss: cross_entropy
  - /optim@_here_: adam_1e3
  - _self_
input: {image_size: 224}
encoder: {in: batch.image, out: {feat.pooled: null}}
head:    {in: feat.pooled, out: {pred.logits: [num_classes]}}
act:     {in: pred.logits, out: {act.probs: null}}
loss:    {in: [pred.logits, batch.cls], out: {loss.cls: null}}
capture: {cls.scores: act.probs}
```

- A packaged module carries no `_target_`. The engine instantiates `TeiaDataModule` and `TeiaNetModule` by default, and their operational defaults (`num_workers`, `split`, …) live in the constructors. A project module may set `_target_` to use its own `LightningDataModule` or `LightningModule` (full control).
- A module's non-node keys are that executor's settings (`batch_size` and `image_size` for the datamodule; `input`, `capture` and the optimizer fields for the netmodule; `enabled`, `split` and `capture` limits for the evalmodule).
- A node is any mapping carrying a `_target_`.
- `@_here_` merges a leaf into the module root, which is how optimizer fields and shared eval fragments are mounted.

### Net-owned input preferences

A preference that the net owns but the data graph consumes, such as input resolution or normalization statistics, is declared under `netmodule.input`. The datamodule interpolates it (`image_size: ${netmodule.input.image_size}`), so swapping the netmodule carries its preference along.

### CLI actions

```text
teia train task=vision-cls                                                   # select a task preset
teia train task=vision-cls netmodule=vision-cls/wide_mlp                 # swap a module
teia train task=vision-cls node/net/encoder@netmodule.encoder=mlp_tabular # swap one mounted node
teia train task=vision-cls optim@netmodule=adam_1e3                # swap a @_here_ mount
teia train task=vision-cls data_root=/ds/pets netmodule.lr=1e-3             # value overrides
```

A node swap replaces the leaf's `_target_` and parameters and keeps the module's wiring (`in`, `out`) and any parameter the module sets. Value overrides of keys present in the composed config stay unprefixed, and an absent key needs `++`.

### data_root

- **The only dataset-location key.** Dataset location is the top-level `data_root=…`, with no aliases. The engine resolves it relative to the project dir (a file path collapses to its parent, and a missing path raises) and passes it to the datamodule as its `data_root` kwarg. Reader and transform nodes resolve their node-level `root` under it. Streaming datamodules that declare `requires_data_root = False` need none.
- **`data=` and `data_dir=` are rejected.** Before composition, the composer fails with "use `data_root=<path>`".
- **Path-valued module selections are rejected.** A module-group selection whose value looks like a path (`datamodule=/x/data.yaml`) is rejected with the same hint.

### teia config init

Writes a thin project overlay only:
- `conf/node/net/head/custom.yaml`, `conf/node/net/loss/custom.yaml` and `conf/node/data/reader/custom.yaml` stubs.
- With `--from task=<name>`, it also writes the fully resolved `conf/{datamodule,netmodule,evalmodule}/mine/<name>.yaml`.

It never duplicates the package tree. `teia train` snapshots only the project overlay, and replay re-composes against the installed package.

## Extending

- **A component config:** place a leaf under `conf/node/<graph>/<kind>/` naming its `_target_`, then mount it from a module.
- **A module:** add it to the matching `<graph>module/<task>/` folder ([task](task.md)).
- **A system variant:** add an option to its system group, such as `trainer/vision.yaml`.

## Constraints

- Fields MUST be content-named (`batch.image`, `feat.pooled`), never positional. See [base/batch](../base/batch.md).
- Every group a user swaps MUST be a primary entry of the root defaults, or a mount whose final package address is overridable as `<group>@<address>=<option>`.
- A node leaf MUST NOT carry a `# @package` header, `in` or `out`. A packaged module MUST NOT carry a `_target_` and MUST live under `<graph>module/<task>/`. This is checked by `teia config lint`.
- A mount alias MUST NOT contain `_group_` or `_name_`: Hydra substitutes those inside package paths, so `@km_by_risk_group_view` silently mounts elsewhere. This is checked by `teia config lint`.
- A task preset MUST follow [task](task.md#constraints).
- There MUST be no custom composition resolver. Overrides use native Hydra only.
- There MUST be no `domain` or `task` string field that routes behavior. The task is identified only by `task._target_`.
- Dataset-derived dims (`num_classes`, `num_targets`) MUST use `out` dim-refs, not `${…}` interpolation.
- An absent key MUST be written `++key=value`. The composer rejects a bare `+key=value`.
- A `key=value` override whose dotted key is absent from the composed config emits a `UserWarning` (typo guard).
- `data=`, `data_dir=` and path-valued module selections MUST fail before composition with a `data_root=` hint.

