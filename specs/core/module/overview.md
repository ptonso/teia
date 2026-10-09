# Module Overview Spec

Related: **Read first** [user_contracts](user_contracts.md), [../../base/structure](../../base/structure.md). **See also** [nodes](nodes.md), [objective](objective.md), [codegen](codegen.md), [dry_run](dry_run.md), [activation_route](activation_route.md), [optimization](optimization.md).

## Overview

`TeiaNetModule` is the single Lightning module that executes the net graph. It is configured as an alias-centric pipeline: each top-level `netmodule.*` alias is one Hydra-composed object plus a flat teia-owned envelope. This spec covers build and runtime behavior and the shape, prediction and step contracts. The config surface itself is in [user_contracts](user_contracts.md).

The graph decomposes into optional named regions of one flat graph: an **input** bundle (stems and fusion, `batch.* → feat.*`), the **middle** (encoder, neck, branch, fusion, `feat.* → feat.*`), and one or more **objective** atoms (`feat.*` and `batch.* → pred/act/loss`, tagged `{name, route}`, see [objective](objective.md)). The composed config is always flat. Multi-input and multi-output runs are just more branches and objectives.

Walk-through: at construction the module scans aliases, infers each node's role from its `out` namespace, and toposorts the forward nodes. At `configure_from_datamodule(dm)` it builds a synthetic workspace from `dm.batch_meta()`, instantiates each node with injected shapes, and runs a meta forward pass to propagate shapes downstream. Activation and loss nodes stay instantiated for `predict_step` and export.

## Language

- **alias**: a top-level `netmodule.*` entry. `netmodule.<alias>.*` are constructor params, and the reserved envelope keys are teia orchestration params at the same flat level.
- **workspace**: the dotted-key namespace (`batch.*`, `feat.*`, `pred.*`, `act.*`, `loss.*`), content-named (`batch.image`, `feat.pooled`).
- **dim-ref**: an `out` shape resolved against the live datamodule at build time.
- **meta forward pass**: a synthetic shape-propagation run during build.

## Map

- Config surface, envelope, role inference: [user_contracts](user_contracts.md). Node base and hooks: [nodes](nodes.md).
- Objectives and aggregation: [objective](objective.md). Prediction route: [activation_route](activation_route.md). Optimization: [optimization](optimization.md).
- Tied and frozen nodes: [bind](bind.md). Build validation: [dry_run](dry_run.md). Generated forward: [codegen](codegen.md).
- Implementation: `teia.core.module` (`net.TeiaNetModule`).

## Contracts

### Alias config

Canonical presets use the purpose aliases `stem`, `encoder`, `neck`, `fusion`, `branch`, `decoder`, `head`, `loss`, suffixed only for multiples (`head_det`). Implementation params live only in node leaves under `conf/node/net/**`. Modules add only the `in`/`out` envelope, which keeps CLI overrides clean (`netmodule.encoder.hidden=256`).

```yaml
netmodule:
  _target_: teia.core.module.TeiaNetModule
  _recursive_: false
  encoder:
    _target_: teia.node.net.one_to_one.agnostic.mlp.MLP
    hidden: 128
    in: batch.image
    out: { feat.pooled: null }
  head:
    _target_: teia.node.net.one_to_one.agnostic.linear.LinearHead
    in: feat.pooled
    out: { pred.logits: [num_classes] }
  act:
    _target_: teia.node.net.activation.sigmoid.SigmoidActivation
    in: pred.logits
    out: { act.probs: null }
  loss:
    _target_: teia.node.net.loss.bce.BCELoss
    in: [pred.logits, batch.cls]
    out: { loss.cls: null }
    weight: 1.0
```

### Build behavior

At construction: scan aliases, treat each `_target_` plus envelope entry as a node, infer role from `out`, parse `in`/`out`, toposort the forward aliases (activation and loss nodes are outside the forward DAG), reject missing producers, duplicate produced keys and cycles, and collect flat non-alias root scalars as optimizer kwargs.

At `configure_from_datamodule(dm)`: build the synthetic workspace, instantiate each forward alias with envelope keys stripped, inject `in_shape` and `out_shape` into `TeiaNode` subclasses (resolving `out` dim-refs against `dm`), run the meta forward pass, then instantiate activation and loss aliases. A loss's `configure_from_datamodule` runs only after the whole build ([nodes](nodes.md)). A `bind` `tied` node meta-forwards its source, and a `frozen` clone is not meta-forwarded ([bind](bind.md)).

### Output shapes

Each `out` entry is `null` (inferred from meta execution), an int or string list or tuple (declared shape, passed as `out_shape`), or a dim-ref, a string naming a `dm` attribute (`pred.logits: [num_classes]` → `int(dm.num_classes)`, `pred.box: [2, num_points]` → `(2, dm.num_points)`). Resolution is generic, not a whitelist: it takes `getattr(dm, name)` when that is an int (a 1-tuple) or an int tuple or list, or a `batch_meta()` key. Multi-output order is the declared map order.

### Prediction path

The activation node (`act.*`) owns the route ([activation_route](activation_route.md)): `activation` (logits to activated, ONNX-traceable) plus `postprocess` (host tail). Heads emit logits only, and loss nodes have no prediction role. A module may declare several activation nodes, and `predict_step`, capture and export iterate them and return a dict keyed by activation-node name, uniform even for one route. The same `activation` and `postprocess` serve infer, capture and ONNX export.

### Step and batch

Batch size is read from the first non-`None` tensor field, so optional leading fields (a batch whose leading field is `None`) never break logging. A datamodule may flag a val or test pass as capture-only (`dm.eval_capture_only`), in which case the step runs `forward` and capture but skips loss and the route sum (interactive rollout eval). Supervised runs leave it unset.

### Optimization

The module always runs Lightning manual optimization ([optimization](optimization.md)). The flat single-optimizer form is shorthand for a one-element `optimizers` list. When `weight_decay > 0`, parameters split into a decay group (`ndim >= 2`) and a no-decay group (biases and normalization scale or shift). `decouple_weight_decay: false` forwards all parameters as one group.

### EMA

`lightning.pytorch.callbacks.WeightAveraging` runs an EMA swapped in for val, test and predict, with `avg_fn` setting the update rule. `teia.core.callbacks.ramped_ema_avg_fn(decay, tau)` gives per-step decay `decay * (1 - exp(-step / tau))`, warming up after about 5·tau steps. Switching to flat decay points `avg_fn._target_` at `torch.optim.swa_utils.get_ema_avg_fn`.

## Constraints

- Reserved root keys (`_target_`, `_recursive_`, `optimizer_target`, `lr_scheduler`, `optimizers`, `preset`, `task`, `objective`, `__*`) MUST NOT be treated as aliases. Every other top-level mapping MUST be a real composed node, and every other flat scalar is an optimizer kwarg. Per-node envelope keys (`in`, `out`, `phase`, `stage`, `weight`, `optimizer`, `detach`, `aux_in`, `last_layer`, `bind`) are the shared set, and anything else on a node is a component kwarg.
- An unresolvable dim-ref (no matching int or int-tuple `dm` attribute and no `batch_meta()` key) MUST be a hard error.
- A missing `batch.*` key in any forward, loss or activation `in` MUST be a hard error, and the dry run never synthesizes undeclared datamodule fields.
- Having no loss node is not a build error. A loss-less netmodule builds and passes `fast_dev_run` silently, and the summary can show `Trainable params: 0`.
- Output dimensionality MUST come from `out` dim-refs, not Hydra interpolation, since dims are runtime properties absent from the composed tree.
- For short runs a flat EMA decay of 0.9999 collapses validation. Use flat 0.999 or the ramped factory.

