# Module Codegen Spec

Related: **Read first** [overview](overview.md), [user_contracts](user_contracts.md). **See also** [nodes](nodes.md), [activation_route](activation_route.md), [dry_run](dry_run.md), [bind](bind.md).

## Overview

`TeiaNetModule` generates its forward, loss and activation methods at `__init__` by building Python function strings from the config and `exec()`-ing them. This removes dict loops that break `torch.compile` while supporting arbitrary DAG topologies that `nn.Sequential` cannot express.

Walk-through: `__init__` validates the graph and generates early functions before node attributes exist. `configure_from_datamodule` regenerates them against the real `nn.Module` attributes, and those are the functions used at train time. The generated function is bound to `self._generated_forward` through `types.MethodType`, and the class-level `forward(batch)` is a thin wrapper that calls through, which is what `torch.compile` wraps.

## Language

- **generated functions**: `_generated_forward`, `_generated_losses`, `_generated_activation`.
- **two-phase build**: a validation pass at `__init__` (graph checks and early generation) and a build pass at `configure_from_datamodule`.
- **key translation**: the rule mapping dotted workspace keys to Python identifiers.

## Map

- `teia.core.module` codegen (`generate_forward_fn` and siblings) and `net.TeiaNetModule`.
- Validation counterpart: [dry_run](dry_run.md). Roles and key namespaces: [user_contracts](user_contracts.md).

## Contracts

### Key translation

| YAML key | Python expression | Notes |
|---|---|---|
| `batch.image` / `batch.bboxes` | `batch.image` / `batch.bboxes` | attribute access on the Batch NamedTuple, unchanged |
| `feat.pooled` / `feat.z` | `feat_pooled` / `feat_z` | local variable, dot to underscore |
| `pred.logits` / `pred.mu` | `pred_logits` / `pred_mu` | local variable, dot to underscore |
| `act.probs` | `act_probs` | local variable in `_generated_activation` |

A node attribute name normalizes `/` to `_` (`encoder/mlp_image` becomes `self.encoder_mlp_image`).

### _generated_forward

Straight-line assignments, one per forward node, calling each node by attribute with its `in` args and unpacking its `out`s. `out: null` is a side-effect call with no assignment. All `pred.*` out-keys are collected in declaration order into the `Pred` return. The `Pred` NamedTuple type is created once in `__init__` and never recreated in the generated function, so a stable type avoids `torch.compile` guard churn.

### _generated_losses(batch, pred)

Builds a `loss_log` dict, calling each loss node with its `batch.*` and `pred.*` args. Each `loss.*` value is logged under `{stage}/{key}`, and the weighted sum across loss nodes becomes `loss_log['loss']`. One `loss.*` out-key returns a scalar `Tensor`, and several return a tuple of scalars in declaration order. The atomic default is one `loss.*` per node. Several are allowed only when one criterion computes them jointly and cannot be cheaply split (a detection criterion emitting `box`, `cls` and `dfl`). Loss nodes never produce `act.*` or `pred.*`.

### _generated_activation(pred)

Excluded from the forward DAG and from `_generated_losses`. It iterates the activation records, calls `act.activation(*pred_logits)` on the `pred.*` entries of each node's `in`, and stores each returned dict under the activation-node name. Host-side `postprocess` runs per route afterwards ([activation_route](activation_route.md)).

### torch.compile

Compatibility holds because the generated code has no dict loops, a stable `Pred` type, static control flow and `setattr`-registered attribute access. A preset whose batches carry Python `list` fields (`ori_shape`, `path`) is the exception: those fields prevent `fullgraph=True`, and a node wrapping a third-party module may be `torch.export`-only.

## Constraints

Validation before generation is fail-fast:

- No duplicate `out` across steps, and every non-`batch.*` `in` MUST be resolvable from a previous `out`.
- A loss `in` MUST NOT start with `feat.`, and all loss `out` MUST start with `loss.`.
- An activation `in` MUST all start with `pred.`, so targets never enter prediction, and all activation `out` MUST start with `act.`.
- Each `out` shape MUST be `null`, an int list or tuple, or a resolvable dim-ref string.
- An N-key `out` MUST match an N-value return, verified empirically by the dry run.
- The generated function MUST be bound to `_generated_forward`, not to `forward`, so Lightning's introspection is undisturbed.

