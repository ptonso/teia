# Module User Contracts Spec

Related: **Read first** [overview](overview.md). **See also** [nodes](nodes.md), [codegen](codegen.md), [activation_route](activation_route.md), [optimization](optimization.md).

## Overview

The single canonical reference for the `TeiaNetModule` public configuration surface and the node contracts it enforces: root-entry kinds, the flat envelope, role inference, the key-namespace phases, loss and activation node contracts, and optimizer kwargs. Other module specs link here instead of restating these rules.

A netmodule config has exactly three root-entry kinds. A mapping with `_target_` is a graph node, a flat non-reserved scalar is an optimizer kwarg, and the rest are reserved keys. Role is inferred from a node's `out` namespace and never declared, and the envelope is teia-owned and never reaches the node constructor.

## Language

- **node alias**: a top-level `netmodule.*` mapping with `_target_` plus flat envelope fields, one graph node.
- **envelope**: teia orchestration metadata (`in`, `out`, `weight`, and routing fields from [optimization](optimization.md)) declared as flat keys beside `_target_` and stripped before construction. Shared vocabulary: `teia.base.envelope.ENVELOPE_KEYS`.
- **role**: forward, activation or loss, inferred from the `out` namespace, with no `kind:` field.
- **dim-ref**: an `out` shape string resolved against the live datamodule at build time ([overview](overview.md)).

## Map

- Root keys and envelope constants: `teia.base.envelope`, `NETMODULE_ROOT_KEYS`.
- Build behavior: [overview](overview.md). Node base class and hooks: [nodes](nodes.md). Routing fields: [optimization](optimization.md). Validation: [dry_run](dry_run.md).

## Contracts

### Root-entry kinds

| Entry kind | How identified | Purpose |
|---|---|---|
| Reserved key | in `NETMODULE_ROOT_KEYS` or `__*` | netmodule-level config |
| Node alias | mapping with `_target_` | a graph node (forward, activation or loss) |
| Optimizer kwarg | flat, non-mapping, non-reserved | forwarded to the optimizer constructor |

`NETMODULE_ROOT_KEYS` = `_target_`, `_recursive_`, `optimizer_target`, `lr_scheduler`, `optimizers`, `preset`, `task`, `objective` (plus any `__*`).

### Envelope fields

| Field | Type | Required | Description |
|---|---|---|---|
| `in` | `str` or `list[str]` | yes | workspace keys the node reads |
| `out` | `{key: shape \| dim-ref \| null}` | yes | keys the node writes, `act.*` for activation and `loss.*` for loss |
| `weight` | `float`, default 1.0 | no | loss-term weight, loss nodes only |

Routing fields (`optimizer`, `detach`, `aux_in`, `last_layer`, per-`out` loss routes) are defined in [optimization](optimization.md).

### Role inference

A `loss.*` entry in `out` makes a loss node, an `act.*` entry makes an activation node, and otherwise (`feat.*` or `pred.*`) a forward node.

### Key-namespace phases

| Prefix | Source | Forward | Activation | Loss |
|---|---|---|---|---|
| `batch.*` | dataloader batch | read | forbidden | read |
| `feat.*` | forward outputs | read/write | forbidden | forbidden |
| `pred.*` | head or forward outputs | read/write | read | read |
| `act.*` | activation outputs | act-only | write | forbidden |
| `loss.*` | loss outputs | loss-only | forbidden | write |

So a loss `in` may reference only `batch.*` and `pred.*`, and all loss `out` start with `loss.`. An activation `in` may reference only `pred.*`, since targets never enter the prediction route, and all activation `out` start with `act.`.

### Loss and activation nodes

- **Loss nodes** are pure backward terms, plain `nn.Module` with `forward(*in_keys) -> Tensor`. They own no prediction route. Auxiliary regularizers (KL, perceptual, semi-supervised) are ordinary loss nodes, and a composite term is decomposed into atomic ones.
- **Activation nodes** own the prediction route (`BaseActivation`, see [activation_route](activation_route.md) and [base/net/activation](../../base/net/activation.md)). Inputs are `pred.*` only plus a runtime `ctx`.

### Factory callable

When a loss needs forward-node information to initialize a criterion, the head can carry a factory in its output NamedTuple, for example `MyTaskRaw(raw=..., init_criterion=self._model.init_criterion)`, and the loss calls `pred_raw.init_criterion()` lazily on first forward. This needs no special hook.

### Optimizer kwargs

`optimizer_target` names the optimizer class as a dotted path, and every other flat non-reserved, non-alias root key is forwarded as-is. The multi-route form uses an `optimizers` list ([optimization](optimization.md)).

### configure_from_datamodule

Optional on forward nodes, called during build after instantiation and before the meta forward pass. It serves structural metadata that is not output width (a detector head reading `dm.kpt_shape`), since width comes from `out` dim-refs. See [dry_run](dry_run.md).

## Constraints

- `weight` MUST apply only to loss aliases and is teia-owned, never a loss constructor param.
- Role MUST be inferred from `out` and never declared.
- Inputs MUST arrive only through positional `in` arguments, with no head injection or bespoke configure protocol.
- A loss `in` referencing `feat.*` MUST be a validation error, and an activation `in` referencing anything but `pred.*` MUST be a validation error.

