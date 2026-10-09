# Module Bind Spec

Related: **Read first** [overview](overview.md). **See also** [codegen](codegen.md), [nodes](nodes.md), [optimization](optimization.md), [user_contracts](user_contracts.md).

## Overview

`bind` decouples a graph node (an application, wiring `in → out`) from the module (the parameters) it uses. By default they are one to one. `bind` lets a node reference another node's module under a declared parameter-coupling policy. It is the minimal generalization that expresses **weight tying** (Siamese towers, actor-critic critics) and **frozen synced clones** (target networks of value-based methods and EMA teachers), with no algorithm-specific core code.

There are two graphs over the nodes: the dataflow graph (`in`/`out`, driving execution order) and the parameter graph (who shares or tracks whose weights). `bind` is the only declaration of parameter-graph edges.

Scenario: a target encoder binds to `encoder` in `frozen` mode with `HardSync(every=500)`. The module clones the encoder, freezes it, excludes it from every optimizer, and copies weights from the online encoder every 500 steps.

## Language

- **definer**: a node building its own module (`_target_` plus kwargs), the default, owning trainable parameters.
- **bind node**: a node with `bind`, referencing a definer's module.
- **tied**: `bind.mode='tied'` shares the definer's live parameters. Gradients flow back into them and it adds zero parameters.
- **frozen**: `bind.mode='frozen'` is a separate, non-trainable deep copy of the definer's module, seeded at build, excluded from every optimizer, and slaved to the definer each training step by a `SyncStrategy`.
- **SyncStrategy**: a pluggable `_target_` driving how a frozen clone tracks its source. Core ships `HardSync(every)` and `EmaSync(tau)`, and the set is open.

## Map

- `teia.core.module.sync`: `SyncStrategy`, `HardSync`, `EmaSync`. `TeiaNetModule._build_bind_node`: the build path.
- Codegen: `generate_forward_fn` ([codegen](codegen.md)). Routing and optimizers: [optimization](optimization.md).

## Contracts

### Declaration

```yaml
encoder_target:
  in: batch.next_obs_state           # this call site's own wiring
  out: { feat.x_target: null }
  bind:
    from: encoder                          # source node (a definer in the same graph)
    mode: frozen                           # tied | frozen
    sync: { _target_: teia.core.module.sync.HardSync, every: 500 }   # frozen only
```

`bind = {from, mode, sync?}` is one flat envelope key beside the node's own `in`/`out` ([user_contracts](user_contracts.md)). When `bind` is present `_target_` is omitted, since the architecture is inferred from the source and cannot drift.

- `bind.from` (required): a forward-node name in the same graph. The bind node is ordered in the toposort after its source, as a dependency added on top of `in` edges.
- `bind.mode` (required): `tied` or `frozen`.
- `bind.sync` (required iff `frozen`): a `SyncStrategy` spec. `HardSync(every: N)` copies `state_dict` every N global steps. `EmaSync(tau)` does `θ_t ← (1−τ)·θ_t + τ·θ_s` each step, copying buffers without averaging.

### SyncStrategy

`sync(target, source, *, global_step) -> None`, in place and without grad, in `teia.core.module.sync`. Sync is intrinsic to `TeiaNetModule`, not a callback. The module derives (target ← source) pairs from `bind` edges and runs each frozen node's strategy in `on_train_batch_end`, after the optimizer step, so the target tracks the just-updated source.

### Build

`tied` builds no module and codegen calls the source attribute directly. `frozen` `copy.deepcopy`s the source, sets `requires_grad=False`, registers it as the node's attribute, instantiates the strategy, records the (target, source, strategy) pair, and returns the clone without running `_meta_forward`. Only `tied` runs `_meta_forward`, on the source over the bind node's `in`, to propagate the call site's output shape. A frozen clone keeps the `in_shape` and `out_shape` inherited from the source. The build path branches three ways (no params, lagged frozen params, own params), which is the full space of a node's parameter identity.

### Codegen, optimizer and export

A `tied` node emits `<out> = self.<bind.from>(<in_args>)`, and `frozen` and own nodes emit `self.<name>(...)`, with the same None-tolerance guards. A `tied` node has no attribute and contributes no params, and gradients from its call site accumulate in the shared module under whichever optimizer owns the definer. `frozen` params are `requires_grad=False` and filtered at optimizer build. Bind nodes off the inference path are pruned as training-only, and an on-path `tied` node resolves to the source module.

## Constraints

- `bind.from` MUST be a forward node defined in the same graph, and cycles (including through the bind edge) are rejected by the toposort.
- A `frozen` clone MUST be excluded from all optimizers, and its only update channel is its `SyncStrategy`.
- Target and EMA pairing MUST be derived from `bind` edges, with no separate pairs config and no target-sync callback.
- `tied` nodes MUST NOT register a module attribute. Introspection (`last_layer`) references the definer, not a tied call site.
- Node and `bind.from` names are normalized alike (`/` and `-` to `_`).
- Gated execution, ensembles and dynamic or recurrent topology stay in node code or inside a node's forward, not in the `bind` contract. A future loop can reuse a `tied` module per step.

