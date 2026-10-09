# Multi-Optimization Spec

Related: **Read first** [overview](overview.md), [user_contracts](user_contracts.md). **See also** [codegen](codegen.md), [nodes](nodes.md), [bind](bind.md), [../runtime](../runtime.md).

## Overview

`TeiaNetModule` supports several parallel optimization routes, one backward and step schedule per named optimizer, as a config-declared, domain-neutral feature with no per-task or per-algorithm Lightning subclass. One root cause unifies the cases: a generator and discriminator, separate actor and critic networks with target networks, and any objective whose terms need disjoint parameters under different schedules.

The module always runs Lightning manual optimization, single optimizer included, so there is exactly one training-step code path. The single-optimizer flat form is shorthand that desugars into a one-element `optimizers` list named `default`.

Walk-through: at construction the module sets `automatic_optimization = False`, parses the `optimizers` list, and builds a loss-to-optimizer route map and a node-to-optimizer ownership map. At each training step the forward runs once, per-term losses are grouped by route, and each active optimizer backwards its own sum, clips, steps and zeroes.

Scenario: a generator with a discriminator declares optimizers `gen` and `disc`, an adversarial node with `optimizer: disc`, and loss out-keys `loss.g_adv: gen` and `loss.d_adv: disc`. The discriminator starts at step 50001.

## Language

- **optimization route**: one optimizer's backward and step schedule, reaching a disjoint parameter set.
- **`optimizers` list**: the ordered, named optimizer specs at netmodule root, where index 0 is `default`.
- **param ownership**: which optimizer owns a node's parameters (the node's `optimizer` field).
- **loss route**: which optimizer backwards a loss term (the `out` map value).
- **detach**: a node's `detach` field, the stop-gradient primitive applied before the node's forward.

## Map

- `teia.core.module.net.TeiaNetModule` (`configure_optimizers`, `training_step`, `_build_one_optimizer`). `PipelineNodeRecord` carries `optimizer`, `loss_routes`, `detach`, `aux_in`, `last_layer`.
- Optimizer implementations live in the component libraries as plain `torch.optim.Optimizer` subclasses. Adversarial math (hinge, warmup, adaptive weight) lives in the component, not core.
- Target-network syncing: [bind](bind.md).

## Contracts

### The optimizers list

```yaml
netmodule:
  _target_: teia.core.module.net.TeiaNetModule
  _recursive_: false
  optimizers:
    - name: gen
      _target_: torch.optim.AdamW
      lr: 1.0e-4
      weight_decay: 1.0e-2
      decouple_weight_decay: true              # per-optimizer norms+bias no-decay split
      start_step: 0                            # default 0
      every: 1                                 # default 1
      grad_clip: { val: 1.0, algorithm: norm } # manual mode ignores trainer clip
      scheduler: { ... }                       # optional per-optimizer lr_scheduler_config
    - name: disc
      _target_: torch.optim.Adam
      lr: 1.0e-4
      start_step: 50001
```

Per entry: `name` (required with more than one optimizer), `_target_` plus flat kwargs, `decouple_weight_decay` (default true, splitting `ndim>=2` decay from `ndim<2` no-decay within this optimizer's params), `start_step` (default 0), `every` (default 1, stepping when `global_step % every == 0`), `grad_clip` (`{val, algorithm}`, required here because Lightning ignores `trainer.gradient_clip_*` in manual mode), and `scheduler` (a `lr_scheduler_config` stepped manually).

The flat form (`optimizer_target`, `lr`, `weight_decay`, `lr_scheduler`) is exactly equivalent to `optimizers: [{name: default, ...}]`.

### Routing

A node's `optimizer` field (a name or an int index, absent meaning 0) sets ownership. On a forward node the named optimizer owns its params, and the default optimizer absorbs all unclaimed params. On a loss node it sets ownership for any params the node carries (a composite adversarial node owning a discriminator) and the default route for its loss out-keys. A loss `out` map value names the optimizer that backwards that term:

```yaml
adversarial:
  optimizer: disc                # param ownership + default route
  in: [pred.reconstructed, batch.image]
  out:
    loss.g_adv: gen               # backwarded on the gen route
    loss.d_adv: disc              # backwarded on the disc route
```

Route resolution: explicit `out` value, then the node's `optimizer` field, then index 0.

### Stop-gradient and adaptive weights

`detach: [key, ...]` `.detach()`-es those in-keys before the node's forward, in both `_generated_forward` and `_generated_losses` (critic targets, distillation, adversarial wiring). Loss-node-only hooks: `aux_in: [loss.recon, loss.perc]` passes already-computed loss-term tensors into the node, the only sanctioned exception to the loss-input rule (`_generated_losses` computes those terms first, keeping their graph), and `last_layer: <forward-alias>` resolves to that alias's last parameter for grad-norm balancing.

### Training step

At `configure_optimizers`, each named optimizer gathers the union of its nodes' params (default absorbs the rest), applies its decay split, and returns the Lightning list form. At `training_step` (generic, not generated), forward runs once and `_generated_losses` produces per-term values. Terms are grouped by route, and per optimizer in declared order, when `global_step >= start_step` and `global_step % every == 0`: `toggle_optimizer`, `manual_backward(route_sum, retain_graph=<True for all but the last active route>)`, optional `clip_gradients`, `step`, `zero_grad`, `untoggle_optimizer`. `accumulate_grad_batches` and scheduler stepping are honored manually. `validation_step` and `test_step` are unchanged and log the summed `loss`.

### Closure and second-order optimizers

Some optimizers re-evaluate the objective at candidate parameters within one update (line search, trust regions), which the plain `manual_backward → step()` path cannot express. They set `requires_closure = True`, and the manual loop then drives them with a closure:

- An optimizer with `requires_closure` is active on cadence regardless of whether it contributes a `loss.*` route term, since its objective is computed inside `step`. Its companion routes (a value-function Adam route) are ordinary routes.
- The loop builds `closure = lambda: opt.objective.evaluate(self, batch)` and calls `opt.step(closure)`, with no `manual_backward` and no route grad clip. The optimizer owns its gradient computation and may call the closure repeatedly.
- The `objective` is an ordinary `{_target_: ...}` object kwarg on the `optimizers` entry. `_build_one_optimizer` instantiates any object-spec kwarg before constructing the optimizer, so collaborators are config-wired and the domain math is the provider's.

The seam is generic: a second-order optimizer such as a trust-region method implements `requires_closure` plus an `objective` protocol (`evaluate(module, batch)`, `new_step()`) in its own module. The same seam admits a plain `torch.optim.LBFGS` with a loss-recomputing closure.

## Extending

To add a route, append a named entry to `optimizers`, set `optimizer` on the nodes it owns, and name it in the `out` map of the loss terms it backwards. To add a second-order optimizer, implement `requires_closure` and an `objective` collaborator and wire it as an object-spec kwarg.

## Constraints

- `optimizers` MUST be a reserved netmodule-root key, an ordered list where index 0 is `default`.
- Param ownership MUST be by node, and the norms and bias no-decay split MUST be applied per optimizer.
- Loss route MUST resolve by `out` value, then the node's `optimizer` field, then index 0.
- Gradient clipping, accumulation and scheduler stepping are netmodule-owned in manual mode. With a shared forward, every backward except the last active route MUST use `retain_graph=True`.
- `aux_in` is the only sanctioned cross-loss read, and only for gradient-norm balancing.
- A `requires_closure` optimizer MUST be driven by `step(closure)`, MUST be active on cadence even with no `loss.*` route, and MUST take its `objective` as a config object-spec. Core stays domain-neutral.
- Adversarial math MUST NOT live in core, which provides only routing, the manual loop, and the `aux_in` and `last_layer` hooks.

