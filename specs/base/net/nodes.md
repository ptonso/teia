# Net Node Spec

Related: **Read first** [batch](../batch.md). **See also** [activation](activation.md), [core/module/nodes](../../core/module/nodes.md).

## Overview

`teia.base` is the torch-only floor shared by the engine and the public component library. For the net graph it owns the node base classes, and no concrete architecture. `TeiaNode` is the base for forward nodes that need shape injection. `ParameterNode` is the helper for a component that needs a registered tensor without importing engine code.

Walk-through: the orchestrator builds each node in topological order. It injects `in_shape` (and optionally `out_shape`), calls `build_module(...)`, then runs two optional build-time hooks. The node's `forward(*args)` returns a tensor or a fixed-length tuple of tensors.

## Language

- **shape injection**: the engine passing `in_shape` and `out_shape` before `build_module`.
- **producer**: the node whose `out` key is another node's `in` key. `batch.*` keys have no producer node.

## Map

- `teia.base.net`: `TeiaNode`, `ParameterNode`, `BaseLoss`, `BaseActivation` (see [activation](activation.md)).
- Orchestration and node envelope: [core/module/nodes](../../core/module/nodes.md).
- Concrete MLPs, encoders, heads, losses and activations live in `teia.node.*`.

## Contracts

### TeiaNode

Subclasses implement `forward(*args)` and return tensors or fixed-length tensor tuples. The engine passes `in_shape` and optional `out_shape` before `build_module(...)`.

### Build-time hooks

`TeiaNode`, `BaseLoss` and `BaseActivation` take part in two hooks that the orchestrator (`TeiaNetModule`) calls once after the node is constructed:

- `configure_from_datamodule(dm) -> None`: datamodule-derived sizing (categorical cardinality, number of classes). It is not defined on the base classes. The orchestrator looks it up with `hasattr` and calls it when a datamodule is available.
- `configure_from_producers(producers: dict[str, TeiaNode]) -> None`: called after every node producing one of this node's `in` keys is built. `producers` maps each `in` key to its producing node, and `batch.*` keys are absent. This is how a loss or activation paired with a wrapped-backbone head gets a build-time handle to that head (for example to call `init_criterion()`) instead of passing a non-Tensor through `forward`. It has a no-op default on the base classes.

Neither hook adds config surface: no new `_target_` and no new conf key.

### Holding a producer handle

`BaseLoss` and `BaseActivation` are `nn.Module`s. A bare `self._head = head` would register the head as a second submodule, duplicating its parameters in `module.parameters()` and `state_dict()` and corrupting the optimizer and checkpoints. Store the handle unregistered, either `object.__setattr__(self, "_head", head)` or a tuple `self._head = (head,)`, and use one form consistently. A criterion built fresh from a head factory (`head.init_criterion()`) is not a shared handle and may be assigned normally.

## Extending

A new node subclasses `TeiaNode` in `teia.node.net.<fan>/<assumes>/` (placement is owned by the `node/net/README.md` taxonomy). Reuse an existing node before writing one, and size from `in_shape` or the two hooks, never from a domain-specific dim name.

## Constraints

- `teia.base` MUST NOT import project packages above `teia.base`.
- Shape injection MUST stay generic: no domain-specific dim names or task branches in `teia.base`.
- A handle obtained through `configure_from_producers` MUST be stored unregistered.
- Concrete component inventories live in `teia.node.*`.

