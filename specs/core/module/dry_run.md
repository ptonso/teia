# Module Dry Run Spec

Related: **Read first** [overview](overview.md). **See also** [user_contracts](user_contracts.md), [nodes](nodes.md), [codegen](codegen.md), [activation_route](activation_route.md), [../../base/batch](../../base/batch.md).

## Overview

The shape-aware construction pass that `TeiaNetModule.configure_from_datamodule(dm)` performs before training. It builds a synthetic workspace, orders and constructs the graph, resolves dim-refs, and fails clearly on invalid wiring, so config errors surface at build time and not as runtime crashes.

Walk-through: the module asks `dm.batch_meta()` for field shapes and materializes zero tensors as `batch.*`. It walks forward aliases in topological order, injects shapes, instantiates each node and runs it once on synthetic tensors to record the actual output shapes. Activation and loss aliases are instantiated afterwards.

## Language

- **synthetic workspace**: zero tensors materialized from `batch_meta()` integer shapes, keyed `batch.*`.
- **meta forward pass**: running each forward node on synthetic tensors to propagate real output shapes.
- **dim-ref**: an `out` shape string resolved against `dm` at build time. It may appear inside a composite shape such as `[2, action_dim]`.

## Map

- `teia.core.module.net.TeiaNetModule.configure_from_datamodule`. The datamodule side of the protocol is in [datamodule](../datamodule.md) and [base/batch](../../base/batch.md).
- Hook ordering per node role: [nodes](nodes.md). The generated functions: [codegen](codegen.md).

## Contracts

### Workspace

The module materializes synthetic tensors for tuple or list integer shapes from `dm.batch_meta()` (`batch.image`, `batch.cls`, `batch.bboxes`). The smoke test fills any remaining declared field with its NamedTuple default, or a type-appropriate empty (`[]`, `{}`, `()`) for list, dict and tuple annotations, falling back to `torch.zeros(B)`.

### Per-alias build

Forward aliases run in topological order derived from `in`/`out`, with declaration order only a deterministic tie-breaker. Per forward alias: copy the composed config, strip the envelope keys, inject `in_shape` and `out_shape` into a `TeiaNode` (resolving any dim-ref against `dm` first, so `pred.logits: [num_classes]` becomes `(int(dm.num_classes),)`), Hydra-instantiate, and run the meta forward, which updates the workspace with the actual output tensors.

Activation and loss aliases are excluded from the forward graph and instantiated after it. They are not run in the forward meta pass, and a `frozen` bind clone is not meta-forwarded either. Loss `configure_from_datamodule` runs after the full build ([nodes](nodes.md)). An activation's `activation` is exercised by a separate prediction-route dry run.

### out interpretation

`null` is inferred from meta execution. An int list or tuple is passed as `out_shape`. A dim-ref string, or any string inside an int or string list, is resolved against `dm` and expanded into the final `out_shape`. Map-key order is the multi-output order.

### Loss and activation rules

Role comes from the `out` namespace. A loss `in` is only `batch.*` and `pred.*`, with weights from `weight` applied by the generated loss function, not the constructor. An activation `in` is only `pred.*`, implementing `activation` plus `postprocess` or `kernel`, with the route keyed by activation-node name ([activation_route](activation_route.md)).

### No-loss modules

A netmodule with no loss node is not rejected: only existing loss nodes are validated. `++trainer.fast_dev_run=true` then completes with no loss term, and the summary can show `Trainable params: 0`.

## Constraints

Build-time validation MUST fail clearly on:

- any `batch.*` key missing from `dm.batch_meta()`;
- a missing producer for any non-`batch.*` `in` (`Node 'head': in_key 'feat.encoded' is not defined...`);
- a duplicate produced `out_key` (`Duplicate out_key 'pred.raw' in module graph.`);
- a cycle in the forward graph (`Cycle detected in module graph involving: encoder, neck`);
- an invalid `out` shape declaration, or an unresolvable dim-ref;
- an exception raised by a node during the meta forward (`Meta dry-run failed for node 'head': ...`).

Generic heads MUST get their width from `out` dim-refs resolved here, and Hydra `${...}` interpolation MUST NOT be used for dataset dims.

