# Activation and Loss Base Spec

Related: **Read first** [nodes](nodes.md). **See also** [core/module/activation_route](../../core/module/activation_route.md), [core/module/capture_map](../../core/module/capture_map.md), [core/module/objective](../../core/module/objective.md).

## Overview

The base prediction-route and loss contracts that every concrete activation and loss implements. An activation turns raw predictions into activated outputs and, through one canonical `kernel`, into final predictions. A loss reads explicit `pred.*` and `batch.*` inputs and returns scalars under `loss.*`. Both stay torch-only and domain-neutral.

Walk-through: a head emits `pred.*` logits, the activation for that route maps them to activated tensors and `postprocess` turns them into predictions using its `kernel`, and the loss consumes `pred.*` plus `batch.*` targets. The kernel is collected verbatim into the export bundle, so infer and export cannot drift.

## Language

- **atom dict**: the flat `{atom: array}` one sample's decode returns. See [activation_route](../../core/module/activation_route.md).
- **kernel**: the inline `@staticmethod` holding the postprocess math.
- **raw passthrough**: an explicit opt-in for a route whose activated outputs already are the prediction.

## Map

- `teia.base.net`: `BaseActivation`, `BaseLoss`, and `teia.base.net.action.ActionActivation`.
- The prediction route: [activation_route](../../core/module/activation_route.md). Concrete activations and losses live in `teia.node.net` (`activation/`, `loss/`).

## Contracts

### BaseActivation

`activation(*pred_logits) -> dict[str, Tensor]` plus `postprocess(activated, ctx) -> list[dict[str, Any]]`, one flat atom dict per sample. `DYNAMIC_AXES` (default `{}`) declares the activated atoms whose ONNX axes vary per input. There is no prediction-kind attribute. What leaves the net graph is decided by the netmodule's `capture:` map ([capture_map](../../core/module/capture_map.md)), and how it is stored by the task contract ([task](../task.md)).

### postprocess and kernel

`postprocess` is final plumbing and MUST NOT be overridden. It delegates to a `kernel: ClassVar[(activated, ctx) -> predictions]` `@staticmethod` defined inline on the concrete class. It is never assigned from an external function, and a shared implementation is called from the kernel body. `params(self) -> dict` supplies the repr-round-trippable literals the kernel needs at export, mirroring the io `kernel`/`params` convention in [data/nodes](../data/nodes.md). A route with no kernel sets `RAW_PASSTHROUGH = True`, and `postprocess` then slices the activated tensors along the batch dim into per-sample atom dicts. Otherwise `postprocess` fails fast, closing the old silent-identity fallback.

### BaseLoss

A loss reads explicit `pred.*` and `batch.*` inputs and returns scalar tensors under `loss.*`. A loss owns no prediction route.

### ActionActivation

The torch-only sub-protocol for activations that sample an action. It extends `BaseActivation` with `sample(*logits, deterministic, generator) -> ActionSample`. Concrete action activations live in the component libraries.

## Extending

To add a route, subclass `BaseActivation` (or `ActionActivation`) in `teia.node.net.activation` and write an inline `kernel` with `params()` returning content-named atoms. Map the atoms the task needs in the netmodule's `capture:` map. No core change is needed.

## Constraints

- Base activation and loss code MUST stay torch-only and domain-neutral.
- Every concrete activation MUST declare exactly one of a `kernel` staticmethod or `RAW_PASSTHROUGH = True`, and `postprocess` raises otherwise.
- `postprocess` MUST NOT be overridden, and MUST return one flat atom dict per sample.
- An activation MUST NOT declare a prediction kind.
- A kernel MUST be defined inline on its class.

