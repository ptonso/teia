# Net Types Spec

Related: **Read first** [layout](layout.md), [../base/net/activation](../base/net/activation.md). **See also** [../docstrings](../docstrings.md), [../core/module/activation_route](../core/module/activation_route.md), [../core/module/capture_map](../core/module/capture_map.md).

## Overview

The typing and prediction contracts every `teia.node.net` component obeys, so a composed module's forward is `torch.compile` and `torch.export` friendly without wrapper shims, and infer, capture and export bind predictions without special cases. [layout](layout.md) fixes where a component lives, and this spec fixes the types crossing its boundaries and the head → activation → loss route.

A head emits logits, the activation owns the logit↔prediction pairing and the distribution or decode, and a loss is a pure backward term. One generic head (`LinearHead`) is reused across routes, its meaning set only by `out` and its paired activation.

Walk-through: a `pred.*` head output flows to its activation, whose `activation` (ONNX-traceable) publishes `act.*` and whose `kernel` (host tail) publishes flat per-sample atoms under `post.<node>.*`. The netmodule's `capture:` map picks which of those leave the net as the task's capture atoms ([../core/module/capture_map](../core/module/capture_map.md)). Losses read the same logits.

## Language

- **node return**: what a node's `forward` outputs, a `Tensor` or a fixed-length tuple or NamedTuple.
- **route**: an activation's `activation` plus `kernel` pair ([../base/net/activation](../base/net/activation.md)). It carries no kind; the task contract types what is captured.
- **frozen bind**: a synced non-trainable clone of another node, the target-network mechanism ([../core/module/bind](../core/module/bind.md)).

## Map

- Placement: [layout](layout.md). Base contracts: [../base/net/nodes](../base/net/nodes.md), [../base/net/activation](../base/net/activation.md).
- Components: `teia.node.net.activation` and `teia.node.net.loss` (flat), and the `<fan>/<assumes>/` node folders.

## Contracts

### Types across a forward boundary

A single-`out` node returns a `Tensor`, and a multi-`out` node returns a fixed-length `typing.NamedTuple` or `tuple[Tensor, ...]`, with arity fixed by the `out` list. Forbidden across a boundary: `dict`, `OrderedDict`, `FeatureDict`, `@dataclass` instances, `Any` fields and variable-length sequences. Config-only structs consumed in `__init__` may stay `@dataclass`. The contract is the same for every `fan` and `assumes`. A node writing `feat.*` returns per-element tokens or a pooled or refined feature, and a node writing `pred.*` returns logits.

Batch types are NamedTuples of `Tensor`, `Optional[Tensor]`, scalar or homogeneous list fields. List fields are export-exempt and used only for `postprocess`.

### Head, activation, loss

Loss `forward(*args) -> Tensor` (scalar per `loss.*`). Activation `activation(*logits) -> dict[str, Tensor]` is pure-tensor and ONNX-traceable, and the inline `kernel(activated, ctx) -> list[dict]` is the host tail returning one flat atom dict per sample (`RAW_PASSTHROUGH` when the activated tensors already are the prediction). Export collects the kernel as bundle Python, so infer equals export and there is no third-party `predict()` bypass. Composite objectives decompose into atomic losses, and a joint criterion that cannot be split may emit several `loss.*` keys from one node.

### Route table

The routes `teia` ships. A multi-head module has several routes, and its capture map draws atoms from any of them. Other distributions add routes under the same contract.

| route | `activation` (ONNX-traceable) | `kernel` (host tail) atoms | typical paired loss reads |
|---|---|---|---|
| `scalar` | identity | none | none |
| `regression` | `{mean, std}` (μ, unit var) | raw passthrough | NLL or MSE over μ logits |
| `classification` | softmax → probs, argmax → label | `scores`, `label` | cross-entropy over class logits |
| `multilabel` | sigmoid → probs | `scores`, thresholded `labels` | BCE over per-label logits |
| `semantic_mask` | softmax → probs, argmax → mask | `mask` resized to the ctx frame | dense cross-entropy |

### Encoder return

Encoders consume the concatenated token set (`feat.*`) or a raw structured tensor and return a single pooled feature as one `Tensor`, owning their own concatenation, `[CLS]` and pooling.

### Target networks and multi-optimizer

A target network is a `bind: {mode: frozen, sync: {_target_: teia.core.module.sync.EmaSync, tau: ...}}` (or `HardSync`) clone read with `detach`. The architecture is declared once on the online node. Multi-optimizer schedules are specified in [../core/module/optimization](../core/module/optimization.md).

## Extending

A new activation declares an inline `kernel` (or `RAW_PASSTHROUGH`) returning content-named atoms. Reuse a route from the table before writing one. A new head is usually not needed, since `LinearHead` with an `out` shape and a paired activation covers most routes.

## Constraints

- `dict`, `OrderedDict`, `FeatureDict`, `@dataclass` returns and `Any`-typed fields MUST NOT cross a `forward` boundary.
- Heads MUST NOT emit activated predictions (softmax, sigmoid, exp, decode). Activation belongs to the activation node.
- A node that wraps a third-party module MUST return `Tensor` or `tuple[Tensor, ...]` from public `forward()` so codegen unpacks correctly.
- An activation MUST NOT declare a prediction kind; its kernel returns flat per-sample atoms. Built-in components follow the [provenance docstring](../docstrings.md) contract.
