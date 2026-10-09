# Activation Route Spec

Related: **Read first** [overview](overview.md), [../../base/net/activation](../../base/net/activation.md). **See also** [capture_map](capture_map.md), [user_contracts](user_contracts.md), [codegen](codegen.md), [nodes](nodes.md), [../infer](../infer.md), [../export](../export.md), [../eval](../eval.md).

## Overview

The activation node owns the standardized **prediction route**: one torch-defined prediction path shared by training, inference, capture and ONNX export. There is no per-task prediction code, and export needs one generic activation-driven driver.

Heads emit logits into `pred.*` and carry no task semantics. The activation node reads `pred.*` and writes `act.*`. A loss is a pure backward term and owns no prediction. The prediction order everywhere is `forward → activate → postprocess`.

Walk-through: `activation` maps logits to activated outputs (pure tensor, ONNX-traceable). `postprocess` is the host-side tail (NMS, decode, rescale) that delegates to a `kernel`, and the same kernel is collected into the export bundle. `postprocess` returns one flat atom dict per sample, published as `post.<alias>.<atom>`, and a multi-head module has several activation nodes. Which atoms leave the net graph is declared by the netmodule's `capture:` map ([capture_map](capture_map.md)).

## Language

- **head**: emits logits (raw real tensors) into `pred.*`.
- **activation**: owns the route over a head's logits, reading `pred.*` and writing `act.*`.
- **loss**: a pure backward term `forward(*in_keys) -> Tensor`.
- **activation stage**: logits to activated outputs, pure-tensor and ONNX-traceable.
- **decode**: the host-side tail (`postprocess` delegating to a `kernel`), not ONNX-traceable.
- **post key**: `post.<alias>.<atom>`, one atom of the `postprocess` output of the activation node mounted as `<alias>`.
- **atom dict**: the flat `{atom: array}` a kernel returns for one sample (`{"boxes": [K,4], "score": [K], "category": [K]}`).
- **dynamic axes**: `DYNAMIC_AXES`, the activation-declared ONNX output axes whose extent varies per input ([export](../export.md)).

## Map

- Base classes: [base/net/activation](../../base/net/activation.md). Assembly: [codegen](codegen.md) and `teia.core.module.net`.
- Consumers: [capture_map](capture_map.md) (the only reader of `act.*` and `post.*` outside the net graph), which feeds [eval](../eval.md) and [infer](../infer.md); [export](../export.md) (traces `forward + activation`, collects each kernel).

## Contracts

### Inputs and outputs

Inputs are the `pred.*` logits (the node's `in` is restricted to `pred.*`) plus a prediction `ctx` for `postprocess`, meaning runtime params and per-sample metadata not expressible as pure tensors (thresholds, `ori_shape`, class counts, keypoint shapes). Training targets are consumed only by losses. Outputs: `activation` returns `dict[str, Tensor]` written under the node's `act.*` out-keys, and `postprocess` returns final predictions (decoded boxes, masks, keypoints or labels with scores).

### Base class summary

`BaseActivation` carries an inline `@staticmethod kernel(activated, ctx) -> list[atom dict]`, `DYNAMIC_AXES`, a `RAW_PASSTHROUGH` opt-in, `activation(*logits)` (default identity) and `postprocess(activated, ctx)` (delegates to `kernel`, else identity if `RAW_PASSTHROUGH`, else raises). A subclass defines `kernel` inline and never overrides `postprocess` or assigns the attribute from an external function. The full contract is in [base/net/activation](../../base/net/activation.md).

### Postprocess output

`kernel(activated, ctx)` returns a list with one flat atom dict per sample, with content-named atoms and no nesting. The module publishes it as `post.<alias>.<atom>`, and [capture_map](capture_map.md) stacks or concatenates the atoms according to the task contract's spec types. An activation declares no prediction kind, and nothing downstream dispatches on what its output means.

`DYNAMIC_AXES: ClassVar[dict[str, dict[int, str]]]` maps an activated out-atom to the ONNX axes whose extent varies per input, for example `{"boxes": {1: "detections"}}`. The default is `{}`.

### Multiple routes

A module may declare several activation nodes. Each one's `postprocess` runs independently, and the `capture:` map files their atoms under the routes the contract names. `predict_step` returns every activation's atom dicts, and infer and capture read them only through the map.

### Module assembly

`TeiaNetModule` builds an activation assembly like `_generated_forward` and `_generated_losses` ([codegen](codegen.md)): for each activation node it calls `act.activation(*pred_logits)` and merges the dict under the activation alias. `postprocess` runs on the host after the (optionally exported) graph, once per route. Activation and loss nodes stay instantiated after `configure_from_datamodule`. `predict_step`, driven by `trainer.predict`, runs `forward → activate → postprocess` over all nodes, with the host injecting the per-batch `ctx` through a `_predict_ctx_builder`.

### Route across stages

```text
train:   forward(batch) -> logits ; loss.forward(logits, targets) -> {'loss.<name>': scalar}  (summed)
predict: forward -> logits ; act.activation(logits) -> activated ; act.postprocess(activated, ctx)
         -> [atom dict per sample] -> post.<alias>.<atom> -> capture map
export:  trace(forward + act.activation) -> ONNX {logits..., activated...} ;
         collect act.kernel -> bundle/runtime/postprocess.py (one per route)
```

### Forward nodes and losses

All forward nodes (heads, decoders, necks, encoders) emit logits only, and the route lives in the activation node. A loss reads `pred.*` and MAY apply a fixed internal link (BCE-with-logits sigmoid, or a recon squash to the target pixel space) but never reads `act.*` and never emits a prediction. When a loss compares against a normalized target, its squash MUST use the same strategy as the activation node and the datamodule, single-sourced by the task recipe. Auxiliary regularizers (KL, perceptual, semi-supervised) are ordinary loss nodes, with no empty `activation() -> {}` stubs.

### Wrapped backends

Tabular activations are mostly `activation`-only link functions (`postprocess` identity for regression, classification and distributions, meaningful for forecast and survival). A detector activation node exposes `activation` (torch decode) and `postprocess` (confidence gather, NMS, rescale, mask and keypoint decode), normalizing a third-party detector into this contract. The export wrapper in `teia.core.export.onnx` traces `forward` then each `act.activation`, emitting `(*logits, *activated)` as named ONNX outputs.

## Constraints

- `activation` MUST be pure-tensor and ONNX-traceable: no data branching, no numpy, no dynamic-shape ops. Its `in` MUST be `pred.*` only.
- Decode is the non-traceable tail and is collected as Python, never traced.
- `kernel` MUST be export-portable per [export](../export.md): a pure `@staticmethod` taking only `(activated, ctx)`, with no `self`, closures or teia-core imports, importing only numpy, torch, torchvision, PIL, the stdlib and other kernels. `collect` fails fast on any violation, and the assembler self-verifies the emitted decode equals `postprocess`.
- Every concrete activation MUST declare exactly one of `kernel` or `RAW_PASSTHROUGH = True`.
- `kernel` MUST return one flat atom dict per sample, with no nested records.
- An activation MUST NOT declare a prediction kind. Consumers read atoms through the `capture:` map only.
- ONNX MUST carry both logits and activated outputs, and the bundle carries `postprocess` as Python.
- There MUST be one prediction path: no task-branched prediction code and no framework-native predictor bypass.

