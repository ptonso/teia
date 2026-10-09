# Capture Map Spec

Related: **Read first** [overview](overview.md), [activation_route](activation_route.md), [../../base/task](../../base/task.md). **See also** [objective](objective.md), [../eval](../eval.md), [../infer](../infer.md).

## Overview

The capture map is the netmodule's declaration of boundary B. It is one mapping from `route.atom` to the net-graph key that produces it. Capture, streaming validation metrics and inference all read predictions through it, so the net graph's internal names (`act.*`, `post.*`) never leak into eval or writer configs.

Walk-through:
1. The netmodule ends with `capture: {cls.scores: act.probs}`.
2. On a validation or test step, `CaptureMap` reads `act.probs` from the step outputs and yields the atom `capture.cls.scores`.
3. It also yields the contract's target fields from the batch, for example `batch.cls` ([base/task](../../base/task.md)).
4. Streaming metrics consume those atoms live. The test pass writes them to the capture store, and `teia infer` hands them to writer nodes.

Scenario:
- Detection maps `det.boxes: post.act.boxes`, `det.score: post.act.score` and `det.category: post.act.category`.
- The activation's `postprocess` runs NMS and returns per-sample atom dicts.
- The map concatenates them into flat rows and derives `det.sample_idx`, because the contract declares those atoms `Ragged(index="det.sample_idx")`.

## Language

- **capture map**: `netmodule.capture`, `{ "<route>.<atom>": "<act.* | post.<alias>.<atom>>" }`.
- **route**: the first segment of a capture key, the namespace a prediction is filed under (`cls`, `det`, `student`). It is chosen by the contract, not by node names.
- **post key**: `post.<alias>.<atom>`, one atom of the `postprocess` output of the activation node mounted as `<alias>`.
- **atom dict**: the flat `{key: array}` that `CaptureMap` yields per step. It holds `capture.<route>.<atom>` and the contract's target `batch.*` fields.

## Map

- Parsing and source checks: `teia.core.module.net` (`TeiaNetModule.capture_map`).
- Extraction: `teia.core.capture.extract.CaptureMap`. Consumers: `teia.core.capture.callback` (store writer and streaming metrics) and `teia.core.infer`.
- The contract the map must cover: [../../base/task](../../base/task.md).

## Contracts

### Map form

```yaml
capture:
  cls.scores: act.probs                 # activated tensor, one row per sample
  det.boxes: post.act.boxes             # postprocess atom, ragged
  det.score: post.act.score
  det.category: post.act.category
```

Each value is either:
- an `act.*` key declared by an activation node's `out`, or
- `post.<alias>.<atom>`, where `<alias>` is an activation node.

Keys are `route.atom`. A netmodule with no `capture` map produces no atoms and cannot be evaluated or inferred.

### Postprocess atoms

When the map references `post.<alias>.*`, `CaptureMap` calls that activation's `postprocess(activated, ctx)`, which returns one flat `{atom: array}` dict per sample ([activation_route](activation_route.md)). The contract's spec for the capture key decides how per-sample values are assembled:
- **`Tensor`**: stacked along the batch dim.
- **`Ragged`**: concatenated. Its `index` atom is derived from the per-sample row counts and is not listed in the map.
- **`Blob`**: kept per sample for the file lane.

### ctx

The `ctx` passed to `postprocess` is `evalmodule.ctx` during capture and streaming validation, and `infer.ctx` during inference, each merged with per-batch restore values the datamodule provides only at inference (`ori_shape`, `ratio_pad`). Capture therefore stays in model space, matching the captured `batch.*` targets.

### Targets

`CaptureMap` adds every `batch.<field>` the contract marks `target=True`, read from the same batch. There is no second ground-truth path.

## Constraints

- Every contract `capture` atom MUST be covered by the map, except `Ragged` index atoms, which are derived. This is checked by `TaskContract.check_net` at pre-flight.
- A map value MUST name an `act.*` out-key or a `post.<alias>.*` of a declared activation node. Anything else fails at build with the offending key.
- `CaptureMap` MUST be the only producer of `capture.*` atoms, shared by streaming validation, the store writer and inference.
- Core MUST NOT interpret an atom's meaning. Assembly depends on the contract's spec type only.

