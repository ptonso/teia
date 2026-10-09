# Input Bundle and Objective Spec

Related: **Read first** [overview](overview.md), [../../base/structure](../../base/structure.md). **See also** [nodes](nodes.md), [activation_route](activation_route.md), [user_contracts](user_contracts.md), [capture_map](capture_map.md), [../task](../task.md), [../eval](../eval.md).

## Overview

The net graph decomposes into a shared **input bundle** (feature machinery, `batch.* → feat.*`) and one or more **objective** atoms (head or decoder, activation, loss, `feat.*` and `batch.* → pred.*`, `act.*`, `loss.*`). This is the compute-plane basis for multi-input, multi-output runs. Both are optional named regions of the one flat `netmodule` graph, and the flat whole-graph view (`composed.yaml`) is canonical. Monolithic authoring, the whole graph in one preset, stays first-class.

An objective is what to optimize. It declares nothing about evaluation. What the eval graph sees is the netmodule's `capture:` map ([capture_map](capture_map.md)), and metrics are wired in the task's evalmodule, so a new metric never touches the objective.

Scenario: a two-objective netmodule self-namespaces `head_det`, `act_det`, `loss_det` and `head_seg`, `act_seg`, `loss_seg`, and its `capture:` map files each activation's atoms under their own route (`det.*`, `seg.*`), so detection and segmentation are evaluated independently.

## Language

- **input bundle**: the shared representation machinery up to `feat.*`, being stem(s), fusion, and the middle (encoder(s), neck, branch).
- **stem**: the first input node (a conf purpose) mapping `(raw element, value-type) → element vectors` before the structure encoder. A stem is a `one_to_one` node wired `batch.* → feat.*`.
- **objective**: an atomic `{head or decoder + activation route + loss}` producing `pred.*`, `act.*` and `loss.*`.

## Map

- Node leaves: `conf/node/net/{stem,embedder,encoder,decoder,neck,fusion,branch,head,act,loss}/*`, mounted by alias. Modules: `conf/netmodule/<task>/<arch>.yaml`.
- Boundary B: [capture_map](capture_map.md). Composition into a run: [../task](../task.md). Evaluation: [eval](../eval.md).

## Contracts

### Objective region

An objective is a naming convention inside one netmodule, not a config entry. Excerpt of `conf/netmodule/vision-cls/mlp.yaml`:

```yaml
defaults:
  - /node/net/head@head: linear_cls
  - /node/net/loss@loss: cross_entropy
  - /node/net/act@act: softmax
  - _self_
head: { in: feat.x, out: { pred.logits: [num_classes] } }
act:  { in: pred.logits, out: { act.probs: null, act.label: null } }
loss: { in: [pred.logits, batch.cls], out: { loss.cls: null }, weight: 1.0 }
capture: { cls.scores: act.probs, cls.label: post.act.label }
```

Multi-objective netmodules self-namespace their aliases (`head_cls`, `act_cls`, `loss_cls`).

### Composition

A netmodule preset assembles an input bundle plus objectives into one flat graph, and the engine builds and dry-runs it like any netmodule graph ([overview](overview.md)). No separate `input` or `objective` fragment groups ship. The whole graph is one preset.

### Objectives and modality

An objective is `(feature shape it reads, target or output structure it supervises)`, not tied to the input modality. Generic objectives (cls, regression, similarity, vector recon) are structure-agnostic. Structured objectives (detection, segmentation, keypoint, captioning, ASR) are bound to a target structure, and their kernels are homed with that structure, but remain atoms wired by keys.

### Assignment bundles

Where label assignment is intrinsic to the loss, the objective bundles several atomic heads sharing one assignment, route and loss. In assignment-based detectors, the assigner matches predictions to targets by a joint class and box score, so which anchor is positive depends jointly on class and box and cannot be split per head. The heads stay atomic (box, cls, coeff, proto as separate nodes) and the objective is the bundle: `det` is bbox plus cls, `inst-seg` is box, cls and mask, `pose` is box, cls and kpt, `obb` is rbox plus cls. Assignment-free objectives (cls, reg, sem-seg, recon, dist) are single-head atoms. The fused loss and assignment node is not decomposed. Because capture is atomic (`boxes`, `category`, `score` as separate keys), the training-time bundle imposes no bundling on eval ([eval](../eval.md)).

### Val metrics

A metric with a streaming driver may publish into the monitor namespace (`<split>/<name>`) for early stopping and checkpoint selection. The same implementation serves the offline eval, so the monitored and reported numbers cannot drift.

### Decoders and tied weights

A decoder is objective-side. The input bundle ends at `feat.*`, and a decoder is the head of a reconstruction or generative objective (`feat.* → pred.recon`). Tied weights are a node-level concern. A tied encoder and decoder is one encode-decode node or a shared `ParameterNode`, identical whether authored monolithically or split, and fragment boundaries never partition parameters.

## Constraints

- The composed netmodule MUST remain a single flat in/out graph. Input bundle and objective are authoring regions, never runtime walls.
- An objective MUST declare nothing about evaluation. Boundary B is the netmodule's `capture:` map, and metrics, plots and tables are wired in the evalmodule.
- Objective aliases MUST be self-namespaced to avoid collisions across composed objectives.
- The input bundle MUST end at `feat.*`, and terminal outputs (including reconstruction and generation) belong to objectives.
- The fused assignment loss node MUST NOT be decomposed into a shared `assign.*` producer.

