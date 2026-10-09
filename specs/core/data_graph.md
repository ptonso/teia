# Data Graph Spec

Related: **Read first** [base/data/nodes](../base/data/nodes.md), [base/batch](../base/batch.md). **See also** [datamodule](datamodule.md), [task](task.md), [module/overview](module/overview.md), [export](export.md), [infer](infer.md).

## Overview

The data graph DSL is the config language that builds a dataset, and the relations between its streams, from atomic data components. It is symmetric to the net graph. Format names such as yolo, coco, labelme, class-dir and clip are recipes over shared components, not packages.

The graph is a flat map of named nodes wired by `in`/`out`, read output-anchored by tracing each `batch.*` back. It is the value of `datamodule` in a composed run. Any node may read any produced key, executed in topological order, with no rigid skeleton.

Walk-through: readers emit indexed streams, a join aligns streams by key, transforms rewrite items, and collate nodes assemble `batch.*`. Phase tags let the engine slice the graph: export takes the `preprocess` subgraph, inference drops `target` and `augment`, and `augment` runs on train only.

Scenario: swapping the label reader and box parser turns a YOLO dataset into a COCO one while the image reader stays unchanged.

## Language

- **data graph**: a flat map of named data nodes wired by keys.
- **workspace**: `stream.*` (indexed streams) → `item.*` (per sample) → `batch.*` (collated). Writers read `pred.*` and `act.*`.
- **node envelope**: `{ _target_, in, out, +optional phase/stage, …params }`, one grammar shared with the `netmodule` graph.
- **provenance attribute**: key, split or class derived from an item's path or columns.
- **slice**: the subgraph the engine extracts by phase for export or inference.
- **rewrite chain**: nodes whose `out_key` re-emits a key they also consume (`letterbox`, augments).

## Map

- Node ABCs, phases and stages: [base/data/nodes](../base/data/nodes.md). Execution: [datamodule](datamodule.md).
- Authoring: node leaves `conf/node/data/*` wired by `datamodule/<task>/<format>` modules ([config](config.md), [task](task.md)).
- Slices: [export](export.md) takes the eval path, [infer](infer.md) takes the rewired graph.
- Graph machinery (`parse_node`, `producer_map`, `_toposort`, contract validation, `build_graph`) lives in `teia.core.datamodule.graph`, and the eval graph reuses it.

## Contracts

### Node envelope

```yaml
<name>: { _target_: <component>, in: <key|[keys]>, out: <key|{key: type}>, +optional phase/stage, ...params }
```

`_target_` is usually injected by an `@alias` group default. Component kwargs are the keys outside the reserved envelope set `{_target_, in, out, phase, stage, weight, optimizer, detach, aux_in, …}`. The composed config is always flat.

The kind (`read`, `transform`, `join`, `reshape`, `collate`, `write`) is inferred from the component's ABC and never written as a wrapper key. `phase` is one of `source`, `preprocess`, `augment`, `target`, `collate`, `writer`, and `stage` one of `plan`, `item`, `stream`, `batch`. Both have per-kind defaults and are written only to override. `stream` is never a default, and a node opts in. The recipes below use a compact shorthand, and the canonical form is the flat envelope with the component bound at `@alias`.

### Provenance and fan-out

Provenance uses `split_from_path`, `class_from_path` (`nested: true`), or a `split` or id column, as `plan` or `target` nodes. Fan-out is native: one `out` consumed by many nodes. `window` is a `plan` reshape (group → order → slice `lookback`/`horizon`).

### Joins

A join aligns samples by key. Aligning an image with its label and aligning an image with a caption from another dataset are the same operator, so a join is fusion and label attachment. Within-item multiplicity (one image, many boxes) is ragged field structure from a parse plus a `ragged` collate, never a join.

### Rewrite chains

A node re-emitting a key it consumes forms a chain rooted at the key's single base producer, ordered by declaration. Downstream consumers bind to the final version, and a join emits its aligned seed keys positionally. Skipping a rewrite (an off-train `augment`) leaves the prior version, so a deterministic consumer downstream of an augment reads the letterbox value on the eval slice and the augmented value on the train slice, with no eval/train fork.

### Recipe: strict YOLO

```yaml
datamodule:
  imgs: { read: files, root: root/, glob: "*.jpg", key: stem, out: stream.img }
  lbls: { read: files, root: root/, glob: "*.txt", key: stem, out: stream.lbl }
  splt: { split_from_path, phase: plan, in: stream.img, out: stream.img.split }
  pair: { join: inner, on: key, in: [stream.img, stream.lbl], out: item.pair }
  img:  { transform: decode_image,     phase: preprocess, in: item.pair.img, out: item.image }
  box:  { transform: parse_yolo_boxes, phase: target,     in: item.pair.lbl, out: [item.xyxy, item.cls] }
  lbox: { transform: letterbox,        phase: preprocess, in: [item.image, item.xyxy], out: [item.image, item.xyxy] }
  fb:   { collate: stack, in: item.image, out: batch.image }
  bb:   { collate: ragged, in: item.xyxy, out: batch.bboxes }
  cb:   { collate: ragged, in: item.cls,  out: batch.cls }
```

COCO or labelme swaps the `lbls` reader and `box` parser. Class-dir is an image reader plus `split_from_path` plus `class_from_path`.

### Recipe: one csv and a schema fan out to typed fields

```yaml
datamodule:
  rows:   { read: csv, source: data.csv, key: col:id, out: stream.row }
  schema: { resolve_schema, phase: plan, in: stream.row, out: plan.schema }
  split:  { split_frame, phase: plan, val_frac: .15, test_frac: .15, in: stream.row, out: stream.row.split }
  num:    { transform: tabular_numerical,   phase: preprocess, schema: plan.schema, in: item.row, out: item.numerical }
  cat:    { transform: tabular_categorical, phase: preprocess, schema: plan.schema, in: item.row, out: item.categorical }
  tgt:    { transform: tabular_target,      phase: target,     schema: plan.schema, in: item.row, out: item.target }
  nb:     { collate: stack, in: item.numerical,   out: batch.numerical }
  cb:     { collate: stack, in: item.categorical, out: batch.categorical }
  tb:     { collate: stack, in: item.target,      out: batch.cls }
```

For time series, insert `{ reshape: window, group: entity, order: time, lookback: 48, horizon: 12 }` before type routing, giving `[B, lookback, F]` and `[B, horizon, F]`, structurally a set of features × a sequence of time. A `reshape` node is a supported export chain root: the chain walk ends there, its pre-seam preprocess steps are marked row-mapped, and the field's payload spec becomes `{"kind": "window", **reshape.payload_spec()}`, so the export client sends a list of rows (see [export](export.md)).

### Recipe: CLIP train (join as fusion)

```yaml
datamodule:
  imgs: { read: files, root: images/,   glob: "*.jpg", key: stem, out: stream.img }
  caps: { read: text_lines, root: captions/, key: stem, out: stream.cap }
  pair: { join: inner, on: key, in: [stream.img, stream.cap], out: item.pair }
  # image preprocess → batch.image ; text preprocess → batch.tokens
```

### Export slicing

Export extracts the `preprocess` subgraph on the eval path to each model-input `batch.*` field, as standalone code or ONNX (preprocess ⊕ model ⊕ postprocess). `target` and `augment` nodes are excluded by phase, and a `reshape` root ends the walk as a reader or join does.

### Inference rewiring and writers

Inference is a small data graph wired to the trained module plus a writer node. CLIP zero-shot classification into a class-dir tree is an image reader, a `literal_list` vocabulary reader, the trained similarity module and a `class_dir` writer, with the class vocabulary reloaded from the trained run (see [infer](infer.md)). For an optional or missing input, an `outer` join emits `present` masks and the input-bundle fusion masks the absent branch.

## Extending

To add a format, compose existing readers, parse transforms and a join under the `dataset` bundle, and add a component only where none exists. New components follow [base/data/nodes](../base/data/nodes.md). Swap at the `@alias` without touching the wiring.

## Constraints

- The graph MUST validate at build: topological sort, and rejection of missing producers, cycles and duplicate produced keys.
- Phase MUST be monotonic: a `batch`-stage node never feeds an `item`-stage node, and an `augment` node never sits on the export or eval path.
- The structural type at each `collate` seam MUST be checked against the consumer.
- A node's kind MUST be inferred from its ABC, never declared as a wrapper key.
- Stateful transforms MUST fit on train only, with state persisted and reloaded, never re-fit on eval or infer.
- Every `batch.*` field MUST be structure-typed.

