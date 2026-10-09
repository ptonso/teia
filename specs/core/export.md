# Export Spec

Related: **Read first** [infer](infer.md), [data_graph](data_graph.md). **See also** [base/data/nodes](../base/data/nodes.md), [base/net/activation](../base/net/activation.md), [module/overview](module/overview.md), [module/codegen](module/codegen.md), [module/activation_route](module/activation_route.md).

## Overview

`teia export` produces a standalone, ONNX-first inference bundle from a saved run. The bundle runs outside the training runtime with no `teia` import, and its decode is byte-identical to what infer and capture run. One exporter (`teia.core.export`) handles every task. Structure, objective, regime and format are wired through the data and net graphs, not through per-task export code.

The runtime package is collected, not templated. Routing happens once at export time, so the bundle is model-specific: one decode and one set of preprocess chains, with no per-task dispatch. Training and export run the identical function by construction, because every node's `kernel` is both executed at infer and capture time and collected verbatim into the bundle.

Walk-through: export rebuilds the module from saved config and weights. It walks each model-input `batch.<field>` backward through the data graph to its data-loading root, collecting one `PreprocessStep` per node. It traces the module to ONNX, collects each selected activation route's decode kernel, bakes parameters as literals, writes the runtime files, and self-verifies both directions.

## Language

- **export kernel** (*kernel*): a pure, teia-free, module-level function performing one slice of preprocess or decode, written so it can be copied verbatim into a bundle.
- **collect**: `collect_callable` reading a kernel via `inspect.getsource`, resolving free names against `fn.__globals__`, and inlining the transitive closure of allowed helpers.
- **assemble**: `teia.core.export.assemble` walking the resolved bundle contract and emitting exactly the preprocess and decode this model uses.
- **bake**: writing model-specific parameters (labels, thresholds, conf/iou, resolved sizes, fitted means, stds, vocabs) as literals (`CTX`, `PARAMS`).
- **preprocess chain**: the ordered `PreprocessStep`s (kernel, baked params, owner, `row_mapped`, `via_bridge`) one model-input field resolves to.
- **boundary / exempt node**: an data node or activation with no `kernel` (a reader, a `target` or `source` parse, an augment, a `RAW_PASSTHROUGH` route). Never collected and never blocks export.
- **restore ↔ ctx**: a preprocess kernel's second return (letterbox `ratio_pad`, `ori_shape`) merges forward into the decode `ctx`. Kernels read restore values as per-sample lists, as infer passes batch fields, so the bundle's `decode` wraps each single-payload value in a one-element list.
- **payload spec**: what an export client sends for one field, `{"kind": "value"}` or `{"kind": "window", **reshape.payload_spec()}` for a `Reshape`-rooted chain, sent as a list of rows.

## Map

- `teia.core.export`: `bundle` (`BundleContract`), `collect` (`collect_callable`, `KERNEL_PACKAGES`), `preprocess_slice` (`resolve_data_graph_chain`, `resolve_interactive_chain`), `assemble` (`write_bundle_runtime`), `kernels` (teia-independent helpers).
- Kernel contracts: [base/data/nodes](../base/data/nodes.md) (preprocess `kernel` and `params()`) and [base/net/activation](../base/net/activation.md) (activation `kernel`).
- The graph being sliced: [data_graph](data_graph.md).

## Contracts

### Single-sourced kernels

Every activation node's `kernel` and every preprocess data node's `kernel` (`(value, params) -> (value, restore)`) run at infer and capture time and are collected into the bundle, so infer equals export by construction. A route with no kernel and `RAW_PASSTHROUGH = True` ships an identity tail. A route with neither fails fast. `params()` supplies literals a bare staticmethod cannot read off `self`.

A kernel is always an inline `@staticmethod def kernel(...)` on its class, named `kernel`, with no exception for shared logic. Sharing happens by the body delegating to a shared helper, never by assigning the attribute. `collect_callable` permits exactly one decorator on the entry (`@staticmethod`), strips it, and renames the entry to `<lowercaseclassname>_kernel` to avoid collisions in one bundle. A shared helper is a dependency, not the entry, and keeps its name. It MUST live in an allowlisted module (`collect.py::KERNEL_PACKAGES`): `teia.core.export.kernels` (teia-independent), and the node-side private math modules for shared decode (`teia.node.net._activation_utils`), plus any module another distribution registers under the `teia.export_kernels` entry-point group.

### Chain resolution

`resolve_data_graph_chain` walks a model-input `batch.<field>` backward through preprocess transforms, across the one `Collate` seam, and through any batch-stage rewrite, to its data-loading root. The root producer's key names the payload key, and the root's kind names the payload spec. A root is a `read` or `join` node, a `Reshape` node, or a `target` or `source` transform that supplies an input field (`ParseLabelme` yielding `item.img_path`). Each terminates the walk like a reader, and a step on the input path must itself be `phase=preprocess`. The interactive regime has one counterpart, `resolve_interactive_chain`, which queries the datamodule's single `ObsTransform` per observation key (`kernel_for(obs_key)`, `params_for(obs_key)`).

### Windowed export

A chain reaching a `Reshape` node is a windowed input. The walk terminates there, every pre-seam item-stage step is marked `row_mapped = True`, and the payload spec is `{"kind": "window", **reshape.payload_spec()}`. The assembler loops the row-mapped steps once per client row, then applies the collate-seam step once over the whole transformed list.

### Multi-input and extras

Inputs are the data graph's model-facing `batch.*` fields (`resolve_forward_batch_fields`, reading `_pipeline_records`). The ONNX wrapper is one fixed-arity `nn.Module` over however many fields the model consumes, with no per-modality adapters. The same list backs `validate_export_kernels`, a dry-run-time check (`TeiaNetModule.configure_from_datamodule`) that fails fast on a live forward-chain node missing a `kernel`, including for `Reshape`-rooted chains, so a broken path surfaces at `teia train` or `teia test` start.

Any data node may declare `export_artifacts() -> dict[str, dict]`, JSON-able extras keyed by filename stem (a fitted schema or vocab). The exporter collects them into `contract.extra_metadata`, and each is owned by the component producing the state.

### Route selection

`export.routes: [...]` (default all `_activation_records`) selects which routes are traced and decoded, for example restricting a recipe to one route. The raw `forward()` output is not filtered by route, so an unselected route's logit may remain as an unused ONNX output, and decode is what is pruned.

### BundleContract

`teia.core.export.bundle.BundleContract`: `task` (the contract `_target_`, or null), `exporter_family`, `inputs: list[InputSpec]` (`name`, `payload_key`, `dtype`, `shape`, `payload`), `preprocess_chains: dict[field, list[PreprocessStep]]` (`kernel`, `params`, `owner`, `row_mapped`), `routes`, `outputs`, `dynamic_axes`, `postprocess` (decode ctx), `labels`, `extra_metadata`. `BUNDLE_VERSION = 3`. There is no `runtime.backend` field, because torch is unconditional.

### ONNX contract

One named input per `InputSpec`. Outputs are the `Pred` fields returned by `forward()` plus each selected route's activated keys, since the tracer runs `forward + activation`. Several selected routes contribute one kernel each, and the bundle's `decode` returns a dict keyed by activation-node name. Each activation's `DYNAMIC_AXES` declares its activated outputs' variable axes (`{"boxes": {1: "detections"}}`). The exporter reads them from the nodes, never from a task or kind list.

### Bundle layout

```text
export/
  README.md
  manifest.json
  onnx/model.onnx
  metadata/   task.json  labels.json  preprocess.json  postprocess.json  <extra_metadata keys>.json
  runtime/
    preprocess.py   # collected per-field kernel chains + baked PARAMS + prepare(payload)
    postprocess.py  # collected decode kernels + baked CTX/ROUTES/_DECODERS + decode(outputs, restore)
    model.py        # generic onnxruntime glue (ExportedModel)
    run.py          # generic CLI entrypoint
```

`manifest.json`'s `specs` map lists every metadata file, including one entry per `extra_metadata` key. `task.json` carries the contract `_target_` (or null), the resolved tensor contract and `dynamic_axes`. `labels.json` carries label names when resolvable. `preprocess.json` is a lossless per-field summary of baked params for inspection only, and `runtime/preprocess.py` is the executable source of truth. `postprocess.json` carries the decode ctx.

### Runtime emission

`write_bundle_runtime`:

- **preprocess**: per input field, collect the chain's kernels (owner-named, deduped by rendered source) and unroll them into `_prepare_<field>(payload)`. A `"value"` field keeps a straight-line unroll, since chains are static at export time; its collate step (`PreprocessStep.collate`) receives `[value]`, so the feed carries the batch dim the ONNX graph expects. A `"window"` field loops its `row_mapped` steps per row. `prepare(payload)` merges every field's feed entry and restore dict.
- **postprocess**: collect each selected route's `kernel`, bake `CTX`, a `ROUTES` map, a `_DECODERS` map, and a generic `decode(outputs, restore)` slicing ONNX outputs per route.
- write the generic `model.py` (converts the feed to numpy for ONNX Runtime) and `run.py` (JSON-serializes array atoms via `tolist`), then self-verify both directions. `_verify_prepare` runs the emitted `prepare` against a real seed and asserts exact per-field equality with the live `on_after_batch_transfer(_collate_fn([...]))` path. It is best-effort, skipping interactive datamodules, datamodules with no static seed, non-`"value"` fields, fields produced mid-graph, and graphs whose seed shape the lookup does not model. Only a clean mismatch raises. `_verify_decode` runs the emitted `decode` on a dummy forward and asserts per-route equality with `model.postprocess`.

### Requirements

Export requires torch and there is no torch-free bundle. The emitted bundle imports torch, torchvision, onnxruntime, numpy and PIL and never teia. There is no `ops` module and no numpy mirror of a kernel.

## Extending

To make a new node exportable, give it an inline `kernel` and `params()` per [base/data/nodes](../base/data/nodes.md) or [base/net/activation](../base/net/activation.md), and put any shared math in an allowlisted helper module. A recipe pins export options (`export.opset`, `export.bundle_name`, `export.routes`) in its `export` config group. A per-architecture ONNX limit, such as an operator with no symbolic at the default opset, is handled by pinning `export.opset` in that recipe's config.

## Constraints

- A bundle MUST NOT import `teia` to run inference, and may import only numpy, torch, torchvision, PIL, onnxruntime and the stdlib. A teia reference inside a collected kernel fails the export.
- Collection MUST fail fast (`ExportCollectError`) on teia-core internals, instance state, closures (other than the one permitted `@staticmethod`), or a non-allowlisted import. A preprocess-phase node reachable from a model input with no `kernel` fails the same way, naming the node.
- `core/export` MUST NOT branch on task, contract or adapter.
- After emitting, the assembler MUST self-verify both directions, and either mismatch aborts the export.
- The ONNX tracer MUST be the dynamo exporter (`torch.export` + FX, `strict=False`), with `dynamic_shapes` as a list of `{axis: torch.export.Dim}`, one per input, and no TorchScript fallback. Because `torch.export` specializes size-0/1 dims, the dynamic batch axis needs an example extent of at least 2. A model containing an `nn.RNNBase` traces at static `batch_size=1`.
- Export MUST emit a single `[teia] export complete` status line, and third-party logging from `torch.onnx` and third-party model hubs is suppressed for its duration.
- Bundle extras MUST be owned by the producing component through `export_artifacts()`, never by export-side domain code.

