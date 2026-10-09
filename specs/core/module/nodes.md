# Pipeline Nodes Spec

Related: **Read first** [user_contracts](user_contracts.md), [../../base/net/nodes](../../base/net/nodes.md). **See also** [overview](overview.md), [codegen](codegen.md), [dry_run](dry_run.md), [activation_route](activation_route.md).

## Overview

What an net-graph node is and the contracts it satisfies: the `TeiaNode` base for forward nodes, the `out` shape forms, the node config file format, the forward calling convention, and the build-hook order. Forward nodes subclass `TeiaNode`. Activation and loss nodes are plain `nn.Module`, since dynamic-shape `pred` and `batch` tensors need no shape injection.

Node configs carry no shape params. Shapes are injected from `in_shape` and `out_shape`, and dataset-dependent output width comes from an `out` dim-ref, so generic heads need no `configure_from_datamodule`. Component construction (a leaf under `conf/node/net/<purpose>/`) and pipeline wiring (the envelope) are separate concerns in separate files.

## Language

- **node**: the atomic pipeline unit, a `TeiaNode` (forward) or plain `nn.Module` (activation or loss), tensors in and tensors out.
- **component construction**: the node leaf under `conf/node/net/<purpose>/`, `_target_` plus constructor params.
- **wiring**: the netmodule's flat envelope keys.
- **shape injection**: the orchestrator passing `in_shape` and `out_shape` into a `TeiaNode` at build.

## Map

- Base classes: [base/net/nodes](../../base/net/nodes.md). Wiring and envelope: [user_contracts](user_contracts.md). Build order: [overview](overview.md).
- Node families live in the component library's `conf/node/net/{stem,embedder,encoder,decoder,neck,fusion,head,act,loss}/` trees, which are the authoritative inventory. Python placement is `node/<fan>/<assumes>/`, and conf purpose is a separate, conf-side concern.

## Contracts

### TeiaNode

Constructed with `in_shape` (a tuple or list of tuples, no batch dim) and optional `out_shape` (tuple, list of tuples or `None`), both injected. Subclasses implement `build_module(**kwargs)`, which builds weights from the shapes, and `forward(*args)`. `out_shape=None` is valid for topology-preserving nodes (transformers, norm, pass-through), which default `self.out_shape = self.in_shape` in `build_module`.

### out forms

```yaml
out: { feat.maps: [256, 16, 16] }          # dict with declared shape (dimensionality change)
out: { z: [16,4,4], pred.mu: [16,4,4], pred.logvar: [16,4,4] }   # multi-output, declared order
out: [feat.encoded]                          # list form, no shapes (topology-preserving)
```

The workspace is always populated from the actual meta dry-run tensor, so downstream nodes get correct shapes whatever form is used.

### Node config file

`conf/node/net/<purpose>/<name>.yaml` holds `_target_` plus flat constructor kwargs, with no `# @package` header:

```yaml
_target_: teia.node.net.one_to_one.grid.cnn.ConvEncoder
channels: [64, 128, 256, 512]
strides:  [2, 2, 2, 2]
activation: silu
norm_type: layer
```

### Calling convention

`in: [a, b, c]` calls `node(var_a, var_b, var_c)`. N `out` entries return an N-tuple, and one returns a single tensor. Activation nodes implement `activation(*pred_logits)` and `postprocess(activated, ctx)` with `in` restricted to `pred.*` ([activation_route](activation_route.md)).

### Build-hook order

`TeiaNetModule` calls the optional `configure_from_datamodule(dm)` and `configure_from_producers(producers)` hooks ([base/net/nodes](../../base/net/nodes.md)), always after every node producing one of the node's `in` keys is built:

- **forward** nodes, in toposort order: instantiate, `configure_from_datamodule(dm)` (if `dm` and the hook exist), `configure_from_producers`, then a meta forward filling the workspace.
- **loss** nodes: instantiate, then `configure_from_producers` in the build pass. `configure_from_datamodule(dm)` runs afterwards, once for all losses, after the whole module is built and export kernels are validated. So for a loss, `configure_from_producers` runs first.
- **activation** nodes: instantiate, `configure_from_datamodule(dm)`, then `configure_from_producers`.

`producers` is built incrementally (forward nodes, then losses, then activations), accumulating `out → built node`, so a loss or activation paired with an earlier forward node (a wrapped-backbone head) can resolve a handle to it. `batch.*` keys never appear in it. A `bind` node registers as a producer under its own `out`s, the source node for `tied` and the frozen clone for `frozen` ([bind](bind.md)).

### Node families

- **encoders, necks, decoders**: `TeiaNode` backbones and feature transforms. Like heads, decoders emit logits only, and the pixel-space squash lives in the activation node and the recon losses.
- **heads**: emit logits only. The generic `LinearHead` obtains width from an `out` dim-ref. Heads with non-tuple output stay specialized. A head that wraps a third-party module flattens its nested training-mode payload into a `tuple[Tensor, ...]` and exposes an `unflatten` method that its paired loss and activation call through a `configure_from_producers` handle.
- **activations**: one per prediction route.
- **losses**: pure backward terms, atomic where possible (`recon` plus `kld` rather than one variational loss). Joint criteria that cannot be split (an assignment-based detector loss) may emit several `loss.*` keys from one node.

## Extending

Add a node by subclassing `TeiaNode` in `teia.node.net.<fan>/<assumes>/`, adding a `conf/node/net/<purpose>/<name>.yaml`, and mounting it in a netmodule through `in`/`out`. Reuse an existing head or encoder before writing one.

## Constraints

- Forward nodes MUST subclass `TeiaNode`. Activation and loss nodes MUST be plain `nn.Module`.
- Node configs MUST NOT carry shape params (`in_channels`, `in_features`, `num_classes`).
- New nodes MUST NOT use `configure_from_encoder` or `configure_from_decoder` (use `self.in_shape`), `_build_if_needed` lazy construction (use `build_module` at init), or `@dataclass` shape-config classes (use flat `build_module(**kwargs)`).
- Decoders and heads MUST emit logits only.

