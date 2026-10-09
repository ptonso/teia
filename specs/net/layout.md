# Net Layout Spec

Related: **Read first** [types](types.md), [../base/structure](../base/structure.md). **See also** [../docstrings](../docstrings.md), [../data/layout](../data/layout.md), [../base/net/nodes](../base/net/nodes.md).

## Overview

The canonical filesystem, import and Hydra group layout for every compute component in `teia.node.net`, and the procedure for placing a new one. Two independent axes govern it. The **Python tree** organizes by hard architectural constraint, what the DAG must respect. The **conf tree** organizes by purpose, how a user wires a component. The same Python module is wired for different purposes by different conf files.

A Python folder is an interchangeability class: two components in the same `<fan>/<assumes>/` folder must be swappable by changing `_target_`. If a proposed distinction does not prevent a swap, it is a file, not a folder. There is no `stem`, `encoder` or `head` folder in Python. A stem is just a `one_to_one` node, and a `LinearHead` (1→1, agnostic) is wired as a stem, an encoder or a head by conf alone.

Scenario: a new attention block over unordered table columns has one input, no positional order, and needs no value-domain: it is `one_to_one/set/`. A leaf under `conf/node/net/encoder/` configures it.

## Language

- **fan**: a node's io cardinality counted from the declared `in`/`out` lists: `zero_to_one`, `one_to_one`, `many_to_one` (N ≥ 2 in), `one_to_many` (N ≥ 2 out), `many_to_many`. In-count and out-count are independently 1-or-N, a 2×2 plus the no-input cell.
- **assumes**: what a node requires of each input, by precedence (first match wins): (1) **value-domain** (`categorical`, `count`, `numerical`), firing only on `one_to_one` raw-column tokenizers; (2) **structure** (`sequence` > `grid` > `set`, plus `graph`), firing anywhere; (3) **residual**, `agnostic` (no requirement) or, multi-input only, `heterogeneous`. `zero_to_one` takes no `assumes` subfolder. A multi-input node applies the ladder to every input and is `heterogeneous` if they disagree.
- **kind folder**: `activation/` or `loss/`, the two flat folders beside the fan folders. Their contract is a route or a key namespace instead of fan × assumes. Every component under `net/` is a node of the net graph, mirroring `teia.base.net` (`TeiaNode`, `BaseActivation`, `BaseLoss`).
- **purpose**: a conf-side label (`stem`, `encoder`, `decoder`, `neck`, `fusion`, `branch`, `head`) that the engine never sees.

## Map

- `teia.node.net.<fan>/<assumes>/`: compute components, each folder with a `README.md`. `teia.node.net.activation` and `teia.node.net.loss`: flat.
- `conf/node/net/{stem,encoder,decoder,neck,fusion,branch,head,act,loss,embedder}/*`: purpose-sorted node leaves.
- Boundary typing for these components: [types](types.md). Another distribution may contribute to the one `teia.node.net.*` namespace under the identical path.

## Contracts

### Python layout

```text
net/
  zero_to_one/
    README.md
    <component>.py         # ParameterNode (0→1); no assumes subfolder
  <fan>/<assumes>/          # fan ∈ {one_to_one, many_to_one, one_to_many, many_to_many}
    README.md               # required, markdown, not imported
    <component>.py          # one standalone component
  activation/  <component>.py       # flat; contract = activation + kernel route
  loss/        <component>.py       # flat; contract = pred.*/batch.* namespace
  _<bundle>_utils.py
```

`assumes ∈ {categorical, count, numerical, sequence, grid, set, graph, agnostic, heterogeneous}`. A leaf `_target_` names the canonical unified path (`teia.node.net.one_to_one.agnostic.mlp.MLP`). Files beginning with `_` are private helpers, exempt from the taxonomy, and may live at any level beside their users.

### Conf purpose layout

```text
conf/node/net/
  stem/*  encoder/*  decoder/*  neck/*  fusion/*  branch/*  head/*   # node leaves by purpose
  act/*  loss/*  optim/*
  embedder/*                                                        # tokenizer leaves (value-type subfolders)
```

A leaf sets `_target_` plus hyperparameters only: no `in`/`out` and no `@package` header. The netmodule mounts it under a name (`/node/net/head@head: linear`) and wires its keys, so one leaf serves every task. Duplicate leaves differing only by wiring are not kept.

### Attention and the assumes ladder

Bare self-attention is permutation-equivariant, so the positional scheme is the assumption. No positional scheme means `set`. 1D order (sinusoidal, learned, RoPE, causal) means `sequence`. 2D+ spatial (patchify, windowed attention) means `grid`. Adjacency-derived bias means `graph`. A per-node op that ignores the adjacency (a shared MLP head, a pack or unpack by node mask) is `agnostic`. Architecture families do not map to folders: one family can appear under several folders (a convolutional and a dense variant), so the folder names the contract and the file names the architecture.

### Fusion and branch

Fusion (`many_to_one`) and branch (`one_to_many`) are first-class families, so multimodal combination is a swappable component (`ConcatFusion`, `SumFusion`, `GatedFusion`). A component that both fuses and transforms inputs is a monolith and is split into a fusion node plus a plain node.

### Placement procedure

1. Forward pass over shape-typed tensors (`TeiaNode`)? If not, `activation/` or `loss/`.
2. **fan**: count declared `in`/`out` entries. If `zero_to_one`, stop.
3. **assumes**: run the precedence ladder over every input, and `heterogeneous` if multi-input inputs disagree.
4. Owns a logit↔prediction pairing (`BaseActivation`)? `activation/` (flat).
5. Reads `pred.*` and `batch.*` and returns a scalar (`BaseLoss`)? `loss/` (flat).

`activation/` and `loss/` stay flat because their contract axis is not fan × assumes: an activation's is its route ([types](types.md)) and a loss's is the key namespace it reads.

## Extending

To add a component, place it by the procedure above, add it as the sole public class of one leaf module with a provenance docstring ([docstrings](../docstrings.md)), list it in the folder `README.md` **Members**, and write the `conf/node/net/<purpose>/<name>.yaml` leaf for it. Reuse an existing node before writing one: one `one_to_one/grid` node serves images, spectrograms and video frames alike. An export kernel lives beside its component.

## Constraints

- The net Python skeleton MUST be exactly the five fan folders plus `activation/` and `loss/`. A fan folder is `zero_to_one/` or holds only `<assumes>/` folders.
- `teia.node.net` holds nodes only. A non-node (optimizer, task contract, callback) MUST live in a sibling of the graph folders (`teia.task`, `teia.callbacks`, and `teia.optim` where a distribution ships optimizers), and its conf leaf outside `conf/node/`.
- No Python folder MAY be named for a role or purpose (`embedder`, `encoder`, `stem`, `head`) or for a domain.
- No folder MAY be named for a rejected token: `boolean` (merged into `categorical`), `temporal` as a value-type (merged into `numerical`), or bare `numerical` outside the value-domain family.
- Every `<fan>/<assumes>/` folder, including empty reserved ones (for example `one_to_one/sequence/`), MUST carry a `README.md` with sections **Contract**, **Why this is a real constraint**, **Members** and **Placement**. It replaces the former `protocol.py`, since markdown cannot drift into being imported.
- A public component MUST live only in a leaf folder and be the sole public class of its module. Barrels and compat re-exports are not component surfaces.
- Purpose MUST live only in `conf/node/net/*`, never as a Python folder or an engine-visible role.
- An export kernel MUST live beside its component, per [../core/export](../core/export.md).
