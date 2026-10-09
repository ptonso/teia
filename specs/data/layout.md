# Data Layout Spec

Related: **Read first** [../base/data/nodes](../base/data/nodes.md). **See also** [../net/layout](../net/layout.md), [../docstrings](../docstrings.md), [../core/data_graph](../core/data_graph.md), [../core/datamodule](../core/datamodule.md).

## Overview

The canonical filesystem and Hydra group layout for every data-plane component in `teia.node.data`, and the procedure for placing a new one. `teia.node.data` is one of two sibling component libraries under `teia.node` (the other is `teia.node.net`), organized by data-node kind, never by modality or format. This spec is the companion to [base/data/nodes](../base/data/nodes.md), which owns the six ABCs. It documents the concrete taxonomy and the recurring authoring patterns the ABCs make possible: stateful fit and cross-sample reads.

A CSV reader and an image-folder reader are both just `reader/` components. Format (csv, jpeg, parquet) discriminates which reader you pick, not where readers live.

## Language

The canonical definitions of **data node**, **workspace**, **key**, **phase** and **stage** are in [base/data/nodes](../base/data/nodes.md).

- **kind folder**: one of the six top-level folders (`reader/`, `transform/`, `join/`, `reshape/`, `collate/`, `writer/`), each housing every concrete component of that ABC regardless of domain.
- **stateful transform**: a `Transform` that fits state at `plan` stage from a materialized `train_stream` (impute, scale, encode, vocab).

## Map

- `teia.node.data.{reader,transform,join,reshape,collate,writer}`. Conf leaves under `conf/node/data/<kind>/*`, and the graphs composing them under `conf/datamodule/<task>/*`.
- Executor and graph: [core/datamodule](../core/datamodule.md), [core/data_graph](../core/data_graph.md).

## Contracts

### Filesystem layout

```text
data/
  reader/<module>.py          # files, image_folder, rows, csv, parquet, sql, audio, coco, labelme
    extension/<component>.py  # ReaderExtension helpers appended to a reader record
  transform/<module>.py       # decode, geometry, normalize, stft, parse_*, labels, schema, ...
    augment/<component>.py    # train-only augment-phase rewrites
  join/<module>.py            # align (KeyJoin)
  reshape/<module>.py         # window (PlanReshape)
  collate/<module>.py         # tensor (Stack, ClassLabels, MultiHotClasses, ...), geometric (StackImages, ConcatBoxes, ...)
  writer/<module>.py          # class_dir, csv
```

Same atomic-module and config-group rules as `teia.node.net` ([../net/layout](../net/layout.md)): one component family per leaf module, `_target_` always the canonical `teia.node.data.<kind>.<component>.<ClassName>`. The conf surface is component-oriented: leaf data components under `conf/node/data/<kind>/*` (`augment/` holds the stochastic train-only transforms), and one wired graph per data format under `conf/datamodule/<task>/<format>.yaml`: readers and parsers (`item.*`), deterministic transforms, augments, and collates (`item.* → batch.*`) emitting the task contract's batch fields. A leaf carries no `in`/`out`; the datamodule wires it.

### Node envelope

One grammar shared with the netmodule graph ([../core/data_graph](../core/data_graph.md)):

```yaml
<name>: { _target_: <component>, in: <key|[keys]>, out: <key|{key: type}>, +optional phase/stage, ...params }
```

The kind is inferred from the component's ABC. `phase` and `stage` keep per-kind defaults and are written only to override. A `ctx.<name>` key lives inside an ordinary `in` list, never a separate field.

### Placement procedure

1. Opens a source and emits an indexed stream (`iter_split`, or `stage: stream` for a live source)? `reader/`.
2. Maps one item to one item? `transform/` (`transform/augment/` if `phase: augment`).
3. Aligns two or more streams by key? `join/`.
4. Redefines item granularity? `reshape/`, as `PlanReshape` (materialized source, single shot) or `StreamReshape` (unbounded source, incremental). The choice follows source cardinality, not convenience.
5. Assembles one atomic `batch.*` field from a list of item values? `collate/`.
6. Materializes a prediction or activation record back to a container? `writer/`.
7. Has no `in`/`out` envelope, phase or stage, so the graph cannot toposort it? It is not a data node.

### Pattern: reader extension

A reader owns one primary format, and a `ReaderExtension` (`reader/extension/base.py`) owns a secondary payload that rides with it. It is a plain helper with no `in`/`out` envelope, mounted as the reader's `extension` argument (`/node/data/reader/extension@<reader>.extension: <leaf>`). The reader builds a record, `extension.extend(key, record, source)` returns it with fields appended, and the reader emits the record's values in insertion order, so the datamodule's `out` list names the extra keys. An extension documents the file format it accepts in its module docstring, appends and never removes, and raises on an entry it cannot resolve.

```python
class ReaderExtension:
    def bind(self, root: Path) -> None: ...                                       # resolved dataset root
    def extend(self, key: str, record: dict, source: Path) -> dict: ...           # append fields
```

### Fit timing

A transform's fit timing has exactly two shapes. Most fit once at `plan` stage from a materialized `train_stream` (impute means, category vocabularies, letterbox stats). A transform whose true source is unbounded reads training state continuously through `self.training`, updating on every call when `True` and freezing when `False`. `state()` and `load_state()` persistence applies identically. A third shape should not be invented without checking that one of these two fits.

### Pattern: stateful transform

```python
class Scale(Transform):
    stateful = True
    def fit(self, train_stream: Iterable[Any]) -> dict: ...     # mean/std over the train split only
    def state(self) -> Any: ...
    def load_state(self, state: Any) -> None: ...
    kernel = staticmethod(scale_kernel)                          # export: pure (value, params) -> value
    def params(self) -> dict[str, Any]: ...
```

### Pattern: PlanReshape

`window` (`teia.node.data.reshape.window.WindowRows`) groups a materialized stream by entity, orders by time, and slices into `lookback` and `horizon` window items. It is called once and carries no state between calls.

### Pattern: cross-sample augment

A `phase: augment` transform depending on other items in the split (mosaic) sets `needs_siblings = True`, and the executor injects `siblings(i)` and `siblings_len` before the train `item` stage. An augment is always an in-place rewrite (`out_key ⊆ in_key`), so skipping it off the train path leaves the deterministic value for downstream consumers.

## Extending

To add a reader, transform, join, reshape, collate or writer, place it with the procedure above, subclass the matching ABC, implement `kernel` plus `params()` for any `preprocess` transform or `Collate`, add a provenance docstring, and write the conf leaf. Imitate the exemplar named in each pattern, and reuse an existing collate and field before adding one.

## Constraints

- The data skeleton MUST be exactly the six kind folders, and no data tree MAY be grouped by domain.
- A component with unbounded input MUST NOT be shoehorned into `PlanReshape`.
- A `stream`-stage node MUST declare `stage: stream` explicitly, and it is never a kind default.
- A node consuming a live external value MUST do so through `ctx.<name>` in its ordinary `in` list, resolved by `bind_context` and `resolve_context`, with no bespoke attribute injection and no `needs_*` boolean. `needs_siblings` is the one existing exception and not a template.
- Every leaf MUST follow the [docstrings](../docstrings.md) contract. Barrels and compat re-exports are not component surfaces.
