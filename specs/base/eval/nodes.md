# Eval Node Spec

Related: **Read first** [data/nodes](../data/nodes.md), [net/nodes](../net/nodes.md). **See also** [core/eval](../../core/eval.md), [eval/layout](../../eval/layout.md), [task](../task.md), [core/module/capture_map](../../core/module/capture_map.md).

## Overview

The eval graph is the third component graph, sibling to the data graph and the net graph. It turns a run's captured predictions and targets into named results, and then into artifact files. This spec defines the `teia.base` contract types for it: small ABCs that concrete components in `teia.node.eval.*` implement, wired by `in`/`out` over the eval workspace. `teia.base` owns the language and ships no concrete metric, plot, table or kernel.

The plane exists because evaluation is not standardizable. Survival analysis and time series each admit open-ended custom statistics, so core defines this contract and executes anything that satisfies it.

There are three node kinds: `Metric`, `View` and `Comparison`. Each declares `in`/`out` and is executed in topological order, like an io or net node.

Walk-through: core captures a run into a route-keyed store. The eval graph reads `capture.*` and `batch.*` atoms and produces `eval.*` results through metrics, and views render those results to files. A comparison reads several finished runs and writes one artifact.

Scenario: a detection evalmodule wires an instance-match metric over `capture.det.boxes`, `capture.det.score`, `capture.det.category` and `batch.bboxes`. An AP metric consumes the matches, and a ranked-bar view draws per-class AP.

## Language

- **eval node**: one atomic component in the eval graph.
- **eval workspace**: the key namespaces the graph reads and writes: `capture.*` (per-run atoms written by core), `batch.*` (captured targets, named as in the data graph), `meta.*` (dataset facts from the manifest), `log.*` (per-epoch logger series), and `eval.*` (results produced by metrics). Views are terminal and produce files, not keys.
- **capture key**: `capture.<route>.<atom>`, as declared by the task contract ([task](../task.md)), where `<atom>` is one content-named quantity (`score`, `category`, `boxes`, `time`, `event`). Never a composite record.
- **driver**: how a metric is fed, streaming (`update`/`compute`, incremental, no disk) or materialized (`from_source`, over a finished run).
- **pairing**: `aligned` (row *i* ↔ row *i*) or `unpaired` (compare aggregate distributions). The structural constraint the metric taxonomy is grouped by.
- **kernel**: a pure similarity or comparison function (box IoU, mask IoU, OKS) passed to a metric as a nested `_target_`. Not an eval node.
- **eval source**: the read handle over one finished run directory.
- **monitor name**: the flat `<split>/<name>` key a metric publishes for early stopping, checkpointing and LR scheduling.

## Map

- `teia.base.eval`: `Metric`, `View`, `Comparison`, `EvalSource`, the contracts here.
- `teia.node.eval.{metric,view,compare,kernel}`: concrete components, laid out in [eval/layout](../../eval/layout.md).
- Capture, the graph runner and the streaming monitor runner: [core/eval](../../core/eval.md). The graph reuses the data graph's `parse_node`, `producer_map`, `_toposort` and contract validation.
- Which atoms exist: the task contract ([task](../task.md)) and the netmodule's `capture:` map ([capture_map](../../core/module/capture_map.md)).

## Contracts

### Node base

Constructed from flat kwargs. `in`/`out` come from the graph envelope, as in the io and net graphs.

```python
class EvalNode:
    in_key: list[str]        # capture.* / batch.* / meta.* / log.* / eval.*
    out_key: list[str]       # eval.*  (empty for View)


class Metric(EvalNode):
    MONITORS: tuple[str, ...] = ()          # monitor names, relative to the split
    REQUIRES_CALLBACK: tuple[str, ...] = () # _target_ paths of callbacks this metric needs

    def from_source(self, source: EvalSource, **inputs) -> dict[str, Any]:
        """Required. Materialized driver: whole-split results keyed by out_key."""

    # Optional streaming driver.
    def update(self, **inputs) -> None: ...
    def compute(self) -> dict[str, Any]: ...
    def reset(self) -> None: ...


class View(EvalNode):
    def render(self, out_dir: Path, **inputs) -> list[Path]: ...


class Comparison(EvalNode):
    def compare(self, sources: list[EvalSource], out_dir: Path) -> list[Path]: ...
```

`MONITORS` is empty when the metric produces only array results for views. `REQUIRES_CALLBACK` is checked by the callback gate in [core/eval](../../core/eval.md), for example a metric that reads `lr-*` CSV columns requires `LearningRateMonitor`.

### Metric drivers

One class serves both drivers. `from_source` is required and `update`/`compute`/`reset` are optional. A metric whose predictions cannot be retained (a generator's output) must stream. A metric that needs the whole split (ROC, calibration bins, threshold search, worst-sample panels) cannot. Absence of `update` is the declaration that a metric cannot stream, with no separate flag. The number early stopping monitors and the number the report prints come from the same code.

### Extra kwargs and the after-pass hook

`ACCEPTS` (class attribute, default empty) lists the optional `from_source` kwargs a node takes beyond its `in` keys. `Metric.after_pass(graph)` is an optional hook called once after the whole graph ran, with the runner's `EvalPass` handle ([core/eval](../../core/eval.md#after-pass-hook)). A metric uses it to rerun the nodes whose `ACCEPTS` names a kwarg it can supply. The default does nothing, and a base pass never depends on it.

### Roles by keys

A metric names the atoms it consumes in `in`, one key per role (`time` and `event` for survival, `bboxes` and `cls` for detection). It never consumes a composite record, and adding a role adds a key.

### Matching

There is no fourth kind for matching. A matcher is a `Metric` whose output is a row table.

### EvalSource

The only handle an eval node gets on a run. It never exposes a live `LightningModule`, datamodule, trainer or dataloader.

```python
class EvalSource:
    run_dir: Path
    def routes(self) -> list[str]: ...             # capture routes in the manifest
    def meta(self, key: str) -> Any: ...           # manifest meta (class_names, ...)
    def has(self, key: str) -> bool: ...           # "capture.det.boxes" present?
    def column(self, key: str) -> Any: ...         # memmap / row lane / blob paths
    def config(self) -> Mapping[str, Any]: ...     # config/composed.yaml
    def curves(self) -> Any: ...                   # logs/*/metrics.csv, per-epoch
    def descriptor(self) -> Mapping[str, Any]: ... # artifacts/run.yaml
```

### Kernels

A kernel is a plain callable with no envelope, passed as a nested `_target_`:

```yaml
match_det:
  _target_: teia.node.eval.metric.aligned.InstanceMatch
  similarity: {_target_: teia.node.eval.kernel.IouXyxy}
  in:  [capture.det.boxes, capture.det.score, capture.det.category, batch.bboxes, batch.cls]
  out: eval.det.matched
```

### Result lifetime

`eval.*` results live in memory for one pass and are discarded, so re-running a view re-runs its upstream metric.

### Views

A view is typed by the geometry of what it draws, not by what the data means. See [eval/layout](../../eval/layout.md).

## Extending

To add a metric, subclass `Metric` under `teia.node.eval.metric.<pairing>/`, implement `from_source`, and add `update`/`compute`/`reset` if it can stream. To add a view, subclass `View` under `teia.node.eval.view.<geometry>/`. Write the leaf in `conf/node/eval/*` and wire it in the task's evalmodule. Reuse an existing kernel before writing one.

## Constraints

- `teia.base.eval` MUST NOT ship a concrete metric, view, comparison or kernel, and MUST NOT import `teia.node`.
- Every `Metric` MUST implement `from_source`. A `Metric` implementing `update` MUST also implement `compute` and `reset`.
- A `Metric` without `update` MUST NOT be wired as a val metric, and the eval-graph validator rejects it at pre-flight.
- A `View` MUST be terminal: no `out_key`, files only.
- An eval node MUST read only its declared `in` keys and MUST NOT reach into live state. Anything that needs live state is capture, not eval.
- An eval node MUST NOT consume a composite record and names atomic `capture.*`, `batch.*`, `meta.*` or `log.*` keys.
- Monitor names published by the active graph MUST be unique within a split, and a duplicate is a hard pre-flight error.
- A kernel MUST be a pure function and MUST NOT declare `in`/`out`.

