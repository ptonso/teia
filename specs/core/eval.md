# Eval Plane Spec (core)

Related: **Read first** [base/eval/nodes](../base/eval/nodes.md), [base/task](../base/task.md). **See also** [module/capture_map](module/capture_map.md), [task](task.md), [runtime](runtime.md), [config](config.md), [cli](cli.md), [deps](deps.md), [eval/layout](../eval/layout.md).

## Overview

This is core's half of the eval plane: what a run captures, when, and in what format; how the evalmodule is validated and executed; and where results enter the monitor namespace. Core is a thin orchestrator here. It persists the atoms the netmodule's `capture:` map declares, typed by the task contract, and runs the evalmodule's graph. It contains no metric or plot implementation. Every metric, view, comparison and kernel lives in `teia.node.eval` ([eval/layout](../eval/layout.md)), and the ABCs they satisfy live in [base/eval/nodes](../base/eval/nodes.md).

The single rule that separates the planes: anything needing live state is capture, and evaluation is pure over artifacts. That one property makes eval re-runnable against a finished run and capable of comparing runs.

Walk-through:
1. **During `trainer.fit` validation**, the capture callback feeds each step's atoms (from `CaptureMap`, [module/capture_map](module/capture_map.md)) to the streaming metrics, and no disk is touched.
2. **On a standalone validate or test pass**, the same callback writes the whole split once into the capture store, each atom in the lane its contract spec type selects. The `test` stage runs a standalone validate pass first whenever an eval node reads the `val` split ([node splits](#node-splits)).
3. **`run_offline_eval`** then executes the evalmodule over an `EvalSource` and writes views flat into `eval/`.
4. **`teia eval`** does the same offline for finished runs.

## Language

- **capture**: persisting the atom dicts of a stage pass to the run's capture store. Core-owned, and it interprets nothing it writes.
- **capture store**: the on-disk artifact one capture pass produces.
- **evalmodule**: the composed eval graph plus its settings ([config](config.md)). Its nodes are the mappings carrying a `_target_`.
- **val metric**: a streaming metric run inside the validation loop, publishing into the monitor namespace.
- **offline eval**: the materialized pass over an `EvalSource`, during `test` or standalone via `teia eval`.
- **run descriptor**: `artifacts/run.yaml`, the machine-readable identity of a finished run.
- **monitor namespace**: the flat `<split>/<name>` key space consumed by early stopping, checkpointing and LR scheduling.

## Map

- `teia.core.capture`: `extract.CaptureMap` (atoms from a step), `callback.CaptureCallback` (streaming and store writing), `store.py` (lanes and manifest), and the generic raster encoders.
- `teia.core.eval`: the graph (`graph.py`), `run_offline_eval` (`runner.py`), `StreamingMonitorRunner` (`streaming.py`), `compare_runner.py`.
- `teia.core.commands.eval`: the `teia eval` command. `teia.core.runtime.run_descriptor`: `run.yaml`.
- Components and the `node/eval/*` leaves: [eval/layout](../eval/layout.md).

## Contracts

### evalmodule settings

`evalmodule` holds the eval graph's nodes and the settings below. Defaults live in the eval runner, and a module sets only what it changes.

```yaml
enabled: true           # run offline eval after test
split: test             # the split captured and evaluated
max_rows: null          # null = whole split; an int caps captured rows uniformly across routes
ctx: {}                 # postprocess ctx during capture (e.g. conf: 0.001, iou: 0.7, max_det: 300)
media: {max_items: 16, max_seconds: 10, max_resolution: 256}   # caps for Blob atoms of media formats
```

There is no universal graph selected by core. Each evalmodule mounts the shared run fragment (`/evalmodule/_shared/run@_here_`: training curves, LR curves, target summary) when it wants it.

### Capture callback

Core adds `CaptureCallback` to the trainer whenever the composed run has an evalmodule with nodes. There is no capture callback to list in `callbacks`. It uses `CaptureMap` for every step, so streaming validation, the store writer and inference see the same atoms.

### Capture store

One directory per split, `artifacts/capture/<split>/`, with three lanes and a completion marker. The lane is chosen by the atom's contract spec ([base/task](../base/task.md)):

- **numeric**: `Tensor` and `Ragged` atoms, as raw bytes appended per batch and read back as `np.memmap`, shape `(rows, *row_shape)`. A `Ragged` atom's `index` column is offset to be global across the split.
- **rows**: JSONL plus an offset index, for `Names` meta and non-numeric atoms (`sample_ids`, paths).
- **blobs**: `Blob` atoms, one file per sample in the spec's `format`. The encoder is picked from the array's dtype: an integer mask becomes a palette PNG, a float image an RGB PNG, and a frame stack a GIF or MP4.
- `manifest.json`: the completion marker, the `{route: {atom: column}}` map, the contract `_target_`, and every contract `meta` value.

### Ground truth

Ground truth is the contract's target fields (`target=True`), read by `CaptureMap` from the same eval batch as the predictions. Both are in model space: the deterministic eval slice of the data graph, with no augment. Capture `ctx` carries no restore values, so decoded predictions stay in that space too. A format difference between targets and predictions, such as normalized `xywh` against pixel `xyxy`, is converted by an eval node. Rescaling to original coordinates happens only at inference ([infer](infer.md)).

### Two execution paths

Inside `trainer.fit` validation, only streaming metrics run and no disk is touched. Whole-split capture happens once, on a standalone validate or test pass. The branch is gated by `TeiaNetModule._in_fit_validation`, so training speed and disk IO are unchanged. When predictions are too large to retain, the contract declares them `Blob` with media caps, or the evalmodule relies on streaming metrics alone.

### Eval-graph execution

```yaml
evalmodule:
  split: test
  match:
    _target_: teia.node.eval.metric.aligned.instance_match.InstanceMatch
    similarity: {_target_: teia.node.eval.kernel.iou_xyxy.IouXyxy}
    overlap_thresholds: [0.5, 0.75]
    in:  [capture.det.boxes, capture.det.score, capture.det.category, capture.det.sample_idx, eval.det.gt_boxes, batch.cls, batch.batch_idx]
    out: eval.det.matched
  ap:
    _target_: teia.node.eval.metric.aligned.ranked_average_precision.RankedAveragePrecision
    in:  [eval.det.matched]
    out: eval.det.ap
```

The executor builds the graph with the data graph's resolver ([data_graph](data_graph.md)), toposorts it, runs each node, and writes view outputs flat into `eval/`. Key resolution:
- `eval.*` reads in-memory results;
- `meta.*` reads the manifest;
- `log.<column>` reads the per-epoch logger series;
- everything else reads a store column.

Intermediate `eval.*` values live in memory for the pass. `run_offline_eval` is the post-train and post-test entry point, and `teia eval` is the standalone one.

### Node splits

An eval node may set `split: <name>` beside `in`/`out`. The runner reads that node's `in` keys from the named capture split and every other node's from `evalmodule.split`; `meta.*` is identical across splits. `required_splits(evalmodule)` is `evalmodule.split` plus every node `split`. The `test` stage captures each required split other than `test` before `trainer.test`, and only `val` (a standalone `trainer.validate` on the same restored weights) is capturable. A required split with no capture store fails fast, naming the path and `teia test --run-dir <run>` as the way to create it, with no fallback to another split.

### After-pass hook

After the base pass, the runner calls `Metric.after_pass(graph)` on every metric node (default: nothing). `graph` is an `EvalPass` handle:

- `records`: the nodes in graph order (`name`, `kind`, `in_key`, `out_key`, `split`), and `component_class(name)`;
- `results` and `out_dir`: the base pass results and output folder;
- `downstream(names)`: the nodes that transitively consume those nodes' `eval.*` outputs, in graph order;
- `rerun(names, *, out_dir, extra)`: executes those nodes again in graph order on a copy of `results`, views writing under `out_dir`, with `extra[name]` added to that metric's `from_source` kwargs. `rerun` never calls `after_pass`.

A node declares the extra kwargs it can take in the class attribute `ACCEPTS`. Which nodes to rerun, with what and where is the calling metric's decision, so core holds no component name. The shipped user is `ThresholdOptimization`: it fits a decision rule on its own `split` (`val`), then reruns every node accepting `rule` plus their consumers into `eval/threshold-opt/` with the fitted rule, while everything else stays flat-only in `eval/`.

### Monitor namespace

One flat space, `<split>/<name>`. Loss nodes contribute `loss` and per-node keys (`loss.cls`), and metrics contribute the names in their `MONITORS`. A multi-route run self-namespaces (`cls.macro_f1`, `det.macro_f1`). Because one metric implementation serves both drivers, the value a monitor sees equals the value the report prints.

### Run descriptor

`artifacts/run.yaml`, written at run start and finalized at run end:

```yaml
schema_version: 2
run_id: <stable id>
group: ${experiment_name}
seed: 0
git_rev: <sha or null>
data_root: <path>
dataset_id: <stable id or null>
task: teia.task.vision_det.VisionDet   # task._target_, or null
routes: {det: [boxes, score, category, sample_idx]}
status: ok | failed | incomplete
stages: [train, test]
```

`status` and `stages` are load-bearing for comparing runs: a run that trained but never reached `test` has no summary, and that must read as a status, not a missing file. Paths in run artifacts are relative to the run directory or `data_root`.

### Pre-flight

`validate_composed_config` runs before any instantiation, from `train_project` and the shared snapshot-stage opener, and gates the eval plane:

1. **Contract gate**: with a `task`, `check_net` over the netmodule's `in` keys and `capture:` map, and `check_eval` over every evalmodule `in` key ([base/task](../base/task.md)).
2. **Dependency gate**: `deps.check_missing` over the `evalmodule` subtree ([deps](deps.md)).
3. **Wiring gate**: every eval node's `in` key has a producer. That is a `capture.*` atom in the netmodule's `capture:` map (or a derived `Ragged` index), a target `batch.*` field, a `meta.*` key, a `log.*` key, or another node's `out`.
4. **Callback gate**: a `Metric` may declare `REQUIRES_CALLBACK`, and each requirement is checked against `callbacks.items`, raising before `trainer.fit` if one is missing.

A second gate runs after the data graph's plan stage, where `meta` is known: `check_data`, and checks that need dataset-derived values.

### teia eval

It is discovered by `pkgutil.iter_modules` over `teia.core.commands.__path__`, so it MUST live in core, because there is no entry-point mechanism for plugin commands ([cli](cli.md)).
- With one `--run-dir`, it rebuilds that run's `eval/` from its own `evalmodule`.
- With several run dirs and no `--compare` override, it runs the zero-config bundle `core/conf/compare/default.yaml`: `metric_table`, `curve_overlay` on `val/loss`, `factor_diff` and `seed_aggregate`, all task-agnostic.

A comparison needing task-specific metric names is a copied bundle passed via `--compare`.

### Run-directory layout

```
<run_dir>/
  config/            composed.yaml, overrides.yaml, hydra.yaml, conf/
  logs/              stage logs + <logger>/metrics.csv (per-epoch)
  artifacts/
    run.yaml         run descriptor
    capture/<split>/ capture store
  eval/              *.{png,csv,yaml} (flat, incl. summary.yaml), walltime.yaml
  checkpoints/  infer/  export/
```

`eval/summary.yaml` is a flat scalar map and the join key for comparing runs. Every view writes flat into `eval/`.

## Constraints

- `teia.core` MUST NOT contain a metric, plot, table, comparison or kernel implementation, and MUST NOT import `teia.node.eval`. It instantiates eval components from `_target_` strings only.
- Core MUST NOT interpret an atom's meaning. There is no task-name set, no route-name dispatch and no prediction-kind branch. Lane and encoding follow the contract spec type and the array dtype only.
- `CaptureMap` MUST be the only producer of `capture.*` atoms ([module/capture_map](module/capture_map.md)).
- A capture key MUST be one content-named atom under `capture.<route>.<atom>`, and composite record types MUST NOT be reintroduced.
- Nothing MUST be written to the capture store inside the `trainer.fit` validation loop.
- A rerun MUST NOT call `after_pass`, and the base pass results MUST stay untouched by it.
- A required capture split that is missing MUST fail fast and MUST NOT fall back to another split.
- An eval node MUST NOT receive a live `LightningModule`, datamodule, trainer or dataloader, and the engine MUST NOT hold them alive for the eval pass.
- Monitor names MUST be unique per split, and duplicates are rejected at pre-flight.
- The contract, dependency, wiring and callback gates MUST run before `trainer.fit`.
- Run artifacts MUST NOT contain absolute paths.
- The walltime report MUST be emitted by every run.

