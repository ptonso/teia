# Eval Layout Spec

Related: **Read first** [../base/eval/nodes](../base/eval/nodes.md), [../core/eval](../core/eval.md). **See also** [../core/deps](../core/deps.md), [../core/task](../core/task.md), [../core/config](../core/config.md).

## Overview

The canonical filesystem and Hydra group layout for every component in `teia.node.eval`, and the procedure for placing a new one. `teia.node.eval` is the third component library beside `teia.node.data` and `teia.node.net`, shipped inside the `teia` distribution. A metric's exotic dependency is therefore a dependency of the runs that use it, never of the package.

Components are grouped by ABC kind and then by pairing, never by domain. A confusion matrix over image classes and one over tabular classes are the same component, and a mask-IoU matcher and a box-IoU matcher are the same component with different kernels. "Computer vision" is a config wiring, not a structural property of the code.

Scenario: to add a novel survival statistic, add one `Metric` subclass under `metric/aligned/`, one leaf conf, and one entry in the task's evalmodule. No engine change is needed.

## Language

Terms owned by [base/eval/nodes](../base/eval/nodes.md): eval node, eval workspace, capture key, driver, pairing, kernel, eval source, monitor name.

- **kind folder**: one of `metric/`, `view/`, `compare/`, each housing every concrete component of that ABC regardless of domain. `kernel/` is a fourth folder but not a kind, since kernels are sub-nodes.
- **payload geometry**: the shape a view draws (ranked bar, matrix heatmap, curve bundle, …), the hard constraint views are grouped by.
- **leaf**: one component's config file, one `_target_`.
- **evalmodule**: a task's wired eval graph preset, `evalmodule/<task>/<profile>`.

## Map

- `teia/node/eval/{metric},view,compare,kernel}`: components. Leaf confs `conf/node/eval/*` and the shared fragments `conf/evalmodule/_shared/{run,decision_rule}.yaml`. Search-path plugin `hydra_plugins/teia_searchpath.py`.
- Contracts the components satisfy: [base/eval/nodes](../base/eval/nodes.md). The runner that executes them: [core/eval](../core/eval.md).

## Contracts

### Filesystem layout

```text
teia/node/eval/
  metric/
    aligned/<component>.py     # confusion, ranked_ap, calibration, instance_match,
                               #   residuals, horizon_error, concordance, km_estimate,
                               #   semantic_confusion, threshold_optimization (per-class thresholds for multi-hot targets, per-class score weights for class-index targets)
    unpaired/<component>.py    # frechet_distance, kernel_distance, distribution_test
    run/<component>.py         # training_curves, gen_diagnostics, target_summary (read no predictions)
  view/<component>.py          # ranked_bar, matrix_heatmap, scatter_plot, multi_curve_bundle,
                               #   calibration_diagram, threshold_curve, grouped_histogram,
                               #   binary_confusion_grid, table, summary
  compare/<component>.py       # metric_table, curve_overlay, seed_aggregate, factor_diff, pareto
  kernel/<component>.py        # iou_xyxy, iou_rotated, iou_mask, oks   (sub-nodes, not eval nodes)
```

One public component per leaf module. `_target_` always names the canonical path `teia.node.eval.<kind>.<component>.<ClassName>`.

### Metric groups

Metrics sub-group by pairing. `aligned/` has per-sample correspondence, and `unpaired/` compares aggregate distributions. `run/` holds metrics that read no predictions: the per-epoch logger CSV (`training_curves`, `gen_diagnostics`) or captured ground-truth atoms alone (`target_summary`). They are offline-only and mounted through the shared run fragment.

There is deliberately no second `assumes` level like `teia.node.net`'s. An net node accepts one typed input, but an eval metric consumes a record of roles, and a single-valued folder label would recreate the bundling that `FIELD_VOCAB` dissolves for `batch.*`. Roles are declared by `in` keys.

Matching is a metric, not a kind. An instance matcher takes declared keys in and publishes one declared key out, and after matching everything downstream is `aligned`.

### Views

Views are grouped by payload geometry, and there are nine: `ranked_bar`, `matrix_heatmap`, `scatter_plot`, `multi_curve_bundle`, `calibration_diagram`, `threshold_curve`, `grouped_histogram`, `binary_confusion_grid`, `table`. A named plot is a wiring, not a class: `per_class_f1_bar_sorted` and `per_class_average_precision_bar_sorted` are one `ranked_bar` wired twice. This keeps a plot inventory of about 90 at about 10 view classes.

### Config tiers

| tier | path | selected by |
|---|---|---|
| **leaf** | `conf/node/eval/{metric,view,kernel}/<name>.yaml` (the `compare/` defaults are `core/conf/compare/default.yaml`) | mounted by an evalmodule |
| **shared fragment** | `conf/evalmodule/_shared/run.yaml` (training curves, LR curves, target summary), `decision_rule.yaml` (the `threshold_optimization` metric on `split: val` plus `decision_rule_table`) | mounted `@_here_` by an evalmodule, never selected |
| **evalmodule** | `conf/evalmodule/<task>/<profile>.yaml` in the component library | `evalmodule=<task>/<profile>`, or a task preset |

Every classification evalmodule mounts `decision_rule`, so threshold optimization is on by default. The feature is the presence of the node: `~evalmodule.threshold_optimization ~evalmodule.decision_rule_table` switches it off. The metric reruns the nodes that accept a decision rule into `eval/threshold-opt/` ([core/eval](../core/eval.md#after-pass-hook)).

An evalmodule is packaged at `evalmodule` and mounts leaves by alias, exactly like a netmodule ([core/config](../core/config.md)):

```yaml
# conf/evalmodule/vision-det/default.yaml
defaults:
  - /evalmodule/_shared/run@_here_
  - /node/eval/metric@match: instance_match
  - /node/eval/kernel@match.similarity: iou_xyxy
  - /node/eval/metric@ap: ranked_average_precision
  - /node/eval/view@ap_bar: ranked_bar
  - _self_
ctx: {conf: 0.001, iou: 0.7, max_det: 300}
match: {in: [capture.det.boxes, capture.det.score, capture.det.category, capture.det.sample_idx, eval.det.gt_boxes, batch.cls, batch.batch_idx], out: eval.det.matched}
ap:    {iou_thresholds: [0.5, 0.75], in: [eval.det.matched], out: eval.det.ap, monitor: true}
ap_bar: {title: Per-class AP, sort: descending, in: {labels: "eval.det.ap:per_category[].category", values: "eval.det.ap:per_category[].average_precision"}}
```

Swapping one plot's option is `evalmodule.ap_bar.sort=ascending`, and swapping its component is `node/eval/view@evalmodule.ap_bar=<leaf>`. Adding a plot is one more entry, and a custom metric is one subclass plus one entry.

### Worked collapses

- **`det`, `obb`, `inst-seg` and `pose` are one component plus four kernels.** They share one algorithm (rank predictions, match under a similarity threshold, integrate precision over recall) and differ only in the overlap measure (`iou_xyxy`, `iou_rotated`, `iou_mask`, `oks`). `InstanceMatch` matches once per threshold in `overlap_thresholds` and returns rows `{score, category, is_true_positive, threshold}` plus the ground-truth count per category. `RankedAveragePrecision` takes recall against that count, so missed instances cost score, and averages over thresholds (COCO `AP@[.50:.95]` for the ten COCO thresholds, with `ap50` and `ap75` reported), so one `RankedAveragePrecision` serves all five.
- **Semantic-segmentation mIoU is the confusion-matrix metric flattened over pixels.** IoU, Dice, pixel accuracy and frequency-weighted IoU derive from the confusion matrix classification already builds.
- **Survival is atomic keys, not a record.** `time`, `event` and a cure model's `cure` are independent capture atoms consumed by whichever metric names them.

### How an evalmodule wires its `in` keys

Every evalmodule follows these conventions, so the module files carry no explanatory comments.

- **Predictions** are read from `capture.<route>.<atom>`, exactly as the task contract declares them (`scores`, `boxes`, `quantiles`, `risk`).
- **Targets** are the contract's `target=True` fields, under their data-graph `batch.*` names, in model space. A format conversion (normalized `xywh` to pixel `xyxy`) is an explicit eval node writing `eval.*`.
- **`log.<column>`** keys read a per-epoch logger series.
- **`meta.*`** keys (`meta.class_names`) come from the capture manifest, not a per-sample column.
- **View `in` is mapping form** (`{kwarg: "producer_key:field.path"}`), since a metric publishes one whole result dict per `out` key and a view needs named arrays out of it. A `[]` segment (`class_rows[].f1`) pivots one field across a list of dicts (`core/eval/runner.py::_walk_field_path`).
- **A node with `monitor: true`** is driven twice from one implementation: the streaming val-metric runner during `trainer.fit`, and `run_eval_graph` at eval time. Regression, forecast and survival gain a val metric this way, so a survival run can early-stop on C-index.
- **Instance-match, semantic-confusion and the generative distance metrics read their columns directly** through `source.column(...)` inside `from_source`, because their predicted and ground-truth atoms share a trailing name (`sample_idx`, `mask_path`, `image_path`) the plain `in`-list convention cannot disambiguate. Their `in` list still names every column for dependency and wiring validation.

### Distribution wiring

`teia` registers its conf tree through a Hydra `SearchPathPlugin` under the `hydra_plugins` namespace package, auto-discovered on any installed distribution:

```python
# hydra_plugins/teia_searchpath.py   (NO __init__.py, namespace package)
class TeiaSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        home = Path(__file__).resolve().parents[1]
        for portion in map(Path, teia.__path__):
            if portion.is_relative_to(home) and (portion / "conf").is_dir():
                search_path.append(provider="teia", path=f"file://{portion / 'conf'}")
```

The conf tree lives inside the importable package (`src/teia/conf`) so it ships in the wheel, and `pyproject.toml` must carry the packaging keys:

```toml
[tool.setuptools.packages.find]
where = ["src", "."]
include = ["teia*", "hydra_plugins*"]

[tool.setuptools.package-data]
"*" = ["**/*.yaml"]
```

`numpy`, `matplotlib`, `pandas` and `scikit-learn` are core dependencies ([core/deps](../core/deps.md)). Everything else is an extra, and extras are cosmetic: `teia.core.deps` resolves a missing module to a nicer install line (`teia[survival]` rather than `lifelines`) and works identically with none declared.

## Extending

Place a new component by ABC kind, then by pairing for a metric or by payload geometry for a view. Add a kernel for a new overlap measure, and reuse `RankedAveragePrecision` or the confusion metric before writing a new one. Write the leaf conf under `conf/node/eval/`, then wire it in the task's evalmodule. A component that seems to need a domain folder is mis-factored or belongs in an evalmodule.

## Constraints

- `teia.node.eval` MUST NOT contain a domain folder (`vision/`, `tabular/`, `survival/`) and MUST NOT name a domain in a class name.
- A component module MUST import its third-party dependencies at module top level, so `collect_missing` can detect them by trial import.
- Extra names are resolved against installed distribution metadata by `teia.core.deps._extra_owner`, with `COMPONENT_DISTRIBUTION = "teia"` the preferred owner.
- A view MUST be one of the payload geometries. A tenth geometry needs a renderer, not a one-off plotting function inside a metric.
- A metric MUST NOT render, and a view MUST NOT compute.
- `hydra_plugins/` MUST NOT contain an `__init__.py`.
- Every leaf module MUST expose exactly one public component, and its `_target_` MUST be the canonical `teia.node.eval.<kind>.<component>.<ClassName>` path.

