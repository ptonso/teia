# Reducer Spec

Related: **Read first** [../core/eval](../core/eval.md), [../base/eval/nodes](../base/eval/nodes.md). **See also** [layout](layout.md), [../core/runtime](../core/runtime.md), [../core/cli](../core/cli.md).

## Overview

The reducer aggregates the metric results of the replicas in a group directory ([core/runtime](../core/runtime.md#replicated-runs)) into one result per `eval.*` key. It reads value shapes and never an atom's meaning, so it serves every metric. Views then render the reduced results once, and draw an interval wherever a std sibling is present.

Walk-through: `teia eval --run-dir <group>` builds the eval graph and runs its metric nodes against each `rep_i` capture store, collecting each replica's `results`. The reducer folds the list of results key by key. The views run once over the reduced results and write into `<group>/eval/`.

Scenario: three replicas publish `eval.cls.confusion` with `accuracy` of 0.91, 0.93 and 0.92. The reduced dict holds `accuracy: 0.92`, `accuracy_std: 0.01` and `accuracy_values: [0.91, 0.93, 0.92]`. A `summary` view writes all three scalars, and a `ranked_bar` wired to a std-aware field draws whiskers.

## Language

- **replica**: one `rep_i` run of a group.
- **reduced result**: the dict the reducer publishes under an `eval.*` key, with the shape of one replica's result plus sibling fields.
- **sibling fields**: for a reduced field `f`, `f_std` (sample std, ddof=1, `0.0` for one replica) and `f_values` (the per-replica values in replica order). The reduced result also carries `n_runs`.
- **pass-through**: a field the reducer cannot align. It holds replica 0's value and `f_values` holds the per-replica list.

## Map

- `teia.core.eval.reduce`: the reducer. `teia.core.eval.runner`: the metric pass per replica and the single view pass over reduced results.
- Views that read siblings: `teia.node.eval.view.*` ([layout](layout.md)).

## Contracts

### Reduce rule

`reduce(results: list[dict[str, Any]]) -> dict[str, Any]` takes one `results` mapping per replica, in replica order, and returns the reduced mapping with the same keys. Each value reduces by shape:

| value in every replica | reduced `f` | siblings |
|---|---|---|
| number (not `bool`) | mean | `f_std`, `f_values` |
| array or nested list of numbers, equal shape | elementwise mean | `f_std`, `f_values` (stacked on axis 0) |
| list of dicts with equal length and equal non-numeric fields row by row | the same rows, each numeric field reduced as above | siblings inside each row |
| dict | recursion on its keys | siblings inside |
| equal non-numeric value (`str`, `bool`, names) | that value | none |
| anything else (unequal shape, unaligned rows, differing strings) | replica 0's value | `f_values` as the per-replica list |

The top level of every reduced dict also carries `n_runs`. One replica yields `f_std = 0.0`, so the same code path handles `n == 1`.

### View intervals

A view whose `render` declares a parameter `<kwarg>_std` receives it resolved from the same producer and field path as `<kwarg>`, with the leaf renamed `<leaf>_std`. The parameter is `None` when the sibling is absent, and the view renders without an interval. A view with no such parameter is unaffected by siblings. Shipped behavior:

- `table`: sibling numeric fields appear as extra columns.
- `summary`: sibling scalars are flat top-level fields, so `accuracy_std` is written beside `accuracy`.
- `ranked_bar`: error bars from its `values_std` parameter.
- `multi_curve_bundle`: curve dicts are rows, so the reduction puts `y_std` inside each curve and the view draws a band around `y` when it is present.

### Aggregate report

The group report is the reduced pass of the group's evalmodule, written to `<group>/eval/`. Each replica's own `eval/` is unchanged.

## Constraints

- The reducer MUST NOT branch on an atom, route or metric name, and MUST NOT import `teia.node.eval`.
- Sample std MUST use ddof=1.
- A mismatch of keys across replica `results` MUST raise `ValueError` naming the key. A mismatch of value shape MUST pass through and MUST NOT raise.
- A reduced result MUST keep the key set and nesting of one replica's result, plus siblings only.
- Metric nodes MUST NOT receive a reduced result as input. Metrics run per replica and views read the reduced results.
- A group with zero finished replicas MUST fail fast naming the group directory.
- Unit tests (`tests/test_eval_reduce.py`) MUST cover scalar, array, aligned rows, unaligned rows, string equality, pass-through and one replica.

## History

- [2026-10-17 replicated runs and a cross-run reducer](../logs/2026-10-17_replicated-runs-and-reducer.md): one shape-based reducer with sibling fields.
