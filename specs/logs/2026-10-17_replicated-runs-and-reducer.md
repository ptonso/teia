# Replicated runs and a cross-run reducer

Affects: [core/config](../core/config.md), [core/cli](../core/cli.md), [core/runtime](../core/runtime.md), [core/eval](../core/eval.md), [eval/reducer](../eval/reducer.md), [eval/layout](../eval/layout.md), [base/eval/nodes](../base/eval/nodes.md).

## Situation

Hyperparameter search was meant to call `teia train` under Hydra's multirun and sweeper. `teia` uses Hydra only as a compositor (`compose` and `initialize_config_dir`), so `-m` and sweeper plugins never launch. Every eval node also saw exactly one run, so a result under noise was a single number. The only cross-run statistic was the `seed_aggregate` comparison node, which read scalars from `eval/summary.yaml` and used the population std.

A sweep that selects on the validation set needs a noise estimate per configuration, and the final report needs mean and std intervals. Both need replicated runs as a first-class feature.

## Alternatives considered

- **Adopt Hydra's launcher and sweeper (`@hydra.main`).** Rejected: it would take over the run directory, job numbering and the argparse front end, and the sweeper has no notion of splits, so the test set would be computed on every trial.
- **A `multirun` flag in the eval config that routes nodes between single-run and replicated behavior.** Rejected: two modes over one output type duplicate every node. Whether a result has replicas is a fact about the data.
- **Make every numeric leaf a `Stat` struct (mean, std, n, values) so views always receive Stats.** Rejected: one code path, but every view and every `in` wiring changes. Sibling fields reach the same result and leave views that ignore them untouched.
- **Keep `seed_aggregate` as the aggregator.** Rejected: it reads only the flat scalar summary, so curves, matrices and tables cannot be reduced, and it uses ddof=0.

## Decision

`runtime.n_runs` repeats a train `n_runs` times into a group directory ([core/runtime](../core/runtime.md#replicated-runs)). One reducer aggregates the per-replica metric results by value shape ([eval/reducer](../eval/reducer.md)), and views draw intervals when the reduced result carries the sibling fields. Each replica keeps its own report and the group adds an aggregate report. `seed_aggregate` is removed because the reducer subsumes it.

## What tipped it

A sweep objective is the mean of a validation metric across replicas, and a final result is reported as mean and std. Both are the same reduction over the same per-replica results, so one reducer serves the search and the report.

## Trade-offs accepted

- Training cost scales with `n_runs`.
- Outputs the reducer cannot align (ragged curves, unmatched row tables) pass through as per-replica lists and views draw replica 0.
- `--from-run-dir` is restricted to a single run, since a continued train of a group has no unambiguous parent.
- Group-versus-group significance tests are left out of this change.
