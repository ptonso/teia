# The sweep command

Affects: [core/sweep](../core/sweep.md), [core/cli](../core/cli.md), [core/config](../core/config.md).

## Situation

Hyperparameter search was meant to reuse Hydra's multirun, but `teia` composes through the Hydra API and never launches a job, so `-m` and sweeper plugins do nothing. Replicated runs ([log](2026-10-17_replicated-runs-and-reducer.md)) give a per-configuration noise estimate, and a search must select on validation only, without computing test.

## Alternatives considered

- **Hydra's launcher and sweeper (`@hydra.main`).** Rejected for the reasons in the replication note, and because the sweeper has no notion of splits.
- **A `--trials` flag, an Optuna study and `--objective`/`--mode` flags.** The user dropped the trials flag, so the budget and the objective became ordinary config entries (`sweep.n_trials`, `sweep.objective`, `sweep.mode`), the same style as `runtime.n_runs`.
- **Optuna (TPE, resumable SQLite study).** Rejected for now: a seeded random search plus the grid needs no new dependency, and the trial ledger in `sweep.yaml` is enough to inspect a search.

## Decision

`teia sweep` takes the same override language as `teia train`. Hydra's own override parser decides which overrides are sweep expressions, all other overrides are the baseline, and each trial runs the train stage with test, report and export skipped. See [core/sweep](../core/sweep.md).

## What tipped it

The user wants one language for a baseline run and a search: take a working `teia train` line and turn any value into a sweep expression.

## Trade-offs accepted

- No pruning, no TPE and no resumable study: a sweep that stops early is rerun.
- Grids multiply fast, since a discrete sweep runs its full product.
- The objective is the best value of a logged series over epochs, so it is only as unbiased as the monitored split, which the user chooses.
