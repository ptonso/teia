# Sweep Spec

Related: **Read first** [cli](cli.md), [config](config.md), [runtime](runtime.md). **See also** [eval](eval.md), [../eval/reducer](../eval/reducer.md).

## Overview

`teia sweep` searches hyperparameters by running the train stage once per trial and scoring each trial on a logged validation series. Its input is the language of `teia train`: every override that is a Hydra sweep expression becomes a search dimension, and every other override is the baseline shared by all trials.

Walk-through: the overrides are parsed with Hydra's override parser and split into the baseline and the dimensions. The baseline is composed once to read the `sweep` keys. Each trial applies one assignment of the dimensions on top of the baseline, runs the train stage (a replicated group when `runtime.n_runs` is above 1) with the post-train test, report and export skipped, and records its score. The best trial's overrides are printed, ready to paste into `teia train`.

Scenario: `teia sweep task=vision-cls data_root=/ds runtime.n_runs=3 sweep.objective=val/loss sweep.mode=min sweep.n_trials=20 'optim.lr=tag(log,interval(1e-5,1e-2))' netmodule.depth=2,4,6` samples 20 trials of a learning rate and a depth, trains each 3 times, and scores each by the mean over replicas of the lowest `val/loss`.

## Language

- **baseline**: the overrides that are not sweep expressions.
- **dimension**: one override whose value is a sweep expression.
- **trial**: one assignment of every dimension, trained and scored once.
- **score**: for a trial, the mean over its replicas of the best value of the objective series over epochs.
- **sweep directory**: the run directory of a sweep, holding `sweep.yaml` and one `trial_<k>` run or group directory per trial.

## Map

- `teia.core.commands.sweep`: the command. `teia.core.runtime.sweep`: parsing, sampling and the trial loop.
- Trials reuse `train_project` and `train_group` ([runtime](runtime.md#replicated-runs)). The objective series comes from the run's logger CSVs ([eval](eval.md)).

## Contracts

### Command

`teia sweep [--project-dir D] [--config-dir D] [--run-name N] <overrides...>`. There is no `--from-run-dir`, `--checkpoint` or `--cfg`.

### Dimensions

Hydra's override grammar decides what is a dimension:

| expression | kind | values |
|---|---|---|
| `a=1,2,3`, `a=choice(...)`, `model=x,y` | discrete | the listed values |
| `a=range(1,10)`, `a=range(0,10,2)` | discrete | the range |
| `a=interval(0.1,0.9)` | continuous | uniform float |
| `a=tag(log,interval(1e-5,1e-2))` | continuous | log-uniform float |
| `a=int(interval(2,8))` | continuous | uniform integer, inclusive |

### Sweep keys

```yaml
sweep:
  n_trials: null    # int >= 1; required when any dimension is continuous, ignored for a pure grid
  objective: null   # a logged series, e.g. val/loss
  mode: null        # min | max
```

### Trial plan

When every dimension is discrete, the trials are the full product in dimension order. Otherwise `n_trials` trials sample every dimension independently (discrete ones uniformly), from a generator seeded by the baseline `runtime.seed`. A trial's overrides are `key=value` strings with the full-precision repr of floats.

### Trial execution

Trial `k` runs with the baseline plus its overrides, `skip_post_test`, `skip_report` and `skip_export` set, under `<sweep_dir>/trial_<k>`. With `runtime.n_runs` above 1 (baseline or dimension) the trial is a group, otherwise a single run. A trial that raises stops the sweep.

### Sweep descriptor

`<sweep_dir>/sweep.yaml`, rewritten after every trial:

```yaml
objective: val/loss
mode: min
baseline: [task=vision-cls, data_root=/ds]
dimensions: {optim.lr: "tag(log,interval(1e-5,1e-2))"}
trials:
  - {name: trial_0, overrides: [optim.lr=0.0003], score: 0.41}
best: trial_0
status: incomplete   # ok after the last trial
```

The command prints the best trial's overrides, one per line.

## Constraints

- The sweep MUST NOT compute test metrics: every trial skips the post-train test, report and export.
- A sweep with no dimension MUST raise.
- `sweep.objective` MUST be set and `sweep.mode` MUST be `min` or `max`, or the sweep raises before any trial.
- A continuous dimension without `sweep.n_trials` MUST raise before any trial.
- Parsing MUST use Hydra's override parser, and the sweep MUST NOT define its own grammar.
- A trial whose run has no series named by the objective MUST raise naming the run directory.
- Unit tests (`tests/test_sweep.py`) MUST cover the split, grid and random plans, scoring in both modes, and the descriptor.
