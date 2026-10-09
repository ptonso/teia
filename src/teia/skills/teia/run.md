# Run, evaluate, infer, export

Every stage takes the same overrides as `train`. Only `train` reads the live project. `test`, `eval`, `infer` and `export` rebuild from the config snapshot saved inside the run directory, so they reproduce exactly what was trained, even if the project's config has changed since.

| Stage | Command | Notes |
|---|---|---|
| Train | `teia train task=<task> data_root=<dir>` | Also validates, tests and runs the evalmodule by default. `--skip-post-test` and `--skip-report` (skips eval) turn those off, `--export` adds an export bundle, and `--run-name` names the run. |
| Continue | `teia train --from-run-dir <run> --from-run-weights best <overrides>` | Starts a new run from a parent's snapshot and weights. Cannot be combined with `--project-dir` or `--config-dir`. |
| Test | `teia test --run-dir <run>` | Also `--checkpoint` and `--skip-report` (skips eval). |
| Evaluate | `teia eval --run-dir <run>` | Rebuilds the eval outputs from captured predictions, with no model. Repeat `--run-dir` to compare runs, and use `--split` to pick the captured split. |
| Infer | `teia infer --run-dir <run> --src <input> --dst <output>` | Predicts on new data with the trained model. |
| Export | `teia export --run-dir <run> --output-dir <dir>` | Writes a self-contained bundle (ONNX model plus the preprocessing and decoding it needs). |

Running `teia <stage> --help` lists the current flags.

## Where things land

The run directory sits under the project's `runs/`. The keys `runs_root`, `project`, `experiment` and `run_name` decide the path, and each stage adds to the same directory:

- `config/`: the snapshot the later stages replay from;
- `logs/`: training logs;
- `eval/`: plots, tables and `summary.yaml`, whose scalars are the run's headline numbers;
- `artifacts/`: the run descriptor and the capture store (the network's captured predictions);
- `infer/` and `export/`, once those stages have run.

## Chained and scripted runs

Multi-phase training is several `teia train` calls, each seeded from the previous run with `--from-run-dir` (a warm-up with a frozen part, then the full run, for example). For a study, give each run its own name, keep the exact command lines in a script, and compare the runs with `teia eval`. Before a long run, use the cheap checks in [validate in layers](SKILL.md#ground-rules).
