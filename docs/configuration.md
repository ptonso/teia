# Configuration

A Teia run is three graphs, composed from config:

- the **datamodule** reads your files (or an environment) and emits batches;
- the **netmodule** is the network: it turns batches into losses and predictions;
- the **evalmodule** turns the captured predictions into metrics, plots and tables.

A **task** (`vision-cls`, `tabular-cls`, ...; contract class `teia.task.vision_cls.VisionCls`) is a contract over what these graphs hand each other. A task preset (`conf/task/<name>.yaml`) selects one module per graph. That gives you three ways to work.

## 1. Use a task preset

```bash
teia train task=vision-cls data_root=/data/pets
```

`task=` picks the preset: a datamodule, a network, the evaluation, and the task contract that validates their wiring. `data_root=` is the only dataset knob.

## 2. Swap inside a module

Finer swaps replace one node inside a module and keep its wiring:

```bash
teia train task=vision-cls node/net/encoder@netmodule.encoder=mlp_tabular data_root=...
```

And any value is a dotted override:

```bash
teia train task=vision-cls datamodule.batch_size=16 epochs=50
```

A key that does not exist yet needs `++` (`++trainer.fast_dev_run=true`).

## 3. Take full control

Drop the task and select (or write) the three modules yourself:

```bash
teia train datamodule=vision-cls/image_folder netmodule=vision-cls/mlp evalmodule=vision-cls/default data_root=...
```

Without `task=` there is no contract check, only the graph's own wiring checks. To start from a task and own everything, materialize it:

```bash
teia config init --from task=vision-cls      # writes conf/{datamodule,netmodule,evalmodule}/mine/vision-cls.yaml
teia train datamodule=mine/vision-cls netmodule=mine/vision-cls evalmodule=mine/vision-cls data_root=...
```

## How the config tree is laid out

```
node/data/<kind>/        leaves: one component each (_target_ + arguments)
node/net/<kind>/
node/eval/<kind>/
datamodule/<name>/       wired graphs
netmodule/<task>/
evalmodule/<task>/
task/<task>.yaml         contract + one default per graph
trainer/ callbacks/ …    run-wide settings
```

A leaf never says where it goes or what it reads. A module mounts leaves under names and wires their `in`/`out` keys. A netmodule ends with a `capture:` map naming which outputs leave the net for evaluation.

## Your project overlay

A `conf/` directory in your project is layered on top of the installed config. Put new files there under new names and select them:

```bash
teia config init                          # stubs: conf/node/net/{head,loss}/custom.yaml, conf/node/data/reader/custom.yaml
teia config list                          # the layers, in order
teia config show netmodule=vision-cls/mlp --resolved
teia config explain task=vision-cls netmodule=vision-cls/mlp
teia config lint
```

A project file at the same path as a packaged option does not override it; give it a new name.

## Determinism

`runtime.determinism.mode` is `prefer` by default (deterministic kernels where available, a warning otherwise). Use `force` to fail on nondeterministic kernels, or `stochastic` to allow them. Do not set `trainer.deterministic`.

## The run directory and later stages

`teia train` composes, trains, tests and evaluates (when a test split exists), and exports with `--export`. Everything lands in `runs/<project>/<experiment>/<run>/`, including the composed config. Later stages replay that snapshot:

```bash
teia test   --run-dir runs/proj/exp/<run>
teia eval   --run-dir runs/proj/exp/<run>            # rebuild eval/ from the captured predictions
teia infer  --run-dir runs/proj/exp/<run> --src new_images/ --dst preds/
teia export --run-dir runs/proj/exp/<run>
```

Multi-phase training is several `teia train` calls, each seeding from the previous run:

```bash
teia train task=vision-cls data_root=/data netmodule.encoder.freeze=true run_name=head-warmup
teia train --from-run-dir runs/proj/exp/head-warmup --from-run-weights best netmodule.encoder.freeze=false
```

---

Next: **[Custom modules](custom_modules.md)**. The full config contract is in [core/config](../specs/core/config.md) and [core/task](../specs/core/task.md).
