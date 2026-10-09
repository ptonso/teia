# Using Teia

Teia is a **Hydra-composed deep-learning runtime**. You do not write a training loop, a data pipeline or an export script. You describe what you want in config, point the runtime at your data and run one terminal command. Teia composes the config, instantiates a datamodule and a model, trains it, evaluates it and snapshots everything into a reproducible run bundle.

Three graphs make a run: a **datamodule** (raw data to typed `batch.*` fields), a **netmodule** (fields to predictions and losses) and an **evalmodule** (captured predictions to metrics and plots). A **task** is a contract over what they hand each other.

```bash
./demo/digits.sh
```

## The mental model

```
compose config  ─►  instantiate  ─►  train  ─►  { test · eval · export }
(package + your      (datamodule        (writes a run bundle under
 optional overlay)    + model)           runs/<project>/<experiment>/<run>/)
```

- **Compose**: Teia merges the config shipped inside the installed package with an optional, thin `conf/` overlay in your project directory. Your overlay adds new names and selects them.
- **Instantiate**: the composed config becomes live objects: a datamodule reading your data, and a `TeiaNetModule` (a graph of model nodes and losses).
- **Train / test / eval / export**: `teia train` runs the fit, then tests and evaluates if a test split exists, and exports an ONNX bundle with `--export`. `test`, `infer`, `eval` and `export` can be replayed later against a saved run.

## Where to go next

- **[Configuration](configuration.md)**: override defaults, swap components, use a project `conf/` overlay, and understand the run bundle and multi-phase training.
- **[Custom modules](custom_modules.md)**: the module-graph contract and the `TeiaNode` / `BaseActivation` / `BaseLoss` interfaces for writing your own network code.

The precise contracts are in the package `specs/` tree.
