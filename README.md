# teia

A Hydra-configured deep-learning runtime. A run is three config-declared component graphs, one for data, one for the network and one for evaluation, executed by a thin Lightning engine. Components are organized by architectural constraint, and a task contract types the seams between the graphs.

## Install

The package is published on PyPI as **`pyteia`**. It installs the `teia` command and the `teia` Python package:

```bash
pip install pyteia
teia --help
python -c "import teia.core"
```

Teia needs Python 3.11 or newer and PyTorch. To pick a PyTorch build for your hardware (CPU only, or a specific CUDA version), install it first by following https://pytorch.org/get-started/locally/, then install `pyteia`.

Optional component dependencies are grouped as extras named after a domain:

```bash
pip install "pyteia[vision]"    # image readers and transforms (torchvision)
pip install "pyteia[audio]"     # audio readers and spectrograms (torchaudio, soundfile)
pip install "pyteia[tabular]"   # SQL and parquet readers (sqlalchemy, pyarrow)
pip install "pyteia[all]"       # all of the above
```

You do not need to guess: when a configured run needs a missing package, teia stops before training and prints the exact install command.

A project that builds on teia lists `pyteia` in its own dependencies, for example `dependencies = ["pyteia>=0.0.1"]` in its `pyproject.toml` (`teia plugin init` writes this for you).

## Quick start

Two demos live in the repository's `demo/` folder and need no dataset download:

```bash
git clone https://github.com/ptonso/teia && cd teia
pip install "pyteia[vision]"
./demo/digits.sh      # scikit-learn digits as an image folder, Flatten + MLP
./demo/tabular.sh     # 3-class table, MLP
```

Each script runs a small `prepare_*.py` that writes the data under `demo/data/`, then runs `teia train` with a datamodule, a netmodule and an evalmodule. Extra arguments are forwarded to `teia train`.

CLI:

- `teia config init|list|show|tree|explain|lint`
- `teia init`
- `teia train`, `teia test`, `teia eval`, `teia infer`, `teia export`
- `teia plugin init`, `teia plugin check`: scaffold a project as a teia plugin and verify it ([specs/plugin](https://github.com/ptonso/teia/blob/main/specs/plugin.md))

A **task** is a contract over what the graphs exchange (`teia.task.*`). A task preset selects a datamodule, a netmodule and an evalmodule. You can also wire the three modules yourself:

```bash
teia train \
  datamodule=tabular-cls/csv \
  netmodule=tabular-cls/numerical_mlp \
  evalmodule=tabular-cls/default \
  data_root=...
```

## Docs

- [Using teia](https://github.com/ptonso/teia/blob/main/docs/usage.md)
- [Configuration](https://github.com/ptonso/teia/blob/main/docs/configuration.md)
- [Custom modules](https://github.com/ptonso/teia/blob/main/docs/custom_modules.md)
- [Specs](https://github.com/ptonso/teia/blob/main/specs/README.md): the contracts for contributors.

## License

Apache License 2.0, see [LICENSE](https://github.com/ptonso/teia/blob/main/LICENSE).
