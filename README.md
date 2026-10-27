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

One demo per task lives in the repository's `demo/` folder and needs no dataset download:

```bash
git clone https://github.com/ptonso/teia && cd teia
pip install "pyteia[vision]"
./demo/vision-cls.sh        # scikit-learn digits as an image folder
./demo/tabular-cls.sh       # 3-class table
./demo/graph-node-cls.sh    # random graphs in the TU Dortmund format
./demo/vision-graph-node-cls.sh # the same graphs with a red or green image per node
./demo/tabular-multi-cls.sh # 3 independent labels over a table
./demo/tabular-reg.sh       # noisy linear target over a table
./demo/tabular-survival.sh  # right-censored event times in whole weeks
./demo/tabular-forecast.sh  # monthly series for three routes
./demo/vision-multi-cls.sh  # digits filed under parity and range labels
./demo/vision-sem-seg.sh    # digits with a stroke mask
./demo/audio-cls.sh         # synthetic tones as an audio folder (needs pyteia[audio])
./demo/audio-multi-cls.sh   # the same tones with pitch and duration labels
./demo/audio-wave-cls.sh    # the same tones, read as raw waveforms
./demo/vision-det.sh        # colored rectangles, boxes on a 4 x 4 grid
./demo/vision-obb.sh        # rotated rectangles as LabelMe polygons
./demo/vision-pose.sh       # rectangles with their four corners as keypoints
./demo/vision-inst-seg.sh   # ellipses with outline polygons
./demo/vision-gen.sh        # linear autoencoder on digits (needs pyteia[fid]; the FID downloads the Inception weights once)
./demo/rl-discrete.sh       # linear Q-function on CartPole (needs pyteia[rl]; the world is gym:CartPole-v1, no data)
./demo/rl-continuous-replay.sh # linear soft actor-critic on Pendulum
./demo/rl-continuous.sh     # linear PPO on Pendulum
```

Each script calls a small script from `demo/prepare/` that writes the data under `demo/data/` (the reinforcement-learning demo needs none, its world is a registered environment id), then runs `teia train task=<task>`. Teia ships a `linear_probe` netmodule per task, a one-layer network that checks the data and the contract and is not meant to perform. You write the netmodule that does, in your project. Extra arguments are forwarded to `teia train`.

CLI:

- `teia config init|list|show|tree|explain|lint`
- `teia init`
- `teia train`, `teia test`, `teia eval`, `teia infer`, `teia export`
- `teia plugin init`, `teia plugin check`: scaffold a project as a teia plugin and verify it ([specs/plugin](https://github.com/ptonso/teia/blob/main/specs/plugin.md))

A **task** is a contract over what the graphs exchange (`teia.task.*`). A task preset selects a datamodule, a linear probe and an evalmodule. You can also wire the three modules yourself:

```bash
teia train \
  datamodule=tabular-cls/csv \
  netmodule=tabular-cls/linear_probe \
  evalmodule=cls/default \
  data_root=...
```

## Docs

- [Using teia](https://github.com/ptonso/teia/blob/main/docs/usage.md)
- [Configuration](https://github.com/ptonso/teia/blob/main/docs/configuration.md)
- [Custom modules](https://github.com/ptonso/teia/blob/main/docs/custom_modules.md)
- [Specs](https://github.com/ptonso/teia/blob/main/specs/README.md): the contracts for contributors.

## License

Apache License 2.0, see [LICENSE](https://github.com/ptonso/teia/blob/main/LICENSE).
