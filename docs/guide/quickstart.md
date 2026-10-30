# Quick start

This page takes you from install to a first run, and lists the tasks teia ships with a demo each. It ends with links to the guides that go deeper.

If you work with an agent, run `teia init` in your project and ask `/teia teach me about this system`. It does the same job interactively.

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

When a configured run needs a missing package, teia stops before training and prints the exact install command.

A project that builds on teia lists `pyteia` in its own dependencies. `teia plugin init` writes that for you.

## First run

One demo per task lives in the repository's `demo/` folder and needs no dataset download:

```bash
git clone https://github.com/ptonso/teia && cd teia
pip install "pyteia[vision]"
./demo/vision-cls.sh
```

Each script calls a small script from `demo/prepare/` that writes the data under `demo/data/` (the reinforcement-learning demos need none, their world is a registered environment id), then runs `teia train task=<task>`. Extra arguments are forwarded to `teia train`.

Teia ships a `linear_probe` netmodule per task, a one-layer network that checks the data and the contract and is not meant to perform. You write the netmodule that does, in your project.

A task preset selects a datamodule, a linear probe and an evalmodule. You can also wire the three modules yourself:

```bash
teia train \
  datamodule=tabular-cls/csv \
  netmodule=tabular-cls/linear_probe \
  evalmodule=cls/default \
  data_root=...
```

## Tasks

A **task** is a contract over what the graphs exchange (`teia.task.*`).

| Task | Demo | Data |
|---|---|---|
| `vision-cls` | `demo/vision-cls.sh` | scikit-learn digits as an image folder |
| `vision-multi-cls` | `demo/vision-multi-cls.sh` | digits filed under parity and range labels |
| `vision-sem-seg` | `demo/vision-sem-seg.sh` | digits with a stroke mask |
| `vision-det` | `demo/vision-det.sh` | colored rectangles, boxes on a 4 x 4 grid |
| `vision-obb` | `demo/vision-obb.sh` | rotated rectangles as LabelMe polygons |
| `vision-pose` | `demo/vision-pose.sh` | rectangles with their four corners as keypoints |
| `vision-inst-seg` | `demo/vision-inst-seg.sh` | ellipses with outline polygons |
| `vision-gen` | `demo/vision-gen.sh` | linear autoencoder on digits (needs `pyteia[fid]`; the FID downloads the Inception weights once) |
| `vision-graph-node-cls` | `demo/vision-graph-node-cls.sh` | random graphs with a red or green image per node |
| `graph-node-cls` | `demo/graph-node-cls.sh` | random graphs in the TU Dortmund format |
| `tabular-cls` | `demo/tabular-cls.sh` | 3-class table |
| `tabular-multi-cls` | `demo/tabular-multi-cls.sh` | 3 independent labels over a table |
| `tabular-reg` | `demo/tabular-reg.sh` | noisy linear target over a table |
| `tabular-survival` | `demo/tabular-survival.sh` | right-censored event times in whole weeks |
| `tabular-forecast` | `demo/tabular-forecast.sh` | monthly series for three routes |
| `audio-cls` | `demo/audio-cls.sh` | synthetic tones as an audio folder (needs `pyteia[audio]`) |
| `audio-multi-cls` | `demo/audio-multi-cls.sh` | the same tones with pitch and duration labels |
| `audio-wave-cls` | `demo/audio-wave-cls.sh` | the same tones, read as raw waveforms |
| `rl-discrete` | `demo/rl-discrete.sh` | linear Q-function on CartPole (needs `pyteia[rl]`) |
| `rl-continuous-replay` | `demo/rl-continuous-replay.sh` | linear soft actor-critic on Pendulum |
| `rl-continuous` | `demo/rl-continuous.sh` | linear PPO on Pendulum |

`teia config tree --root task` prints the tasks of your installed version.

## Commands

- `teia config init|list|show|tree|explain|lint`
- `teia init`: creates the project `conf/` and links the teia skill
- `teia train`, `teia sweep`, `teia test`, `teia eval`, `teia infer`, `teia export`
- `teia plugin init`, `teia plugin check`: scaffold a project as a plugin and verify it

The full surface is in [core/cli](https://github.com/ptonso/teia/blob/main/specs/core/cli.md).

## Next

- [Using teia](../usage.md): the mental model of a run.
- [Configuration](../configuration.md): overrides, swapping components, the project overlay.
- [Custom modules](../custom_modules.md): writing your own network code.
