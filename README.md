# teia

Teia is a laboratory for neural networks. You plug in a task and a dataset, and teia trains, evaluates and exports models through the same pipeline every time. You change the part under test (the architecture, the dataset stack, the loss) with a config edit, and everything else stays identical.

It is built on PyTorch, with [Lightning](https://lightning.ai) running the training loop and [Hydra](https://hydra.cc) composing the configs. A run is three config-declared graphs of small components: one for data, one for the network and one for evaluation. A task contract types what the graphs hand to each other.

## Is teia for you?

Teia fits if you:

- compare architectures, datasets or training recipes and need the comparison to be fair and repeatable
- run many experiments and want each one fully described by its config
- want to spend your effort on your own contribution (a layer, a loss, a method) while training, evaluation and export stay stable
- work with an agent and want it to follow an explicit, shared structure

Teia asks you (or your agent) to learn its vocabulary: tasks, graphs, nodes and wiring. In return, every project that speaks it shares one pipeline and one evaluation.

## The ideas

- **Tasks.** A task is a contract over what the graphs exchange. Tasks cover vision, audio, tabular, graph and reinforcement-learning problems, from classification and detection to survival and control. See the [task catalog](https://github.com/ptonso/teia/blob/main/docs/guide/quickstart.md#tasks).
- **Configs are graphs.** Components are generic and the wiring in the config is where the experiment lives. See [configuration](https://github.com/ptonso/teia/blob/main/docs/configuration.md).
- **Everything swaps.** Replace a datamodule, a network or an evaluation from a config file or the command line, and the other parts stay fixed. Every run is seeded and snapshots its composed config.
- **Projects are plugins.** `teia init` turns your repository into a plugin, and its configs and components become visible to teia as overrides. See [plugin](https://github.com/ptonso/teia/blob/main/specs/plugin.md).

## Three ways to work

1. **Instant training.** Every task ships a linear probe network, so a dataset in a standard format trains and evaluates with one command. The probe checks your data and your contract.
2. **Wrapped components.** A wrapper turns code from an existing library into a teia component. It plugs into the same graphs and swaps by configuration. Wrappers live in your plugin or in a wrapper plugin ([ecosystem](https://github.com/ptonso/teia/blob/main/specs/ecosystem.md)).
3. **Custom architectures.** You write your own components and configs in your plugin. You go deep only on your contributions.

## Get started

```bash
pip install pyteia
teia init
```

`teia init` links the teia skill into your project for your agent. An agent that loads the skill knows the CLI, the contracts and the conventions, and can prepare your data, wire a run, write a component or scaffold a plugin. Then ask it:

```
/teia teach me about this system
```

If you prefer the traditional route, follow the [quick start](https://github.com/ptonso/teia/blob/main/docs/guide/quickstart.md). Teia needs Python 3.11 or newer and PyTorch.

## Docs

- [Quick start](https://github.com/ptonso/teia/blob/main/docs/guide/quickstart.md): install, the demos and the task catalog.
- [Using teia](https://github.com/ptonso/teia/blob/main/docs/usage.md): the mental model of a run.
- [Configuration](https://github.com/ptonso/teia/blob/main/docs/configuration.md): overrides, swapping components, the project overlay.
- [Custom modules](https://github.com/ptonso/teia/blob/main/docs/custom_modules.md): writing your own network code.
- [Specs](https://github.com/ptonso/teia/blob/main/specs/README.md): the contracts for contributors.

## License

Apache License 2.0, see [LICENSE](https://github.com/ptonso/teia/blob/main/LICENSE).
