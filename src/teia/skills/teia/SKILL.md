---
name: teia
description: Work with Teia, a Hydra-configured deep-learning runtime, inside a project. Learn what tasks, modules and components are installed by asking the CLI, prepare data for a task, wire train, eval, infer and export runs and experiments, write new configs or Python components that respect Teia's contracts, including general, well-documented reimplementations of paper techniques, make the project an installable Teia plugin, wrap baselines for a fair comparison, and get a plugin ready to publish. Use whenever a project uses Teia (`teia train`, `teia plugin`, a `conf/` overlay, `task=...`).
---

# Working with Teia

Teia turns a training run into configuration. You do not write a training loop. You say which data to use and which parts to assemble, and Teia composes the run, trains it, evaluates it, and can later infer and export from it. This skill teaches you how Teia is organised and how to question it, so that you can help with three kinds of jobs: run a task's linear probe, assemble experiments out of parts that already exist, and write new configs or components when the part you need is missing.

A project that uses Teia is itself a **plugin**: an installable package that keeps its components and configs under its own name. Anyone who installs Teia and the plugin can run its method with the same commands and score it with the same evaluation as everything else, and other projects reuse its components by installing it.

The CLI describes the installed version exactly, so it is your source of truth for what exists. Whenever you need a fact about a task, a module or a component, ask it or open the installed code ([inspect](inspect.md) shows how) before assuming. A two-second question is cheaper than a confident guess.

## How a run is built

A run is three graphs of small components, called nodes. Each node reads and writes named keys, and the graphs hand over to each other at two well-defined seams:

```
your data ─▶ datamodule ──batch.*, meta.*──▶ netmodule ──capture.*──▶ evalmodule ─▶ eval/
             (data nodes)      seam A       (net nodes)    seam B    (metrics, views)
```

- The **datamodule** reads your files (or an interactive generator) and produces `batch.*`, the tensors and targets a model needs, plus `meta.*`, facts learned from the data such as class names or dimensions.
- The **netmodule** is the network. It turns a batch into losses for training and into predictions. It ends with a `capture:` map that names which outputs leave the network for evaluation.
- The **evalmodule** holds metrics and views (plots, tables). They read what the network captured, the targets and the meta, and write the `eval/` folder of the run.

Configuration mirrors that design in three layers:

| Layer | Where it lives | What it is |
|---|---|---|
| **node leaf** | `node/{data,net,eval}/<purpose>/<name>.yaml` | One component: a `_target_` plus its constructor arguments. It says nothing about wiring, so it can be reused anywhere. |
| **module** | `datamodule/` and `netmodule/` with one folder per task, `evalmodule/` with one folder per capture family (`cls`, `det`, ...) | A wired graph. It mounts leaves under names and connects their `in` and `out` keys. |
| **task** | `task/<name>.yaml` plus a Python contract class | A preset choosing the task's datamodule, its `linear_probe` netmodule and its evalmodule, and a contract stating what crosses seam A and seam B. |

The contract is what makes swapping safe. Every module in a task's folders promises to speak that task's contract, and Teia checks the promise before a run starts. So "which architectures can I try for this task?" is answered by listing the task's folder, and any of them can replace the default. Options are gathered from layers: Teia's core, every installed plugin (this project's included, once installed), and the project's local `conf/`.

## Three ways to work

1. **Run a task's probe.** Choose a task and point it at the data: `teia train task=<task> data_root=<dir>`. The run uses the task's `linear_probe`, a one-layer network that checks the data and the contract end to end and is not meant to perform. Teia ships no other netmodule.
2. **Write and swap components.** Keep the task, and replace the probe with a netmodule the project writes, a single node, or a value from the command line. This is how experiments are built, and the contract keeps them valid. A project netmodule starts as a copy of the probe: `teia config show netmodule=<task>/linear_probe`.
3. **Extend.** Write a new config option, a new component in Python, or a new task contract when nothing installed fits. Extensions live in the project's plugin package, so they ship with it.

Each step down costs more to maintain, so go only as far as the job needs. A request like "check that my data runs" lives at level 1, and "train a model on my data" starts by writing a netmodule at level 2. "Compare another backbone" or "sweep the learning rate" lives at level 2. "We have our own loss" or "our data has a different format" lives at level 3, and before you accept that, use [inspect](inspect.md) to check whether something installed already does the job.

## Ground rules

- **Write only inside the project**: its plugin package `src/<package>/` (Python and configs, see [make-a-plugin](make-a-plugin.md#layout)) and its local `conf/`, which holds personal overrides that do not ship. Installed packages are read-only. A component that needs changing becomes a new component in the project, with a new name.
- **Wiring and tuned recipes live in the project.** Never write a module, a task preset, a trainer, callbacks or an optimizer recipe into an installed package. A component proposed for Teia moves with its reference leaf only, and the wiring that used it stays in the project that wrote it.
- **Reuse by installing, or by a frozen copy.** A component that another published package owns is used through its dotted path, and the project lists that package as a dependency. Code that lives only in the author's private workshop never becomes a dependency: the project takes a frozen copy under its own package, and the copy is the project's code from then on. Code that does not speak Teia is wrapped once.
- **Every config path is new.** A file at the path of an installed option stops every run that selects it, with an error naming both copies, and `teia config lint` reports it. Give new files descriptive names of their own and select them.
- **Validate in layers, cheapest first.** `teia config lint` checks the shape of config files. `teia train <overrides> -c job` prints the composed config, which shows what your overrides did, but it does not check the task contract or run any node. `teia train <overrides> --dry-run` runs the real pre-flight (contract, wiring, dependencies), builds the graphs on the real data and runs one batch, so it catches most mistakes. A pass shows the wiring works, and says nothing about whether the model learns. Last, `teia plugin check` verifies the project is a well-formed plugin.
- **Read errors as instructions.** Contract and wiring errors name the class, the node and the key at fault. Fix that key instead of trying variations.
- **Finish with a short report**: files touched, commands run and what they showed, and open questions. A claim about the data or a result states how you checked it, so the user can tell a measurement from an impression.

## Guides

- [inspect](inspect.md): the start of every job. Ask the CLI what tasks, modules and components exist, read what they do, and audit whether a component you need already exists.
- [use-a-task](use-a-task.md): the user wants a run from a task's probe, including getting their data into the shape the task expects.
- [swap](swap.md): the user wants experiments from parts that already exist, and needs the command line for them.
- [write-config](write-config.md): no existing option fits, and the change is configuration: a leaf, a module, or a task preset.
- [write-component](write-component.md): the job needs a Python class that does not exist yet, and it has to sit in the right place and honour its base contract.
- [run](run.md): run, test, evaluate, infer or export, and find the outputs.
- [make-a-plugin](make-a-plugin.md): the project is not yet an installable plugin, or `teia plugin check` reports findings.
- [baselines](baselines.md): the user wants to compare against existing methods, and needs to decide how each one runs and where its code lives.
- [publish](publish.md): the project is about to become public, or another person must be able to reproduce it.
