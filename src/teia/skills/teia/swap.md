# Swap components and build experiments

Once a task runs, most experiments are variations on it: another architecture, another optimizer, a different resolution, a learning-rate sweep. Teia lets you express each as a command-line change while everything else stays as the task set it. The contract does the safety work: any module in the task's folders speaks the task's contract, so it is a valid replacement by construction.

## The forms

Selections and overrides are ordinary Hydra syntax, given after the command, and they apply after the task's own choices. Placeholders below are filled with names from [inspect](inspect.md).

| Goal | Form |
|---|---|
| Use a task's defaults | `task=<task>` |
| Replace a whole graph with another module for the task | `netmodule=<task>/<name>`, and likewise `datamodule=` or `evalmodule=` |
| Replace one node inside a module and keep its wiring | `node/<graph>/<purpose>@<module>.<alias>=<leaf>`, for example `node/net/head@netmodule.head=<leaf>` |
| Replace a leaf mounted at the module's root (an optimizer, for instance) | `node/net/<purpose>@netmodule=<leaf>` |
| Point at the dataset | `data_root=<dir>` |
| Change a value that exists | `netmodule.lr=1e-3`, `datamodule.batch_size=16` |
| Add a value that does not exist yet | `++trainer.fast_dev_run=true` |
| Work without a task (no contract check) | `datamodule=<x> netmodule=<y> evalmodule=<z>` |

A few things worth knowing:

- A node swap replaces the leaf's `_target_` and arguments and keeps the module's wiring. If the new leaf needs different `in` or `out`, override those in the same command, or write a module ([write-config](write-config.md)).
- A bare `+key=value` is rejected, and a missing key needs `++`. A plain `key=value` whose key does not exist prints a warning about a typo. That usually means a wrong path, so read it.
- The name of the key to override is what `config show` prints. Networks that require a particular input declare it in their own config (for example under `input:`), and the datamodule reads it, so switching architectures can switch the input with it. Look at the module before assuming a datamodule knob.
- A module from another task's folder is accepted when it satisfies the selected contract. Pre-flight names the key that does not.

## A good rhythm for one experiment

1. **List the candidates** in the task's folders, and read two or three of them ([inspect](inspect.md)) so that the choice is informed instead of alphabetical.
2. **Write the command** with the smallest set of overrides that expresses the idea.
3. **Check what it changes**: `teia config explain <overrides>` should show only the groups you meant to change, and `teia train <overrides> -c job` lets you read the composed result of the module you swapped ([validate in layers](SKILL.md#ground-rules)).
4. **Run the one-batch pre-flight**, then the real run ([run](run.md)).

## Several runs

To compare configurations, script one command per variant and keep the exact command lines in the script: the script is the record of the experiment. Give every run its own name (`project=`, `experiment=`, `--run-name`), so that runs land in separate directories. Compare finished runs with `teia eval --run-dir A --run-dir B`, which reads their captured predictions and summaries without a model. Multi-stage training is a chain of runs, each seeded from the previous one ([run](run.md)).

When no combination of existing parts expresses the idea, the next step is [write-config](write-config.md), and only after that [write-component](write-component.md).
