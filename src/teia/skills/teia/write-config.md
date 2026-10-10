# Write new config

Use this when the components you need exist but no option combines them the way you want: another wiring, other settings, or a whole new module or task. New files go in the project's plugin package, `src/<package>/conf/`, at the same group paths the installed packages use ([make-a-plugin](make-a-plugin.md#layout)), and the command line selects them. A name must be new: a path that an installed package already ships stops the run that selects it.

The most reliable way to write a config file is to start from a real one that is close. Installed options were written to the shapes the engine and the linter expect, so copying one keeps you inside those shapes:

1. Find the closest option ([inspect](inspect.md)) and print it with `teia config show <group>=<option>`.
2. Save it under a new, descriptive name at the matching path in `src/<package>/conf/`.
3. Change what differs, and select it (`netmodule=<task>/<new-name>`, or `node/net/head@netmodule.head=<new-name>` for a leaf).
4. Validate: `teia config lint`, then a one-batch pre-flight ([validate in layers](SKILL.md#ground-rules)).

Two scaffolds save typing, and both write into the local `conf/`, which never ships. Move a file into the plugin package once it should. `teia config init` writes placeholder leaf stubs whose `_target_` names something that does not exist yet, so keep only the stubs you fill in. `teia config init --from task=<task>` writes the task's three modules fully resolved into `conf/{datamodule,netmodule,evalmodule}/mine/<task>.yaml`. You own those outright: select them with `datamodule=mine/<task> netmodule=mine/<task> evalmodule=mine/<task>`, and add `task=<task>` to keep the contract check.

## The three shapes

Each layer of configuration has its own shape, and `teia config lint` reports departures with a message saying what is expected.

**A leaf** (`node/<graph>/<purpose>/<name>.yaml`) holds a `_target_` and constructor arguments and nothing else. It has no `in` or `out`, and no `# @package` line. Keeping wiring out of the leaf is what lets one leaf serve many modules and tasks.

**A module** (`<graph>module/<task>/<name>.yaml`) mounts leaves and wires them:

- Mounting is a `defaults:` entry: `- /node/net/head@head: <leaf>` mounts a leaf under the name `head`, and `@_here_` merges a leaf's keys into the module's own root.
- Wiring is `in` and `out` under each mounted name (a loss also has `weight`). `in` lists the keys the node reads, and `out` maps each key it writes to a shape, or `null` when the shape is inferred. Shapes may use dimension names, such as a class count, which Teia resolves from the datamodule's `meta`. Prefer these names over copying numbers or writing `${...}` interpolations.
- A netmodule ends with the `capture:` map from each atom of the task's contract to an activation output (`act.<key>`) or a decoded atom (`post.<alias>.<atom>`). It is the network's side of seam B, so read the contract ([inspect](inspect.md#reading-what-comes-back)) to see which atoms are required.
- A mounted name must not contain `_group_` or `_name_`, because Hydra rewrites those. Lint catches it.
- A netmodule can declare preferences that the data graph consumes, such as the image size or normalization statistics, under `input:`. The datamodule reads them with an interpolation (`image_size: ${netmodule.input.image_size}`), so swapping the netmodule carries the preference along.
- Keys that are not nodes configure the module's executor: `batch_size` and `image_size` for a datamodule, `input`, `capture` and the optimizer fields for a netmodule, `enabled`, `split` and `capture` limits for an evalmodule. A node is any mapping that carries a `_target_`.
- A module in the project may name its own `_target_` to run a custom executor. Installed modules never do, so most modules should not.

**A task preset** (`task/<task>.yaml`) exists once per task, in the package that ships the task's default, so a project adds modules to an installed task and ships a preset only for a task it introduces. It starts with `# @package _global_`, sets `task._target_` to a contract class, selects the default module of each graph with `override /datamodule: ...` (and likewise `netmodule`, `evalmodule`), and may set run-wide options such as callbacks or trainer settings. It defines no nodes itself. Copy an installed preset and change the selections.

## Avoid `++`

Hydra's `+key=value` and `++key=value` force a key into the composed config, and they hide typos because they accept any path. Teia rejects a bare `+`, and no guide uses `++`. Put the setting in a file instead: give the leaf, the module or the preset the key, so that it exists and a plain `key=value` override works and warns on a typo. A run-time switch that needs a command-line form gets a real key and a flag, as `--dry-run` does for `trainer.fast_dev_run`; if you find yourself wanting `++`, say so in the report.

Other mistakes that lint and the composer do not always stop:

- Name fields by content (`batch.image`, `feat.pooled`), never by position.
- Take dataset-derived sizes from `out` dimension names, never from `${...}` interpolations.
- Do not add a `domain` or `task` string that routes behaviour. The task is identified by `task._target_` alone.

## A new task

A new task is warranted when the data or the predictions differ in kind from every installed contract, so that no existing module set could satisfy it. It has three parts, and you can build them one at a time:

1. **A contract class** in `src/<package>/task/<task>.py`. Its name must not exist in Teia or another installed plugin, and `teia plugin check` reports a clash. Read the `teia.base.task` docstring for the vocabulary, and open a contract of a similar task in `teia.task` as your template ([inspect](inspect.md#browsing-the-component-libraries)). Declare the batch fields and meta facts the datamodule will produce, and the capture atoms the network will produce.
2. **A module per graph** in `conf/<graph>module/<your-task>/`, each written as above, with the netmodule's `capture:` map covering the contract.
3. **A preset** in `conf/task/<your-task>.yaml` pointing at them and at your contract's dotted path.

Any node those modules need that does not exist is a component ([write-component](write-component.md)). Verify with `--dry-run`: the contract checks that the modules and the data agree, and its errors name what is missing.
