# Write a component

Use this when the [audit](inspect.md#does-it-already-exist) found nothing installed that does the job. A component is a Python class plus a config option that points at it. Teia's components are deliberately uniform: each kind has a base class stating what the engine calls, each sits in a folder that says what it can be swapped with, and each copies the shape of its neighbours. Your job is to fit into that uniformity. A component that looks like its neighbours is easy to review, easy to swap, and might later graduate into the library.

## 1. Find your kind and its base class

The kind follows from where the node sits in the graph:

| The node… | It is a… | Base class |
|---|---|---|
| reads `batch.*` or `feat.*` and writes `feat.*` or `pred.*` | forward node | `TeiaNode` in `teia.base.net` |
| writes any `loss.*` key | loss | `BaseLoss` in `teia.base.net` |
| turns `pred.*` into `act.*`, and decodes it into per-sample atoms | activation | `BaseActivation` in `teia.base.net` |
| reads or transforms data before batching (files, columns, augmentation, batching, writing predictions) | data node | `Reader`, `Transform`, `Join`, `Reshape`, `Collate` or `Writer` in `teia.base.data` |
| computes a number or a table from captured predictions | metric | `Metric` in `teia.base.eval` |
| renders results into a file | view | `View` in `teia.base.eval` |

Read the base class before writing anything. Locate it with `python -c "import teia.base.net as m; print(m.__file__)"` (and the same for `teia.base.data`, `teia.base.eval`). Base classes are short, and their docstrings state what the engine calls, what it injects (input and output shapes, for instance), and which methods you must not override.

## 2. Place it where its neighbours are

Folders are how Teia records what a component assumes, so the right folder is part of the contract.

- **Forward nodes** are sorted by two questions. *Fan* is how many inputs and outputs it declares. *Assumes* is what must be true of its input for the math to be defined (an ordered axis, a spatial grid, a permutation-symmetric set, any tensor). The rule of thumb is that two components in one folder can replace each other without touching the wiring. The placement ladder lives in the `README.md` at the top of the `node/net` tree, and each leaf folder's README states its own contract. Read them and walk the ladder with your node's declared `in` and `out`. If the answer is not obvious, put the question to the user.
- **Activations and losses** live in flat folders, because what distinguishes them is the keys they read and write, not their fan.
- **Io nodes** and **eval nodes** are grouped by kind (reader, transform, collate…; metric, view…). Metrics also split by whether predictions pair one to one with targets or are compared as distributions. List the folder and read two members ([inspect](inspect.md#browsing-the-component-libraries)) to see where yours belongs.

The file goes in the project's plugin package at the same relative path the library uses: `src/<package>/node/net/<fan>/<assumes>/<name>.py`, and likewise under `node/data/` and `node/eval/` ([make-a-plugin](make-a-plugin.md#layout)). Installed packages are read-only, and the package name keeps you out of the shared `teia.*` namespace. Because the plugin is installed editable, the new module resolves in `_target_` as soon as it is saved.

## 3. Use a neighbour as your template

Find the closest existing member of the same folder, and read the whole file before you write yours ([inspect](inspect.md#reading-what-comes-back)). Then copy its shape:

- the module docstring: a title line, then `Source:` (either `common knowledge`, or a list of `- title:` / `url:` / `year:` entries for the paper), then `Description:` with two or three sentences. A component that runs code from another repository adds an `Upstream` block ([baselines](baselines.md#where-the-code-lives));
- one public class per module, named for the component;
- the constructor pattern: variations a config can express become arguments with defaults, so that one class serves many options;
- how it reads its inputs: forward and loss methods receive inputs positionally, in the order of the node's `in` list, and envelope keys such as `in`, `out`, `weight` and `detach` are stripped before construction, so no constructor argument may reuse those names;
- for an activation, the pattern the base docstring lays out: a traceable `activation` method, and an inline `kernel` that returns one flat dict of atoms per sample, with `params()` supplying any literals the kernel needs. What the kernel returns are the atoms your netmodule's `capture:` map will name.

If your idea needs a contract the neighbours do not follow, that is a sign you are in the wrong folder, or on a task boundary that needs a new contract ([write-config](write-config.md#a-new-task)).

## 4. Wire it in

Write a leaf in the owner folder that points at the class (`_target_: <package>.node.net...<Class>`) with its arguments, mount it in a module with the right `in` and `out`, and select the module ([write-config](write-config.md)). Reuse the surrounding installed leaves for everything else, so that the diff between your experiment and an installed setup is the component alone.

## 5. Check it

1. Instantiate the class alone and run it on a random tensor shaped like its input. This catches shape and dtype mistakes before Teia's machinery is involved.
2. Run `teia config lint`, then the one-batch pre-flight on the real data ([validate in layers](SKILL.md#ground-rules)). Wiring errors name the node and key: a missing producer, a duplicate output key, a cycle, an unresolvable dimension name, or a capture atom that the contract requires and the map lacks.
3. If Teia reports a missing third-party package, it prints the `pip install` line. Install it, add it to the plugin's `pyproject.toml`, and rerun.
4. Run `teia plugin check`: it confirms the docstring, the base class, the owner folder and the declared dependencies.

Then tell the user what you wrote, where it sits and why that folder, which neighbour you modelled it on, and how you checked it.
