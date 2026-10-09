# Find out what exists

Teia can describe itself, and you should let it. The CLI reads the installed packages and the project together, so its answers match what a run will actually see. Work from the wide view to the narrow one, and stop as soon as you know enough.

## The questions, and the commands that answer them

| You want to know | Ask |
|---|---|
| What commands and flags exist | `teia --help`, `teia <command> --help` |
| Where options come from | `teia config list` prints the layers in search order. A layer named after a package means that plugin is installed, the project's own plugin included once it is installed ([make-a-plugin](make-a-plugin.md)). `project (none)` means the project has no local `conf/` yet, and `teia init` creates it. |
| Which tasks exist | `teia config tree --root task` |
| Which modules fit a task | `teia config tree --root netmodule/<task>`, and the same with `datamodule/<task>` or `evalmodule/<task>` |
| Which components exist for a purpose | `teia config tree --root node/net/<purpose>`, or `node/data/<kind>`, `node/eval/<kind>`. Without `--root` you see the whole tree, and `--node-only` hides the option files. |
| What a task promises | `teia config show task=<task>` |
| How a module is wired | `teia config show netmodule=<task>/<name>` (add `--resolved` for the composed result) |
| What one component is configured with | `teia config show node/net/<purpose>=<leaf>` |
| What my overrides changed | `teia config explain <overrides>` lists each group that changed and which override or preset set it |
| The exact config a run would use | `teia train <overrides> -c job` (`-c hydra` shows Hydra's own view, including search order) |
| Whether my config files are well shaped | `teia config lint` |
| Whether the project is a well-formed plugin | `teia plugin check` |

Everything above lists what is installed right now, so trust it over anything you remember about Teia.

## Reading what comes back

**A task** is a preset plus a contract. `config show task=<task>` shows the default module of each graph and a `task._target_` naming a Python class. Open that class: it lists the `batch.*` and `meta.*` keys the datamodule promises (seam A) and the `capture.*` atoms the network promises (seam B). Read it first when you need to understand what a task is for, what data it consumes, and what it predicts. The vocabulary it uses (tensor, ragged, blob, dimension, names) is defined in the `teia.base.task` module docstring.

**A module** shows two things: which leaves it mounts (`defaults:` entries such as `/node/net/head@head: <leaf>`), and how it wires them (each alias's `in` and `out`). Read the wiring the way you read a diagram: `in` keys come from a producer, `out` keys feed a consumer, and the netmodule's `capture:` map connects to the contract.

**A leaf** is a `_target_` and its arguments. To see what the class does, locate its source and read it:

```bash
python -c "import <module.path>; print(<module.path>.__file__)"
```

Component modules open with a docstring saying what the component is and where the idea comes from. Base classes (`teia.base.net`, `teia.base.data`, `teia.base.eval`) are short and state what the engine calls on each kind of node.

## Browsing the component libraries

Components live in Python namespace packages, so one folder name can be contributed by several installed distributions. List the real directories, then browse them like any code tree:

```bash
python -c "import teia.node.net.node as m; print(*[p for p in m.__path__ if p.startswith('/')], sep='\n')"
```

The same works for `teia.node.data`, `teia.task` and `teia.node.eval`. Folders in these trees are not decoration. They record what components assume about their inputs and what they can be swapped with, and the folders that need explaining carry a `README.md`. [write-component](write-component.md) uses them for placement.

## Does it already exist?

Before proposing anything new, run this audit. It usually ends at step 2 or 3, which saves a lot of work.

1. **By purpose.** List the purpose folder for the kind of thing you need (`node/net/<purpose>`, `node/eval/metric`, and so on) and read the names. Show two or three candidates and compare their arguments with what you need.
2. **By contract.** If the need is "another way to do X within this task", list the task's module folder. A module you can already select may do it.
3. **By text.** Search the layers' conf directories (printed by `teia config list`) and the component source directories for the class name, the technique's name, or a distinctive argument. Module docstrings name the paper under `Source:` and, for wrapped code, the repository under `Upstream:`, so search for those too. Include the project's own package and local `conf/`, since an earlier session may have written it.
4. **By near miss.** If something similar exists, ask whether a constructor argument, a node swap or a different wiring gets there. A new component is the last resort.

Say what you found in your report: "exists, use this", "exists nearly, configure it like this", or "does not exist, here is what I will write and where". A component found in another installed package is used through its dotted path, and that package goes into the project's dependencies. Then take the matching path: [swap](swap.md), [write-config](write-config.md) or [write-component](write-component.md).
