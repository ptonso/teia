# Make the project a plugin

A plugin is the shape a Teia project takes so that it can be installed, reused and published. Its Python lives in one package named after the project, so anyone reading `_target_: headvlm.node.net...` knows which package to install. Its configs use the same group paths as Teia's own, and a small search-path file registers them with Teia when the package is installed. Config paths are shared by every installed package, so each name must be new: a run that selects a path two packages ship stops with an error naming both, and `teia config lint` reports every clash.

Start every new project this way. An existing project with a root `conf/` and loose Python moves into the same shape.

## Layout

```
<project>/
  pyproject.toml                         depends on pyteia; ships hydra_plugins/ and the conf yaml
  LICENSE
  hydra_plugins/<package>_searchpath.py  registers src/<package>/conf (no __init__.py in hydra_plugins/)
  src/<package>/
    node/{data,net,eval}/...             components, at the same relative paths as teia.node
    task/<task>.py                       task contracts the project introduces
    conf/
      node/<graph>/<kind>/<name>.yaml   leaf                      -> node/<graph>/<kind>@...=<name>
      {data,net}module/<task>/<name>.yaml  module                 -> <graph>module=<task>/<name>
      evalmodule/<family>/<profile>.yaml   evaluation, new families only -> evalmodule=<family>/<profile>
      task/<task>.yaml                  preset, new tasks only    -> task=<task>
  conf/                                  local overrides only; never shipped
```

The package name is the distribution name with `-` replaced by `_`. A task has one preset across all packages, shipped with its default, so a plugin adds modules to an installed task and ships a preset only for a task it introduces.

## Steps

1. **Scaffold.** From the project root, run `teia plugin init --name <distribution>`. It creates the files of the layout that are missing and keeps every existing file. If a `pyproject.toml` already exists, `teia plugin check` will name what it lacks.
2. **Add a `LICENSE`** matching the `license` field. Ask the user which license, and suggest a permissive one (Apache-2.0, MIT) unless the project wraps copyleft code ([baselines](baselines.md#where-the-code-lives)).
3. **Move existing work in.** For a project that already has configs and Python:
   - Python components go to `src/<package>/node/...` at the path their kind belongs to ([write-component](write-component.md#2-place-it-where-its-neighbours-are)), contracts to `src/<package>/task/`.
   - Config files go to the same path in `src/<package>/conf/`. A preset for an installed task becomes the modules it selected plus a run script with those selections (`task=vision-cls netmodule=vision-cls/<name>`). A preset for a task the project introduces is named after it (`task/<task>.yaml`).
   - Update every `_target_` to the new dotted path, every `defaults:` mount to the new option name (`/node/net/encoder@encoder: <name>`), and every command line and run script to the new selections.
   - Keep in the root `conf/` only personal overrides that should not ship, such as the `mine/` files `teia config init` writes.
4. **Install it editable:** `pip install -e .` inside the project's environment. `teia config list` now shows a layer named after the package.
5. **Check:** `teia plugin check`. Fix each finding and run it again until it reports no error. Each line names a rule and the file at fault. The table below covers the fixes that are not obvious from the message.
6. **Run it:** a one-batch pre-flight with the plugin's own modules, `teia train task=<task> netmodule=<task>/<name> data_root=<dir> --dry-run`.

| Rule | Usual fix |
|---|---|
| `installed`, `registered` | reinstall with `pip install -e .` in the same environment that runs `teia`; keep `hydra_plugins/` without an `__init__.py` |
| `lint` with `[duplicate]` | another package ships the same config path: rename yours to a descriptive name (`resnet18_imagenet`) and update every reference to it |
| `import` | add the distribution that provides the module to `dependencies`, or to an optional extra when only some components need it |
| `docstring` | copy the module docstring shape of a neighbour ([write-component](write-component.md#3-use-a-neighbour-as-your-template)) |
| `base-class` | subclass the Teia base for the node's kind ([write-component](write-component.md#1-find-your-kind-and-its-base-class)) |
| `task-name` | rename the preset after the task it selects; list tasks with `teia config tree --root task` |
| `task-duplicate` | another package already defines that task: reuse its contract if it means the same thing, or give yours a distinct name |
| `name-duplicate`, `source-duplicate` (warnings) | read the other component; if it is the same method, use it and delete yours |
| `provider` | the project uses a `teia.*` component from a package other than Teia, which blocks publishing ([publish](publish.md)) |

## Report

List the files moved or created, the option names that changed (old and new, since the user's scripts and notes use them), the final `teia plugin check` result and the pre-flight result.
