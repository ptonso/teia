# Make the project a plugin

A plugin is the shape a Teia project takes so that it can be installed, reused and published. Its Python lives in one package named after the project, and every config file it ships sits in a folder with that same name, the **owner folder**. Those two names make ownership visible: anyone reading `_target_: headvlm.node.net...` or `netmodule=vision-det/headvlm/graph` knows which package to install, and two plugins can never collide. A small search-path file registers the configs with Teia when the package is installed.

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
      node/<graph>/<kind>/<package>/<name>.yaml   leaf      -> node/<graph>/<kind>@...=<package>/<name>
      <graph>module/<task>/<package>/<name>.yaml  module    -> <graph>module=<task>/<package>/<name>
      task/<package>/<task>.yaml                  preset    -> task=<package>/<task>
  conf/                                  local overrides only; never shipped
```

The package name is the distribution name with `-` replaced by `_`. The owner folder always sits right after the fixed part of a group path, as the three config lines show.

## Steps

1. **Scaffold.** From the project root, run `teia plugin init --name <distribution>`. It creates the files of the layout that are missing and keeps every existing file. If a `pyproject.toml` already exists, `teia plugin check` will name what it lacks.
2. **Add a `LICENSE`** matching the `license` field. Ask the user which license, and suggest a permissive one (Apache-2.0, MIT) unless the project wraps copyleft code ([baselines](baselines.md#where-the-code-lives)).
3. **Move existing work in.** For a project that already has configs and Python:
   - Python components go to `src/<package>/node/...` at the path their kind belongs to ([write-component](write-component.md#2-place-it-where-its-neighbours-are)), contracts to `src/<package>/task/`.
   - Config files go to the matching path in `src/<package>/conf/`, with the owner folder inserted. A task preset is named after the task it selects (`task/<package>/vision-cls.yaml`), whatever dataset it is meant for.
   - Update every `_target_` to the new dotted path, every `defaults:` mount to the new option name (`/node/net/encoder@encoder: <package>/<name>`), and every command line and run script to the new selections.
   - Keep in the root `conf/` only personal overrides that should not ship, such as the `mine/` files `teia config init` writes.
4. **Install it editable:** `pip install -e .` inside the project's environment. `teia config list` now shows a layer named after the package.
5. **Check:** `teia plugin check`. Fix each finding and run it again until it reports no error. Each line names a rule and the file at fault. The table below covers the fixes that are not obvious from the message.
6. **Run it:** a one-batch pre-flight with the plugin's own preset, `teia train task=<package>/<task> data_root=<dir> ++trainer.fast_dev_run=true`.

| Rule | Usual fix |
|---|---|
| `installed`, `registered` | reinstall with `pip install -e .` in the same environment that runs `teia`; keep `hydra_plugins/` without an `__init__.py` |
| `conf-owner` | move the file into the owner folder and update every reference to its option name |
| `import` | add the distribution that provides the module to `dependencies`, or to an optional extra when only some components need it |
| `docstring` | copy the module docstring shape of a neighbour ([write-component](write-component.md#3-use-a-neighbour-as-your-template)) |
| `base-class` | subclass the Teia base for the node's kind ([write-component](write-component.md#1-find-your-kind-and-its-base-class)) |
| `task-name` | rename the preset after the task it selects; list tasks with `teia config tree --root task` |
| `task-duplicate` | another package already defines that task: reuse its contract if it means the same thing, or give yours a distinct name |
| `name-duplicate`, `source-duplicate` (warnings) | read the other component; if it is the same method, use it and delete yours |
| `provider` | the project uses a `teia.*` component from a package other than Teia, which blocks publishing ([publish](publish.md)) |

## Report

List the files moved or created, the option names that changed (old and new, since the user's scripts and notes use them), the final `teia plugin check` result and the pre-flight result.
