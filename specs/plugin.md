# Plugin Spec

Related: **Read first** [ecosystem](ecosystem.md), [architecture](architecture.md). **See also** [stability](stability.md), [core/config](core/config.md), [core/task](core/task.md), [docstrings](docstrings.md).

## Overview

A plugin is an installable package that extends `teia` with components, configs and task contracts. Its Python lives in its own package, and its configs use the same group layout as `teia`. Installing a plugin registers its configs on the Hydra search path, so `teia train` selects them like any built-in option. A config path is unique across `teia`, every installed plugin and the project, and a run that selects a path shipped twice fails ([core/config](core/config.md#config-layers)).

`teia plugin check` verifies a plugin folder against this spec. Static rules read the source tree. Environment rules need the plugin installed in the current environment, and compare it with `teia` and every other installed plugin.

Scenario: a project named `myproj` adds an encoder and a module for `vision-cls`. After `pip install -e .`, `teia train task=vision-cls netmodule=vision-cls/conv data_root=/ds/pets` runs it, and `node/net/encoder@netmodule.encoder=conv` mounts the encoder inside any other netmodule.

## Language

- **package name**: the import name, the distribution name with `-` replaced by `_` (`my-proj` imports as `my_proj`).
- **component module**: a Python module under `<package>/node/` or `<package>/task/`.
- **provider**: the installed distribution a module or config file comes from.
- **finding**: one rule violation, an error or a warning.

## Map

```
myproj/
  pyproject.toml                      depends on pyteia, packages src/ and hydra_plugins/, ships *.yaml
  LICENSE
  hydra_plugins/
    myproj_searchpath.py              registers src/myproj/conf (named after the distribution; no __init__.py)
  src/myproj/skills/<name>/SKILL.md     optional: agent skills, linked into projects by `teia init`
  src/myproj/
    node/{data,net,eval}/...          components, laid out like teia.node
    task/<task>.py                    task contracts the project introduces
    conf/
      node/net/encoder/conv.yaml           node leaf                    -> conv
      netmodule/vision-cls/conv.yaml       module                       -> vision-cls/conv
      task/my-task.yaml                    preset of a task it adds     -> task=my-task
```

`teia plugin init [path] --name <distribution>` creates the files above that are missing and never overwrites one. The checker is `teia.core.plugin` and the command `teia plugin check [path] [--publish]`.

## Contracts

### Package and configs

- The Python package is `src/<package>/`, and it is the only package under `src/`.
- Components follow the same layout as `teia.node` ([data/layout](data/layout.md), [net/layout](net/layout.md), [eval/layout](eval/layout.md)) and subclass the matching base: data nodes a `teia.base.data` node class, net nodes `TeiaNode`, `BaseActivation` or `BaseLoss`, eval nodes a `teia.base.eval` node class. A sub-component mounted inside another node subclasses the extension point `teia` ships for it. Eval kernels are plain callables.
- Configs live in `src/<package>/conf/`, in the same group layout as `teia` ([core/config](core/config.md)). A path that another package already ships is a clash, so a new config gets a descriptive name of its own, such as `resnet18_imagenet`.

- The search-path plugin is `hydra_plugins/<distribution>_searchpath.py`, with `-` replaced by `_`, a `SearchPathPlugin` that appends the plugin's `conf/` with provider `<package>`. `hydra_plugins/` has no `__init__.py`.

### Task contracts

A project that needs a new task writes its contract in `<package>/task/<task>.py` and ships its preset `task/<task>.yaml`. A plugin ships a preset only for a task it introduces, and adds modules to the tasks of other packages (a shared-namespace plugin may ship the preset of a `teia` task that `teia` itself leaves without one). A task name is unique across `teia` and every installed plugin. A released contract follows [stability](stability.md).

### Rules

Static rules, read from the source tree:

| Rule | Checks |
|---|---|
| `pyproject` | `pyproject.toml` has a build system, a name, a version, a license, and depends on `pyteia` |
| `license-file` | a `LICENSE` file exists |
| `package` | `src/<package>/` exists and is the only package under `src/` |
| `searchpath` | `hydra_plugins/<distribution>_searchpath.py` exists, `hydra_plugins/__init__.py` does not, and the build packages both `hydra_plugins` and the conf yaml |
| `target` | every `_target_` into the plugin's own package names a module and a symbol that exist |
| `docstring` | every component module carries the provenance docstring, an `Upstream` block when present names a repository, a commit and a license, and every public top-level symbol has a docstring ([docstrings](docstrings.md)) |

Environment rules, which need the plugin installed:

| Rule | Checks |
|---|---|
| `installed` | the plugin is installed from this folder |
| `registered` | its `conf/` is on the search path |
| `import` | every module it imports is the plugin itself, the standard library, provided by `teia` or its dependencies, or provided by a declared dependency (a declared optional dependency may be absent) |
| `target-import` | every `_target_` in its configs imports |
| `base-class` | every node leaf, except eval kernels, targets a subclass of a class the `pyteia` distribution ships: a base class or an extension point |
| `task-duplicate` | no task it introduces has the name of a task in `teia` or another plugin |
| `task-name` | every task preset is named after the task it selects, a task with a contract in `teia` or in the plugin |
| `name-duplicate` (warning) | no component class shares its name with a component of another provider |
| `source-duplicate` (warning) | no component module cites the same source title as a component of the same kind (`node/<graph>/<kind>`) from another provider |
| `provider` | every `teia.*` module it imports, targets, or uses through a composed preset or module (a mounted leaf of another package included) comes from the `pyteia` distribution (an error with `--publish`, a warning without it) |
| `compose` | every task preset and module it ships composes |
| `lint` | `teia config lint` reports nothing in its files, a config path that another provider also ships (`duplicate`) included |

`--publish` also turns `license-file` into an error. The command prints one line per finding and exits non-zero when any error remains.

### Shared namespace

A private plugin maintained together with `teia` may ship its Python inside the `teia.*` namespace, so its components move into `teia` with no path change. It is checked with `--namespace teia`, which skips `package`. A published plugin always uses its own package.

## Extending

A new rule is a function in `teia.core.plugin` returning findings, registered in the static or environment list, with a row in the tables above and a test in `tests/test_plugin_check.py`.

## Constraints

- A plugin MUST ship its Python in its own package, unless it is checked with `--namespace teia`.
- A plugin MUST NOT ship a config path that `teia` or another installed plugin ships, and, unless it is checked with `--namespace teia`, MUST NOT ship a preset for a task it does not introduce.
- A plugin MUST register its configs through a search-path plugin and MUST NOT add `hydra_plugins/__init__.py`.
- A plugin MUST NOT introduce a task whose name already exists in `teia` or another installed plugin.
- `teia plugin check --publish` MUST report no error before a plugin is published.
