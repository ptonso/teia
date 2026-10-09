# Ecosystem Spec

Related: **Read first** [architecture](architecture.md), [stability](stability.md). **See also** [plugin](plugin.md), [base/task](base/task.md), [docstrings](docstrings.md).

## Overview

`teia` is the hub of a network of packages that speak one language: the base classes, the task contracts and the evaluation. A research project that follows the language is a **plugin**, an installable package that adds components, configs and tasks. Anyone who installs `teia` and a plugin runs the plugin's method with the same commands, on the same data pipeline, scored by the same evaluation as everything else.

Every piece of code has exactly one home. Reuse happens by installing the package that owns a component, never by copying it. Code that does not speak the language is wrapped once, and the wrapper becomes its single home.

Walk-through: a researcher starts a private project as a plugin, builds a new component, and compares it with existing methods on a teia task. When the paper is published, the project becomes public. A second project that wants the new component installs the first project at a tagged version. If the component becomes standard, it moves into `teia`.

## Language

- **plugin**: an installable package that depends on `pyteia` (the distribution that installs `teia`) and adds components, configs or task contracts under its own name ([plugin](plugin.md)).
- **home**: the one package that owns a piece of code.
- **building block**: a component any project could use (an encoder, a pooling layer, an adapter that loads the models of one external library).
- **contribution**: what a research project proposes, its new components, task contracts and configs.
- **baseline**: an existing method a project compares against.
- **wrapper**: a component that makes code which does not speak the language usable as a teia component.
- **upstream**: the external repository a wrapper or baseline comes from.

## Map

| Package | Holds | Released |
|---|---|---|
| `teia` | the language, the task contracts and the zoo of standard building blocks ([stability](stability.md)) | on PyPI as `pyteia` |
| research plugin | one project's contribution, its baselines and their configs | with the paper, as a tagged repository or on PyPI |
| wrapper plugin | the wrappers of one external library, under that library's license | on PyPI or as a tagged repository |
| private plugin | components still on test, private to their author | never |

## Contracts

### Where code lives

| Kind | Home |
|---|---|
| engine, base classes, task contracts that several projects share | `teia` |
| standard building block | `teia`, once it is generic, tested and permissively licensed |
| contribution | its research plugin, permanently |
| task contract a project introduces | its research plugin, until other projects depend on it ([stability](stability.md)) |
| wrapper used by one project | that project |
| wrapper used by several projects | its own wrapper plugin, licensed like the library it wraps |
| baseline config and glue | the research plugin that runs it |

A baseline is building blocks plus the config that reproduces one paper's setup. The blocks follow their own row of the table, and the config stays in the project.

### Reuse

- A component that speaks the language is reused by installing its home at a tagged version. It is never copied.
- Code that does not speak the language is wrapped once. Before writing a wrapper, look for an installed component whose docstring names the same upstream ([docstrings](docstrings.md)).
- A home that disappears or stops working with the current `teia` is forked, and the fork becomes the new home.

### Running a baseline

A comparison is fair when every method is scored on the same split by the same evaluation. Each way below guarantees that, and they differ in what else is shared.

| Way | What `teia` does | Also equal | When |
|---|---|---|---|
| trained | trains and scores it | data pipeline and training loop | the claim is about a component, so only the component may differ |
| frozen | runs it on the test data and scores its outputs | data pipeline | pretrained models, external pipelines, API calls |
| imported | scores predictions the baseline saved with its own code | nothing else | the baseline runs in its own framework |

Choose the cheapest way that keeps equal whatever the claim is about. A trained baseline is checked by loading the original weights and comparing outputs with the original code. The baselines a reviewer is most likely to challenge also reproduce the original paper's number under its own settings, and that result is kept with the project.

### Licenses

- `teia` is Apache-2.0 and accepts only code under a compatible permissive license.
- Code under a copyleft license (GPL, AGPL) lives only in a wrapper plugin under that same license, or in a project that accepts that license.
- A wrapper records the upstream repository, commit and license in its docstring ([docstrings](docstrings.md)).

### Publishing a research plugin

A project is ready to publish when:
1. It installs and runs with only released packages: `teia`, other published plugins and released libraries.
2. Every dependency is pinned to a released version or a tagged commit.
3. `teia plugin check --publish` reports no error ([plugin](plugin.md)).
4. Every baseline's upstream commit is recorded, together with how the baseline was run.

## Extending

A new research project starts as a plugin from day one ([plugin](plugin.md)). A component moves into `teia` when it is generic, tested, permissively licensed and used by more than one project, and the move is released as a minor version ([stability](stability.md)).

## Constraints

- A component MUST have exactly one home, and a package MUST NOT ship a copy of a component another package owns.
- A published plugin MUST NOT depend on an unreleased package.
- `teia` MUST NOT contain code under a copyleft license.
- A wrapper MUST record its upstream repository, commit and license.
