# Stability and Versioning Spec

Related: **Read first** [architecture](architecture.md). **See also** [ecosystem](ecosystem.md), [base/task](base/task.md), [core/task](core/task.md), [docstrings](docstrings.md).

## Overview

`teia` ships as one distribution with three layers that change in different ways. The **language** (the engine and the base classes) almost never grows and is the part everything depends on. **Task contracts** form an open set: new tasks are added freely, and a task, once released, is frozen together with its evalmodule and its `linear_probe`, because they have become a shared language between the data graph, the net graph and the evaluation. The **zoo** (components with their reference leaves, and the format datamodules) also grows freely, and its behavior and defaults improve over time. `teia` ships no tuned recipe, so no architecture result is promised ([ecosystem](ecosystem.md#where-code-lives)).

The version number tells a user one thing: whether a project that ran on the previous version still runs. Changed numbers are not a breaking change. A run reproduces exactly from its config snapshot and the pinned `teia` version.

Scenario: a release changes the default `dropout` of the `MLP` component, and with it the `mlp` reference leaf. Every project still runs, so it is a minor release, listed under "Results may change". A project whose netmodule relies on that default and needs last year's numbers pins the older version, and its old runs replay from their snapshots.

## Language

- **language**: `teia.base`, `teia.core`, the batch-field vocabulary, the CLI and the config and run layouts.
- **task contract**: a class in `teia.task` typing what crosses the two graph boundaries ([base/task](base/task.md)), shipped with its preset, evalmodule and `linear_probe` ([core/task](core/task.md)).
- **zoo**: `teia.node`, `teia.callbacks`, their reference leaves under `src/teia/conf/node/`, the format datamodules and the system groups under `src/teia/conf/`.
- **released**: published on PyPI under a version number.
- **breaking change**: a change after which a config, project or run snapshot that worked on the previous version raises an error.

## Map

- Language: `teia.base`, `teia.core`, `src/teia/core/conf/`.
- Task contracts: `teia.task`, with `conf/task/`, `conf/evalmodule/<task>/` and `conf/netmodule/<task>/linear_probe.yaml`.
- Zoo: `teia.node`, `teia.callbacks`, `conf/node/`, `conf/datamodule/<task>/`, the system groups under `conf/`.

## Contracts

### Layers

| Layer | Grows | Once released |
|---|---|---|
| language | rarely | changes only in a breaking release |
| task contracts, their evalmodules and `linear_probe` netmodules | freely, one per task | frozen: fields, types, route and the probe's wiring change only with the contract |
| zoo | freely | behavior and defaults may change in any minor release |

### Version numbers

Versions follow `MAJOR.MINOR.PATCH`.

| Change | Release |
|---|---|
| A breaking change in any layer: a language change, a removed or renamed component, config option or task, a constructor argument removed or made required, a changed field in a released task contract | major |
| An addition: a new component, config, task contract or extra | minor |
| A change in results with no error: a changed component or reference-leaf default, a changed format-datamodule default, a changed metric formula | minor |
| A fix that changes no interface | patch |

Before `1.0.0`, the minor digit plays the role of major and the patch digit covers everything else. The project is in beta until then, so a minor release may also change a released task contract in place, listed under **Breaking** in `CHANGELOG.md`.

### Task contracts

A task that needs different fields, types or routes is a new contract with its own name. A research project that needs a new task defines the contract in its own plugin ([plugin](plugin.md)); it moves into `teia.task` when other projects start to depend on it.

### Changelog

`CHANGELOG.md` has one section per release. Any change that can alter results is listed under **Results may change**, with the task or component it affects.

## Extending

To add a component, add it in the zoo with its reference leaf and release it as a minor version. To add a task, add its contract, preset, evalmodule, format datamodule and `linear_probe` together ([core/task](core/task.md#extending)). To change a default, change the component or its leaf and list it under "Results may change". To change a released task contract, add a new contract instead, except before `1.0.0` ([version numbers](#version-numbers)).

## Constraints

- A change that makes a previously working config, project or run snapshot raise an error MUST be released as a major version.
- From `1.0.0` on, a released task contract MUST NOT change its fields, types or route.
- A release that breaks a config, project, run snapshot or task contract MUST list each break under "Breaking" in `CHANGELOG.md`.
- A release that can change results MUST list each such change under "Results may change" in `CHANGELOG.md`.
- The language MUST NOT import the zoo statically ([architecture](architecture.md)).
- A `linear_probe` netmodule MUST change only when its task contract changes.
