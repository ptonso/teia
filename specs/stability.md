# Stability and Versioning Spec

Related: **Read first** [architecture](architecture.md). **See also** [ecosystem](ecosystem.md), [base/task](base/task.md), [core/task](core/task.md), [docstrings](docstrings.md).

## Overview

`teia` ships as one distribution with three layers that change in different ways. The **language** (the engine and the base classes) almost never grows and is the part everything depends on. **Task contracts** form an open set: new tasks are added freely, and a task, once released, is frozen because it has become a shared language between the data graph, the net graph and the evaluation. The **zoo** (components, configs, task presets and default evalmodules) also grows freely, and its behavior and defaults improve over time.

The version number tells a user one thing: whether a project that ran on the previous version still runs. Changed numbers are not a breaking change. A run reproduces exactly from its config snapshot and the pinned `teia` version.

Scenario: a release replaces the `vision-cls` default netmodule with a wider one. Every project still runs, so it is a minor release, listed under "Results may change". A project that needs last year's numbers pins the older version, and its old runs replay from their snapshots.

## Language

- **language**: `teia.base`, `teia.core`, the batch-field vocabulary, the CLI and the config and run layouts.
- **task contract**: a class in `teia.task` typing what crosses the two graph boundaries ([base/task](base/task.md)).
- **zoo**: `teia.node`, `teia.callbacks` and everything under `src/teia/conf/` (node leaves, modules, task presets, default evalmodules).
- **released**: published on PyPI under a version number.
- **breaking change**: a change after which a config, project or run snapshot that worked on the previous version raises an error.

## Map

| Layer | Lives in | Grows | Once released |
|---|---|---|---|
| language | `teia.base`, `teia.core`, `src/teia/core/conf/` | rarely | changes only in a breaking release |
| task contracts | `teia.task` | freely, one module per task | frozen: fields, types and route never change |
| zoo | `teia.node`, `teia.callbacks`, `src/teia/conf/` | freely | behavior and defaults may change in any minor release |

## Contracts

### Version numbers

Versions follow `MAJOR.MINOR.PATCH`.

| Change | Release |
|---|---|
| A breaking change in any layer: a language change, a removed or renamed component, config option or task, a constructor argument removed or made required, a changed field in a released task contract | major |
| An addition: a new component, config, task contract or extra | minor |
| A change in results with no error: a new task preset or default evalmodule, a changed component default, a changed metric formula | minor |
| A fix that changes no interface | patch |

Before `1.0.0`, the minor digit plays the role of major and the patch digit covers everything else.

### Task contracts

A task that needs different fields, types or routes is a new contract with its own name. A research project that needs a new task defines the contract in its own plugin ([plugin](plugin.md)); it moves into `teia.task` when other projects start to depend on it.

### Changelog

`CHANGELOG.md` has one section per release. Any change that can alter results is listed under **Results may change**, with the task or component it affects.

## Extending

To add a component or task, add it in the zoo or in `teia.task` and release it as a minor version. To change a default, change the preset and list it under "Results may change". To change a released task contract, add a new contract instead.

## Constraints

- A change that makes a previously working config, project or run snapshot raise an error MUST be released as a major version.
- A released task contract MUST NOT change its fields, types or route.
- A release that can change results MUST list each such change under "Results may change" in `CHANGELOG.md`.
- The language MUST NOT import the zoo statically ([architecture](architecture.md)).
