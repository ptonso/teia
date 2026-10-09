# Dependency Spec

Related: **Read first** [overview](overview.md), [config](config.md). **See also** [runtime](runtime.md), [eval](eval.md).

## Overview

How teia manages dependencies across a flexible component landscape: which dependencies live in the engine, how optional component dependencies are grouped without a rigid install surface, and how a missing-dependency failure becomes an actionable install command. No dependency data is hardcoded in code.

There are three layers, each with one source of truth. Core dependencies live in `teia`'s `pyproject.toml`. Flexible groups are pyproject extras, read at runtime from installed distribution metadata. Component-particular dependencies are declared nowhere and resolved reactively by name.

Walk-through: components import optional packages plainly (`import cv2`). When an instantiate or engine stage fails with a dependency-shaped error, teia trial-imports every `_target_` module in the composed config, collects everything missing, matches those names against the extras of the installed distributions, and prints one exact command that solves the run.

## Language

- **core dependency**: a package required by `teia` itself and pinned in `[project.dependencies]` (hydra, lightning, torch, numpy, matplotlib, pillow, pandas, scikit-learn, onnx, rich). `teia.core.*` imports these directly.
- **flexible group**: a small bundle of optional dependencies commonly installed together, declared as a pyproject extra. `pyteia` names its groups after the domain whose components need them (`pyteia[vision]`), and a group is not exhaustive.
- **component-particular dependency**: a heavy, research or github-only dependency used by one component (one detector library, one chosen external simulator). It belongs to no group and is suggested on its own, only when that component is selected.
- **run-scoped resolution**: resolving a failure using only the `_target_` modules present in the composed config.

## Map

- `teia.core.deps`: the public surface below. `teia.base.deps` re-exports `MissingDependencyError` only.
- Call sites: `instantiate` and the engine stage runner (`raise_with_dependency_context`), and `validate_composed_config` for the eval subtree.
- Eval component placement: [eval/layout](../eval/layout.md).

## Contracts

### Public surface

`teia.core.deps` exposes:

- `MissingDependencyError(message, *, missing_modules=(), install_commands=(), notes=())`
- `require_dependency(package_name, feature, import_roots=()) -> None`, a shim that raises a fail-fast error with the declared package name.
- `has_module(module_name) -> bool`
- `render_dependency_message(*, feature, missing_modules=(), install_commands=(), groups=None, notes=())`
- `collect_missing(config) -> list[str]`
- `check_missing(config, *, feature) -> None`: `collect_missing` plus a raised `MissingDependencyError` with rendered guidance, used by the eval pre-flight gate.
- `dependency_error_for(exc, config, *, feature="Teia run") -> MissingDependencyError | None`
- `raise_with_dependency_context(exc, config, *, feature="Teia run") -> None`

### Group registry

Flexible groups are the `[project.optional-dependencies]` of the installed distributions. Teia reads the union of every installed distribution that contributes components, through `importlib.metadata.requires`, and skips absent distributions. The same code path serves a full development install and a published install with a smaller set. For illustration, a component library may declare:

```toml
[project.optional-dependencies]
vision = ["torchvision"]
audio = ["torchaudio", "soundfile"]
tabular = ["sqlalchemy", "pyarrow"]
all       = ["<dist>[vision,audio,tabular]"]
```

### Resolution on failure

1. Find the dependency cause (`ModuleNotFoundError`, `ImportError`, `MissingDependencyError`) in the exception chain. Non-dependency failures pass through unchanged.
2. Trial-import each distinct `_target_` module in the composed config and aggregate every top-level import it lacks, plus the original cause's missing module.
3. For each missing import root, match `normalize(root)` (`_` to `-`, lowercase) against the distributions in the extras. A **matched** root collects its extra and suggests `pip install <owner>[<extras>]`. An **unmatched** root is suggested directly, using the declared package name when available, else the normalized root.
4. Render one command for the run, joining grouped extras and direct packages into one `pip install` line.

```text
Configured run is missing optional dependencies.
Missing import(s): torchaudio, some_research_pkg.
python3 -m pip install pyteia[audio] some-research-pkg
```

A github-only or vendored component carries no code table. It imports the package plainly and MAY raise `MissingDependencyError(..., notes=(...))` at its own import site with a custom hint, which is merged into the message.

### The eval pre-flight

Eval nodes are instantiated inside `run_offline_eval`, after `trainer.fit`. So `validate_composed_config` (`_validate_eval_deps`) runs `check_missing` over the composed `eval` subtree and raises eagerly, before training. The mechanism is unchanged and only the moment moves. No other subtree is pre-scanned.

## Constraints

- No dependency name, import-root override or group membership MUST be hardcoded in `teia.core.*`. The only dependency data is the extras, read through `importlib.metadata`.
- Validation MUST be dependency-agnostic except for the `eval` subtree. Resolution otherwise runs only after a failure.
- Core dependencies MUST NOT be routed through optional-dependency resolution.
- An eval component MUST import its third-party dependencies at module top level. `collect_missing` trial-imports the module, so an import deferred into a method makes the dependency go undetected.
- `render_dependency_message` MUST emit exact solving commands and never fall back to `[all]`.
- An extra declared by several distributions MUST be attributed to `pyteia` first and to the other distributions in name order, so the suggested command names an installable distribution.

