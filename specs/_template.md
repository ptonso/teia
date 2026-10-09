# Spec Template

Related: **Read first** [architecture](architecture.md). **See also** [README](README.md).

## Overview

How a spec is written in this package: the live spec, a contract for a current or target state. It also fixes where each kind of spec lives. This spec is itself a live spec.

## Language

- **live spec**: a spec describing the current or target state of a contract, architecture or component taxonomy. It stays in sync with code and never presents a state other than its declared one as current.
- **`Related:` line**: the line directly under the title listing **Read first** specs (in reading order) and **See also** neighbours, as relative markdown links.

## Map

| What the spec documents | Lives in |
|---|---|
| Architecture, shared terms, domain-word rules | `specs/architecture.md`, `specs/terminology.md` |
| Torch-only base contracts: batch-field vocabulary, structure, data, net and eval ABCs, task contract | `specs/base/` |
| The engine: graph executors, config, runtime, capture, eval, infer, export | `specs/core/`, `specs/eval/` |
| Component libraries: data, net, docstring convention | `specs/data/`, `specs/net/`, `specs/docstrings.md` |

## Contracts

### Live spec

Sections, in order. A section is omitted only when it has nothing durable to hold.

```markdown
# <Area> Spec

Related: **Read first** [spec](path/to/spec.md). **See also** [spec](path/to/spec.md).

## Overview
## Language
## Map
## Contracts
## Extending
## Constraints
## History
```

### Placement

A spec lives in the folder of the thing it documents and links only within this package.

## Extending

To add a spec, place it by the Map, follow the live-spec sections, link it from `README.md`, and add it to a neighbour's `Related:` line.

## Constraints

- Every spec MUST open with `# <Title>` then a `Related:` line before its first `##` section.
- Every spec MUST be reachable from `README.md`, and every link MUST be a relative markdown link that resolves.
- A live spec MUST NOT describe a future or superseded state as current, and MUST NOT carry migration language, status or TODOs.
- Specs MUST be tracked by git.
