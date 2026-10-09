# Zoo Provenance Docstrings

Related: **Read first** [net/layout](net/layout.md). **See also** [plugin](plugin.md), [ecosystem](ecosystem.md), [net/types](net/types.md), [data/layout](data/layout.md), [eval/layout](eval/layout.md).

## Overview

Every component source file, in `teia.node` and in every plugin ([plugin](plugin.md)), carries stable provenance at the top of the file. The convention applies to every contract folder alike: data, net, eval, task. Component docstrings are provenance and behavior contracts, not changelogs.

## Contracts

### Module template

The first module statement MUST be a docstring in this shape:

```python
"""
<Component or component family name>.

Source:
  - title: "<paper/title or Common Knowledge>"
    url: "<canonical URL when available>"
    year: <year>

Description:
  <Two or three sentences.>

Adaptations:
  - <Conceptual or behavioral implementation difference.>
"""
```

`Adaptations` is omitted when the implementation closely follows the cited idea. `Source` is a YAML list and may hold several entries when a component combines a paper-level idea with a pinned upstream implementation.

### Compact form

For basic tensor adapters, losses, activations, package barrels and small utility wrappers with no meaningful original source:

```python
"""
<Component or component family name>.

Source: common knowledge

Description:
  <Two or three sentences.>
"""
```

### Upstream

A wrapper, a component that runs code from an external repository ([ecosystem](ecosystem.md)), adds an `Upstream` block after `Source`:

```python
Upstream:
  repo: "<repository URL>"
  commit: "<commit hash or tag>"
  license: "<SPDX identifier>"
```

### Symbol docstrings

Public top-level classes and functions MUST have a docstring. Math-heavy symbols SHOULD state the operation inline with compact formulas (objectives, sampling, decoding, assignment, schedules, normalization, geometry). Routine methods (simple `build_module`, `configure_from_datamodule`, pass-through `forward`) need none unless they implement non-obvious math.

## Constraints

- Cite canonical paper URLs first, and cite official or pinned upstream implementation URLs when behavior derives from library code.
- Include only conceptual or behaviorally relevant adaptations.
- The folder-level `<fan>/<assumes>/README.md` files ([net/layout](net/layout.md)) are exempt from the template.
- The check is `tests/test_node_docstrings.py`.
