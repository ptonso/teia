"""Multi-run comparison execution: the counterpart to ``run_eval_graph`` for ``Comparison``
nodes, which take every source run at once rather than one.

A compare graph is a flat map of independent nodes with no toposort or producer resolution.
Each node is instantiated from its ``_target_`` and run once against every source, unlike the
offline eval graph's wired ``in``/``out`` chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from teia.core.eval.runner import RunEvalSource
from teia.core.instantiate import instantiate

__all__ = ["run_compare"]

_RESERVED_KEYS = frozenset({"in", "out"})


def run_compare(run_dirs: list[str | Path], *, compare: Mapping[str, Any], split: str = "test", out_dir: str | Path) -> list[Path]:
    """Instantiate every node in ``compare`` (a flat ``{name: {_target_, **kwargs}}`` map) and
    run its ``.compare(sources, out_dir)`` against every ``run_dirs`` entry, each read via its
    own :class:`~teia.core.eval.runner.RunEvalSource`."""
    sources = [RunEvalSource(run_dir, split=split) for run_dir in run_dirs]
    resolved_out_dir = Path(out_dir)
    written: list[Path] = []
    for name, entry in compare.items():
        if not isinstance(entry, Mapping) or "_target_" not in entry:
            continue
        component_cfg = {key: value for key, value in entry.items() if key not in _RESERVED_KEYS}
        component = instantiate(component_cfg)
        if not hasattr(component, "compare"):
            raise ValueError(f"compare node '{name}': {entry['_target_']} is not a Comparison component.")
        written.extend(component.compare(sources, resolved_out_dir))
    return written
