"""Shared node-envelope grammar for both graphs (io + nn).

A node in either graph is an alias whose body is ``{<component-ref>, in, out, …optional-meta}``.
The component's kind (io) or role (nn) is inferred from its base class / ``out`` namespace, never
declared. ``ENVELOPE_KEYS`` is the one reserved-key vocabulary both ``teia.core.datamodule.graph`` and
``teia.core.module.net`` parse against. See teia:core/config.md, core/module/user_contracts.md,
base/data/nodes.md.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

ENVELOPE_KEYS: frozenset[str] = frozenset(
    {
        "_target_",
        "_recursive_",
        "in",
        "out",
        "phase",
        "stage",
        "weight",
        "optimizer",
        "detach",
        "aux_in",
        "last_layer",
        "bind",
    }
)

#: Keys stripped when building a component's instantiate config. ``_target_``/``_recursive_`` stay
#: (Hydra/``instantiate`` needs them) — the rest is pure Teia orchestration metadata.
_META_KEYS: frozenset[str] = ENVELOPE_KEYS - {"_target_", "_recursive_"}


def component_config(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The ``{_target_, **kwargs}`` instantiate spec: ``entry`` minus the orchestration envelope."""
    return {key: value for key, value in entry.items() if key not in _META_KEYS}


def check_envelope_collision(cls: type | None, entry: Mapping[str, Any], *, name: str) -> None:
    """Fail fast if a component's own constructor parameter is shadowed by a reserved envelope key.

    Only catches signatures that name the parameter explicitly (not ``**kwargs`` sinks) — the one
    blind spot of a flat envelope grammar. A shadowed parameter is silently absorbed as graph
    metadata instead of reaching the component, which is worse than a loud failure.
    """
    if cls is None:
        return
    target = getattr(cls, "build_module", None) or getattr(cls, "__init__", None)
    if target is None:
        return
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return
    shadowed = sorted(p for p in params if p in ENVELOPE_KEYS and p in entry)
    if shadowed:
        raise ValueError(
            f"Node '{name}': parameter(s) {shadowed} collide with reserved envelope keys "
            f"{sorted(ENVELOPE_KEYS)}; rename the component parameter — the envelope key always wins."
        )


__all__ = ["ENVELOPE_KEYS", "component_config", "check_envelope_collision"]
