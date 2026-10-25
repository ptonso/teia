"""Role binding for stream-mode buffers: the keys a buffer reads come from its mapping-form ``in``, and the keys it
writes come from its positional ``out``."""

from __future__ import annotations

from typing import Any


def bind_roles(node: Any, *, required: tuple[str, ...], optional: tuple[str, ...] = (), outs: tuple[str, ...]) -> tuple[dict[str, str], dict[str, str]]:
    roles = dict(node.in_roles or {})
    missing = [role for role in required if role not in roles]
    unknown = sorted(set(roles) - set(required) - set(optional))
    if missing or unknown:
        raise ValueError(f"{type(node).__name__} binds `in` roles {list(required)} (optional {list(optional)}); missing {missing}, unknown {unknown}.")
    if len(node.out_key) != len(outs):
        raise ValueError(f"{type(node).__name__} writes {list(outs)} through `out`; got {node.out_key}.")
    return roles, dict(zip(outs, node.out_key))
