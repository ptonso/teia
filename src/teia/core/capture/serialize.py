"""Tensor-to-plain serialization for capture writers.

``to_serializable`` produces the small human-facing payloads (JSON/YAML) the image and threshold
capture callbacks persist alongside their numeric columns.
"""

from __future__ import annotations

from typing import Any


def to_serializable(value: Any) -> Any:
    if value is None:
        return []
    detached = value
    for name in ("detach", "cpu"):
        method = getattr(detached, name, None)
        if callable(method):
            detached = method()
    tolist = getattr(detached, "tolist", None)
    if callable(tolist):
        return tolist()
    return detached
