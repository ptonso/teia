"""``KeyJoin`` — align indexed streams by key into per-item aligned raw tuples.

The executor calls the join at plan stage with one ``{key → raw}`` dict per input stream (in ``in_key``
order) and assigns each returned row positionally to the join's ``out_key``
(under ``outer``: one raw per stream, then one ``bool`` present flag per stream). ``how=inner`` keeps keys
present in every stream (e.g. image ⨝ label); ``left`` keeps the first stream's keys (missing label →
``None`` raw, an empty-label image); ``outer`` keeps the union.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

from teia.base.data import Join


class KeyJoin(Join):
    """Align indexed streams by key (``inner``/``left``/``outer``); ``outer`` appends per-input present flags."""
    def __init__(self, *, how: str = "inner", on: str = "key", prefix: bool = False) -> None:
        if how not in ("inner", "left", "outer"):
            raise ValueError(f"KeyJoin.how must be inner/left/outer; got {how!r}.")
        if on != "key":
            raise ValueError(f"KeyJoin.on must be 'key'; got {on!r}.")
        self.how = how
        self.prefix = prefix

    def __call__(self, *streams: dict[Any, Any]) -> list[tuple]:
        key_sets = [set(stream) for stream in streams]
        keys = {"inner": set.intersection, "left": lambda *k: k[0], "outer": set.union}[self.how](*key_sets)
        rows = [tuple(stream.get(key) for stream in streams) for key in sorted(keys)]
        if self.how != "outer":
            return rows
        return [row + tuple(key in ks for ks in key_sets) for row, key in zip(rows, sorted(keys))]
