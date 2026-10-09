"""Plan-stage tabular sliding-window reshape.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any, Iterable

from teia.base.data import PlanReshape
from teia.base.schema import SchemaError, Record
from teia.node.data._frame import Window


class WindowRows(PlanReshape):
    """Group rows by entity, sort by time, and emit sliding ``(history, horizon)`` ``Window`` records."""

    def __init__(
        self,
        *,
        lookback: int | None = None,
        horizon: int | None = None,
        group: str | None = None,
        order: str | None = None,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.group = group
        self.order = order

    def payload_spec(self) -> dict[str, Any]:
        return {"lookback": self.lookback, "horizon": self.horizon, "group": self.group, "order": self.order}

    def __call__(self, rows: list[Record]) -> Iterable[Window]:
        if any(row.meta.get("is_lagged") for row in rows):
            raise SchemaError("window reshape cannot consume pre-lagged tabular data.")
        if rows and any(row.time is None for row in rows):
            raise SchemaError("window reshape requires a time_col in data.yaml.")
        lookback = int(self.lookback or (rows[0].meta.get("lookback") if rows else 1) or 1)
        horizon = int(self.horizon or (rows[0].meta.get("horizon") if rows else 1) or 1)
        groups: dict[Any, list[Record]] = {}
        for row in rows:
            groups.setdefault(row.entity, []).append(row)
        for group_rows in groups.values():
            ordered = sorted(group_rows, key=lambda row: (row.time is None, row.time))
            for idx in range(len(ordered) - lookback - horizon + 1):
                history = ordered[idx : idx + lookback]
                future = ordered[idx + lookback : idx + lookback + horizon]
                yield Window(
                    history=history,
                    horizon=future,
                    row_id=f"{history[0].row_id}:{future[-1].row_id}",
                    entity=history[0].entity,
                    time=history[-1].time,
                )
