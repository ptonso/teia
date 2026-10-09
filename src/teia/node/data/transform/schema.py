"""Schema-bound transforms — contract: require a resolved column schema (``requires_schema``).

Every node fans out from / rewrites a columnar ``Record`` (or windowed sequence of records) using the
train-fitted ``Schema`` + ``PreprocessState``, so the graph MUST contain a ``ResolveSchema`` plan node —
the executor rejects the graph at build otherwise (``requires_schema``). No modality: any source that
resolves to typed named columns uses these. Grouped by role: ``ResolveSchema`` (publish schema),
``SchemaField``/``Numerical``/``Categorical``/… (fan out atomic fields), ``Target`` (targets),
``Impute``/``Scale``/``Encode`` (in-place record rewrites), ``NumericalNoise``/``CategoricalSwap``
(augments).

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from teia.base.data import Transform
from teia.base.schema import INDEX_SLOTS, NUMERIC_SLOTS, PreprocessState, Record, Schema
from teia.node.data._frame import (
    Window,
    column_meta,
    expand_temporal,
    fit_preprocess_state,
    records_frame,
    resolve_root,
    resolve_schema,
)


def _expand_temporal_kernel(value: Any, features: list[str]) -> list[float]:
    """Teia-free (stdlib-only) twin of ``_frame.expand_temporal`` — collectable into a bundle
    (the pandas-based original lives in a different module, so the collector can't inline it)."""
    import math
    from datetime import datetime

    if not value:
        return [0.0] * len(features)
    try:
        ts = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return [0.0] * len(features)
    table = {
        "sin_dow": math.sin(2 * math.pi * ts.weekday() / 7.0),
        "cos_dow": math.cos(2 * math.pi * ts.weekday() / 7.0),
        "sin_month": math.sin(2 * math.pi * (ts.month - 1) / 12.0),
        "cos_month": math.cos(2 * math.pi * (ts.month - 1) / 12.0),
        "elapsed": ts.timestamp() / (365.25 * 24 * 3600),
    }
    return [float(table.get(name, 0.0)) for name in features]


class _SchemaBound(Transform):
    stateful = True
    requires_schema = True

    def __init__(self, *, task: str | None = None, root: str | None = None, schema_file: str = "schema.yaml") -> None:
        self.task = task
        self.root = root
        self.schema_file = schema_file
        self.data_root: str | None = None
        self.project_dir: str | None = None
        self.schema: Schema | None = None
        self.preprocess_state: PreprocessState | None = None
        self.fitted = False

    def fit(self, train_stream: Iterable[Any]) -> None:
        records: list[Record] = []
        for item in train_stream:
            if isinstance(item, Record):
                records.append(item)
            elif isinstance(item, Window):
                records.extend(item.history)
                records.extend(item.horizon)
        if not records:
            raise ValueError(f"{type(self).__name__} cannot fit without train rows.")
        frame = records_frame(records)
        first = records[0]
        targets = list(first.target.keys())
        root = resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)
        schema_path = root / self.schema_file
        task = str(self.task or first.meta.get("task") or "")
        self.schema = resolve_schema(
            train_frame=frame,
            columns=column_meta(frame),
            targets=targets,
            task=task,
            schema_path=schema_path,
            entity_col=first.meta.get("entity_col"),
            time_col=first.meta.get("time_col"),
            lookback=first.meta.get("lookback"),
            horizon=first.meta.get("horizon"),
        )
        self.preprocess_state = fit_preprocess_state(
            train_frame=frame,
            schema=self.schema,
            schema_path=schema_path,
            split_assignment=first.meta.get("split_controls"),
        )
        self.fitted = True

    def state(self) -> dict[str, Any]:
        self._require_fit()
        return {"schema": self.schema, "preprocess_state": self.preprocess_state}

    def load_state(self, state: Any) -> None:
        self.schema = state["schema"] if isinstance(state, dict) else state.schema
        self.preprocess_state = state["preprocess_state"] if isinstance(state, dict) else state.preprocess_state
        self.fitted = True

    def runtime_dims(self) -> dict[str, Any]:
        if self.schema is None or self.preprocess_state is None:
            return {}
        dims: dict[str, Any] = {"num_targets": self.schema.num_targets(), "target_names": list(self.schema.targets)}
        if self.schema.task == "cls" and self.schema.targets:
            target = self.schema.targets[0]
            vocab = self.preprocess_state.columns[target].vocab or {}
            labels = [value for value, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
            dims.update(num_classes=len(labels), num_labels=len(labels), class_names=[str(v) for v in labels])
        elif self.schema.task == "multi-cls":
            dims.update(num_classes=len(self.schema.targets), num_labels=len(self.schema.targets), class_names=list(self.schema.targets))
        return dims

    def _require_fit(self) -> tuple[Schema, PreprocessState]:
        if self.schema is None or self.preprocess_state is None:
            raise RuntimeError(f"{type(self).__name__} has no fitted tabular state.")
        return self.schema, self.preprocess_state


class ResolveSchema(_SchemaBound):
    """Fit and publish the dataset schema/preprocess state."""

    def __call__(self, row: Record) -> Schema:
        del row
        return self._require_fit()[0]

    def export_artifacts(self) -> dict[str, Any]:
        schema, state = self._require_fit()
        return {"schema": schema.to_dict(), "preprocess": {name: col.to_dict() for name, col in state.columns.items()}}


class SchemaField(_SchemaBound):
    """Base fan-out node: one schema type-slot's columns → a flat per-row list of values."""

    def __init__(self, *, slot: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.slot = slot

    def __call__(self, item: Record | Window) -> list[Any]:
        p = self.params()
        rows = _history(item)
        values = [type(self).kernel(row.cells, p)[0] for row in rows]
        return values[0] if isinstance(item, Record) else values

    def params(self) -> dict[str, Any]:
        schema, state = self._require_fit()
        columns = schema.groups.get(self.slot, [])
        if self.slot in NUMERIC_SLOTS:
            return {
                "columns": columns,
                "means": [state.columns[column].mean for column in columns],
                "stds": [state.columns[column].std for column in columns],
                "impute_values": [state.columns[column].impute_value for column in columns],
            }
        if self.slot in INDEX_SLOTS:
            return {"columns": columns, "vocabs": [dict(state.columns[column].vocab or {}) for column in columns]}
        if self.slot == "temporal":
            return {
                "columns": columns,
                "features": [list(state.columns[column].temporal_features or []) for column in columns],
            }
        return {"columns": columns}


class Numerical(SchemaField):
    """``numerical`` columns → standardized ``(value - mean) / std`` floats, imputed at the column mean."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="numerical", **kwargs)

    @staticmethod
    def kernel(cells: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
        out = []
        for col, mean, std, imp in zip(
            params["columns"], params["means"], params["stds"], params["impute_values"]
        ):
            v = cells.get(col)
            raw = imp if v is None else float(v)
            out.append((raw - mean) / std if std else raw)
        return out, {}


class Count(SchemaField):
    """``count`` columns (non-negative integers) → standardized floats; shares ``Numerical``'s kernel."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="count", **kwargs)

    kernel = Numerical.kernel


class Categorical(SchemaField):
    """``categorical`` columns → their fitted-vocab integer index (``0`` for unseen values)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="categorical", **kwargs)

    @staticmethod
    def kernel(cells: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[list[int], dict[str, Any]]:
        out = [int(vocab.get(cells.get(col), 0)) for col, vocab in zip(params["columns"], params["vocabs"])]
        return out, {}


class Boolean(SchemaField):
    """``boolean`` columns → their fitted-vocab integer index; shares ``Categorical``'s kernel."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="boolean", **kwargs)

    kernel = Categorical.kernel


class Temporal(SchemaField):
    """``temporal`` columns → their expanded cyclical/elapsed features (see ``_frame.expand_temporal``)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="temporal", **kwargs)

    @staticmethod
    def kernel(cells: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
        out: list[float] = []
        for col, features in zip(params["columns"], params["features"]):
            out.extend(_expand_temporal_kernel(cells.get(col), features))
        return out, {}


class PresenceMask(_SchemaBound):
    """Base presence-mask node: one schema type-slot's columns → a flat ``1.0``/``0.0`` observed-mask."""

    def __init__(self, *, slot: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.slot = slot

    def __call__(self, item: Record | Window) -> list[float]:
        p = self.params()
        rows = _history(item)
        values = [type(self).kernel(row.cells, p)[0] for row in rows]
        return values[0] if isinstance(item, Record) else values

    def params(self) -> dict[str, Any]:
        schema, state = self._require_fit()
        columns = schema.groups.get(self.slot, [])
        if self.slot == "temporal":
            widths = [len(state.columns[col].temporal_features or []) for col in columns]
        else:
            widths = [1] * len(columns)
        return {"columns": columns, "widths": widths}

    @staticmethod
    def kernel(cells: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
        out: list[float] = []
        for col, width in zip(params["columns"], params["widths"]):
            present = 0.0 if cells.get(col) is None else 1.0
            out.extend([present] * width)
        return out, {}


class NumericalMask(PresenceMask):
    """``numerical`` columns → their presence mask (one flag per column)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="numerical", **kwargs)


class TemporalMask(PresenceMask):
    """``temporal`` columns → their presence mask (one flag per expanded temporal feature)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slot="temporal", **kwargs)


class Static(_SchemaBound):
    """No-op schema-bound placeholder: fits (so the graph stays schema-consistent) but emits nothing."""

    def __call__(self, item: Record | Window) -> list[float]:
        del item
        self._require_fit()
        return []

    def params(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def kernel(cells: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]:
        del cells, params
        return [], {}


class Target(_SchemaBound):
    """Emits one training target field (``cls``/``time``/``event``/regression vector) from the schema's targets."""

    def __init__(self, *, field: str = "target", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.field = field

    def __call__(self, item: Record | Window) -> Any:
        schema, state = self._require_fit()
        rows = _horizon(item) if isinstance(item, Window) and self.field == "target" else _history(item)
        values = [_record_target(row, self.field, schema, state) for row in rows]
        return values[0] if isinstance(item, Record) else values


class NumericalNoise(_SchemaBound):
    """Training-time augment: adds ``N(0, sigma)`` Gaussian jitter to every ``numerical`` column value."""

    def __init__(self, *, sigma: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def __call__(self, record: Record) -> Record:
        schema, _ = self._require_fit()
        cells = dict(record.cells)
        for col in schema.groups.get("numerical", []):
            value = cells.get(col)
            if isinstance(value, (int, float)):
                cells[col] = float(value) + random.gauss(0.0, self.sigma)
        return Record(cells=cells, target=record.target, split=record.split, row_id=record.row_id, entity=record.entity, time=record.time, meta=record.meta)


class CategoricalSwap(_SchemaBound):
    """Training-time augment: with probability ``p``, swaps a ``categorical`` value for a random other class."""

    def __init__(self, *, p: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.p = float(p)

    def __call__(self, record: Record) -> Record:
        schema, state = self._require_fit()
        cells = dict(record.cells)
        for col in schema.groups.get("categorical", []):
            card = getattr(state.columns.get(col), "cardinality", None)
            if card and card > 2 and random.random() < self.p:
                cells[col] = random.randint(1, int(card) - 1)
        return Record(cells=cells, target=record.target, split=record.split, row_id=record.row_id, entity=record.entity, time=record.time, meta=record.meta)


class Impute(_SchemaBound):
    """In-place: fills missing values in numeric-slot columns with their fitted impute value (mean)."""

    def __call__(self, record: Record) -> Record:
        schema, state = self._require_fit()
        cells = dict(record.cells)
        for slot in NUMERIC_SLOTS:
            for col in schema.groups.get(slot, []):
                fitted = state.columns.get(col)
                if cells.get(col) is None:
                    cells[col] = float(getattr(fitted, "impute_value", 0.0) or 0.0)
        return Record(cells=cells, target=record.target, split=record.split, row_id=record.row_id, entity=record.entity, time=record.time, meta=record.meta)


class Scale(_SchemaBound):
    """In-place: standardizes numeric-slot columns to ``(value - mean) / std`` using the fitted stats."""

    def __call__(self, record: Record) -> Record:
        schema, state = self._require_fit()
        cells = dict(record.cells)
        for slot in NUMERIC_SLOTS:
            for col in schema.groups.get(slot, []):
                fitted = state.columns.get(col)
                value = float(cells.get(col, getattr(fitted, "impute_value", 0.0) or 0.0) or 0.0)
                cells[col] = (value - float(fitted.mean)) / float(fitted.std)
        return Record(cells=cells, target=record.target, split=record.split, row_id=record.row_id, entity=record.entity, time=record.time, meta=record.meta)


class Encode(_SchemaBound):
    """In-place: vocab-encodes index-slot columns and expands temporal columns to their cyclical features."""

    def __call__(self, record: Record) -> Record:
        schema, state = self._require_fit()
        cells = dict(record.cells)
        for slot in INDEX_SLOTS:
            for col in schema.groups.get(slot, []):
                vocab = getattr(state.columns.get(col), "vocab", None) or {}
                cells[col] = int(vocab.get(cells.get(col), 0))
        for col in schema.groups.get("temporal", []):
            fitted = state.columns.get(col)
            cells[col] = expand_temporal(cells.get(col), getattr(fitted, "temporal_features", None))
        return Record(cells=cells, target=record.target, split=record.split, row_id=record.row_id, entity=record.entity, time=record.time, meta=record.meta)


def _history(item: Record | Window) -> list[Record]:
    return [item] if isinstance(item, Record) else item.history


def _horizon(item: Record | Window) -> list[Record]:
    return [item] if isinstance(item, Record) else item.horizon


def _record_target(record: Record, field: str, schema: Schema, state: PreprocessState) -> Any:
    targets = list(schema.targets)
    if field == "cls" and schema.task == "cls":
        vocab = state.columns[targets[0]].vocab or {}
        return int(vocab.get(record.target.get(targets[0]), 0))
    if field == "cls":
        return [float(bool(record.target.get(col, 0))) for col in targets]
    if field == "time":
        return float(record.target.get(targets[0], 0.0) or 0.0)
    if field == "event":
        return int(bool(record.target.get(targets[1], 0) if len(targets) > 1 else 0))
    return [float(record.target.get(col, 0.0) or 0.0) for col in targets]
