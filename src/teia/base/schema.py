"""Schema-bound record contracts — value types of the columnar/frame data format.

Contract floor for data components that require a resolved column schema (``requires_schema``). Three
structures, carrying no modality — any format that resolves to named typed columns uses them:

- ``Record``          — one per-row item emitted by a row reader, consumed by schema-bound transforms.
- ``Schema``          — dataset-adjacent declarative schema (``schema.yaml``); owns column meaning,
                        auto-created/auto-filled but authoritative.
- ``PreprocessState`` — train-fitted statistics/vocabularies, run-owned, persisted into run/export
                        artifacts (never written back into schema.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SchemaError(ValueError):
    """Raised for hard data-contract violations (missing targets, bad declared types)."""


# Canonical core type-slots. Ordering here defines the per-type column order
# inside every batch tensor and inside ``Schema.groups``.
CORE_SLOTS: tuple[str, ...] = ("numerical", "categorical", "count", "temporal", "boolean")
EXTENSION_SLOTS: tuple[str, ...] = ("text", "ordinal", "high-cardinality")
ALL_SLOTS: tuple[str, ...] = CORE_SLOTS + EXTENSION_SLOTS

# Slots whose cells are integer lookup indices (reserved index 0 == missing/unknown).
INDEX_SLOTS: frozenset[str] = frozenset({"categorical", "boolean", "ordinal"})
# Slots whose cells are continuous floats fed through impute + scale.
NUMERIC_SLOTS: frozenset[str] = frozenset({"numerical", "count"})

# Fixed temporal expansion recipe (see contracts.md §5).
TEMPORAL_FEATURES: tuple[str, ...] = ("sin_dow", "cos_dow", "sin_month", "cos_month", "elapsed")


@dataclass
class Record:
    """One row as it flows through Materialize → Transform → collater.

    ``cells`` holds raw (pre-transform) values on emit from IO; the Transform pillar
    rewrites them in place into model-ready values (scaled floats, int indices, or
    expanded temporal feature lists). Presence flags live in ``meta['present']``.
    """

    cells: dict[str, Any]
    target: dict[str, Any] = field(default_factory=dict)
    split: str = "train"
    row_id: str | int | None = None
    entity: Any | None = None
    time: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def present(self, column: str) -> float:
        return float(self.meta.get("present", {}).get(column, 1.0))

    def set_present(self, column: str, value: float) -> None:
        self.meta.setdefault("present", {})[column] = float(value)


@dataclass
class ColumnMeta:
    """Raw, observed column metadata surfaced by an IO adapter."""

    name: str
    observed_dtype: str
    declared_type: str | None = None


@dataclass
class ColumnSchema:
    """Declarative meaning for one column, serialized into schema.yaml."""

    name: str
    type_slot: str
    hash_buckets: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type_slot": self.type_slot}
        if self.hash_buckets is not None:
            payload["hash_buckets"] = self.hash_buckets
        return payload

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "ColumnSchema":
        return cls(
            name=name,
            type_slot=str(payload["type_slot"]),
            hash_buckets=payload.get("hash_buckets"),
        )


@dataclass
class Schema:
    """Declarative, dataset-adjacent schema (``schema.yaml``)."""

    columns: dict[str, ColumnSchema]
    targets: list[str]
    task: str
    groups: dict[str, list[str]] = field(default_factory=dict)
    entity_col: str | None = None
    time_col: str | None = None
    lookback: int | None = None
    horizon: int | None = None

    def feature_count(self, type_slot: str) -> int:
        return len(self.groups.get(type_slot, []))

    def num_targets(self) -> int:
        return len(self.targets)

    def input_columns(self) -> list[str]:
        return [name for slot in CORE_SLOTS for name in self.groups.get(slot, [])]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "targets": list(self.targets),
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }
        forecast = {
            key: value
            for key, value in (
                ("entity_col", self.entity_col),
                ("time_col", self.time_col),
                ("lookback", self.lookback),
                ("horizon", self.horizon),
            )
            if value is not None
        }
        if forecast:
            payload["forecast"] = forecast
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Schema":
        columns = {
            name: ColumnSchema.from_dict(name, col)
            for name, col in (payload.get("columns") or {}).items()
        }
        forecast = payload.get("forecast") or {}
        return cls(
            columns=columns,
            targets=list(payload.get("targets") or []),
            task=str(payload.get("task", "")),
            groups={},
            entity_col=forecast.get("entity_col"),
            time_col=forecast.get("time_col"),
            lookback=forecast.get("lookback"),
            horizon=forecast.get("horizon"),
        )


@dataclass
class FittedColumnState:
    """Train-fitted preprocessing state for a single column."""

    name: str
    # numerical / count
    mean: float | None = None
    std: float | None = None
    impute_value: float | None = None
    # categorical / boolean / ordinal (and classification targets)
    vocab: dict[Any, int] | None = None
    cardinality: int | None = None
    # temporal
    temporal_features: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        for key in ("mean", "std", "impute_value", "cardinality", "temporal_features"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.vocab is not None:
            # JSON keys must be strings; record the original repr so we can round-trip.
            payload["vocab"] = {str(k): v for k, v in self.vocab.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedColumnState":
        vocab = payload.get("vocab")
        return cls(
            name=str(payload["name"]),
            mean=payload.get("mean"),
            std=payload.get("std"),
            impute_value=payload.get("impute_value"),
            vocab={k: int(v) for k, v in vocab.items()} if vocab is not None else None,
            cardinality=payload.get("cardinality"),
            temporal_features=payload.get("temporal_features"),
        )

    def expanded_width(self) -> int:
        """Number of model-ready feature columns this column contributes."""
        if self.temporal_features is not None:
            return len(self.temporal_features)
        return 1


@dataclass
class PreprocessState:
    """Train-fitted preprocessing artifact (run-owned, persisted to artifacts)."""

    columns: dict[str, FittedColumnState]
    schema_path: str
    fitted_on: str = "train"
    split_assignment: dict[str, Any] | None = None

    def cardinality(self, column: str) -> int:
        state = self.columns.get(column)
        if state is None or state.cardinality is None:
            raise KeyError(f"No fitted cardinality for column {column!r}")
        return int(state.cardinality)

    def num_classes(self, target: str | None = None) -> int:
        if target is not None:
            return self.cardinality(target)
        # Single-target convenience: first column carrying a cardinality.
        for state in self.columns.values():
            if state.cardinality is not None:
                return int(state.cardinality)
        raise KeyError("No classification cardinality fitted in preprocess state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fitted_on": self.fitted_on,
            "schema_path": self.schema_path,
            "split_assignment": self.split_assignment,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessState":
        return cls(
            columns={
                name: FittedColumnState.from_dict(col)
                for name, col in (payload.get("columns") or {}).items()
            },
            schema_path=str(payload.get("schema_path", "")),
            fitted_on=str(payload.get("fitted_on", "train")),
            split_assignment=payload.get("split_assignment"),
        )
