"""Frame-format support: manifest/split/schema/preprocess helpers shared by schema-bound data nodes.

Private support for the columnar (dataframe/row) data format — readers, schema-bound transforms, the
window reshape, and the csv writer. Carries no modality: any source that resolves to typed named columns
reuses it. Filesystem-root resolution lives in :mod:`teia.node.data._paths`.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from teia.node.data._paths import resolve_root  # noqa: F401  (re-exported for schema-bound importers)

from teia.base.schema import (
    CORE_SLOTS,
    INDEX_SLOTS,
    NUMERIC_SLOTS,
    TEMPORAL_FEATURES,
    ColumnMeta,
    ColumnSchema,
    FittedColumnState,
    SchemaError,
    PreprocessState,
    Record,
    Schema,
)

if TYPE_CHECKING:
    import pandas as pd

SPLITS = ("train", "val", "test")
MAX_CATEGORICAL_CARD = 20


@dataclass
class Manifest:
    """Parsed ``data.yaml``: split/source locations, id/entity/time columns, targets, forecast lag."""

    data_root: Path
    task: str | None = None
    targets: list[str] | None = None
    splits: dict[str, str] | None = None
    source: str | None = None
    id_col: str | None = None
    entity_col: str | None = None
    time_col: str | None = None
    event_col: str | None = None
    lookback: int | None = None
    horizon: int | None = None
    lag: dict[str, Any] | None = None


@dataclass
class Window:
    """One sliding-window sample: ``history`` rows predicting the ``horizon`` rows that follow them."""

    history: list[Record]
    horizon: list[Record]
    row_id: Any = None
    entity: Any = None
    time: Any = None


def load_manifest(data_root: Path, manifest_name: str = "data.yaml") -> Manifest:
    """Read ``data.yaml`` under ``data_root`` (missing file → an all-default ``Manifest``)."""
    payload: dict[str, Any] = {}
    path = data_root / manifest_name
    if path.exists():
        payload = yaml.safe_load(path.read_text()) or {}
    lag = payload.get("lag") or {}
    targets = list(payload.get("targets") or [])
    if not targets and isinstance(lag, dict) and lag.get("horizon_cols"):
        targets = [str(name) for name in lag["horizon_cols"]]
    if not targets and payload.get("time_col") and payload.get("event_col"):
        targets = [str(payload["time_col"]), str(payload["event_col"])]
    forecast = payload.get("forecast") or {}
    return Manifest(
        data_root=data_root,
        task=payload.get("task"),
        targets=targets,
        splits={str(k): str(v) for k, v in (payload.get("splits") or {}).items()},
        source=payload.get("source"),
        id_col=payload.get("id_col"),
        entity_col=forecast.get("entity_col") or payload.get("entity_col"),
        time_col=forecast.get("time_col") or payload.get("time_col"),
        event_col=payload.get("event_col"),
        lookback=forecast.get("lookback") or payload.get("lookback"),
        horizon=forecast.get("horizon") or payload.get("horizon"),
        lag=lag or None,
    )


def split_frame(frame: "pd.DataFrame", split: str, controls: dict[str, Any]) -> "pd.DataFrame":
    """Deterministic train/val/test row split for a whole-file source (seeded shuffle, optional stratify)."""
    import numpy as np

    if split == "infer":
        return frame.reset_index(drop=True)
    val_frac = float(controls.get("val_frac", 0.15) or 0.0)
    test_frac = float(controls.get("test_frac", 0.15) or 0.0)
    rng = np.random.default_rng(int(controls.get("seed", 42) or 0))
    positions = np.arange(len(frame))
    stratify_by = controls.get("stratify_by")
    if stratify_by and stratify_by in frame.columns:
        ordered: list[int] = []
        for _, group in frame.groupby(stratify_by, sort=True):
            idx = group.index.to_numpy()
            rng.shuffle(idx)
            ordered.extend(idx.tolist())
        positions = np.array(ordered, dtype=int)
    else:
        rng.shuffle(positions)
    n_test = int(round(len(frame) * test_frac))
    n_val = int(round(len(frame) * val_frac))
    selected = {
        "test": positions[:n_test],
        "val": positions[n_test : n_test + n_val],
        "train": positions[n_test + n_val :],
    }[split]
    return frame.iloc[sorted(selected.tolist())].reset_index(drop=True)


def to_python(value: Any) -> Any:
    """A pandas/numpy scalar (or ``NaN``) → its plain Python value (``None`` for missing)."""
    import pandas as pd

    try:
        if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (ValueError, AttributeError):
            return value
    return value


def records_frame(records: list[Record]) -> "pd.DataFrame":
    """``Record`` list → a flat ``DataFrame`` of their merged ``cells``/``target`` columns."""
    import pandas as pd

    return pd.DataFrame([{**record.cells, **record.target} for record in records])


def column_meta(frame: "pd.DataFrame") -> list[ColumnMeta]:
    """Observed ``(name, dtype)`` metadata for every column in ``frame``, pre-schema-resolution."""
    return [ColumnMeta(str(name), str(frame[name].dtype)) for name in frame.columns]


def infer_type_slot(series: "pd.Series", meta: ColumnMeta) -> str:
    """Guess a column's schema type slot (boolean/categorical/temporal/count/numerical) from its values."""
    import pandas as pd

    non_null = series.dropna()
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    distinct = set(non_null.unique().tolist()) if len(non_null) else set()
    if distinct and distinct <= {0, 1, True, False, "0", "1", "true", "false", "True", "False"}:
        if len(distinct) <= 2:
            return "boolean"
    textual = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "temporal"
    if textual and len(non_null):
        parsed = pd.to_datetime(non_null.head(50), errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return "temporal"
    if isinstance(dtype, pd.CategoricalDtype) or textual:
        return "categorical"
    if pd.api.types.is_integer_dtype(dtype):
        if int(non_null.nunique()) <= MAX_CATEGORICAL_CARD:
            return "categorical"
        return "count" if len(non_null) and float(non_null.min()) >= 0 else "numerical"
    return "numerical"


def validate_declared_schema(declared: ColumnSchema, series: "pd.Series") -> None:
    """Raise ``SchemaError`` if observed values contradict a user-declared ``ColumnSchema`` type slot."""
    import pandas as pd

    slot = declared.type_slot
    non_null = series.dropna()
    if not len(non_null):
        return
    if slot in ("numerical", "count"):
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().mean() > 0.1:
            raise SchemaError(f"Column {declared.name!r} is declared {slot!r} but is not numeric.")
        if slot == "count" and float(coerced.min()) < 0:
            raise SchemaError(f"Column {declared.name!r} is declared 'count' but contains negatives.")
    elif slot == "temporal":
        parsed = pd.to_datetime(non_null, errors="coerce")
        if parsed.isna().mean() > 0.1:
            raise SchemaError(f"Column {declared.name!r} is declared 'temporal' but is not parseable.")
    elif slot == "boolean" and int(non_null.nunique()) > 2:
        raise SchemaError(f"Column {declared.name!r} is declared 'boolean' but has more than two values.")


def build_groups(meta: list[ColumnMeta], columns: dict[str, ColumnSchema], targets: list[str]) -> dict[str, list[str]]:
    """Non-target column names bucketed by their schema type slot (``CORE_SLOTS``)."""
    groups: dict[str, list[str]] = {slot: [] for slot in CORE_SLOTS}
    target_set = set(targets)
    for item in meta:
        col = columns.get(item.name)
        if col is not None and item.name not in target_set:
            groups.setdefault(col.type_slot, []).append(item.name)
    return groups


def load_schema_yaml(schema_path: Path) -> Schema | None:
    """Parsed ``Schema`` from ``schema_path``, or ``None`` if it doesn't exist yet."""
    if not schema_path.exists():
        return None
    return Schema.from_dict(yaml.safe_load(schema_path.read_text()) or {})


def write_schema_yaml(schema_path: Path, schema: Schema) -> None:
    """Serialize ``schema`` to ``schema_path`` as YAML, creating parent directories as needed."""
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(yaml.safe_dump(schema.to_dict(), sort_keys=False))


def resolve_schema(
    *,
    train_frame: "pd.DataFrame",
    columns: list[ColumnMeta],
    targets: list[str],
    task: str,
    schema_path: Path,
    entity_col: str | None = None,
    time_col: str | None = None,
    lookback: int | None = None,
    horizon: int | None = None,
) -> Schema:
    """Load-or-create ``schema.yaml``: validate declared columns against observed data, autofill new ones."""
    if task != "infer" and not targets:
        raise SchemaError(f"No training targets declared for task {task!r}; declare `targets:` in data.yaml.")
    observed = {meta.name for meta in columns}
    schema = load_schema_yaml(schema_path)
    existed = schema is not None
    if schema is None:
        schema = Schema(columns={}, targets=list(targets), task=task, groups={})
    missing = [name for name in schema.columns if name not in observed]
    if missing:
        raise SchemaError(f"schema.yaml declares columns not present in the dataset: {sorted(missing)}")
    autofilled = False
    for meta in columns:
        existing = schema.columns.get(meta.name)
        if existing is None:
            schema.columns[meta.name] = ColumnSchema(meta.name, infer_type_slot(train_frame[meta.name], meta))
            autofilled = True
        else:
            validate_declared_schema(existing, train_frame[meta.name])
    schema.task = task
    schema.targets = list(targets)
    schema.groups = build_groups(columns, schema.columns, targets)
    schema.entity_col = entity_col
    schema.time_col = time_col
    schema.lookback = lookback
    schema.horizon = horizon
    if autofilled or not existed:
        write_schema_yaml(schema_path, schema)
        warnings.warn(f"Created or updated tabular schema.yaml at {schema_path}.", UserWarning, stacklevel=2)
    return schema


def fit_preprocess_state(
    *, train_frame: "pd.DataFrame", schema: Schema, schema_path: Path, split_assignment: dict[str, Any] | None = None
) -> PreprocessState:
    """Fit per-column stats (mean/std, vocab, temporal features) on the train split, per the schema's slots."""
    fitted: dict[str, FittedColumnState] = {}
    targets = set(schema.targets)
    for name, col in schema.columns.items():
        state = FittedColumnState(name=name)
        series = train_frame[name] if name in train_frame.columns else None
        if name in targets:
            if schema.task == "cls" and series is not None:
                vocab = _class_vocab(series)
                state.vocab = vocab
                state.cardinality = len(vocab)
            fitted[name] = state
            continue
        if series is None:
            fitted[name] = state
            continue
        if col.type_slot in NUMERIC_SLOTS:
            mean, std = _fit_numeric(series)
            state.mean = mean
            state.std = std
            state.impute_value = mean
        elif col.type_slot in INDEX_SLOTS:
            vocab = _input_vocab(series)
            state.vocab = vocab
            state.cardinality = len(vocab) + 1
        elif col.type_slot == "temporal":
            state.temporal_features = list(TEMPORAL_FEATURES)
        fitted[name] = state
    return PreprocessState(
        columns=fitted,
        schema_path=str(schema_path),
        fitted_on="train",
        split_assignment=split_assignment,
    )


def expand_temporal(value: Any, features: list[str] | None) -> list[float]:
    """A timestamp value → its requested cyclical/elapsed ``features`` as floats (zeros if unparseable)."""
    names = features or []
    if value is None:
        return [0.0] * len(names)
    import pandas as pd

    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return [0.0] * len(names)
    table = {
        "sin_dow": math.sin(2 * math.pi * ts.dayofweek / 7.0),
        "cos_dow": math.cos(2 * math.pi * ts.dayofweek / 7.0),
        "sin_month": math.sin(2 * math.pi * (ts.month - 1) / 12.0),
        "cos_month": math.cos(2 * math.pi * (ts.month - 1) / 12.0),
        "elapsed": float((ts.value / 1e9) / (365.25 * 24 * 3600)),
    }
    return [float(table.get(name, 0.0)) for name in names]


def _fit_numeric(series: "pd.Series") -> tuple[float, float]:
    import pandas as pd

    values = pd.to_numeric(series, errors="coerce").dropna()
    if not len(values):
        return 0.0, 1.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return mean, 1.0 if std == 0.0 or std != std else std


def _input_vocab(series: "pd.Series") -> dict[Any, int]:
    return {value: idx + 1 for idx, value in enumerate(sorted(series.dropna().unique().tolist(), key=str))}


def _class_vocab(series: "pd.Series") -> dict[Any, int]:
    return {value: idx for idx, value in enumerate(sorted(series.dropna().unique().tolist(), key=str))}
