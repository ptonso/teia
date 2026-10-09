"""Streaming, semantic-free capture store: one directory per run/split.

A capture is written once, per batch, during a stage's callback and read back by eval nodes.
Neither side ever holds the whole split, so capture cost is O(batch) on write and O(rows
touched) on read regardless of split size. The store never interprets what a column means; it
persists whatever atomic keys the caller declares.

Three lanes, one per shape of captured data:

- **numeric**: one fixed-shape row per sample (class scores, targets). Raw C-contiguous bytes
  appended per batch, read back as an ``np.memmap``.
- **rows**: ragged or non-numeric per sample (image paths, per-image instance lists whose box
  count varies). JSONL plus an int64 offset sidecar, so row *i* is one seek. This is the
  access pattern of ``visual_examples``, which wants 6 rows out of thousands.
- **blobs**: per-sample rasters (predicted masks, generated images), already files on disk; the
  store only records where they are.

``manifest.json`` is written last: its presence means the capture is complete, and its absence
is a hard error at read time (report generation never re-runs inference). It also carries a
``routes`` map ``{route -> {atom -> column}}``, a convenience index letting a route's consumer
look up every atom it needs (predictions under ``capture.<route>.<atom>``, captured targets
under ``batch.<atom>``) by short field name, regardless of which column each atom lives in.

See packages/teia/specs/core/eval.md §4.2.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

from teia.core.utils import ensure_dir

SCHEMA_VERSION = 3
MANIFEST_NAME = "manifest.json"


def store_dir(run_dir: str | Path, split: str, *, subdir: str = "capture") -> Path:
    return Path(run_dir) / "artifacts" / subdir / split


class CaptureWriter:
    """Append-only writer for one run/split capture.

    Every column must be declared before it is appended to, and an append whose shape
    disagrees with the declaration is an error. A capture that silently changed width halfway
    through a split would produce a plausible-looking but wrong report.
    """

    def __init__(self, root: str | Path, *, split: str) -> None:
        self.root = Path(root)
        self.split = str(split)
        self._columns: dict[str, dict[str, Any]] = {}
        self._handles: dict[str, Any] = {}
        self._row_offsets: dict[str, list[int]] = {}
        self._counts: dict[str, int] = {}
        self._meta: dict[str, Any] = {}
        self._routes: dict[str, dict[str, str]] = {}
        self._closed = False
        ensure_dir(self.root)

    def declare_numeric(self, name: str, *, shape: tuple[int, ...], dtype: str = "float32") -> None:
        import numpy as np

        self._reject_duplicate(name)
        path = self.root / "columns" / f"{name}.bin"
        ensure_dir(path.parent)
        self._columns[name] = {
            "lane": "numeric",
            "dtype": str(np.dtype(dtype)),
            "shape": [int(dim) for dim in shape],
            "path": f"columns/{name}.bin",
        }
        self._handles[name] = path.open("wb")
        self._counts[name] = 0

    def declare_rows(self, name: str) -> None:
        self._reject_duplicate(name)
        path = self.root / "rows" / f"{name}.jsonl"
        ensure_dir(path.parent)
        self._columns[name] = {
            "lane": "rows",
            "path": f"rows/{name}.jsonl",
            "index": f"rows/{name}.idx",
        }
        self._handles[name] = path.open("wb")
        self._row_offsets[name] = [0]
        self._counts[name] = 0

    def declare_blob(self, name: str, *, ext: str = "png") -> None:
        self._reject_duplicate(name)
        ensure_dir(self.root / "blobs" / name)
        self._columns[name] = {"lane": "blob", "dir": f"blobs/{name}", "ext": str(ext)}
        self._counts[name] = 0

    def append_numeric(self, name: str, values: Any) -> None:
        """Append ``[B, *shape]`` rows. Peak memory is one batch; the OS page cache buffers."""
        import numpy as np

        column = self._require(name, "numeric")
        expected = tuple(column["shape"])
        array = np.ascontiguousarray(np.asarray(values, dtype=column["dtype"]))
        if array.ndim == len(expected):  # a single row
            array = array.reshape((1, *expected))
        if array.shape[1:] != expected:
            raise ValueError(
                f"Capture column {name!r} was declared with row shape {expected}, got {array.shape[1:]}."
            )
        array.tofile(self._handles[name])
        self._counts[name] += int(array.shape[0])

    def append_rows(self, name: str, items: list[Any]) -> None:
        self._require(name, "rows")
        handle = self._handles[name]
        offsets = self._row_offsets[name]
        for item in items:
            handle.write(json.dumps(item, separators=(",", ":"), default=str).encode("utf-8"))
            handle.write(b"\n")
            offsets.append(handle.tell())
        self._counts[name] += len(items)

    def blob_path(self, name: str, row: int, *, stem: str | None = None) -> Path:
        column = self._require(name, "blob")
        leaf = f"{stem or f'{row:06d}'}.{column['ext']}"
        return self.root / column["dir"] / leaf

    def record_blob(self, name: str) -> None:
        """Count a blob written at :meth:`blob_path` (the caller owns the encoding)."""
        self._require(name, "blob")
        self._counts[name] += 1

    def set_meta(self, **values: Any) -> None:
        self._meta.update(values)

    def declare_route(self, route: str, **atoms: str) -> None:
        """Index an activation-route's atom names onto already-declared capture columns.

        Purely a convenience map for :meth:`CaptureStore.routes`/:meth:`CaptureStore.slice`;
        the columns themselves are already usable by their fully-qualified name regardless
        of whether a route indexes them.
        """
        for column in atoms.values():
            if column not in self._columns:
                raise KeyError(f"Route {route!r} references undeclared capture column {column!r}.")
        self._routes[route] = dict(atoms)

    def close(self) -> Path:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        for name, offsets in self._row_offsets.items():
            (self.root / f"rows/{name}.idx").write_bytes(_pack_offsets(offsets))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "split": self.split,
            "n_rows": max(self._counts.values(), default=0),
            "columns": self._columns,
            "routes": self._routes,
            **self._meta,
        }
        path = self.root / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        self._closed = True
        return path

    def __enter__(self) -> "CaptureWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self._closed:
            self.close()

    def _reject_duplicate(self, name: str) -> None:
        if name in self._columns:
            raise KeyError(f"Capture column {name!r} is already declared.")

    def _require(self, name: str, lane: str) -> dict[str, Any]:
        column = self._columns.get(name)
        if column is None:
            raise KeyError(f"Capture column {name!r} was not declared.")
        if column["lane"] != lane:
            raise TypeError(f"Capture column {name!r} is a {column['lane']} column, not {lane}.")
        return column


class RowColumn:
    """Lazy view over a JSONL column; decodes only the rows actually indexed."""

    def __init__(self, path: Path, offsets: Any) -> None:
        self._path = path
        self._offsets = offsets

    def __len__(self) -> int:
        return max(0, len(self._offsets) - 1)

    def __getitem__(self, index: int) -> Any:
        count = len(self)
        resolved = index + count if index < 0 else index
        if not 0 <= resolved < count:
            raise IndexError(f"Row {index} out of range for {self._path.name} ({count} rows).")
        start = int(self._offsets[resolved])
        length = int(self._offsets[resolved + 1]) - start
        with self._path.open("rb") as handle:
            handle.seek(start)
            return json.loads(handle.read(length))

    def __iter__(self) -> Iterator[Any]:
        with self._path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


class RouteSlice(Mapping):
    """The ``{atom -> data}`` view an eval node consuming one route's atoms sees, resolved lazily."""

    def __init__(self, store: "CaptureStore", fields: dict[str, str]) -> None:
        self._store = store
        self._fields = dict(fields)

    def __getitem__(self, key: str) -> Any:
        return self._store.column(self._fields[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)


class CaptureStore:
    """Reader for a capture directory. The constructor touches only ``manifest.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        path = self.root / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"Report capture is missing or incomplete ({path}). Re-run the stage with the "
                "capture callback enabled; report generation does not re-run inference."
            )
        self.manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        self._columns: dict[str, dict[str, Any]] = self.manifest.get("columns") or {}
        self._cache: dict[str, Any] = {}

    @property
    def n_rows(self) -> int:
        return int(self.manifest.get("n_rows") or 0)

    @property
    def split(self) -> str:
        return str(self.manifest.get("split") or "")

    @property
    def class_names(self) -> list[str]:
        return [str(name) for name in (self.manifest.get("class_names") or [])]

    def meta(self, key: str, default: Any = None) -> Any:
        return self.manifest.get(key, default)

    def has(self, name: str) -> bool:
        return name in self._columns

    def column(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        spec = self._columns.get(name)
        if spec is None:
            raise KeyError(f"Capture has no column {name!r} (have: {sorted(self._columns)}).")
        lane = spec["lane"]
        resolved = self._numeric(name, spec) if lane == "numeric" else self._rows(name, spec) if lane == "rows" else self._blobs(spec)
        self._cache[name] = resolved
        return resolved

    def numeric(self, name: str) -> Any:
        if self._columns.get(name, {}).get("lane") != "numeric":
            raise TypeError(f"Capture column {name!r} is not a numeric column.")
        return self.column(name)

    def rows(self, name: str) -> RowColumn:
        if self._columns.get(name, {}).get("lane") != "rows":
            raise TypeError(f"Capture column {name!r} is not a rows column.")
        return self.column(name)

    def blob_path(self, name: str, row: int, *, stem: str | None = None) -> Path:
        spec = self._columns.get(name)
        if spec is None or spec["lane"] != "blob":
            raise TypeError(f"Capture column {name!r} is not a blob column.")
        return self.root / spec["dir"] / f"{stem or f'{row:06d}'}.{spec['ext']}"

    def routes(self) -> list[str]:
        return sorted(self.manifest.get("routes") or {})

    def slice(self, route: str) -> RouteSlice:
        fields = (self.manifest.get("routes") or {}).get(route)
        if fields is None:
            raise KeyError(f"Capture has no route {route!r} (have: {self.routes()}).")
        return RouteSlice(self, fields)

    def close(self) -> None:
        self._cache.clear()

    def _numeric(self, name: str, spec: dict[str, Any]) -> Any:
        import numpy as np

        path = self.root / spec["path"]
        row_shape = tuple(int(dim) for dim in spec["shape"])
        itemsize = np.dtype(spec["dtype"]).itemsize
        row_items = 1
        for dim in row_shape:
            row_items *= dim
        rows = path.stat().st_size // (itemsize * row_items) if row_items else 0
        if rows == 0:
            return np.empty((0, *row_shape), dtype=spec["dtype"])
        return np.memmap(path, dtype=spec["dtype"], mode="r", shape=(rows, *row_shape))

    def _blobs(self, spec: dict[str, Any]) -> list[str]:
        """A blob column reads as its file paths, in row order."""
        return [str(path) for path in sorted((self.root / spec["dir"]).glob(f"*.{spec['ext']}"))]

    def _rows(self, name: str, spec: dict[str, Any]) -> RowColumn:
        import numpy as np

        index_path = self.root / spec["index"]
        offsets = np.frombuffer(index_path.read_bytes(), dtype="<i8")
        return RowColumn(self.root / spec["path"], offsets)


def _pack_offsets(offsets: list[int]) -> bytes:
    import numpy as np

    return np.asarray(offsets, dtype="<i8").tobytes()
