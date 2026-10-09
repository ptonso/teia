"""``FrameRows`` — the row-reader base: read a columnar container → one ``Record`` item per row.

Contract: emits schema-bound ``Record`` items keyed by ``row_id``. Format subclasses
(``CsvRows``/``ParquetRows``/``SqlRows``) only supply ``read_source``; split/manifest/target resolution
is shared here. No modality — any tabular container reuses it.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from teia.base.data import Reader
from teia.base.schema import SchemaError, Record
from teia.node.data._frame import load_manifest, resolve_root, split_frame, to_python

if TYPE_CHECKING:
    import pandas as pd


class FrameRows(Reader):
    """Row-reader base: read a columnar container → one ``Record`` item per row. See module summary."""

    default_extensions: tuple[str, ...] = (".csv",)

    def __init__(
        self,
        *,
        root: str | None = None,
        manifest: str = "data.yaml",
        split_controls: dict[str, Any] | None = None,
        id_col: str | None = None,
        key: str | None = None,
        infer_source: str | None = None,
        source: str | None = None,
        targets: list[str] | None = None,
        **reader_opts: Any,
    ) -> None:
        self.root = root
        self.manifest_name = manifest
        self.split_controls = dict(split_controls or {})
        self.id_col = id_col or (key[4:] if isinstance(key, str) and key.startswith("col:") else key)
        self.infer_source = infer_source
        self.source = source
        self.targets_override = list(targets or [])
        self.reader_opts = reader_opts
        self.data_root: str | None = None
        self.project_dir: str | None = None
        self.task: str | None = None
        self._frames: dict[str, pd.DataFrame] = {}

    def read_source(self, path: Path) -> "pd.DataFrame":
        raise NotImplementedError

    def write_source(self, frame: "pd.DataFrame", path: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support write-back.")

    def _root(self) -> Path:
        return resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)

    def _manifest(self):
        return load_manifest(self._root(), self.manifest_name)

    def _resolve_path(self, source: str) -> Path:
        path = Path(source)
        return path if path.is_absolute() else self._root() / path

    def _convention_source(self, split: str) -> str | None:
        for ext in self.default_extensions:
            if (self._root() / f"{split}{ext}").exists():
                return f"{split}{ext}"
        return None

    def _load_frame(self, split: str) -> "pd.DataFrame":
        manifest = self._manifest()
        if split == "infer" and self.infer_source:
            return self.read_source(self._resolve_path(self.infer_source)).reset_index(drop=True)
        explicit = (manifest.splits or {}).get(split) or self._convention_source(split)
        if explicit is not None:
            return self.read_source(self._resolve_path(explicit)).reset_index(drop=True)
        source = self.source or manifest.source
        if source is None:
            raise SchemaError(
                f"No source resolved for split {split!r} under {self._root()}; provide data.yaml `splits:`, "
                f"`source:`, or `<split>{self.default_extensions[0]}`."
            )
        return split_frame(self.read_source(self._resolve_path(source)), split, self.split_controls).reset_index(drop=True)

    def frame(self, split: str) -> "pd.DataFrame":
        if split not in self._frames:
            self._frames[split] = self._load_frame(split)
        return self._frames[split]

    def iter_split(self, split: str) -> Iterable[tuple[Any, Record]]:
        frame = self.frame(split)
        manifest = self._manifest()
        targets = [] if split == "infer" else (self.targets_override or list(manifest.targets or []))
        target_set = set(targets)
        id_col = self.id_col or manifest.id_col
        entity_col = manifest.entity_col
        time_col = manifest.time_col
        for idx, row in frame.iterrows():
            cells: dict[str, Any] = {}
            target: dict[str, Any] = {}
            for name in frame.columns:
                value = to_python(row[name])
                (target if name in target_set else cells)[str(name)] = value
            row_id = to_python(row[id_col]) if id_col and id_col in frame.columns else int(idx)
            record = Record(
                cells=cells,
                target=target,
                split=split,
                row_id=row_id,
                entity=to_python(row[entity_col]) if entity_col and entity_col in frame.columns else None,
                time=to_python(row[time_col]) if time_col and time_col in frame.columns else None,
                meta={
                    "entity_col": entity_col,
                    "time_col": time_col,
                    "lookback": manifest.lookback,
                    "horizon": manifest.horizon,
                    "is_lagged": bool(manifest.lag),
                    "split_controls": self.split_controls or None,
                },
            )
            yield row_id, record
