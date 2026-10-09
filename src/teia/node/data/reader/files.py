"""``FilesReader`` — split/subdir file globber keyed by stem (image or label streams).

The generic file-per-item reader for the split-prefixed ``<root>/<split>/<subdir>/<glob>`` layout used by
yolo/coco/labelme datasets. Emits ``(stem, path)`` so a downstream ``join`` aligns image + label streams.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from teia.base.data import Reader
from teia.base.data.cache import IndexCacheMixin

from teia.node.data._paths import canonical_split_id, resolve_root

_SPLIT_DIRS = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


class FilesReader(Reader, IndexCacheMixin):
    """Glob files under ``<root>/<split>/<subdir>`` keyed by stem (image or label stream)."""
    cache_subdir = "vision-index"

    def __init__(self, *, subdir: str = "images", glob: str = "*", root: str | None = None) -> None:
        self.subdir = subdir
        self.glob = glob
        self.root = root
        self.data_root: str | None = None
        self.project_dir: str | None = None
        self.image_size: Any = None

    def _root(self) -> Path:
        return resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)

    def _split_dir(self, split: str) -> Path | None:
        root = self._root()
        for name in _SPLIT_DIRS.get(canonical_split_id(split), [split]):
            candidate = root / name / self.subdir
            if candidate.is_dir():
                return candidate
        return None

    def cache_roots(self) -> list[Path]:
        dirs = [d for split in _SPLIT_DIRS for d in [self._split_dir(split)] if d is not None]
        return dirs or [self._root()]

    def cache_key_parts(self) -> tuple:
        return (str(self._root()), self.subdir, self.glob, "files")

    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        directory = self._split_dir(split)
        if directory is None:
            return
        for path in sorted(directory.glob(self.glob)):
            if path.is_file():
                yield path.stem, str(path)
