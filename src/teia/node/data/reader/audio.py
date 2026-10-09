"""``AudioFolderReader`` — class-folder / split-prefixed audio reader.

The audio twin of ``ImageFolderReader``: it scans an audio tree once (cached under
``.teia-cache/audio-index``), derives each file's ``(split, class_name)`` from its path, and emits
per-split ``(stem, (path, class_name))``. Precomputed-feature files (``.npy``/``.pt``) are picked up
alongside encoded audio so a graph can bypass decoding. The class string is encoded to an index by a
downstream stateful ``EncodeLabels`` transform, which owns the vocabulary + ``num_classes``.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from teia.base.data import Reader
from teia.base.data.cache import IndexCacheMixin

from teia.node.data._paths import (
    AUDIO_EXTENSIONS,
    ClassNameMode,
    canonical_split_id,
    iter_files,
    resolve_class_name,
    resolve_root,
    split_and_class_parts,
)

#: Scanned suffixes: encoded audio plus precomputed-feature tensors (``DecodeAudio`` passes those through).
_SCAN_EXTENSIONS = AUDIO_EXTENSIONS | {".npy", ".pt"}


class AudioFolderReader(Reader, IndexCacheMixin):
    """Class-folder / split-prefixed audio reader; derives ``(split, class_name)`` from the path."""
    cache_subdir = "audio-index"

    def __init__(
        self,
        *,
        root: str | None = None,
        class_name_mode: ClassNameMode = "basename",
        multi_label: bool = False,
    ) -> None:
        self.root = root
        self.class_name_mode = class_name_mode
        self.multi_label = bool(multi_label)
        # Injected by the executor from shared datamodule fields.
        self.data_root: str | None = None
        self.project_dir: str | None = None
        self._index: dict[str, list[tuple[str, Path, str | None]]] | None = None

    def _root(self) -> Path:
        return resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)

    def cache_roots(self) -> list[Path]:
        root = self._root()
        split_dirs = [root / n for n in ("train", "val", "valid", "validation", "test") if (root / n).is_dir()]
        return split_dirs or [root]

    def cache_key_parts(self) -> tuple:
        return (str(self._root()), self.class_name_mode, f"audio_folder-{self.multi_label}")

    def _scan(self) -> dict[str, list[tuple[str, Path, str | None]]]:
        root = self._root()
        index: dict[str, list[tuple[str, Path, str | None]]] = {}
        for path in iter_files(root, _SCAN_EXTENSIONS):
            split, class_parts = split_and_class_parts(path.relative_to(root))
            class_name = resolve_class_name(class_parts, class_name_mode=self.class_name_mode)
            index.setdefault(canonical_split_id(split), []).append((path.stem, path, class_name))
        return index

    def _ensure_index(self) -> dict[str, list[tuple[str, Path, str | None]]]:
        if self._index is None:
            self._index = self.load_index()
            if self._index is None:
                self._index = self._scan()
                self.write_index(self._index)
        return self._index

    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        entries = self._ensure_index().get(split, [])
        if not self.multi_label:
            for stem, path, class_name in entries:
                yield stem, (str(path), class_name)
            return
        # Multi-label: same-stem files under different class dirs are one item with many labels.
        grouped: dict[str, tuple[Path, list[str]]] = {}
        for stem, path, class_name in entries:
            rep, names = grouped.setdefault(stem, (path, []))
            if class_name is not None and class_name not in names:
                names.append(class_name)
        for stem, (path, names) in grouped.items():
            yield stem, (str(path), sorted(names))
