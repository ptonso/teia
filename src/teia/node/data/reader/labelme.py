"""``LabelmeReader`` — one item per LabelMe JSON, boxes and labels only.

Expected layout: one folder per split under the dataset root, each holding LabelMe files and their images::

    <root>/<train|val|test>/<name>.json      LabelMe JSON: imagePath, shapes[] (rectangle or polygon)
    <root>/<train|val|test>/<imagePath>      the image named by ``imagePath``

A ``rectangle`` shape gives its box directly, a ``polygon`` gives its axis-aligned bounds. Emits, per
file, ``(image_path, boxes[N,4] xywh pixels, labels: list[str])``, in this order, plus the fields any
``extension`` appends (for example ``edges`` from ``LabelmeGraph``). The label of a box is the shape
``label``, optionally renamed through ``class_map``. A label missing from a given ``class_map`` raises.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from teia.base.data import Reader
from teia.node.data._paths import iter_files, resolve_root
from teia.node.data.reader.extension.base import ReaderExtension

_SPLIT_DIRS = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}


class LabelmeReader(Reader):
    """LabelMe JSON files → one ``(image_path, boxes, labels[, ...extension fields])`` per file."""

    def __init__(
        self,
        *,
        root: str | None = None,
        class_map: dict[str, str] | None = None,
        extension: ReaderExtension | None = None,
    ) -> None:
        self.root = root
        self.class_map = dict(class_map) if class_map is not None else None
        self.extension = extension
        self.data_root: str | None = None
        self.project_dir: str | None = None

    def _root(self) -> Path:
        return resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)

    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        root = self._root()
        if self.extension is not None:
            self.extension.bind(root)
        for name in _SPLIT_DIRS.get(split, ()):
            for path in iter_files(root / name, {".json"}):
                yield path.stem, self._item(path)

    def _item(self, path: Path) -> tuple[Any, ...]:
        payload = json.loads(path.read_text())
        boxes, labels = [], []
        for shape in payload["shapes"]:
            xs, ys = zip(*shape["points"])
            boxes.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
            labels.append(shape["label"] if self.class_map is None else self.class_map[shape["label"]])
        record: dict[str, Any] = {
            "image_path": str((path.parent / payload["imagePath"]).resolve()),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": labels,
            "ann_ids": list(range(len(boxes))),
        }
        if self.extension is not None:
            record = self.extension.extend(path.stem, record, path)
        record.pop("ann_ids")
        return tuple(record.values())
