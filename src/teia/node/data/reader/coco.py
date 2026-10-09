"""``CocoReader`` — one item per image from a COCO annotation file, boxes and labels only.

Expected layout (all paths relative to the dataset root, ``data_root`` unless ``root`` is set)::

    <root>/<annotations[split]>     COCO JSON: images[], annotations[] (bbox = [x, y, w, h] pixels), categories[]
    <root>/<images_dir>/<file_name> the images named by ``images[].file_name``

``annotations`` maps each split (``train``, ``val``, ``test``) to its annotation file. Emits, per image,
``(image_path, boxes[N,4] xywh pixels, labels: list[str])``, in this order, plus the fields any
``extension`` appends (for example ``edges`` from ``CocoGraph``). The label of a box is its category
``name`` or ``supercategory`` (``label_from``), optionally renamed through ``class_map``. A label missing
from a given ``class_map`` raises.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from teia.base.data import Reader
from teia.node.data._paths import resolve_root
from teia.node.data.reader.extension.base import ReaderExtension


class CocoReader(Reader):
    """COCO detection annotations → one ``(image_path, boxes, labels[, ...extension fields])`` per image."""

    def __init__(
        self,
        *,
        annotations: dict[str, str],
        root: str | None = None,
        images_dir: str = "images",
        label_from: str = "name",
        class_map: dict[str, str] | None = None,
        extension: ReaderExtension | None = None,
    ) -> None:
        if label_from not in ("name", "supercategory"):
            raise ValueError(f"CocoReader.label_from must be 'name' or 'supercategory'; got {label_from!r}.")
        self.annotations = dict(annotations)
        self.root = root
        self.images_dir = images_dir
        self.label_from = label_from
        self.class_map = dict(class_map) if class_map is not None else None
        self.extension = extension
        self.data_root: str | None = None
        self.project_dir: str | None = None

    def _root(self) -> Path:
        return resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)

    def _label(self, category: dict[str, Any]) -> str:
        label = category[self.label_from]
        return label if self.class_map is None else self.class_map[label]

    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        if split not in self.annotations:
            return
        root = self._root()
        source = root / self.annotations[split]
        payload = json.loads(source.read_text())
        categories = {c["id"]: c for c in payload["categories"]}
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        if self.extension is not None:
            self.extension.bind(root)
        for image in sorted(payload["images"], key=lambda i: i["id"]):
            anns = sorted(by_image[image["id"]], key=lambda a: a["id"])
            record: dict[str, Any] = {
                "image_path": str(root / self.images_dir / image["file_name"]),
                "boxes": torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4),
                "labels": [self._label(categories[a["category_id"]]) for a in anns],
                "ann_ids": [a["id"] for a in anns],
            }
            if self.extension is not None:
                record = self.extension.extend(image["file_name"], record, source)
            record.pop("ann_ids")
            yield image["file_name"], tuple(record.values())
