"""``CocoGraph`` — a board graph for a ``CocoReader``, read from one ``graph.json``.

Format (one file for the whole dataset, keyed by the COCO image ``file_name``)::

    {
      "board_001.jpg": {"edges": [{"src": 12, "dst": 87}, {"src": 12, "dst": 90}]},
      "board_002.jpg": {"edges": []}
    }

``src`` and ``dst`` are COCO annotation ids (``annotations[].id``) of boxes of that image. Each
undirected edge is stored once, and symmetry is applied downstream. Extra keys on an edge object are
allowed and ignored today, so edge attributes can be added later without changing the format. An image
missing from the file, or an id that is not an annotation of the image, raises.

Adds ``edges`` (``[E, 2]`` int64 node indices, in the reader's node order) to the record.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from teia.node.data.reader.extension.base import ReaderExtension, edge_tensor


class CocoGraph(ReaderExtension):
    """COCO image ``file_name`` → edge list, from ``path`` (relative to the reader root)."""

    def __init__(self, *, path: str = "graph.json") -> None:
        self.path = path
        self._graph: dict[str, Any] | None = None

    def extend(self, key: str, record: dict[str, Any], source: Path) -> dict[str, Any]:
        if self._graph is None:
            self._graph = json.loads((self.root / self.path).read_text())
        if key not in self._graph:
            raise KeyError(f"{self.root / self.path} has no entry for image {key!r}.")
        edges = self._graph[key]["edges"]
        return {**record, "edges": edge_tensor(key, edges, record["ann_ids"])}
