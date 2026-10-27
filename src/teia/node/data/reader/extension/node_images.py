"""``NodeImages`` — one image file per node of a TU graph, appended to a ``TuReader`` record.

Format: next to the ``*_A.txt`` file of a split, the folder ``nodes/`` holds one image per node, named by the
1-based TU node id (``nodes/17.png`` is the image of node 17 of that split). ``images_dir`` and ``suffix`` rename
the folder and the extension. A node without an image raises, naming the first missing file.

Adds ``node_image_paths`` (one path per node, in node order) to the record.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.node.data.reader.extension.base import ReaderExtension


class NodeImages(ReaderExtension):
    """Append the image path of every node of a TU graph."""

    def __init__(self, *, images_dir: str = "nodes", suffix: str = ".png") -> None:
        self.images_dir = images_dir
        self.suffix = suffix

    def extend(self, key: Any, record: dict[str, Any], source: Path) -> dict[str, Any]:
        folder = source.parent / self.images_dir
        paths = [folder / f"{node_id}{self.suffix}" for node_id in record["node_ids"]]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"graph {key!r}: node image {missing[0]} is missing ({len(missing)} of {len(paths)} nodes).")
        return {**record, "node_image_paths": [str(path) for path in paths]}
