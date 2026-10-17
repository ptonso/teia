"""``LabelmeGraph`` — a board graph for a ``LabelmeReader``, appended inside each LabelMe JSON.

Format (an ``edges`` key added next to ``shapes`` in the LabelMe file)::

    {
      "imagePath": "board_001.png",
      "shapes": [{"label": "resistor", "shape_type": "rectangle", "points": [[10, 10], [40, 30]]}, ...],
      "edges": [{"src": 0, "dst": 3}, {"src": 3, "dst": 7}]
    }

``src`` and ``dst`` are indices into ``shapes``. Each undirected edge is stored once, and symmetry is
applied downstream. Extra keys on an edge object are allowed and ignored today. A file without an
``edges`` key, or an index outside ``shapes``, raises.

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


class LabelmeGraph(ReaderExtension):
    """Read the ``edges`` key of the LabelMe JSON the reader just parsed."""

    def extend(self, key: str, record: dict[str, Any], source: Path) -> dict[str, Any]:
        payload = json.loads(source.read_text())
        if "edges" not in payload:
            raise KeyError(f"{source} has no 'edges' key.")
        return {**record, "edges": edge_tensor(key, payload["edges"], record["ann_ids"])}
