"""``ReaderExtension`` — append fields to the record a reader emits, without changing the reader.

A reader owns one primary format (COCO annotations, LabelMe JSON). An extension owns one *secondary*
payload that rides along with it, such as a graph of edges between the annotated boxes. The reader
builds a ``record`` dict per sample, hands it to ``extend``, and emits the record's values in insertion
order, so an extension that appends ``edges`` adds one ``item.*`` key to the reader's ``out`` list.

An extension is a plain helper, not an data node: it has no ``in``/``out`` envelope and is mounted as the
reader's ``extension`` argument.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class ReaderExtension:
    """Append fields to a reader record. Subclasses document the file format they accept."""

    def bind(self, root: Path) -> None:
        """Receive the reader's resolved dataset root once, before the first ``extend``."""
        self.root = root

    def extend(self, key: str, record: dict[str, Any], source: Path) -> dict[str, Any]:
        """Return ``record`` with extra fields appended. ``source`` is the file the reader parsed."""
        raise NotImplementedError


def edge_tensor(key: str, edges: list[dict[str, Any]], ids: list[Any]) -> torch.Tensor:
    """``[{"src": id, "dst": id}, ...]`` → ``[E, 2]`` int64 node indices into ``ids``. Unknown ids raise."""
    index = {node_id: i for i, node_id in enumerate(ids)}
    rows = []
    for edge in edges:
        for end in ("src", "dst"):
            if edge[end] not in index:
                raise ValueError(f"graph edge {edge!r} of {key!r}: {end} {edge[end]!r} is not a node id of this sample.")
        rows.append([index[edge["src"]], index[edge["dst"]]])
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 2)
