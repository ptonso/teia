"""``TuReader`` — one item per graph from a TU Dortmund graph dataset, with node attributes and node labels.

Expected layout (``<root>/<split>/`` holds one TU dataset per split, ``<DS>`` is its name prefix)::

    <DS>_A.txt                 edges, one ``src, dst`` pair per line, 1-based node ids over the whole split
    <DS>_graph_indicator.txt   line i holds the 1-based graph id of node i
    <DS>_node_labels.txt       line i holds the class of node i, an integer or a name
    <DS>_node_attributes.txt   line i holds the comma-separated feature vector of node i

The split folder must hold exactly one ``*_graph_indicator.txt``, which names the prefix. ``edges`` picks
the edge file ``<DS>_<edges>.txt``, so several graphs over the same nodes (``<DS>_A_knn4.txt`` next to
``<DS>_A.txt``) live in one folder and a run selects one. A missing required file raises with its path.
``_graph_labels.txt`` and the edge files of the benchmark format are ignored. A split folder that does
not exist yields no items.

Emits, per graph, ``(node_feat[N,F] float32, edges[E,2] int64, labels: list[str])`` in this order, plus the
fields any ``extension`` appends. Node ids are renumbered to 0-based indices within the graph, and node labels
are strings, so a downstream ``NameListEncode`` builds the class list. TU stores each undirected edge twice,
and symmetry is applied downstream either way. An extension receives the record with ``node_ids``, the
1-based TU node id of each node, and the edge file path as its source.

Source:
  - title: "TUDataset: A collection of benchmark datasets for learning with graphs"
    url: "https://chrsmrrs.github.io/datasets/"
    year: 2020

Description:
  Reads the plain-text TU Dortmund format, the common interchange format of graph benchmark datasets,
  into one item per graph. It carries no vision: node features are the attribute vectors of the files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from teia.base.data import Reader
from teia.node.data._paths import resolve_root
from teia.node.data.reader.extension.base import ReaderExtension

_REQUIRED = ("graph_indicator", "node_labels", "node_attributes")


class TuReader(Reader):
    """TU Dortmund graph files → one ``(node_feat, edges, labels)`` per graph, edges read from ``<DS>_<edges>.txt``."""

    def __init__(self, *, root: str | None = None, edges: str = "A", extension: ReaderExtension | None = None) -> None:
        self.root = root
        self.edges = edges
        self.extension = extension
        self.data_root: str | None = None
        self.project_dir: str | None = None

    def _root(self) -> Path:
        return resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)

    @staticmethod
    def _prefix(folder: Path) -> str:
        found = sorted(folder.glob("*_graph_indicator.txt"))
        if len(found) != 1:
            raise ValueError(f"{folder} must hold exactly one '*_graph_indicator.txt' TU file; found {[p.name for p in found]}.")
        return found[0].name[: -len("_graph_indicator.txt")]

    def iter_split(self, split: str) -> Iterable[tuple[Any, Any]]:
        folder = self._root() / split
        if not folder.is_dir():
            return
        prefix = self._prefix(folder)
        paths = {"A": folder / f"{prefix}_{self.edges}.txt", **{name: folder / f"{prefix}_{name}.txt" for name in _REQUIRED}}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"TU file {path} is missing.")

        edges = np.loadtxt(paths["A"], delimiter=",", dtype=np.int64, ndmin=2).reshape(-1, 2) - 1
        graph_of = np.loadtxt(paths["graph_indicator"], dtype=np.int64, ndmin=1)
        labels = np.loadtxt(paths["node_labels"], dtype=str, ndmin=1)
        attributes = np.loadtxt(paths["node_attributes"], delimiter=",", dtype=np.float32, ndmin=2)
        if not len(graph_of) == len(labels) == len(attributes):
            raise ValueError(
                f"{folder}: graph_indicator ({len(graph_of)}), node_labels ({len(labels)}) and "
                f"node_attributes ({len(attributes)}) must have one line per node."
            )
        if np.any(np.diff(graph_of) < 0):
            raise ValueError(f"{paths['graph_indicator']} must list graph ids in non-decreasing order.")
        if np.any(graph_of[edges[:, 0]] != graph_of[edges[:, 1]]):
            raise ValueError(f"{paths['A']} holds an edge between nodes of two different graphs.")

        if self.extension is not None:
            self.extension.bind(self._root())
        ids, first = np.unique(graph_of, return_index=True)
        edge_graph = graph_of[edges[:, 0]]
        order = np.argsort(edge_graph, kind="stable")
        edges, edge_graph = edges[order], edge_graph[order]
        for graph_id, start in zip(ids.tolist(), first.tolist()):
            end = start + int(np.sum(graph_of == graph_id))
            mine = edges[edge_graph == graph_id] - start
            record: dict[str, Any] = {
                "node_feat": torch.from_numpy(attributes[start:end]),
                "edges": torch.from_numpy(mine).reshape(-1, 2),
                "labels": labels[start:end].tolist(),
                "node_ids": list(range(start + 1, end + 1)),
            }
            if self.extension is not None:
                record = self.extension.extend(graph_id, record, paths["A"])
            record.pop("node_ids")
            yield graph_id, tuple(record.values())
