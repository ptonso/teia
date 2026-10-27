"""Write random graphs in the TU Dortmund format: <root>/<split>/SYN_{A,graph_indicator,node_labels,node_attributes}.txt.

A node is class 1 when its first feature plus the mean first feature of its neighbours is positive, so a
graph model can beat a model that reads each node alone.
"""

import sys
from pathlib import Path

import numpy as np

SPLITS = {"train": 300, "val": 60, "test": 60}
FEATURES = 4


def write_split(folder: Path, graphs: int, rng: np.random.Generator) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    edges, indicator, labels, attributes = [], [], [], []
    offset = 0
    for graph_id in range(1, graphs + 1):
        n = int(rng.integers(5, 31))
        x = rng.normal(size=(n, FEATURES)).astype(np.float32)
        adj = np.triu(rng.random((n, n)) < 0.2, 1)
        adj = adj | adj.T
        degree = np.maximum(adj.sum(1), 1)
        score = x[:, 0] + (adj @ x[:, 0]) / degree
        src, dst = np.nonzero(adj)
        edges += [f"{offset + i + 1}, {offset + j + 1}" for i, j in zip(src, dst)]
        indicator += [str(graph_id)] * n
        labels += [str(int(s > 0)) for s in score]
        attributes += [", ".join(f"{v:.6f}" for v in row) for row in x]
        offset += n
    for name, lines in {"A": edges, "graph_indicator": indicator, "node_labels": labels, "node_attributes": attributes}.items():
        (folder / f"SYN_{name}.txt").write_text("\n".join(lines) + "\n")


root = Path(sys.argv[1])
rng = np.random.default_rng(0)
for split, graphs in SPLITS.items():
    write_split(root / split, graphs, rng)
print(f"graphs written to {root}")
