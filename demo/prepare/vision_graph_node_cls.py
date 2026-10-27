"""Write random graphs whose nodes carry an image, in the TU Dortmund format plus nodes/<id>.png per split.

A node is ``red`` or ``green``, the color of its 16 px image under pixel noise, and its 4 features are pure noise,
so only the image path can learn the class. Besides the random graph (SYN_A.txt) every split carries a ring
graph over the same nodes (SYN_A_ring.txt), selected with datamodule.graphs.edges=A_ring.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = {"train": 200, "val": 40, "test": 40}
FEATURES, SIZE = 4, 16
COLORS = {"red": (200, 40, 40), "green": (40, 200, 40)}


def write_split(folder: Path, graphs: int, rng: np.random.Generator) -> None:
    (folder / "nodes").mkdir(parents=True, exist_ok=True)
    edges, ring, indicator, labels, attributes = [], [], [], [], []
    offset = 0
    for graph_id in range(1, graphs + 1):
        n = int(rng.integers(4, 13))
        adj = np.triu(rng.random((n, n)) < 0.3, 1)
        src, dst = np.nonzero(adj | adj.T)
        edges += [f"{offset + i + 1}, {offset + j + 1}" for i, j in zip(src, dst)]
        ring += [f"{offset + i + 1}, {offset + (i + d) % n + 1}" for i in range(n) for d in (1, n - 1)]
        for node in range(offset + 1, offset + n + 1):
            name = list(COLORS)[int(rng.integers(2))]
            pixels = np.clip(np.array(COLORS[name]) + rng.normal(0, 40, (SIZE, SIZE, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(folder / "nodes" / f"{node}.png")
            labels.append(name)
            attributes.append(", ".join(f"{v:.6f}" for v in rng.normal(size=FEATURES)))
        indicator += [str(graph_id)] * n
        offset += n
    for name, lines in {"A": edges, "A_ring": ring, "graph_indicator": indicator, "node_labels": labels, "node_attributes": attributes}.items():
        (folder / f"SYN_{name}.txt").write_text("\n".join(lines) + "\n")


root = Path(sys.argv[1])
rng = np.random.default_rng(0)
for split, graphs in SPLITS.items():
    write_split(root / split, graphs, rng)
print(f"graphs with node images written to {root}")
