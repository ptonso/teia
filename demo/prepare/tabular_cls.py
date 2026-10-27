"""Write a 3-class table (Gaussian blobs in 4-D) as <root>/{train,val,test}.csv plus data.yaml."""

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
centers = np.array([[0, 0, 0, 0], [3, 3, 0, 0], [0, 0, 3, 3]], dtype=float)
for split, n in {"train": 600, "val": 150, "test": 150}.items():
    labels = rng.integers(0, 3, n)
    features = centers[labels] + rng.normal(size=(n, 4))
    with open(root / f"{split}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f0", "f1", "f2", "f3", "label"])
        writer.writerows([[round(float(v), 4) for v in row] + [int(c)] for row, c in zip(features, labels)])
(root / "data.yaml").write_text("task: cls\ntargets: [label]\nsplits:\n  train: train.csv\n  val: val.csv\n  test: test.csv\n")
print(f"table written to {root}")
