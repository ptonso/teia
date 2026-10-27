"""Write a regression table (a noisy linear target over Gaussian features in 4-D) as <root>/{train,val,test}.csv plus data.yaml."""

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
weights = np.array([2.0, -1.0, 0.5, 3.0])
for split, n in {"train": 600, "val": 150, "test": 150}.items():
    features = rng.normal(size=(n, 4))
    target = features @ weights + 1.0 + 0.1 * rng.normal(size=n)
    with open(root / f"{split}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f0", "f1", "f2", "f3", "y"])
        writer.writerows([[round(float(v), 4) for v in row] + [round(float(t), 4)] for row, t in zip(features, target)])
(root / "data.yaml").write_text("task: reg\ntargets: [y]\nsplits:\n  train: train.csv\n  val: val.csv\n  test: test.csv\n")
print(f"table written to {root}")
