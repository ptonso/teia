"""Write a 3-label table (Gaussian features in 4-D) as <root>/{train,val,test}.csv plus data.yaml.

The labels are ``a`` (f0 > 0), ``b`` (f1 > 0) and ``c`` (f2 + f3 > 0), each independent of the others.
"""

import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for split, n in {"train": 600, "val": 150, "test": 150}.items():
    features = rng.normal(size=(n, 4))
    labels = np.stack([features[:, 0] > 0, features[:, 1] > 0, features[:, 2] + features[:, 3] > 0], axis=1).astype(int)
    with open(root / f"{split}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f0", "f1", "f2", "f3", "a", "b", "c"])
        writer.writerows([[round(float(v), 4) for v in row] + [int(v) for v in lab] for row, lab in zip(features, labels)])
(root / "data.yaml").write_text("task: multi-cls\ntargets: [a, b, c]\nsplits:\n  train: train.csv\n  val: val.csv\n  test: test.csv\n")
print(f"table written to {root}")
